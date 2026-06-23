"""Model loading and QLoRA adapter setup for the Phase 0 gate."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import get_total_parameters as get_mlx_lm_total_parameters

DEFAULT_MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_TRAINABLE_LEAVES = frozenset({"lora_a", "lora_b"})


@dataclass(frozen=True)
class LoRAConfig:
    """Minimal LoRA knobs for Phase 0."""

    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0
    target_suffixes: tuple[str, ...] = DEFAULT_LORA_TARGET_SUFFIXES


@dataclass(frozen=True)
class Phase0Report:
    """Counts emitted by the Phase 0 smoke gate."""

    model_id: str
    layer_count: int
    lora_target_keys: tuple[str, ...]
    quantized_linear_count: int
    lora_module_count: int
    expected_lora_module_count: int
    total_params: int
    trainable_params: int
    trainable_percent: float
    trainable_paths: tuple[str, ...]


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Return the transformer block list exposed by MLX-LM models."""

    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose a transformer layer list.")
    return list(layers)


def discover_lora_target_keys(
    model: nn.Module,
    target_suffixes: Iterable[str] = DEFAULT_LORA_TARGET_SUFFIXES,
) -> tuple[str, ...]:
    """Discover per-layer module paths whose final segment matches LoRA targets."""

    suffixes = tuple(target_suffixes)
    keys: set[str] = set()
    for layer in get_transformer_layers(model):
        for name, module in layer.named_modules():
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)) and name.endswith(suffixes):
                keys.add(name)
    if not keys:
        suffix_text = ", ".join(suffixes)
        raise ValueError(f"No LoRA target modules found for suffixes: {suffix_text}")
    return tuple(sorted(keys))


def assert_targets_on_every_layer(model: nn.Module, target_keys: Iterable[str]) -> None:
    """Ensure the discovered target set exists on every transformer block."""

    expected = set(target_keys)
    for index, layer in enumerate(get_transformer_layers(model)):
        layer_keys = {name for name, _ in layer.named_modules()}
        missing = sorted(expected.difference(layer_keys))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Layer {index} is missing LoRA target modules: {missing_text}")


def inject_lora_adapters(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Inject MLX-LM LoRA layers into all attention and MLP projections."""

    target_keys = discover_lora_target_keys(model, config.target_suffixes)
    assert_targets_on_every_layer(model, target_keys)
    layer_count = len(get_transformer_layers(model))
    linear_to_lora_layers(
        model,
        num_layers=layer_count,
        config={
            "rank": config.rank,
            "scale": config.scale,
            "dropout": config.dropout,
            "keys": set(target_keys),
        },
    )
    return target_keys


def trainable_leaf_paths(model_or_params: Any) -> tuple[str, ...]:
    """Flatten trainable params from a module or already-filtered parameter tree."""

    params = (
        model_or_params.trainable_parameters()
        if hasattr(model_or_params, "trainable_parameters")
        else model_or_params
    )
    return tuple(str(path) for path, _ in tree_flatten(params))


def count_parameters(params: Any) -> int:
    """Count array elements in a parameter tree."""

    return int(sum(cast(Any, array).size for _, array in tree_flatten(params)))


def assert_only_lora_trainable(model_or_params: Any) -> tuple[str, ...]:
    """Assert every trainable leaf is a LoRA adapter tensor."""

    paths = trainable_leaf_paths(model_or_params)
    if not paths:
        raise ValueError("No trainable parameters found after LoRA setup.")
    invalid = [
        path
        for path in paths
        if path.rsplit(".", maxsplit=1)[-1] not in LORA_TRAINABLE_LEAVES
    ]
    if invalid:
        invalid_text = ", ".join(invalid)
        raise ValueError(f"Only LoRA adapter params may be trainable; found: {invalid_text}")
    return paths


def adapter_param_tree(model: nn.Module) -> dict[str, Any]:
    """Return the currently trainable adapter parameter tree."""

    return model.trainable_parameters()


def count_modules(model: nn.Module, module_type: type[Any]) -> int:
    """Count modules by type across an MLX module tree."""

    return sum(1 for _, module in model.named_modules() if isinstance(module, module_type))


def build_phase0_report(
    model_id: str,
    model: nn.Module,
    target_keys: tuple[str, ...],
) -> Phase0Report:
    """Build the Phase 0 gate report after adapter injection and freezing."""

    trainable_paths = assert_only_lora_trainable(model)
    layer_count = len(get_transformer_layers(model))
    total_params = int(get_mlx_lm_total_parameters(model))
    trainable_params = count_parameters(model.trainable_parameters())
    expected_lora_module_count = layer_count * len(target_keys)
    lora_module_count = count_modules(model, LoRALinear)
    if lora_module_count != expected_lora_module_count:
        raise ValueError(
            "Unexpected LoRA module count: "
            f"expected {expected_lora_module_count}, found {lora_module_count}"
        )
    if trainable_params <= 0:
        raise ValueError("LoRA setup produced zero trainable parameters.")

    return Phase0Report(
        model_id=model_id,
        layer_count=layer_count,
        lora_target_keys=target_keys,
        quantized_linear_count=count_modules(model, nn.QuantizedLinear),
        lora_module_count=lora_module_count,
        expected_lora_module_count=expected_lora_module_count,
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_percent=(trainable_params / total_params) * 100.0,
        trainable_paths=trainable_paths,
    )


def load_policy_with_lora(
    model_id: str = DEFAULT_MODEL_ID,
    config: LoRAConfig | None = None,
) -> tuple[nn.Module, Any, Phase0Report]:
    """Load a 4-bit MLX-LM policy and attach trainable LoRA adapters."""

    lora_config = config or LoRAConfig()
    loaded = load(model_id)
    model = loaded[0]
    tokenizer = loaded[1]
    target_keys = inject_lora_adapters(model, lora_config)
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])
    model.eval()
    report = build_phase0_report(model_id, model, target_keys)
    return model, tokenizer, report


def encode_prompt(tokenizer: Any, prompt: str) -> mx.array:
    """Encode a smoke-test prompt into a batch of token ids."""

    text = prompt
    if getattr(tokenizer, "chat_template", None) is not None:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("Tokenizer produced no tokens for the smoke prompt.")
    return mx.array([token_ids], dtype=mx.int32)
