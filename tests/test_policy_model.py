from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.tuner.lora import LoRALinear

from mlxrl.policy.model import (
    DEFAULT_LORA_TARGET_SUFFIXES,
    LoRAConfig,
    assert_lora_on_every_layer,
    assert_only_lora_trainable,
    count_parameters,
    inject_lora_adapters,
    lora_module_counts_by_layer,
    strict_lora_config,
)


class ToyDenseAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class ToyDenseMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class ToyDenseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(4)
        self.self_attn = ToyDenseAttention()
        self.post_attention_layernorm = nn.RMSNorm(4)
        self.mlp = ToyDenseMlp()


class ToyLinearAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dt_bias = mx.zeros(shape=(4,))
        self.in_proj_qkv = nn.Linear(4, 12, bias=False)
        self.in_proj_z = nn.Linear(4, 4, bias=False)
        self.norm = nn.RMSNorm(4)


class ToyHybridLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_attn = ToyLinearAttention()


class ToyModel(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = layers


def test_assert_only_lora_trainable_accepts_adapter_leaves() -> None:
    params = {
        "layers": [
            {
                "self_attn": {
                    "q_proj": {
                        "lora_a": mx.array([1.0, 2.0]),
                        "lora_b": mx.array([0.0]),
                    }
                }
            }
        ]
    }

    paths = assert_only_lora_trainable(params)

    assert paths == (
        "layers.0.self_attn.q_proj.lora_a",
        "layers.0.self_attn.q_proj.lora_b",
    )
    assert count_parameters(params) == 3


def test_assert_only_lora_trainable_rejects_base_weights() -> None:
    params = {"layers": [{"self_attn": {"q_proj": {"weight": mx.array([1.0])}}}]}

    with pytest.raises(ValueError, match="Only LoRA adapter params may be trainable"):
        assert_only_lora_trainable(params)


def test_assert_only_lora_trainable_rejects_empty_tree() -> None:
    with pytest.raises(ValueError, match="No trainable parameters"):
        assert_only_lora_trainable({})


def test_inject_lora_adapters_auto_supports_heterogeneous_layers() -> None:
    model = ToyModel([ToyHybridLayer(), ToyDenseLayer(), ToyHybridLayer()])

    target_keys = inject_lora_adapters(model, LoRAConfig(rank=2, scale=2.0))

    assert set(target_keys) == {
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    }
    assert lora_module_counts_by_layer(model) == (2, 7, 2)
    assert_lora_on_every_layer(model)
    hybrid_layer = cast(Any, model.layers[0])
    dense_layer = cast(Any, model.layers[1])
    assert isinstance(hybrid_layer.linear_attn.in_proj_qkv, LoRALinear)
    assert isinstance(dense_layer.self_attn.q_proj, LoRALinear)
    assert count_parameters(model.trainable_parameters()) > 0
    assert_only_lora_trainable(model)


def test_inject_lora_adapters_explicit_mode_keeps_homogeneous_dense_targets() -> None:
    model = ToyModel([ToyDenseLayer(), ToyDenseLayer()])

    target_keys = inject_lora_adapters(model, strict_lora_config(rank=2))

    assert set(target_keys) == {
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    }
    assert lora_module_counts_by_layer(model) == (7, 7)


def test_inject_lora_adapters_explicit_mode_stays_strict_on_hybrid_layers() -> None:
    model = ToyModel([ToyHybridLayer(), ToyDenseLayer()])

    with pytest.raises(ValueError, match="Layer 0 is missing LoRA target modules"):
        inject_lora_adapters(
            model,
            LoRAConfig(rank=2, target_suffixes=DEFAULT_LORA_TARGET_SUFFIXES),
        )
