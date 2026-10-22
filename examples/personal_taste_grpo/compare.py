"""Compare one name from the base model with one from a GRPO adapter."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from taste_reward import (
    clean_candidate_name,
    copies_prompt_word,
    naming_prompt,
    render_no_think_prompt,
)

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def sample_name(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    source_prompt: str,
    attempts: int,
    seed: int,
    sampling: Any,
    generate_group_rollouts: Any,
) -> dict[str, Any]:
    rendered = render_no_think_prompt(tokenizer, prompt)
    completions = generate_group_rollouts(
        model=model,
        tokenizer=tokenizer,
        prompts=[rendered],
        group_size=max(1, attempts),
        config=sampling,
        seed=seed,
        use_chat_template=False,
    )
    candidates = []
    for index, completion in enumerate(completions):
        name = clean_candidate_name(completion.text)
        copied = copies_prompt_word(name, source_prompt) if name else False
        row = {"attempt": index, "name": name, "raw": completion.text, "prompt_copy": copied}
        candidates.append(row)
        if name and " " not in name and not copied:
            return {**row, "candidates": candidates}
    fallback = next((row for row in candidates if row["name"]), candidates[0])
    return {**fallback, "candidates": candidates, "fallback": True}


def load_adapter(
    model: Any, adapter_path: Path, mx: Any, tree_flatten: Any, tree_unflatten: Any
) -> int:
    arrays = mx.load(str(adapter_path))
    current = dict(tree_flatten(model.trainable_parameters()))
    leaves = []
    for saved_key, value in arrays.items():
        key = str(saved_key).replace("__", ".")
        if key not in current:
            raise ValueError(f"Adapter leaf {key!r} is absent from the model.")
        if tuple(value.shape) != tuple(current[key].shape):
            raise ValueError(
                f"Adapter shape mismatch for {key}: {value.shape} != {current[key].shape}"
            )
        leaves.append((key, value))
    missing = sorted(set(current) - {key for key, _ in leaves})
    if missing:
        raise ValueError(f"Adapter is missing {len(missing)} leaves, e.g. {missing[:3]}")
    model.update(tree_unflatten(leaves))
    mx.eval(model.trainable_parameters())
    return len(leaves)


def main() -> None:
    args = parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    run = config["run"]
    sampling_config = config["sampling"]
    seed = int(args.seed if args.seed is not None else run["seed"])

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load

    from mlxrl.policy import LoRAConfig, load_policy_with_lora
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts

    sampling = SamplingConfig(
        max_tokens=int(run["max_completion_len"]),
        temperature=float(sampling_config["temperature"]),
        top_p=float(sampling_config["top_p"]),
        min_p=float(sampling_config["min_p"]),
        top_k=int(sampling_config["top_k"]),
    )
    prompt = naming_prompt(args.prompt)
    base_model, base_tokenizer = load(str(run["model_id"]), trust_remote_code=False)
    if hasattr(base_model, "eval"):
        base_model.eval()
    base = sample_name(
        model=base_model,
        tokenizer=base_tokenizer,
        prompt=prompt,
        source_prompt=args.prompt,
        attempts=args.attempts,
        seed=seed,
        sampling=sampling,
        generate_group_rollouts=generate_group_rollouts,
    )
    del base_model
    mx.clear_cache()

    grpo_model, grpo_tokenizer, report = load_policy_with_lora(
        model_id=str(run["model_id"]),
        config=LoRAConfig(
            rank=int(run["rank"]),
            scale=float(run["scale"]),
            dropout=float(run["dropout"]),
            grad_checkpoint=bool(run["gradient_checkpointing"]),
        ),
    )
    loaded = load_adapter(grpo_model, args.adapter, mx, tree_flatten, tree_unflatten)
    grpo = sample_name(
        model=grpo_model,
        tokenizer=grpo_tokenizer,
        prompt=prompt,
        source_prompt=args.prompt,
        attempts=args.attempts,
        seed=seed,
        sampling=sampling,
        generate_group_rollouts=generate_group_rollouts,
    )
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "model": run["model_id"],
                "adapter": str(args.adapter),
                "adapter_leaves": loaded,
                "trainable_params": report.trainable_params,
                "base": base,
                "grpo": grpo,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
