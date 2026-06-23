"""Command line entrypoints for mlxrl."""

from __future__ import annotations

import argparse

import mlx.core as mx

from mlxrl.policy.model import DEFAULT_MODEL_ID, LoRAConfig, encode_prompt, load_policy_with_lora


def _phase0_smoke(args: argparse.Namespace) -> int:
    model, tokenizer, report = load_policy_with_lora(
        model_id=args.model,
        config=LoRAConfig(rank=args.rank, scale=args.scale, dropout=args.dropout),
    )
    tokens = encode_prompt(tokenizer, args.prompt)
    logits = model(tokens)
    mx.eval(logits)  # Phase 0 gate sync: materialize the smoke forward before reporting.

    print(f"model_id: {report.model_id}")
    print(f"layer_count: {report.layer_count}")
    print(f"lora_target_keys: {', '.join(report.lora_target_keys)}")
    print(f"quantized_linear_count: {report.quantized_linear_count}")
    print(f"lora_module_count: {report.lora_module_count}")
    print(f"expected_lora_module_count: {report.expected_lora_module_count}")
    print(f"total_params: {report.total_params}")
    print(f"trainable_params: {report.trainable_params}")
    print(f"trainable_percent: {report.trainable_percent:.6f}")
    print(f"logits_shape: {tuple(logits.shape)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="mlxrl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("phase0-smoke", help="Run the Phase 0 model-load gate.")
    smoke.add_argument("--model", default=DEFAULT_MODEL_ID)
    smoke.add_argument("--prompt", default="What is 2+2?")
    smoke.add_argument("--rank", type=int, default=8)
    smoke.add_argument("--scale", type=float, default=20.0)
    smoke.add_argument("--dropout", type=float, default=0.0)
    smoke.set_defaults(func=_phase0_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

