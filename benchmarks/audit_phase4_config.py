#!/usr/bin/env python
"""Audit Phase 4 benchmark configs and fast-path availability.

Attribution: inspects installed mlx-tune and mlx-lm-lora sources to classify
benchmark comparability. No third-party implementation code is vendored here.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from benchmarks.run_phase4 import (
        DEFAULT_TARGETS,
        add_common_args,
        dist_version,
        version_at_least,
    )
except ModuleNotFoundError:
    from run_phase4 import (  # type: ignore[no-redef]
        DEFAULT_TARGETS,
        add_common_args,
        dist_version,
        version_at_least,
    )


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GRPO_TARGETS = ("mlxrl", "mlx-tune", "mlx-lm-lora")
GEN_ONLY_TARGETS = ("mlx-lm", "mlx-lm-g4")


@dataclass(frozen=True)
class ConfigAuditRow:
    target: str
    status: str
    version: str
    model: str
    load_4bit: str
    batch_size: str
    group_size: str
    max_context: str
    max_completion: str
    temperature: str
    top_p: str
    top_k: str
    min_p: str
    beta: str
    lora_targets: str
    lora_layers: str
    fast_path: str
    caveat: str


@dataclass(frozen=True)
class FastPathInspection:
    mlx_tune_prefix_cache: bool
    mlx_tune_stream_generate: bool
    mlx_tune_generation_uses_top_p: bool
    mlx_tune_generation_uses_top_k: bool
    mlx_tune_generation_uses_min_p: bool
    mlx_lm_lora_batch_generate: bool
    mlx_lm_lora_sampler_knobs: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audit_phase4_config.py")
    add_common_args(parser)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--output", default="benchmarks/results/gate4_config_audit.md")
    parser.add_argument("--json-output", default="benchmarks/results/gate4_config_audit.json")
    return parser


def inspect_fast_paths() -> FastPathInspection:
    import mlx_lm_lora.trainer.grpo_trainer as lora_grpo
    import mlx_tune.rl_trainers as tune_rl
    from mlx_tune import GRPOTrainer

    tune_trainer_source = inspect.getsource(GRPOTrainer)
    tune_generate_source = inspect.getsource(tune_rl.generate_with_log_probs)
    lora_generate_source = inspect.getsource(lora_grpo.generate_grpo)
    return FastPathInspection(
        mlx_tune_prefix_cache=(
            "build_shared_prompt_cache" in tune_trainer_source
            and "fork_prompt_cache" in tune_trainer_source
            and "prompt_cache=fork" in tune_trainer_source
        ),
        mlx_tune_stream_generate="stream_generate" in tune_generate_source,
        mlx_tune_generation_uses_top_p="top_p" in tune_generate_source,
        mlx_tune_generation_uses_top_k="top_k" in tune_generate_source,
        mlx_tune_generation_uses_min_p="min_p" in tune_generate_source,
        mlx_lm_lora_batch_generate="batch_generate" in lora_generate_source,
        mlx_lm_lora_sampler_knobs=all(
            token in lora_generate_source for token in ("top_p", "top_k", "min_p")
        ),
    )


def build_audit_rows(
    args: argparse.Namespace,
    targets: Sequence[str],
    inspection: FastPathInspection,
) -> list[ConfigAuditRow]:
    rows: list[ConfigAuditRow] = []
    for target in targets:
        if target == "mlxrl":
            rows.append(mlxrl_row(args))
        elif target == "mlx-tune":
            rows.append(mlx_tune_row(args, inspection))
        elif target == "mlx-lm-lora":
            rows.append(mlx_lm_lora_row(args, inspection))
        elif target == "mlx-lm":
            rows.append(mlx_lm_row(args, completions_per_prompt=1))
        elif target == "mlx-lm-g4":
            rows.append(mlx_lm_row(args, completions_per_prompt=args.group_size, target=target))
        else:
            rows.append(
                ConfigAuditRow(
                    target=target,
                    status="fail",
                    version="-",
                    model=args.model,
                    load_4bit="-",
                    batch_size="-",
                    group_size="-",
                    max_context="-",
                    max_completion="-",
                    temperature="-",
                    top_p="-",
                    top_k="-",
                    min_p="-",
                    beta="-",
                    lora_targets="-",
                    lora_layers="-",
                    fast_path="-",
                    caveat="unknown benchmark target",
                )
            )
    return rows


def mlxrl_row(args: argparse.Namespace) -> ConfigAuditRow:
    return ConfigAuditRow(
        target="mlxrl",
        status="match",
        version=dist_version("mlxrl") or "local",
        model=args.model,
        load_4bit="yes",
        batch_size=str(args.batch_size),
        group_size=str(args.group_size),
        max_context=str(args.max_context),
        max_completion=str(args.max_tokens),
        temperature=str(args.temperature),
        top_p=str(args.top_p),
        top_k=str(args.top_k),
        min_p=str(args.min_p),
        beta=str(args.beta),
        lora_targets=", ".join(LORA_TARGETS),
        lora_layers="all transformer layers",
        fast_path="prefix-cached grouped rollout, compiled decode",
        caveat=common_wired_note(args),
    )


def mlx_tune_row(args: argparse.Namespace, inspection: FastPathInspection) -> ConfigAuditRow:
    installed_version = dist_version("mlx-tune")
    version = installed_version or "-"
    fast_path_ok = (
        version_available_or_at_least(installed_version, "0.5.1")
        and inspection.mlx_tune_prefix_cache
        and inspection.mlx_tune_stream_generate
    )
    sampler_mismatch = not (
        inspection.mlx_tune_generation_uses_top_p
        and inspection.mlx_tune_generation_uses_top_k
        and inspection.mlx_tune_generation_uses_min_p
    )
    caveats = [common_wired_note(args)]
    if sampler_mismatch:
        caveats.append(
            "installed mlx-tune generation consumes temperature only; "
            "top_p/top_k/min_p are not used"
        )
    return ConfigAuditRow(
        target="mlx-tune",
        status="package-speed only" if fast_path_ok else "fail",
        version=version,
        model=args.model,
        load_4bit="yes",
        batch_size=str(args.batch_size),
        group_size=str(args.group_size),
        max_context=str(args.max_context),
        max_completion=str(args.max_tokens),
        temperature=str(args.temperature),
        top_p=f"{args.top_p} (configured, unused by generation)",
        top_k=f"{args.top_k} (configured, unused by generation)",
        min_p=f"{args.min_p} (configured, unused by generation)",
        beta=str(args.beta),
        lora_targets=", ".join(LORA_TARGETS),
        lora_layers="all target modules via get_peft_model",
        fast_path="prefix cache across group + stream_generate"
        if fast_path_ok
        else "prefix-cache fast path not confirmed",
        caveat="; ".join(caveat for caveat in caveats if caveat),
    )


def mlx_lm_lora_row(args: argparse.Namespace, inspection: FastPathInspection) -> ConfigAuditRow:
    installed_version = dist_version("mlx-lm-lora")
    version = installed_version or "-"
    config_ok = (
        version_available_or_at_least(installed_version, "2.1.0")
        and inspection.mlx_lm_lora_sampler_knobs
    )
    return ConfigAuditRow(
        target="mlx-lm-lora",
        status="package-speed only" if config_ok else "fail",
        version=version,
        model=args.model,
        load_4bit="yes",
        batch_size=str(args.batch_size),
        group_size=str(args.group_size),
        max_context=str(args.max_context),
        max_completion=str(args.max_tokens),
        temperature=str(args.temperature),
        top_p=str(args.top_p),
        top_k=str(args.top_k),
        min_p=str(args.min_p),
        beta=str(args.beta),
        lora_targets="package all-LoRA config",
        lora_layers="num_layers=-1",
        fast_path="mlx-lm batch_generate with sampler knobs"
        if inspection.mlx_lm_lora_batch_generate
        else "batch generation not confirmed",
        caveat=join_caveats(
            common_wired_note(args),
            "uses package GRPO/reference semantics, "
            "not mlxrl single-model adapter-disable reference",
        ),
    )


def mlx_lm_row(
    args: argparse.Namespace,
    completions_per_prompt: int,
    target: str = "mlx-lm",
) -> ConfigAuditRow:
    return ConfigAuditRow(
        target=target,
        status="gen-only",
        version=dist_version("mlx-lm") or "-",
        model=args.model,
        load_4bit="yes",
        batch_size=str(args.batch_size),
        group_size=str(completions_per_prompt),
        max_context=str(args.max_context),
        max_completion=str(args.max_tokens),
        temperature=str(args.temperature),
        top_p=str(args.top_p),
        top_k=str(args.top_k),
        min_p=str(args.min_p),
        beta="n/a",
        lora_targets="n/a",
        lora_layers="n/a",
        fast_path="mlx-lm stream_generate",
        caveat=join_caveats(common_wired_note(args), "generation-only baseline"),
    )


def common_wired_note(args: argparse.Namespace) -> str:
    if args.wired_limit_mb > 0:
        return f"wired_limit_mb={args.wired_limit_mb}"
    return "wired_limit_mb not overridden by audit command"


def join_caveats(*values: str) -> str:
    return "; ".join(value for value in values if value)


def version_available_or_at_least(version: str | None, minimum: str) -> bool:
    if version is None:
        return True
    return version_at_least(version, minimum)


def render_markdown(rows: Sequence[ConfigAuditRow]) -> str:
    headers = [
        "target",
        "status",
        "version",
        "model",
        "4bit",
        "batch",
        "G",
        "ctx",
        "max comp",
        "temp",
        "top_p",
        "top_k",
        "min_p",
        "beta",
        "LoRA",
        "fast path",
        "caveat",
    ]
    rendered_rows = [
        [
            row.target,
            row.status,
            row.version,
            row.model,
            row.load_4bit,
            row.batch_size,
            row.group_size,
            row.max_context,
            row.max_completion,
            row.temperature,
            row.top_p,
            row.top_k,
            row.min_p,
            row.beta,
            row.lora_layers,
            row.fast_path,
            row.caveat,
        ]
        for row in rows
    ]
    lines = [
        "# Gate 4 Config Audit",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(table_cell(value) for value in row) + " |"
        for row in rendered_rows
    )
    return "\n".join(lines) + "\n"


def table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    targets = tuple(target.strip() for target in args.targets.split(",") if target.strip())
    inspection = inspect_fast_paths()
    rows = build_audit_rows(args, targets, inspection)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows), encoding="utf-8")
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {output}")
    print(f"json: {json_output}")
    if any(row.status == "fail" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
