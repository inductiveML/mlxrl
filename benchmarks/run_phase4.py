#!/usr/bin/env python
"""Phase 4 benchmark harness for mlxrl and MLX baselines.

Rollout tok/s is defined once for every target: generated completion tokens
only, excluding prompt/prefix tokens and warmup/failed steps, divided by the
wall-clock seconds spent in the generation phase.

The controller interleaves full benchmark passes across targets, e.g.
``mlxrl, mlx-lm, mlx-lm-g4, mlx-tune, mlx-lm-lora, mlxrl, ...``. Built-in
workers cover ``mlxrl``, ``mlx-lm``, and ``mlx-lm-g4``. External RL baselines
are command adapters: the command should either write JSON to
``$MLXRL_BENCH_OUTPUT`` or print JSON on stdout with the same metric fields used
by this script.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = ("mlxrl", "mlx-lm", "mlx-lm-g4", "mlx-tune", "mlx-lm-lora")
DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
MIN_SYNCED_GRADIENT_STEP_SECONDS = 0.010
JSON_FIELDS = {
    "target",
    "pass_index",
    "status",
    "model",
    "steps",
    "warmup_steps",
    "generated_completion_tokens",
    "prompt_tokens",
    "total_forward_tokens",
    "tok_s_denominator",
    "rollout_tokens",
    "rollout_seconds",
    "rollout_tok_s",
    "gradient_steps",
    "gradient_seconds",
    "gradient_step_s",
    "samples",
    "end_to_end_seconds",
    "samples_s",
    "it_s",
    "peak_memory_gb",
    "mlx_version",
    "mlx_lm_version",
    "tool_version",
    "note",
}


@dataclass(frozen=True)
class BenchResult:
    target: str
    pass_index: int
    status: str
    model: str
    steps: int
    warmup_steps: int
    generated_completion_tokens: int = 0
    prompt_tokens: int = 0
    total_forward_tokens: int = 0
    tok_s_denominator: int = 0
    rollout_tokens: int = 0
    rollout_seconds: float = 0.0
    rollout_tok_s: float | None = None
    gradient_steps: int = 0
    gradient_seconds: float = 0.0
    gradient_step_s: float | None = None
    samples: int = 0
    end_to_end_seconds: float = 0.0
    samples_s: float | None = None
    it_s: float | None = None
    peak_memory_gb: float | None = None
    mlx_version: str | None = None
    mlx_lm_version: str | None = None
    tool_version: str | None = None
    note: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_phase4.py")
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("run", help="Run interleaved benchmark passes.")
    add_common_args(controller)
    controller.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated targets. Built-ins: mlxrl, mlx-lm, mlx-lm-g4.",
    )
    controller.add_argument("--passes", type=int, default=2)
    controller.add_argument("--output", default="benchmarks/results/phase4_results.jsonl")
    controller.add_argument("--summary", default="benchmarks/results/phase4_summary.md")
    controller.add_argument("--allow-missing-baselines", action="store_true")
    controller.add_argument("--mlx-tune-command", default=None)
    controller.add_argument("--mlx-lm-lora-command", default=None)
    controller.add_argument("--require-iogpu-wired-limit", action="store_true")
    controller.set_defaults(func=run_controller)

    worker = subparsers.add_parser("worker", help="Internal worker entrypoint.")
    add_common_args(worker)
    worker.add_argument("--target", choices=["mlxrl", "mlx-lm", "mlx-lm-g4"], required=True)
    worker.add_argument("--pass-index", type=int, required=True)
    worker.add_argument("--output", required=True)
    worker.set_defaults(func=run_worker)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--wired-limit-mb", type=int, default=0)


def run_controller(args: argparse.Namespace) -> int:
    targets = tuple(target.strip() for target in args.targets.split(",") if target.strip())
    validate_controller_args(args, targets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []
    with output.open("w", encoding="utf-8") as handle:
        for pass_index in range(args.passes):
            for target in targets:
                result = run_one_target(args, target, pass_index)
                results.append(result)
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                handle.flush()
                print(format_result_line(result), flush=True)

    summary.write_text(render_summary(results), encoding="utf-8")
    print(f"\nsummary: {summary}")
    print(f"jsonl: {output}")
    if any(result.status != "ok" for result in results):
        return 0 if args.allow_missing_baselines else 1
    return 0


def validate_controller_args(args: argparse.Namespace, targets: Sequence[str]) -> None:
    if args.passes < 1:
        raise SystemExit("--passes must be at least 1.")
    if args.warmup_steps < 0 or args.steps < 1:
        raise SystemExit("--steps must be positive and --warmup-steps non-negative.")
    if args.warmup_steps >= args.steps:
        raise SystemExit("--warmup-steps must be smaller than --steps.")
    if args.batch_size != 1:
        raise SystemExit("Phase 4 standard config requires --batch-size 1.")
    if args.require_iogpu_wired_limit and args.wired_limit_mb:
        current = read_iogpu_wired_limit_mb()
        if current is not None and current != args.wired_limit_mb:
            raise SystemExit(
                "iogpu.wired_limit_mb mismatch: "
                f"expected {args.wired_limit_mb}, observed {current}."
            )
    unknown = [target for target in targets if target not in DEFAULT_TARGETS]
    if unknown:
        raise SystemExit(f"Unknown target(s): {', '.join(unknown)}")


def run_one_target(args: argparse.Namespace, target: str, pass_index: int) -> BenchResult:
    if target in {"mlxrl", "mlx-lm", "mlx-lm-g4"}:
        return run_internal_worker(args, target, pass_index)
    command = external_command(args, target)
    if command is None:
        return missing_result(args, target, pass_index, "baseline command not configured")
    return run_external_command(args, target, pass_index, command)


def run_internal_worker(args: argparse.Namespace, target: str, pass_index: int) -> BenchResult:
    with tempfile.TemporaryDirectory(prefix="mlxrl-bench-") as tmpdir:
        output = Path(tmpdir) / f"{target}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--target",
            target,
            "--pass-index",
            str(pass_index),
            "--output",
            str(output),
            *common_worker_args(args),
        ]
        env = bench_env(args, output)
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            note = (completed.stderr or completed.stdout).strip()[-500:]
            return missing_result(args, target, pass_index, note or "worker failed")
        result = result_from_json(output.read_text(encoding="utf-8"), target, pass_index)
        return enforce_gradient_timing_sanity(result)


def run_external_command(
    args: argparse.Namespace,
    target: str,
    pass_index: int,
    command_template: str,
) -> BenchResult:
    with tempfile.TemporaryDirectory(prefix="mlxrl-bench-") as tmpdir:
        output = Path(tmpdir) / f"{target}.json"
        command = expand_command(command_template, args, target, pass_index, output)
        env = bench_env(args, output)
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            note = (completed.stderr or completed.stdout).strip()[-500:]
            return missing_result(args, target, pass_index, note or "baseline failed")
        if output.exists():
            result = result_from_json(output.read_text(encoding="utf-8"), target, pass_index)
        else:
            result = result_from_json(completed.stdout, target, pass_index)
        return enforce_gradient_timing_sanity(validate_external_versions(result))


def common_worker_args(args: argparse.Namespace) -> list[str]:
    return [
        "--model",
        args.model,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--group-size",
        str(args.group_size),
        "--max-context",
        str(args.max_context),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--min-p",
        str(args.min_p),
        *(
            ["--use-chat-template"]
            if getattr(args, "use_chat_template", False)
            else []
        ),
        "--seed",
        str(args.seed),
        "--beta",
        str(args.beta),
        "--learning-rate",
        str(args.learning_rate),
        "--rank",
        str(args.rank),
        "--scale",
        str(args.scale),
        "--dropout",
        str(args.dropout),
        "--wired-limit-mb",
        str(args.wired_limit_mb),
    ]


def bench_env(args: argparse.Namespace, output: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MLXRL_BENCH_OUTPUT"] = str(output)
    env["MLXRL_BENCH_MODEL"] = args.model
    env["MLXRL_BENCH_STEPS"] = str(args.steps)
    env["MLXRL_BENCH_WARMUP_STEPS"] = str(args.warmup_steps)
    env["MLXRL_BENCH_BATCH_SIZE"] = str(args.batch_size)
    env["MLXRL_BENCH_GROUP_SIZE"] = str(args.group_size)
    env["MLXRL_BENCH_MAX_CONTEXT"] = str(args.max_context)
    env["MLXRL_BENCH_MAX_TOKENS"] = str(args.max_tokens)
    env["MLXRL_BENCH_TOP_K"] = str(args.top_k)
    env["MLXRL_BENCH_MIN_P"] = str(args.min_p)
    env["MLXRL_BENCH_WIRED_LIMIT_MB"] = str(args.wired_limit_mb)
    return env


def external_command(args: argparse.Namespace, target: str) -> str | None:
    default_command = (
        "{python} benchmarks/external_baseline_worker.py "
        "--target {target} "
        "--pass-index {pass_index} "
        "--output {output} "
        "--model {model} "
        "--steps {steps} "
        "--warmup-steps {warmup_steps} "
        "--batch-size {batch_size} "
        "--group-size {group_size} "
        "--max-context {max_context} "
        "--max-tokens {max_tokens} "
        "--temperature {temperature} "
        "--top-p {top_p} "
        "--top-k {top_k} "
        "--min-p {min_p} "
        "--seed {seed} "
        "--beta {beta} "
        "--learning-rate {learning_rate} "
        "--rank {rank} "
        "--scale {scale} "
        "--dropout {dropout} "
        "--wired-limit-mb {wired_limit_mb}"
    )
    if target == "mlx-tune":
        return args.mlx_tune_command or default_command
    if target == "mlx-lm-lora":
        return args.mlx_lm_lora_command or default_command
    return None


def expand_command(
    command_template: str,
    args: argparse.Namespace,
    target: str,
    pass_index: int,
    output: Path,
) -> list[str]:
    values = {
        "target": target,
        "python": sys.executable,
        "pass_index": pass_index,
        "output": str(output),
        "model": args.model,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "max_context": args.max_context,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "scale": args.scale,
        "dropout": args.dropout,
        "wired_limit_mb": args.wired_limit_mb,
    }
    return [part.format(**values) for part in shlex.split(command_template)]


def run_worker(args: argparse.Namespace) -> int:
    if args.target == "mlxrl":
        result = run_mlxrl_worker(args)
    elif args.target in {"mlx-lm", "mlx-lm-g4"}:
        result = run_mlx_lm_worker(args)
    else:
        raise SystemExit(f"Unknown worker target {args.target!r}.")
    Path(args.output).write_text(json.dumps(asdict(result), sort_keys=True), encoding="utf-8")
    return 0


def run_mlxrl_worker(args: argparse.Namespace) -> BenchResult:
    import mlx.core as mx
    import mlx.optimizers as optim

    from mlxrl.algo.grpo import GRPOAlgorithm
    from mlxrl.cli import _completion_rewards, _gsm8k_prompt
    from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
    from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
    from mlxrl.rollout.naive import SamplingConfig
    from mlxrl.rollout.optimized import generate_prefix_cached_group_rollouts
    from mlxrl.train.grpo import batch_from_rollouts, optimizer_step

    set_mlx_wired_limit(mx, args.wired_limit_mb)
    model, tokenizer, _ = load_policy_with_lora(
        model_id=args.model,
        config=LoRAConfig(rank=args.rank, scale=args.scale, dropout=args.dropout),
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    pad_token_id = pad_token_id_from_tokenizer(tokenizer)
    sampling = SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
    )
    algorithm = GRPOAlgorithm()

    rollout_seconds = 0.0
    gradient_seconds = 0.0
    generated_completion_tokens = 0
    prompt_tokens = 0
    samples = 0
    measured_start = 0.0
    for step in range(args.steps):
        if step == args.warmup_steps:
            mx.synchronize()  # Benchmark sync: discard warmup work before timing.
            mx.reset_peak_memory()
            measured_start = time.perf_counter()
        prompt, answer = _gsm8k_prompt(args, step)
        rollout_start = time.perf_counter()
        completions = generate_prefix_cached_group_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            group_size=args.group_size,
            config=sampling,
            seed=args.seed + step,
            use_chat_template=False,
            compile_decode_step=True,
            batch_groups=True,
        )
        mx.synchronize()  # Benchmark sync: finish rollout kernels before phase timing.
        rollout_elapsed = time.perf_counter() - rollout_start

        rewards, _, _ = _completion_rewards(completions, [answer])
        gradient_start = time.perf_counter()
        batch = batch_from_rollouts(
            model=model,
            completions=completions,
            rewards=rewards,
            group_size=args.group_size,
            pad_token_id=pad_token_id,
            compute_reference=args.beta != 0.0,
            algorithm=algorithm,
        )
        metrics = optimizer_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            beta=args.beta,
            pad_token_id=pad_token_id,
            algorithm=algorithm,
        )
        del metrics
        mx.synchronize()  # Benchmark sync: finish optimizer step before phase timing.
        gradient_elapsed = time.perf_counter() - gradient_start

        if step >= args.warmup_steps:
            rollout_seconds += rollout_elapsed
            gradient_seconds += gradient_elapsed
            generated_completion_tokens += generated_tokens_from_rollouts(completions)
            prompt_tokens += prompt_tokens_from_rollouts(completions)
            samples += len(completions)

    mx.synchronize()  # Benchmark sync: finish measured run before memory read.
    measured_seconds = time.perf_counter() - measured_start
    measured_steps = args.steps - args.warmup_steps
    tok_s_denominator = generated_completion_tokens
    return BenchResult(
        target="mlxrl",
        pass_index=args.pass_index,
        status="ok",
        model=args.model,
        steps=measured_steps,
        warmup_steps=args.warmup_steps,
        generated_completion_tokens=generated_completion_tokens,
        prompt_tokens=prompt_tokens,
        total_forward_tokens=prompt_tokens + generated_completion_tokens,
        tok_s_denominator=tok_s_denominator,
        rollout_tokens=generated_completion_tokens,
        rollout_seconds=rollout_seconds,
        rollout_tok_s=safe_div(tok_s_denominator, rollout_seconds),
        gradient_steps=measured_steps,
        gradient_seconds=gradient_seconds,
        gradient_step_s=safe_div(gradient_seconds, measured_steps),
        samples=samples,
        end_to_end_seconds=measured_seconds,
        samples_s=safe_div(samples, measured_seconds),
        it_s=safe_div(measured_steps, measured_seconds),
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        mlx_version=dist_version("mlx"),
        mlx_lm_version=dist_version("mlx-lm"),
        tool_version=dist_version("mlxrl"),
    )


def run_mlx_lm_worker(args: argparse.Namespace) -> BenchResult:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    from mlxrl.cli import _gsm8k_prompt

    set_mlx_wired_limit(mx, args.wired_limit_mb)
    mx.random.seed(args.seed)
    model, tokenizer = mlx_lm.load(args.model)
    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
    )
    completions_per_prompt = args.group_size if args.target == "mlx-lm-g4" else 1
    rollout_seconds = 0.0
    generated_completion_tokens = 0
    prompt_tokens = 0
    samples = 0
    measured_start = 0.0
    for step in range(args.steps):
        if step == args.warmup_steps:
            mx.synchronize()  # Benchmark sync: discard warmup generation before timing.
            mx.reset_peak_memory()
            measured_start = time.perf_counter()
        prompt, _ = _gsm8k_prompt(args, step)
        prompt_token_count = encoded_prompt_token_count(tokenizer, prompt)
        start = time.perf_counter()
        step_tokens = 0
        for _ in range(completions_per_prompt):
            token_count = 0
            for response in mlx_lm.stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
            ):
                token_count = max(token_count, int(response.generation_tokens))
            step_tokens += token_count
        mx.synchronize()  # Benchmark sync: finish mlx-lm generation before timing.
        if step >= args.warmup_steps:
            rollout_seconds += time.perf_counter() - start
            generated_completion_tokens += step_tokens
            prompt_tokens += prompt_token_count * completions_per_prompt
            samples += completions_per_prompt

    mx.synchronize()  # Benchmark sync: finish measured generation before memory read.
    measured_seconds = time.perf_counter() - measured_start
    measured_steps = args.steps - args.warmup_steps
    tok_s_denominator = generated_completion_tokens
    return BenchResult(
        target=args.target,
        pass_index=args.pass_index,
        status="ok",
        model=args.model,
        steps=measured_steps,
        warmup_steps=args.warmup_steps,
        generated_completion_tokens=generated_completion_tokens,
        prompt_tokens=prompt_tokens,
        total_forward_tokens=prompt_tokens + generated_completion_tokens,
        tok_s_denominator=tok_s_denominator,
        rollout_tokens=generated_completion_tokens,
        rollout_seconds=rollout_seconds,
        rollout_tok_s=safe_div(tok_s_denominator, rollout_seconds),
        samples=samples,
        end_to_end_seconds=measured_seconds,
        samples_s=safe_div(samples, measured_seconds),
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        mlx_version=dist_version("mlx"),
        mlx_lm_version=dist_version("mlx-lm"),
        tool_version=dist_version("mlx-lm"),
        note=(
            "generation-only baseline; "
            f"sequential G={completions_per_prompt}; gradient fields are not applicable"
        ),
    )


def set_mlx_wired_limit(mx: Any, wired_limit_mb: int) -> None:
    if wired_limit_mb > 0:
        mx.set_wired_limit(wired_limit_mb * 1024 * 1024)


def generated_tokens_from_rollouts(completions: Sequence[Any]) -> int:
    return sum(len(completion.completion_tokens) for completion in completions)


def prompt_tokens_from_rollouts(completions: Sequence[Any]) -> int:
    by_prompt: dict[int, int] = {}
    for completion in completions:
        by_prompt.setdefault(int(completion.prompt_index), len(completion.prompt_tokens))
    return sum(by_prompt.values())


def encoded_prompt_token_count(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool = False,
) -> int:
    text = prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return len(tokenizer.encode(text))


def read_iogpu_wired_limit_mb() -> int | None:
    completed = subprocess.run(
        ["sysctl", "-n", "iogpu.wired_limit_mb"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def result_from_json(raw: str, target: str, pass_index: int) -> BenchResult:
    data = extract_json_object(raw)
    filtered = {key: data.get(key) for key in JSON_FIELDS if key in data}
    filtered.setdefault("target", target)
    filtered.setdefault("pass_index", pass_index)
    filtered.setdefault("status", "ok")
    generated = int(
        filtered.get("generated_completion_tokens")
        or filtered.get("rollout_tokens")
        or 0
    )
    prompt_tokens = int(filtered.get("prompt_tokens") or 0)
    filtered.setdefault("generated_completion_tokens", generated)
    filtered.setdefault("prompt_tokens", prompt_tokens)
    filtered.setdefault("total_forward_tokens", prompt_tokens + generated)
    filtered.setdefault("tok_s_denominator", generated)
    filtered.setdefault("rollout_tokens", generated)
    if filtered.get("rollout_tok_s") is None:
        filtered["rollout_tok_s"] = safe_div(
            float(filtered["tok_s_denominator"]),
            float(filtered.get("rollout_seconds") or 0.0),
        )
    return BenchResult(**filtered)


def extract_json_object(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("No JSON metrics were produced.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def missing_result(
    args: argparse.Namespace,
    target: str,
    pass_index: int,
    note: str,
) -> BenchResult:
    return BenchResult(
        target=target,
        pass_index=pass_index,
        status="missing",
        model=args.model,
        steps=args.steps - args.warmup_steps,
        warmup_steps=args.warmup_steps,
        mlx_version=dist_version("mlx"),
        mlx_lm_version=dist_version("mlx-lm"),
        note=note,
    )


def validate_external_versions(result: BenchResult) -> BenchResult:
    notes: list[str] = []
    local_mlx = dist_version("mlx")
    local_mlx_lm = dist_version("mlx-lm")
    if result.mlx_version is not None and result.mlx_version != local_mlx:
        notes.append(f"mlx version mismatch: {result.mlx_version} != {local_mlx}")
    if result.mlx_lm_version is not None and result.mlx_lm_version != local_mlx_lm:
        notes.append(f"mlx-lm version mismatch: {result.mlx_lm_version} != {local_mlx_lm}")
    if result.target == "mlx-tune":
        if result.tool_version is None:
            notes.append("mlx-tune tool_version missing; require >= 0.5.1")
        elif not version_at_least(result.tool_version, "0.5.1"):
            notes.append(f"mlx-tune {result.tool_version} < 0.5.1")
    if not notes:
        return result
    note = "; ".join([result.note, *notes]).strip("; ")
    return replace(result, status="missing", note=note)


def enforce_gradient_timing_sanity(result: BenchResult) -> BenchResult:
    if (
        result.gradient_steps <= 0
        or result.gradient_step_s is None
        or result.gradient_step_s >= MIN_SYNCED_GRADIENT_STEP_SECONDS
    ):
        return result
    note = "; ".join(
        [
            result.note,
            "rejected synced grad_s_step "
            f"{result.gradient_step_s:.6f}s < {MIN_SYNCED_GRADIENT_STEP_SECONDS:.3f}s",
        ]
    ).strip("; ")
    return replace(
        result,
        status="missing",
        gradient_seconds=0.0,
        gradient_step_s=None,
        note=note,
    )


def version_at_least(observed: str, minimum: str) -> bool:
    return version_tuple(observed) >= version_tuple(minimum)


def version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.replace("-", ".").split("."):
        if piece.isdigit():
            parts.append(int(piece))
        elif parts:
            break
    return tuple(parts or [0])


def render_summary(results: Sequence[BenchResult]) -> str:
    headers = [
        "target",
        "pass",
        "comparison",
        "status",
        "rollout tok/s",
        "gen toks",
        "prompt toks",
        "tok/s denom",
        "grad s/step",
        "samples/s",
        "it/s",
        "peak GB",
        "note",
    ]
    rows = [
        [
            result.target,
            str(result.pass_index + 1),
            comparison_label(result.target),
            result.status,
            fmt(result.rollout_tok_s),
            str(result.generated_completion_tokens),
            str(result.prompt_tokens),
            str(result.tok_s_denominator),
            fmt(result.gradient_step_s),
            fmt(result.samples_s),
            fmt(result.it_s),
            fmt(result.peak_memory_gb),
            table_cell(result.note),
        ]
        for result in results
    ]
    lines = [
        "# Phase 4 Benchmark Summary",
        "",
        "Rollout tok/s definition: generated completion tokens only, excluding "
        "prompt/prefix tokens and warmup steps, divided by synchronized generation "
        "phase wall-clock seconds.",
        "",
        "Sync guarantee: every timed rollout, optimizer, and full-run boundary calls "
        "`mx.eval` or `mx.synchronize` on the relevant MLX outputs/state before "
        "stopping the timer. Synced gradient steps under 10 ms are rejected.",
        "",
        "Comparison labels: `apples-to-apples` is reserved for mlxrl's own GRPO "
        "semantics/config; `package-speed only` means the package fast path is useful "
        "but training or sampling semantics differ; `gen-only` has no gradient phase.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    lines.extend(reproducibility_lines(results))
    return "\n".join(lines)


def comparison_label(target: str) -> str:
    if target == "mlxrl":
        return "apples-to-apples"
    if target in {"mlx-tune", "mlx-lm-lora"}:
        return "package-speed only"
    if target in {"mlx-lm", "mlx-lm-g4"}:
        return "gen-only"
    return "unknown"


def reproducibility_lines(results: Sequence[BenchResult]) -> list[str]:
    by_target: dict[str, list[BenchResult]] = {}
    for result in results:
        if result.status == "ok":
            by_target.setdefault(result.target, []).append(result)
    lines = ["## Two-Pass Reproducibility", ""]
    for target, target_results in by_target.items():
        if len(target_results) < 2:
            lines.append(f"- {target}: only {len(target_results)} successful pass(es).")
            continue
        first, second = target_results[0], target_results[1]
        delta = relative_delta(first.it_s or first.samples_s, second.it_s or second.samples_s)
        lines.append(f"- {target}: throughput relative delta {fmt(delta)}")
    return lines


def format_result_line(result: BenchResult) -> str:
    return (
        f"{result.target}[pass={result.pass_index + 1}] {result.status} "
        f"rollout_tok_s={fmt(result.rollout_tok_s)} "
        f"tok_s_denom={result.tok_s_denominator} "
        f"gen_tokens={result.generated_completion_tokens} "
        f"prompt_tokens={result.prompt_tokens} "
        f"grad_s_step={fmt(result.gradient_step_s)} "
        f"samples_s={fmt(result.samples_s)} "
        f"it_s={fmt(result.it_s)} peak_gb={fmt(result.peak_memory_gb)}"
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:240]


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def relative_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    denominator = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denominator


def dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0] not in {"run", "worker"}:
        raw_args = ["run", *raw_args]
    args = parser.parse_args(raw_args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
