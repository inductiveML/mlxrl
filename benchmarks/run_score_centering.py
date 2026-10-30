"""Paired multi-seed 4-bit QLoRA screening for experimental score centering.

Each run uses a fresh process/model/optimizer. PG and SC use identical
group-centered advantages, fp32 trainer probabilities, and sampler capture;
the only objective difference is the detached score-centering correction.
Stock GRPO/DAPO comparisons keep their original advantage/loss/probability
paths and disable capture; they are contextual comparisons, not that ablation.
This local arithmetic benchmark is not a reproduction of the paper's results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim

from mlxrl.algo.grpo import DAPOAlgorithm, GRPOAlgorithm, _token_loss_metrics
from mlxrl.algo.score_centering import ScoreCenteringAlgorithm
from mlxrl.algorithm import AlgorithmLossMetrics
from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
from mlxrl.train.grpo import batch_from_rollouts, optimizer_step

TRAIN = (
    (19, 7, 4),
    (23, 3, 8),
    (17, 4, 6),
    (12, 6, -5),
    (28, 2, 9),
    (31, 3, -8),
    (26, 4, 7),
    (14, 9, -3),
)
HELD_OUT = (
    (16, 7, 2),
    (22, 3, 5),
    (18, 4, -7),
    (13, 6, -4),
    (27, 2, 8),
    (32, 3, -9),
    (25, 4, 6),
    (15, 9, -2),
)
PREFIX = (
    "Calculate the expression. Return only the integer answer.\n"
    "17 * 2 + 3 = 37\n9 * 4 - 2 = 34\n13 * 3 + 4 = 43\n"
)


class PGBaseline(ScoreCenteringAlgorithm):
    """Same probability/data path, with the centering term omitted."""

    def compute_sampler_loss(self, **kwargs: Any) -> AlgorithmLossMetrics:
        logp = kwargs["policy_logprobs"]
        return _token_loss_metrics(
            token_policy_gradient=-kwargs["advantages"][:, None] * logp,
            ratio=mx.exp(logp - kwargs["sampler"].logprobs),
            policy_logprobs=logp,
            reference_logprobs=kwargs["reference_logprobs"],
            mask=kwargs["completion_mask"],
            beta=kwargs["beta"],
        )


class BenchmarkTokenizer:
    """Stop at the first newline so continuations contain one answer only."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.chat_template = None
        self.eos_token_id = tokenizer.eos_token_id
        newline = tokenizer.encode("\n")
        if len(newline) != 1:
            raise ValueError("Benchmark requires a tokenizer with a single newline token.")
        self.eos_token_ids = [*getattr(tokenizer, "eos_token_ids", []), newline[0]]

    def encode(self, text: str) -> Any:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


def prompt(case: tuple[int, int, int]) -> str:
    a, b, c = case
    return PREFIX + f"{a} * {b} {'+' if c >= 0 else '-'} {abs(c)} = "


def correct(text: str, case: tuple[int, int, int]) -> bool:
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", text)
    return bool(match and int(match[1]) == case[0] * case[1] + case[2])


def arithmetic_reward(text: str, case: tuple[int, int, int]) -> float:
    """Bounded distance reward keeps wrong-but-different answers informative."""

    match = re.fullmatch(r"\s*([+-]?\d+)\s*", text)
    if match is None:
        return -1.0
    target = case[0] * case[1] + case[2]
    relative_error = min(abs(int(match[1]) - target) / max(abs(target), 1), 1.0)
    return 1.0 - 2.0 * relative_error


def evaluate(model: Any, tokenizer: Any, sampling: SamplingConfig, seed: int) -> float:
    rows = generate_group_rollouts(
        model,
        tokenizer,
        [prompt(c) for c in HELD_OUT],
        2,
        replace(sampling, record_sampler_top_k=0),
        seed=seed,
        use_chat_template=False,
    )
    return sum(correct(row.text, HELD_OUT[row.prompt_index]) for row in rows) / len(rows)


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, allow_nan=False), flush=True)


def worker(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    config = json.loads((model_path / "config.json").read_text())
    quantization = config.get("quantization", config.get("quantization_config", {}))
    if quantization.get("bits") != 4:
        raise ValueError("This benchmark requires a verified 4-bit model.")
    mx.random.seed(args.seed)
    model, raw_tokenizer, report = load_policy_with_lora(
        str(model_path),
        LoRAConfig(rank=8, scale=20, dropout=0),
    )
    tokenizer = BenchmarkTokenizer(raw_tokenizer)
    if args.method in {"pg", "score-centering"}:
        algorithm = (PGBaseline if args.method == "pg" else ScoreCenteringAlgorithm)(
            sampler_top_k=args.capture_k,
        )
    else:
        algorithm = (
            GRPOAlgorithm() if args.method == "grpo" else DAPOAlgorithm(dynamic_sampling=False)
        )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    sampling = SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=1.0 if args.condition == "unfiltered" else 0.7,
        top_p=1.0 if args.condition == "unfiltered" else 0.95,
        record_sampler_top_k=getattr(algorithm, "sampler_top_k", 0),
    )
    identity = {"method": args.method, "seed": args.seed, "condition": args.condition}
    initial_accuracy = evaluate(model, tokenizer, sampling, 10000 + args.seed)
    # Warm up sampling and forward loss; the first backward is counted in timing.
    warm = generate_group_rollouts(
        model,
        tokenizer,
        [prompt(TRAIN[0])],
        2,
        sampling,
        seed=args.seed,
        use_chat_template=False,
    )
    warm_batch = batch_from_rollouts(
        model,
        warm,
        [0.0, 0.0],
        2,
        pad_token_id_from_tokenizer(raw_tokenizer),
        algorithm,
        compute_reference=False,
    )
    # Evaluate the loss only; warm-up must not mutate the model or Adam state.
    from mlxrl.train.grpo import grpo_metrics_from_batch

    warm_metrics = grpo_metrics_from_batch(
        model, warm_batch, 0.0, pad_token_id_from_tokenizer(raw_tokenizer), algorithm
    )
    mx.eval(warm_metrics.loss)
    del warm, warm_batch, warm_metrics
    mx.reset_peak_memory()
    elapsed_total, tokens_total, nonzero_groups = 0.0, 0, 0
    rewards_history, gradient_norms = [], []
    emit(
        {
            "kind": "start",
            **identity,
            "initial_accuracy": initial_accuracy,
            "quantization": quantization,
            "trainable_parameters": report.trainable_params,
        }
    )
    for step in range(args.steps):
        cases = [TRAIN[(2 * step + offset) % len(TRAIN)] for offset in range(2)]
        start = time.perf_counter()
        rows = generate_group_rollouts(
            model,
            tokenizer,
            [prompt(c) for c in cases],
            4,
            sampling,
            seed=args.seed * 1000 + step,
            use_chat_template=False,
        )
        rewards = [arithmetic_reward(row.text, cases[row.prompt_index]) for row in rows]
        accuracy = sum(correct(row.text, cases[row.prompt_index]) for row in rows) / len(rows)
        batch = batch_from_rollouts(
            model,
            rows,
            rewards,
            4,
            pad_token_id_from_tokenizer(raw_tokenizer),
            algorithm,
            compute_reference=False,
        )
        metrics = optimizer_step(
            model,
            optimizer,
            batch,
            0.0,
            pad_token_id_from_tokenizer(raw_tokenizer),
            algorithm,
        )
        elapsed = time.perf_counter() - start
        values = asdict(metrics)
        if any(isinstance(x, float) and not math.isfinite(x) for x in values.values()):
            raise RuntimeError(f"Non-finite training diagnostics at step {step}.")
        token_count = sum(len(row.completion_tokens) for row in rows)
        elapsed_total += elapsed
        tokens_total += token_count
        rewards_history.append(accuracy)
        if metrics.gradient_norm is not None:
            gradient_norms.append(metrics.gradient_norm)
        nonzero_groups += sum(len(set(rewards[i : i + 4])) > 1 for i in (0, 4))
        mismatch = None
        if batch.sampler is not None:
            deltas = batch.old_policy_logprobs - batch.sampler.logprobs
            mismatch = float(mx.sum(mx.abs(deltas) * batch.mask) / mx.sum(batch.mask))
        emit(
            {
                "kind": "step",
                **identity,
                "step": step + 1,
                "seconds": elapsed,
                "tokens": token_count,
                "sampler_trainer_logprob_mae": mismatch,
                "training_accuracy": accuracy,
                **values,
            }
        )
    final_accuracy = evaluate(model, tokenizer, sampling, 10000 + args.seed)
    emit(
        {
            "kind": "summary",
            **identity,
            "steps": args.steps,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "first_10_training_accuracy": sum(rewards_history[:10]) / min(10, args.steps),
            "last_10_training_accuracy": sum(rewards_history[-10:]) / min(10, args.steps),
            "nonzero_advantage_groups": nonzero_groups,
            "total_groups": 2 * args.steps,
            "max_gradient_norm": max(gradient_norms, default=None),
            "tokens": tokens_total,
            "training_seconds": elapsed_total,
            "tokens_per_second": tokens_total / elapsed_total,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "all_steps_finite": True,
            "learning_signal_sufficient": nonzero_groups >= max(1, args.steps // 2),
        }
    )


def source_hash() -> str:
    digest = hashlib.sha256()
    for file in sorted(Path("mlxrl").rglob("*.py")):
        digest.update(str(file).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def profile_capture(args: argparse.Namespace) -> None:
    """Paired, alternating-order capture timings with fixed model and token draws."""

    mx.random.seed(7)
    model, tokenizer, _ = load_policy_with_lora(args.model, LoRAConfig(rank=8, scale=20))
    tokenizer = BenchmarkTokenizer(tokenizer)
    base = SamplingConfig(max_tokens=args.max_tokens, temperature=0.7, top_p=0.95)
    prompts = [prompt(case) for case in TRAIN[:2]]
    for capture in (0, args.capture_k):
        generate_group_rollouts(
            model,
            tokenizer,
            prompts,
            4,
            replace(base, record_sampler_top_k=capture),
            seed=7,
            use_chat_template=False,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as output:
        output.write(
            json.dumps(
                {
                    "kind": "capture_manifest",
                    "source_sha256": source_hash(),
                    "model_revision": Path(args.model).name,
                }
            )
            + "\n"
        )
        for seed in args.seeds:
            for repeat in range(4):
                tokens_by_capture = {}
                for capture in (0, args.capture_k) if repeat % 2 == 0 else (args.capture_k, 0):
                    mx.reset_peak_memory()
                    start = time.perf_counter()
                    rows = generate_group_rollouts(
                        model,
                        tokenizer,
                        prompts,
                        4,
                        replace(base, record_sampler_top_k=capture),
                        seed=seed,
                        use_chat_template=False,
                    )
                    elapsed = time.perf_counter() - start
                    tokens_by_capture[capture] = [r.completion_tokens for r in rows]
                    record = {
                        "kind": "capture_timing",
                        "seed": seed,
                        "repeat": repeat,
                        "capture_k": capture,
                        "seconds": elapsed,
                        "tokens": sum(len(r.completion_tokens) for r in rows),
                        "peak_memory_gb": mx.get_peak_memory() / 1e9,
                    }
                    output.write(json.dumps(record) + "\n")
                    del rows
                if tokens_by_capture[0] != tokens_by_capture[args.capture_k]:
                    raise AssertionError("Capturing sampler probabilities changed token draws.")
        output.write(
            json.dumps(
                {
                    "kind": "capture_summary",
                    "paired_runs": 4 * len(args.seeds),
                    "all_tokens_identical": True,
                }
            )
            + "\n"
        )
    print(f"Capture profile complete: {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local pinned 4-bit model snapshot.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--capture-k", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pg", "score-centering", "grpo", "dapo"],
        default=["pg", "score-centering", "grpo", "dapo"],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["unfiltered", "default"],
        default=["unfiltered", "default"],
    )
    parser.add_argument("--output", default="benchmarks/results/score_centering_4bit.jsonl")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--method", choices=["pg", "score-centering", "grpo", "dapo"], help=argparse.SUPPRESS
    )
    parser.add_argument("--condition", choices=["unfiltered", "default"], help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--profile-capture", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.max_tokens < 1 or args.capture_k < 1:
        parser.error("--max-tokens and --capture-k must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.worker:
        worker(args)
        return
    if args.profile_capture:
        profile_capture(args)
        return
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as output:
        manifest = {
            "kind": "manifest",
            "source_sha256": source_hash(),
            "model_revision": Path(args.model).name,
            "runtime": {n: importlib.metadata.version(n) for n in ["mlx", "mlx-lm"]},
            "platform": platform.platform(),
            "arguments": {**vars(args), "model": f"Qwen3-0.6B-4bit@{Path(args.model).name}"},
            "train_cases": TRAIN,
            "held_out_cases": HELD_OUT,
            "scope": "Short local 4-bit arithmetic screening; no long-run stability claim.",
            "reward": "1 - 2 * min(abs(prediction-target)/max(abs(target),1),1); malformed=-1",
        }
        output.write(json.dumps(manifest) + "\n")
        output.flush()
        for condition in args.conditions:
            for seed in args.seeds:
                for method in args.methods:
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.run_score_centering",
                        "--worker",
                        "--model",
                        args.model,
                        "--method",
                        method,
                        "--condition",
                        condition,
                        "--seed",
                        str(seed),
                        "--steps",
                        str(args.steps),
                        "--max-tokens",
                        str(args.max_tokens),
                        "--capture-k",
                        str(args.capture_k),
                        "--learning-rate",
                        str(args.learning_rate),
                    ]
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
                    assert process.stdout is not None
                    for line in process.stdout:
                        record = json.loads(line)
                        output.write(line)
                        output.flush()
                        if record["kind"] != "step" or record["step"] % 10 == 0:
                            print(line.strip(), flush=True)
                    if process.wait() != 0:
                        raise RuntimeError(f"Benchmark failed: {condition}, seed {seed}, {method}")


if __name__ == "__main__":
    main()
