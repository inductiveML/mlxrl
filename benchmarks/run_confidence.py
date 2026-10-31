#!/usr/bin/env python3
"""Matched 4-bit CONF screening: GRPO/SC-GRPO, each with and without CONF.

All arms use identical structured prompts, normalized rewards, sequence-mean
loss, optimizer settings and evaluation examples. This is a short arithmetic
screening, not a reproduction of RAMP's LFM2.5 variable-chain experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.algo.score_centering import ScoreCenteringAlgorithm
from mlxrl.confidence import ConfidenceWarmStart
from mlxrl.config import ConfidenceConfig
from mlxrl.data.confidence import arithmetic_steps
from mlxrl.data.rewards import extract_answer
from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
from mlxrl.rollout.naive import SamplingConfig
from mlxrl.rollout.optimized import generate_prefix_cached_group_rollouts
from mlxrl.train.grpo import batch_from_rollouts, optimizer_step

METHODS = ("grpo", "grpo-conf", "sc-grpo", "sc-grpo-conf")
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
EVAL = (
    (16, 7, 2),
    (22, 3, 5),
    (18, 4, -7),
    (13, 6, -4),
    (27, 2, 8),
    (32, 3, -9),
    (25, 4, 6),
    (15, 9, -2),
)


def prompt(case):
    a, b, c = case
    return (
        "Calculate (a + b) + c. Write two calculation lines and the answer.\n"
        "Question: a=2, b=3, c=4\n"
        "step: x = 2 + 3 = 5\nstep: y = x + 4 = 9\n<answer>9</answer>\n\n"
        "Question: a=8, b=5, c=-2\n"
        "step: x = 8 + 5 = 13\nstep: y = x - 2 = 11\n<answer>11</answer>\n\n"
        f"Question: a={a}, b={b}, c={c}\n"
    )


def reward(text, case):
    try:
        predicted = float(extract_answer(text) or "nan")
    except ValueError:
        predicted = float("nan")
    target = sum(case)
    if not math.isfinite(predicted):
        return -1.0, False
    return 1 - 2 * min(abs(predicted - target) / max(abs(target), 1), 1), predicted == target


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def worker(args):
    path = Path(args.model).resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("quantization", config.get("quantization_config", {})).get("bits") != 4:
        raise ValueError("CONF screening requires verified 4-bit weights.")
    mx.random.seed(args.seed)
    model, tokenizer, _ = load_policy_with_lora(str(path), LoRAConfig(rank=8, scale=20, dropout=0))
    centered = args.method.startswith("sc-")
    algorithm = (
        ScoreCenteringAlgorithm(
            sampler_top_k=32, normalize_rewards=True, loss_reduction="sequence_mean"
        )
        if centered
        else GRPOAlgorithm(loss_reduction="sequence_mean")
    )
    controller = ConfidenceWarmStart(ConfidenceConfig(enabled=args.method.endswith("-conf")))
    sampling = SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        record_sampler_top_k=32 if centered else 0,
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    pad = pad_token_id_from_tokenizer(tokenizer)
    identity = {"method": args.method, "seed": args.seed}
    elapsed = 0.0

    def evaluate(update):
        rows = generate_prefix_cached_group_rollouts(
            model,
            tokenizer,
            [prompt(c) for c in EVAL],
            1,
            SamplingConfig(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0),
            seed=999,
            use_chat_template=False,
        )
        accuracy = sum(reward(r.text, EVAL[r.prompt_index])[1] for r in rows) / len(rows)
        emit(
            {
                "kind": "eval",
                **identity,
                "update": update,
                "accuracy": accuracy,
                "training_seconds": elapsed,
            }
        )
        return accuracy

    initial = evaluate(0)
    for update in range(1, args.steps + 1):
        cases = [TRAIN[((update - 1) * 4 + i) % len(TRAIN)] for i in range(4)]
        mx.synchronize()
        start = time.perf_counter()
        rows = generate_prefix_cached_group_rollouts(
            model,
            tokenizer,
            [prompt(c) for c in cases],
            2,
            sampling,
            seed=args.seed * 1000 + update,
            use_chat_template=False,
        )
        rewards = [reward(r.text, cases[r.prompt_index])[0] for r in rows]
        source = batch_from_rollouts(
            model,
            rows,
            rewards,
            2,
            pad,
            algorithm,
            compute_reference=False,
            record_confidence=controller.active(update),
        )
        score_start = time.perf_counter()
        units = (
            [arithmetic_steps(tokenizer, r.completion_tokens) for r in rows]
            if controller.active(update)
            else []
        )
        # Continuous task reward defines calibration's mixed groups in this pilot.
        prepared, confidence = controller.apply(source, units, update=update, group_size=2)
        score_seconds = time.perf_counter() - score_start
        metrics = optimizer_step(
            model, optimizer, prepared, 0.0, pad, algorithm, micro_batch_size=2
        )
        mx.synchronize()
        seconds = time.perf_counter() - start
        elapsed += seconds
        fields = {**asdict(metrics), **asdict(confidence)}
        if any(isinstance(x, float) and not math.isfinite(x) for x in fields.values()):
            raise RuntimeError("Nonfinite training diagnostics.")
        emit(
            {
                "kind": "step",
                **identity,
                **fields,
                "seconds": seconds,
                "credit_preparation_seconds": score_seconds,
                "tokens": sum(len(r.completion_tokens) for r in rows),
                "training_accuracy": sum(reward(r.text, cases[r.prompt_index])[1] for r in rows)
                / len(rows),
                "first_completion": rows[0].text if update in (1, 8, 9) else None,
            }
        )
        if update == 8:
            evaluate(update)
    adapter_values = [v for _, v in tree_flatten(model.trainable_parameters())]
    if not all(bool(mx.all(mx.isfinite(v))) for v in adapter_values):
        raise RuntimeError("Nonfinite final adapter weights.")
    adapter_norm = float(mx.sqrt(sum(mx.sum(v * v) for v in adapter_values)))
    final = evaluate(args.steps)
    emit(
        {
            "kind": "summary",
            **identity,
            "initial_accuracy": initial,
            "final_accuracy": final,
            "training_seconds": elapsed,
            "controller_state": controller.state_dict(),
            "adapter_norm": adapter_norm,
            "adapter_finite": True,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, default=Path("reference_outputs/confidence.jsonl"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if (
        args.steps < 9
        or args.max_tokens < 16
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error(
            "Use at least 9 updates, 16 completion tokens and a finite positive learning rate."
        )
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        parser.error("Output already exists; retain prior experiments and choose a new path.")
    source = hashlib.sha256()
    for file in sorted(Path("mlxrl").rglob("*.py")):
        source.update(str(file).encode() + b"\0" + file.read_bytes())
    order = [(seed, method) for seed in args.seeds for method in args.methods]
    random.Random(20261101).shuffle(order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        model_path = Path(args.model).resolve()
        model_files = {}
        for model_file in sorted(model_path.glob("*.safetensors")):
            digest = hashlib.sha256()
            with model_file.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            model_files[model_file.name] = digest.hexdigest()
        manifest = {
            "kind": "manifest",
            "source_sha256": source.hexdigest(),
            "model_revision": model_path.name,
            "model_files_sha256": model_files,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm")},
            "platform": platform.platform(),
            "order": order,
            "steps": args.steps,
            "max_tokens": args.max_tokens,
            "learning_rate": args.learning_rate,
            "train_cases": TRAIN,
            "eval_cases": EVAL,
            "scope": "Short arithmetic screening; does not establish a 2x speedup.",
        }
        output.write(json.dumps(manifest) + "\n")
        output.flush()
        for seed, method in order:
            print(f"Running {method} seed={seed}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.run_confidence",
                    "--worker",
                    "--model",
                    args.model,
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--learning-rate",
                    str(args.learning_rate),
                ],
                stdout=output,
                check=True,
            )


if __name__ == "__main__":
    main()
