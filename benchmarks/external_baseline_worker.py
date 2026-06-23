#!/usr/bin/env python
"""External Phase 4 baseline workers for installed MLX RL packages."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_phase4 import (
    BenchResult,
    dist_version,
    encoded_prompt_token_count,
    enforce_gradient_timing_sanity,
    safe_div,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="external_baseline_worker.py")
    parser.add_argument("--target", choices=["mlx-tune", "mlx-lm-lora"], required=True)
    parser.add_argument("--pass-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--min-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--wired-limit-mb", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> BenchResult:
    if args.target == "mlx-tune":
        return run_mlx_tune(args)
    if args.target == "mlx-lm-lora":
        return run_mlx_lm_lora(args)
    raise ValueError(f"Unsupported target: {args.target}")


def prompt_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    from mlxrl.cli import _gsm8k_prompt

    prompt_args = SimpleNamespace(use_chat_template=False)
    rows: list[dict[str, str]] = []
    for step in range(args.steps):
        prompt, answer = _gsm8k_prompt(prompt_args, step)
        rows.append({"prompt": prompt, "answer": answer, "text_completion": prompt})
    return rows


def scalar_reward(completion: str, answer: str) -> float:
    from mlxrl.data.rewards import accuracy_reward, format_reward

    return accuracy_reward(completion, answer=answer) + 0.1 * format_reward(completion)


def scalar_reward_with_tiebreaker(completion: str, answer: str) -> float:
    return scalar_reward(completion, answer) + 1e-6 * len(completion.split())


def set_wired_limit(mx: Any, wired_limit_mb: int) -> None:
    if wired_limit_mb > 0:
        mx.set_wired_limit(wired_limit_mb * 1024 * 1024)


def measured_step_count(args: argparse.Namespace) -> int:
    return args.steps - args.warmup_steps


def sync_timed_outputs(mx: Any, *values: Any) -> None:
    array_type = type(mx.array(0))
    arrays = list(flatten_mx_arrays(values, array_type))
    if arrays:
        mx.eval(*arrays)  # Benchmark eval: materialize timed phase outputs before timing.
    mx.synchronize()  # Benchmark sync: no lazy MLX work may cross a timing boundary.


def flatten_mx_arrays(values: Any, array_type: type[Any]) -> Sequence[Any]:
    if isinstance(values, array_type):
        return [values]
    if isinstance(values, dict):
        arrays: list[Any] = []
        for value in values.values():
            arrays.extend(flatten_mx_arrays(value, array_type))
        return arrays
    if isinstance(values, list | tuple):
        arrays = []
        for value in values:
            arrays.extend(flatten_mx_arrays(value, array_type))
        return arrays
    state = getattr(values, "state", None)
    if state is not None:
        return flatten_mx_arrays(state, array_type)
    return []


def generated_token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        count = 1
        for dimension in shape:
            count *= int(dimension)
        return count
    if hasattr(value, "tolist"):
        return generated_token_count(value.tolist())
    if isinstance(value, list | tuple):
        if not value:
            return 0
        if all(isinstance(item, int) for item in value):
            return len(value)
        return sum(generated_token_count(item) for item in value)
    return 0


def completion_tokens_from_full_ids(
    generated_ids: Any,
    prompt_ids: Any,
    log_probs: Any,
) -> int:
    if prompt_ids is not None:
        return max(0, generated_token_count(generated_ids) - generated_token_count(prompt_ids))
    shape = getattr(log_probs, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return generated_token_count(generated_ids)


def prompt_tokens_for_indices(
    prompt_token_lengths: Sequence[int],
    indices: Any,
    *,
    fallback_index: int,
) -> int:
    if not prompt_token_lengths:
        return 0
    values = flattened_ints(indices)
    if not values:
        return prompt_token_lengths[min(fallback_index, len(prompt_token_lengths) - 1)]
    return sum(
        prompt_token_lengths[index]
        for index in sorted(set(values))
        if 0 <= index < len(prompt_token_lengths)
    )


def flattened_ints(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        return flattened_ints(value.tolist())
    if isinstance(value, int):
        return [value]
    if isinstance(value, list | tuple):
        output: list[int] = []
        for item in value:
            output.extend(flattened_ints(item))
        return output
    return []


def base_result(
    args: argparse.Namespace,
    *,
    rollout_tokens: int,
    prompt_tokens: int,
    rollout_seconds: float,
    gradient_steps: int,
    gradient_seconds: float,
    samples: int,
    end_to_end_seconds: float,
    peak_memory_gb: float | None,
    note: str = "",
) -> BenchResult:
    measured_steps = measured_step_count(args)
    tok_s_denominator = rollout_tokens
    result = BenchResult(
        target=args.target,
        pass_index=args.pass_index,
        status="ok",
        model=args.model,
        steps=measured_steps,
        warmup_steps=args.warmup_steps,
        generated_completion_tokens=rollout_tokens,
        prompt_tokens=prompt_tokens,
        total_forward_tokens=prompt_tokens + rollout_tokens,
        tok_s_denominator=tok_s_denominator,
        rollout_tokens=rollout_tokens,
        rollout_seconds=rollout_seconds,
        rollout_tok_s=safe_div(tok_s_denominator, rollout_seconds),
        gradient_steps=gradient_steps,
        gradient_seconds=gradient_seconds,
        gradient_step_s=safe_div(gradient_seconds, max(gradient_steps, 1)),
        samples=samples,
        end_to_end_seconds=end_to_end_seconds,
        samples_s=safe_div(samples, end_to_end_seconds),
        it_s=safe_div(measured_steps, end_to_end_seconds),
        peak_memory_gb=peak_memory_gb,
        mlx_version=dist_version("mlx"),
        mlx_lm_version=dist_version("mlx-lm"),
        tool_version=dist_version(args.target),
        note=note,
    )
    return enforce_gradient_timing_sanity(result)


def run_mlx_lm_lora(args: argparse.Namespace) -> BenchResult:
    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx_lm_lora.trainer.grpo_trainer as grpo_trainer
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm_lora.trainer.datasets import CacheDataset, GRPODataset
    from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs
    from mlx_lm_lora.utils import from_pretrained

    set_wired_limit(mx, args.wired_limit_mb)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rows = prompt_rows(args)

    rollout_seconds_by_step: list[float] = []
    rollout_tokens_by_step: list[int] = []
    rollout_prompt_tokens_by_step: list[int] = []
    prompt_token_lengths: list[int] = []
    original_generate = grpo_trainer.generate_grpo

    def timed_generate_grpo(*gen_args: Any, **gen_kwargs: Any) -> tuple[Any, Any, Any]:
        start = time.perf_counter()
        completions, texts, batch_indices = original_generate(*gen_args, **gen_kwargs)
        sync_timed_outputs(mx, completions)
        rollout_seconds_by_step.append(time.perf_counter() - start)
        rollout_tokens_by_step.append(sum(int(completion.shape[0]) for completion in completions))
        rollout_prompt_tokens_by_step.append(
            prompt_tokens_for_indices(
                prompt_token_lengths,
                batch_indices,
                fallback_index=len(rollout_prompt_tokens_by_step),
            )
        )
        return completions, texts, batch_indices

    def mlxrl_reward_func(
        prompts: Sequence[str],
        completions: Sequence[str],
        answer: Sequence[str],
        types: Sequence[Any] | None = None,
    ) -> list[float]:
        del prompts, types
        return [
            scalar_reward_with_tiebreaker(completion, expected) + 1e-6 * index
            for index, (completion, expected) in enumerate(zip(completions, answer, strict=True))
        ]

    class StepTimer(TrainingCallback):
        def __init__(self) -> None:
            self.previous = time.perf_counter()
            self.measured_start: float | None = None
            self.measured_end: float | None = None
            self.step_seconds: list[float] = []

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            iteration = int(train_info["iteration"])
            mx.synchronize()  # Benchmark sync: finish package step before callback timing.
            now = time.perf_counter()
            if args.warmup_steps == 0 and self.measured_start is None:
                mx.synchronize()  # Benchmark sync: establish measured boundary after setup.
                mx.reset_peak_memory()
                self.measured_start = now
                self.previous = now
            if iteration == args.warmup_steps:
                mx.synchronize()  # Benchmark sync: discard external warmup before timing.
                mx.reset_peak_memory()
                self.measured_start = time.perf_counter()
                self.previous = self.measured_start
                return
            if iteration > args.warmup_steps:
                self.step_seconds.append(now - self.previous)
                self.previous = now
                self.measured_end = now

    with tempfile.TemporaryDirectory(prefix="mlxrl-mlx-lm-lora-") as tmpdir:
        adapter_dir = Path(tmpdir) / "adapters"
        model, tokenizer, adapter_file = from_pretrained(
            model=args.model,
            new_adapter_path=str(adapter_dir),
            lora_config={
                "rank": args.rank,
                "dropout": args.dropout,
                "scale": args.scale,
                "num_layers": -1,
            },
        )
        prompt_token_lengths = [
            encoded_prompt_token_count(tokenizer, row["prompt"]) for row in rows
        ]
        ref_model, _ = load(args.model)
        ref_model.freeze()
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        dataset = GRPODataset(rows, tokenizer, text_completion_key="text_completion")
        timer = StepTimer()
        grpo_trainer.generate_grpo = timed_generate_grpo
        try:
            grpo_trainer.train_grpo(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                train_dataset=CacheDataset(dataset),
                val_dataset=CacheDataset(GRPODataset([], tokenizer)),
                reward_funcs=[mlxrl_reward_func],
                args=GRPOTrainingArgs(
                    batch_size=args.batch_size,
                    iters=args.steps,
                    val_batches=0,
                    steps_per_report=1,
                    steps_per_eval=args.steps + 1,
                    steps_per_save=args.steps + 1,
                    adapter_file=adapter_file,
                    max_seq_length=args.max_context,
                    max_completion_length=args.max_tokens,
                    beta=args.beta,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    gradient_accumulation_steps=1,
                    importance_sampling_level=None,
                    grpo_loss_type="grpo",
                ),
                training_callback=timer,
            )
        finally:
            grpo_trainer.generate_grpo = original_generate

    measured_rollout_seconds = rollout_seconds_by_step[args.warmup_steps :]
    measured_rollout_tokens = rollout_tokens_by_step[args.warmup_steps :]
    measured_prompt_tokens = rollout_prompt_tokens_by_step[args.warmup_steps :]
    rollout_seconds = sum(measured_rollout_seconds)
    end_to_end_seconds = (
        (timer.measured_end - timer.measured_start)
        if timer.measured_start is not None and timer.measured_end is not None
        else sum(timer.step_seconds)
    )
    gradient_seconds = max(0.0, sum(timer.step_seconds) - rollout_seconds)
    samples = measured_step_count(args) * args.group_size
    return base_result(
        args,
        rollout_tokens=sum(measured_rollout_tokens),
        prompt_tokens=sum(measured_prompt_tokens),
        rollout_seconds=rollout_seconds,
        gradient_steps=measured_step_count(args),
        gradient_seconds=gradient_seconds,
        samples=samples,
        end_to_end_seconds=end_to_end_seconds,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        note="gradient phase is measured as step remainder after package rollout hook",
    )


def run_mlx_tune(args: argparse.Namespace) -> BenchResult:
    import mlx.core as mx
    import mlx_tune._perf as perf
    import mlx_tune.rl_trainers as rl_trainers
    from mlx_tune import FastLanguageModel, GRPOConfig, GRPOTrainer

    set_wired_limit(mx, args.wired_limit_mb)
    mx.random.seed(args.seed)
    rows = prompt_rows(args)
    reward_index = 0

    rollout_seconds_by_call: list[float] = []
    rollout_tokens_by_call: list[int] = []
    gradient_seconds_by_step: list[float] = []
    step_seconds: list[float] = []
    current_step_start = time.perf_counter()
    measured_start: float | None = None
    measured_end: float | None = None
    completed_steps = 0

    original_build_cache = rl_trainers.build_shared_prompt_cache
    original_generate = rl_trainers.generate_with_log_probs
    original_compiled_step = perf.compiled_step

    def timed_build_shared_prompt_cache(*cache_args: Any, **cache_kwargs: Any) -> Any:
        nonlocal current_step_start
        current_step_start = time.perf_counter()
        start = time.perf_counter()
        cache = original_build_cache(*cache_args, **cache_kwargs)
        sync_timed_outputs(mx, cache)
        rollout_seconds_by_call.append(time.perf_counter() - start)
        rollout_tokens_by_call.append(0)
        return cache

    def timed_generate_with_log_probs(*gen_args: Any, **gen_kwargs: Any) -> tuple[Any, Any]:
        start = time.perf_counter()
        generated_ids, log_probs = original_generate(*gen_args, **gen_kwargs)
        sync_timed_outputs(mx, generated_ids, log_probs)
        rollout_seconds_by_call.append(time.perf_counter() - start)
        prompt_ids = gen_args[2] if len(gen_args) > 2 else gen_kwargs.get("prompt_ids")
        rollout_tokens_by_call.append(
            completion_tokens_from_full_ids(generated_ids, prompt_ids, log_probs)
        )
        return generated_ids, log_probs

    def timed_compiled_step(step_fn: Any, state: list[Any], **compile_kwargs: Any) -> Any:
        compiled = original_compiled_step(step_fn, state, **compile_kwargs)

        def timed_step(*step_args: Any, **step_kwargs: Any) -> Any:
            nonlocal completed_steps, measured_start, measured_end
            gradient_start = time.perf_counter()
            output = compiled(*step_args, **step_kwargs)
            sync_timed_outputs(mx, output, state)
            gradient_elapsed = time.perf_counter() - gradient_start
            now = time.perf_counter()
            completed_steps += 1
            if completed_steps == args.warmup_steps:
                mx.reset_peak_memory()
                measured_start = now
            elif completed_steps > args.warmup_steps:
                gradient_seconds_by_step.append(gradient_elapsed)
                step_seconds.append(now - current_step_start)
                measured_end = now
            return output

        return timed_step

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_context,
        load_in_4bit=True,
    )
    prompt_token_lengths = [
        encoded_prompt_token_count(tokenizer, row["prompt"]) for row in rows
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=max(1, int(args.scale * args.rank)),
        lora_dropout=args.dropout,
    )
    def benchmark_reward(completion: str, answer: str) -> float:
        nonlocal reward_index
        reward_index += 1
        return scalar_reward_with_tiebreaker(completion, answer) + 1e-6 * reward_index

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rows,
        tokenizer=tokenizer,
        reward_fn=benchmark_reward,
        args=GRPOConfig(
            loss_type="grpo",
            beta=args.beta,
            num_generations=args.group_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_completion_length=args.max_tokens,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            max_steps=args.steps,
            logging_steps=args.steps + 1,
            save_steps=args.steps + 1,
            max_seq_length=args.max_context,
        ),
    )

    if args.warmup_steps == 0:
        mx.synchronize()  # Benchmark sync: establish measured boundary after setup.
        mx.reset_peak_memory()
        measured_start = time.perf_counter()

    rl_trainers.build_shared_prompt_cache = timed_build_shared_prompt_cache
    rl_trainers.generate_with_log_probs = timed_generate_with_log_probs
    perf.compiled_step = timed_compiled_step
    try:
        trainer.train()
    finally:
        rl_trainers.build_shared_prompt_cache = original_build_cache
        rl_trainers.generate_with_log_probs = original_generate
        perf.compiled_step = original_compiled_step

    calls_per_step = args.group_size + 1
    warmup_calls = args.warmup_steps * calls_per_step
    rollout_seconds = sum(rollout_seconds_by_call[warmup_calls:])
    rollout_tokens = sum(rollout_tokens_by_call[warmup_calls:])
    prompt_tokens = sum(prompt_token_lengths[args.warmup_steps : args.steps])
    end_to_end_seconds = (
        (measured_end - measured_start)
        if measured_start is not None and measured_end is not None
        else sum(step_seconds)
    )
    samples = measured_step_count(args) * args.group_size
    return base_result(
        args,
        rollout_tokens=rollout_tokens,
        prompt_tokens=prompt_tokens,
        rollout_seconds=rollout_seconds,
        gradient_steps=len(gradient_seconds_by_step),
        gradient_seconds=sum(gradient_seconds_by_step),
        samples=samples,
        end_to_end_seconds=end_to_end_seconds,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        note="mlx-tune native trainer timed via package hooks",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    Path(args.output).write_text(json.dumps(asdict(result), sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
