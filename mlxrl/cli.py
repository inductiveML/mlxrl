"""Command line entrypoints for mlxrl."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim

from mlxrl.algo.grpo import (
    DAPOAlgorithm,
    DrGRPOAlgorithm,
    GRPOAlgorithm,
    GSPOAlgorithm,
    PolicyAlgorithm,
    approximate_kl,
    group_normalize_rewards,
    grpo_loss,
)
from mlxrl.data.gsm8k import (
    MINI_GSM8K,
    format_gsm8k_answer_only_prompt,
    format_gsm8k_prompt,
)
from mlxrl.data.rewards import accuracy_reward, format_reward
from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
from mlxrl.policy.model import (
    DEFAULT_MODEL_ID,
    LoRAConfig,
    encode_prompt,
    load_policy_with_lora,
)
from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
from mlxrl.rollout.optimized import generate_prefix_cached_group_rollouts
from mlxrl.train.grpo import (
    GRPOBatch,
    batch_from_rollouts,
    grpo_metrics_from_batch,
    optimizer_step,
    reward_trend,
)


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


def _phase1_toy_gate(_: argparse.Namespace) -> int:
    policy = mx.log(mx.array([[0.20, 0.50], [0.40, 0.25]], dtype=mx.float32))
    old_policy = policy
    reference = mx.log(mx.array([[0.25, 0.40], [0.50, 0.20]], dtype=mx.float32))
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)
    advantages = group_normalize_rewards(rewards, group_size=2)
    mask = mx.ones_like(policy)
    beta = 0.1
    metrics = grpo_loss(policy, old_policy, reference, advantages, mask, beta)

    expected_advantages = mx.array([-1.0, 1.0], dtype=mx.float32)
    expected_kl = approximate_kl(policy, reference)
    expected_loss = mx.mean(-expected_advantages[:, None] + beta * expected_kl)
    mx.eval(  # Toy gate sync: materialize scalars before Python tolerance checks.
        metrics.loss,
        metrics.kl,
        advantages,
        expected_loss,
    )

    advantage_error = mx.max(mx.abs(advantages - expected_advantages)).item()
    loss_error = abs(float(metrics.loss.item()) - float(expected_loss.item()))
    tolerance = 1e-6
    print(f"advantages: {advantages.tolist()}")
    print(f"expected_advantages: {expected_advantages.tolist()}")
    print(f"loss: {float(metrics.loss.item()):.8f}")
    print(f"expected_loss: {float(expected_loss.item()):.8f}")
    print(f"kl: {float(metrics.kl.item()):.8f}")
    print(f"advantage_error: {advantage_error:.8e}")
    print(f"loss_error: {loss_error:.8e}")
    if advantage_error > tolerance or loss_error > tolerance:
        raise SystemExit("Phase 1 toy gate failed.")
    print("phase1_toy_gate: passed")
    return 0


def _completion_reward(text: str, answer: str) -> tuple[float, float, float]:
    accuracy = accuracy_reward(text, answer=answer)
    fmt = format_reward(text)
    total = accuracy + 0.1 * fmt
    return total, accuracy, fmt


def _gsm8k_prompt(args: argparse.Namespace, index: int) -> tuple[str, str]:
    example = MINI_GSM8K[index % len(MINI_GSM8K)]
    prompt = (
        format_gsm8k_prompt(example)
        if args.use_chat_template
        else format_gsm8k_answer_only_prompt(example)
    )
    return prompt, example.answer


def _sampling_config(args: argparse.Namespace) -> SamplingConfig:
    return SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
    )


def _clip_value(raw: str, default: float) -> float | None:
    if raw == "default":
        return default
    if raw.lower() in {"none", "off", "disabled"}:
        return None
    return float(raw)


def _algorithm_from_args(args: argparse.Namespace) -> PolicyAlgorithm:
    name = args.algorithm.lower().replace("_", "-")
    if name == "grpo":
        return GRPOAlgorithm()
    if name in {"dr-grpo", "drgrpo"}:
        return DrGRPOAlgorithm(
            normalize_rewards=args.drgrpo_normalize_rewards,
            loss_reduction=args.drgrpo_loss_reduction,
            max_tokens=args.drgrpo_max_tokens or args.max_tokens,
        )
    if name == "dapo":
        return DAPOAlgorithm(
            clip_low=_clip_value(args.clip_low, 0.2),
            clip_high=_clip_value(args.clip_high, 0.28),
        )
    if name == "gspo":
        return GSPOAlgorithm(
            importance=args.gspo_importance,
            clip_low=_clip_value(args.clip_low, 3e-4),
            clip_high=_clip_value(args.clip_high, 4e-4),
        )
    raise ValueError(f"Unknown algorithm {args.algorithm!r}.")


def _completion_rewards(
    completions: Sequence[Any],
    answers: Sequence[str],
) -> tuple[list[float], float, float]:
    rewards: list[float] = []
    accuracies: list[float] = []
    formats: list[float] = []
    for completion in completions:
        answer = answers[int(completion.prompt_index)]
        total, accuracy, fmt = _completion_reward(completion.text, answer)
        rewards.append(total)
        accuracies.append(accuracy)
        formats.append(fmt)
    return rewards, sum(accuracies) / len(accuracies), sum(formats) / len(formats)


def _padded_tokens(
    rows: Sequence[Sequence[int]],
    pad_token_id: int,
) -> mx.array:
    max_len = max(len(row) for row in rows)
    return mx.array(
        [list(row) + [pad_token_id] * (max_len - len(row)) for row in rows],
        dtype=mx.int32,
    )


def _rollout_timed(
    rollout_fn: Callable[..., tuple[Any, ...]],
    label: str,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], float, float]:
    mx.reset_peak_memory()
    start = time.perf_counter()
    completions = rollout_fn(**kwargs)
    mx.synchronize()  # Timing sync: finish rollout kernels before wall-clock/peak-memory read.
    elapsed = time.perf_counter() - start
    peak_gb = mx.get_peak_memory() / 1e9
    samples_per_second = len(completions) / elapsed if elapsed > 0 else float("inf")
    print(
        f"{label}_elapsed_s: {elapsed:.6f} "
        f"{label}_samples_per_s: {samples_per_second:.6f} "
        f"{label}_peak_gb: {peak_gb:.6f}"
    )
    return completions, elapsed, peak_gb


def _build_equivalence_batch(
    model: Any,
    completions: Sequence[Any],
    rewards: Sequence[float],
    group_size: int,
    beta: float,
    pad_token_id: int,
    use_checkpoint: bool,
) -> tuple[GRPOBatch, mx.array]:
    batch = batch_from_rollouts(
        model=model,
        completions=completions,
        rewards=rewards,
        group_size=group_size,
        pad_token_id=pad_token_id,
        use_checkpoint=use_checkpoint,
    )
    metrics = grpo_metrics_from_batch(
        model,
        batch,
        beta,
        pad_token_id,
        use_checkpoint=use_checkpoint,
    )
    mx.eval(  # Equivalence sync: materialize loss before Python tolerance comparison.
        metrics.loss,
        metrics.kl,
        batch.old_policy_logprobs,
        batch.reference_logprobs,
    )
    return batch, metrics.loss


def _phase1_gsm8k(args: argparse.Namespace) -> int:
    model, tokenizer, _ = load_policy_with_lora(
        model_id=args.model,
        config=LoRAConfig(rank=args.rank, scale=args.scale, dropout=args.dropout),
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    pad_token_id = pad_token_id_from_tokenizer(tokenizer)
    sampling = _sampling_config(args)
    algorithm = _algorithm_from_args(args)
    print(f"algorithm: {algorithm.name}")

    reward_history: list[float] = []
    kl_history: list[float] = []
    final_batch = None
    final_metrics = None
    final_completions = None

    for step in range(args.steps):
        prompt, answer = _gsm8k_prompt(args, step)
        completions = generate_group_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            group_size=args.group_size,
            config=sampling,
            seed=args.seed + step,
            use_chat_template=args.use_chat_template,
        )
        rewards: list[float] = []
        accuracies: list[float] = []
        formats: list[float] = []
        for completion in completions:
            total, accuracy, fmt = _completion_reward(completion.text, answer)
            rewards.append(total)
            accuracies.append(accuracy)
            formats.append(fmt)

        batch = batch_from_rollouts(
            model=model,
            completions=completions,
            rewards=rewards,
            group_size=args.group_size,
            pad_token_id=pad_token_id,
            use_checkpoint=args.checkpoint_completion_forward,
            algorithm=algorithm,
        )
        metrics = optimizer_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            beta=args.beta,
            pad_token_id=pad_token_id,
            use_checkpoint=args.checkpoint_completion_forward,
            algorithm=algorithm,
        )
        reward_history.append(metrics.mean_reward)
        kl_history.append(metrics.kl)
        final_batch = batch
        final_metrics = metrics
        final_completions = completions
        print(
            f"step={step + 1:02d} "
            f"reward={metrics.mean_reward:.4f} "
            f"accuracy={sum(accuracies) / len(accuracies):.4f} "
            f"format={sum(formats) / len(formats):.4f} "
            f"kl={metrics.kl:.6f} "
            f"clip={metrics.clip_fraction:.6f} "
            f"loss={metrics.loss:.6f}",
            flush=True,
        )

    first_reward, last_reward = reward_trend(reward_history)
    print(f"reward_first_window: {first_reward:.6f}")
    print(f"reward_last_window: {last_reward:.6f}")
    print(f"kl_max: {max(kl_history):.6f}")

    if final_batch is None or final_metrics is None or final_completions is None:
        raise RuntimeError("No Phase 1 batches were produced.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    max_prompt_len = max(len(completion.prompt_tokens) for completion in final_completions)
    max_completion_len = max(len(completion.completion_tokens) for completion in final_completions)
    prompt_tokens = [
        list(completion.prompt_tokens)
        + [pad_token_id] * (max_prompt_len - len(completion.prompt_tokens))
        for completion in final_completions
    ]
    completion_tokens = [
        list(completion.completion_tokens)
        + [pad_token_id] * (max_completion_len - len(completion.completion_tokens))
        for completion in final_completions
    ]
    mx.savez(
        str(output),
        prompt_tokens=mx.array(prompt_tokens, dtype=mx.int32),
        completion_tokens=mx.array(completion_tokens, dtype=mx.int32),
        old_policy_logprobs=final_batch.old_policy_logprobs,
        reference_logprobs=final_batch.reference_logprobs,
        mask=final_batch.mask,
        rewards=final_batch.rewards,
        advantages=final_batch.advantages,
        loss=mx.array(final_metrics.loss, dtype=mx.float32),
        kl=mx.array(final_metrics.kl, dtype=mx.float32),
        clip_fraction=mx.array(final_metrics.clip_fraction, dtype=mx.float32),
    )
    print(f"reference_output: {output}")
    return 0


def _phase2_equivalence(args: argparse.Namespace) -> int:
    model, tokenizer, _ = load_policy_with_lora(
        model_id=args.model,
        config=LoRAConfig(rank=args.rank, scale=args.scale, dropout=args.dropout),
    )
    pad_token_id = pad_token_id_from_tokenizer(tokenizer)
    sampling = _sampling_config(args)
    prompt_answers = [
        _gsm8k_prompt(args, args.prompt_index + index)
        for index in range(args.num_prompts)
    ]
    prompts = [prompt for prompt, _ in prompt_answers]
    answers = [answer for _, answer in prompt_answers]
    rollout_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompts": prompts,
        "group_size": args.group_size,
        "config": sampling,
        "seed": args.seed,
        "use_chat_template": args.use_chat_template,
    }
    phase2_rollout_kwargs = {
        **rollout_kwargs,
        "compile_decode_step": args.compile_decode_step,
        "batch_groups": args.batch_groups,
    }

    phase1_completions, phase1_elapsed, phase1_peak = _rollout_timed(
        generate_group_rollouts,
        "phase1",
        **rollout_kwargs,
    )
    phase2_completions, phase2_elapsed, phase2_peak = _rollout_timed(
        generate_prefix_cached_group_rollouts,
        _phase2_label(args.compile_decode_step, args.batch_groups),
        **phase2_rollout_kwargs,
    )

    phase1_tokens = tuple(completion.completion_tokens for completion in phase1_completions)
    phase2_tokens = tuple(completion.completion_tokens for completion in phase2_completions)
    token_equal = phase1_tokens == phase2_tokens

    phase1_rewards, phase1_accuracy, phase1_format = _completion_rewards(
        phase1_completions,
        answers,
    )
    phase2_rewards, phase2_accuracy, phase2_format = _completion_rewards(
        phase2_completions,
        answers,
    )
    phase1_batch, phase1_loss = _build_equivalence_batch(
        model=model,
        completions=phase1_completions,
        rewards=phase1_rewards,
        group_size=args.group_size,
        beta=args.beta,
        pad_token_id=pad_token_id,
        use_checkpoint=args.checkpoint_completion_forward,
    )
    phase2_batch, phase2_loss = _build_equivalence_batch(
        model=model,
        completions=phase2_completions,
        rewards=phase2_rewards,
        group_size=args.group_size,
        beta=args.beta,
        pad_token_id=pad_token_id,
        use_checkpoint=args.checkpoint_completion_forward,
    )

    policy_logprob_error = mx.max(
        mx.abs(phase1_batch.old_policy_logprobs - phase2_batch.old_policy_logprobs)
    )
    reference_logprob_error = mx.max(
        mx.abs(phase1_batch.reference_logprobs - phase2_batch.reference_logprobs)
    )
    loss_error = mx.abs(phase1_loss - phase2_loss)
    mx.eval(  # Equivalence sync: materialize comparison scalars before Python checks.
        policy_logprob_error,
        reference_logprob_error,
        loss_error,
    )

    print(f"token_equal: {token_equal}")
    print(f"phase1_reward_mean: {sum(phase1_rewards) / len(phase1_rewards):.6f}")
    print(f"phase2_reward_mean: {sum(phase2_rewards) / len(phase2_rewards):.6f}")
    print(f"phase1_accuracy: {phase1_accuracy:.6f} phase2_accuracy: {phase2_accuracy:.6f}")
    print(f"phase1_format: {phase1_format:.6f} phase2_format: {phase2_format:.6f}")
    print(f"policy_logprob_error: {float(policy_logprob_error.item()):.8e}")
    print(f"reference_logprob_error: {float(reference_logprob_error.item()):.8e}")
    print(f"loss_error: {float(loss_error.item()):.8e}")
    print(f"phase1_loss: {float(phase1_loss.item()):.8f}")
    print(f"phase2_loss: {float(phase2_loss.item()):.8f}")
    print(f"speedup_samples_per_s: {phase1_elapsed / phase2_elapsed:.6f}")
    print(f"peak_memory_delta_gb: {phase2_peak - phase1_peak:.6f}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(
            str(output),
            phase1_completion_tokens=_padded_tokens(phase1_tokens, pad_token_id),
            phase2_completion_tokens=_padded_tokens(phase2_tokens, pad_token_id),
            phase1_old_policy_logprobs=phase1_batch.old_policy_logprobs,
            phase2_old_policy_logprobs=phase2_batch.old_policy_logprobs,
            phase1_reference_logprobs=phase1_batch.reference_logprobs,
            phase2_reference_logprobs=phase2_batch.reference_logprobs,
            phase1_loss=phase1_loss,
            phase2_loss=phase2_loss,
        )
        print(f"phase2_reference_output: {output}")

    if not token_equal:
        raise SystemExit("Phase 2 equivalence failed: generated tokens differ.")
    if float(policy_logprob_error.item()) > args.tolerance:
        raise SystemExit("Phase 2 equivalence failed: policy logprobs differ.")
    if float(reference_logprob_error.item()) > args.tolerance:
        raise SystemExit("Phase 2 equivalence failed: reference logprobs differ.")
    if float(loss_error.item()) > args.tolerance:
        raise SystemExit("Phase 2 equivalence failed: losses differ.")
    print("phase2_equivalence: passed")
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

    toy = subparsers.add_parser("phase1-toy-gate", help="Run the hand-computed GRPO math gate.")
    toy.set_defaults(func=_phase1_toy_gate)

    gsm8k = subparsers.add_parser(
        "phase1-gsm8k",
        help="Run the naive Phase 1 GRPO loop on built-in GSM8K-style samples.",
    )
    gsm8k.add_argument("--model", default=DEFAULT_MODEL_ID)
    gsm8k.add_argument("--steps", type=int, default=20)
    gsm8k.add_argument("--group-size", type=int, default=2)
    gsm8k.add_argument("--max-tokens", type=int, default=64)
    gsm8k.add_argument("--temperature", type=float, default=0.7)
    gsm8k.add_argument("--top-p", type=float, default=0.95)
    gsm8k.add_argument("--min-p", type=float, default=0.0)
    gsm8k.add_argument("--top-k", type=int, default=0)
    gsm8k.add_argument("--seed", type=int, default=7)
    gsm8k.add_argument("--beta", type=float, default=0.04)
    gsm8k.add_argument("--learning-rate", type=float, default=1e-5)
    gsm8k.add_argument(
        "--algorithm",
        choices=["grpo", "dr-grpo", "dapo", "gspo"],
        default="grpo",
    )
    gsm8k.add_argument(
        "--clip-low",
        default="default",
        help="Algorithm clip-low epsilon, or 'none' to disable clipping.",
    )
    gsm8k.add_argument(
        "--clip-high",
        default="default",
        help="Algorithm clip-high epsilon, or 'none' to disable clipping.",
    )
    gsm8k.add_argument(
        "--gspo-importance",
        choices=["sequence", "token"],
        default="sequence",
    )
    gsm8k.add_argument("--drgrpo-normalize-rewards", action="store_true")
    gsm8k.add_argument(
        "--drgrpo-loss-reduction",
        choices=["sequence_max_tokens", "token_mean"],
        default="sequence_max_tokens",
    )
    gsm8k.add_argument("--drgrpo-max-tokens", type=int, default=None)
    gsm8k.add_argument("--rank", type=int, default=8)
    gsm8k.add_argument("--scale", type=float, default=20.0)
    gsm8k.add_argument("--dropout", type=float, default=0.0)
    gsm8k.add_argument("--use-chat-template", action="store_true")
    gsm8k.add_argument("--checkpoint-completion-forward", action="store_true")
    gsm8k.add_argument("--output", default="reference_outputs/phase1_reference.npz")
    gsm8k.set_defaults(func=_phase1_gsm8k)

    phase2 = subparsers.add_parser(
        "phase2-equivalence",
        help="Compare naive Phase 1 rollouts against the Phase 2 prefix-cache rollout.",
    )
    phase2.add_argument("--model", default=DEFAULT_MODEL_ID)
    phase2.add_argument("--prompt-index", type=int, default=0)
    phase2.add_argument("--num-prompts", type=int, default=1)
    phase2.add_argument("--group-size", type=int, default=4)
    phase2.add_argument("--max-tokens", type=int, default=32)
    phase2.add_argument("--temperature", type=float, default=0.7)
    phase2.add_argument("--top-p", type=float, default=0.95)
    phase2.add_argument("--min-p", type=float, default=0.0)
    phase2.add_argument("--top-k", type=int, default=0)
    phase2.add_argument("--seed", type=int, default=7)
    phase2.add_argument("--beta", type=float, default=0.08)
    phase2.add_argument("--rank", type=int, default=8)
    phase2.add_argument("--scale", type=float, default=20.0)
    phase2.add_argument("--dropout", type=float, default=0.0)
    phase2.add_argument("--use-chat-template", action="store_true")
    phase2.add_argument("--checkpoint-completion-forward", action="store_true")
    phase2.add_argument("--compile-decode-step", action="store_true")
    phase2.add_argument("--batch-groups", action="store_true")
    phase2.add_argument("--tolerance", type=float, default=1e-4)
    phase2.add_argument("--output", default="reference_outputs/phase2_prefix_reference.npz")
    phase2.set_defaults(func=_phase2_equivalence)
    return parser


def _phase2_label(compile_decode_step: bool, batch_groups: bool) -> str:
    if compile_decode_step and batch_groups:
        return "phase2_batched_compiled"
    if batch_groups:
        return "phase2_batched"
    if compile_decode_step:
        return "phase2_compiled"
    return "phase2_prefix"


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
