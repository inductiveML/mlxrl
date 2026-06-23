"""Command line entrypoints for mlxrl."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from mlxrl.algo.grpo import approximate_kl, group_normalize_rewards, grpo_loss
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
from mlxrl.train.grpo import batch_from_rollouts, optimizer_step, reward_trend


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


def _phase1_gsm8k(args: argparse.Namespace) -> int:
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

    reward_history: list[float] = []
    kl_history: list[float] = []
    final_batch = None
    final_metrics = None
    final_completions = None

    for step in range(args.steps):
        example = MINI_GSM8K[step % len(MINI_GSM8K)]
        prompt = (
            format_gsm8k_prompt(example)
            if args.use_chat_template
            else format_gsm8k_answer_only_prompt(example)
        )
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
            total, accuracy, fmt = _completion_reward(completion.text, example.answer)
            rewards.append(total)
            accuracies.append(accuracy)
            formats.append(fmt)

        batch = batch_from_rollouts(
            model=model,
            completions=completions,
            rewards=rewards,
            group_size=args.group_size,
            pad_token_id=pad_token_id,
        )
        metrics = optimizer_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            beta=args.beta,
            pad_token_id=pad_token_id,
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
    )
    print(f"reference_output: {output}")
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
    gsm8k.add_argument("--rank", type=int, default=8)
    gsm8k.add_argument("--scale", type=float, default=20.0)
    gsm8k.add_argument("--dropout", type=float, default=0.0)
    gsm8k.add_argument("--use-chat-template", action="store_true")
    gsm8k.add_argument("--output", default="reference_outputs/phase1_reference.npz")
    gsm8k.set_defaults(func=_phase1_gsm8k)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
