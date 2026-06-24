"""Command line entrypoints for mlxrl."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.optimizers as optim

from mlxrl.algo.gigpo import GiGPOAlgorithm
from mlxrl.algo.grpo import (
    DAPOAlgorithm,
    DrGRPOAlgorithm,
    GRPOAlgorithm,
    GSPOAlgorithm,
    RLOOAlgorithm,
    approximate_kl,
    group_normalize_rewards,
    grpo_loss,
)
from mlxrl.algorithm import Algorithm
from mlxrl.config import (
    AlgorithmConfig,
    ConfigError,
    TrainConfig,
    memory_fit,
    normalize_algorithm_name,
    validate_output_path,
)
from mlxrl.data.gsm8k import (
    MINI_GSM8K,
    format_gsm8k_answer_only_prompt,
    format_gsm8k_prompt,
)
from mlxrl.data.rewards import accuracy_reward, format_reward
from mlxrl.env import EnvFactory, RecurringStateTextEnv, SingleTurnRewardEnv
from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
from mlxrl.policy.model import (
    DEFAULT_MODEL_ID,
    LoRAConfig,
    encode_prompt,
    load_policy_with_lora,
)
from mlxrl.rollout.agentic import RolloutMode, generate_agentic_trajectories
from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
from mlxrl.rollout.optimized import generate_prefix_cached_group_rollouts
from mlxrl.train.grpo import (
    GRPOBatch,
    batch_from_rollouts,
    grpo_metrics_from_batch,
    optimizer_step,
    reward_trend,
)
from mlxrl.train.trajectory import (
    batch_from_trajectories,
    optimizer_step_trajectory,
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
    print(
        "lora_modules_per_layer: "
        + ", ".join(str(count) for count in report.lora_modules_per_layer)
    )
    print(f"quantized_linear_count: {report.quantized_linear_count}")
    print(f"lora_module_count: {report.lora_module_count}")
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


def _algorithm_from_args(args: argparse.Namespace) -> Algorithm:
    return _algorithm_from_algorithm_config(
        AlgorithmConfig(
            name=args.algorithm,
            clip_low=None if args.clip_low == "default" else _clip_value(args.clip_low, 0.2),
            clip_high=None if args.clip_high == "default" else _clip_value(args.clip_high, 0.28),
            dapo_dynamic_sampling=args.dapo_dynamic_sampling,
            gspo_importance=args.gspo_importance,
            drgrpo_normalize_rewards=args.drgrpo_normalize_rewards,
            drgrpo_loss_reduction=args.drgrpo_loss_reduction,
            drgrpo_max_tokens=args.drgrpo_max_tokens,
        ),
        max_completion_len=args.max_tokens,
        dapo_default_clip_low=0.2 if args.clip_low == "default" else None,
        dapo_default_clip_high=0.28 if args.clip_high == "default" else None,
        gspo_default_clip_low=3e-4 if args.clip_low == "default" else None,
        gspo_default_clip_high=4e-4 if args.clip_high == "default" else None,
    )


def _algorithm_from_algorithm_config(
    algorithm: AlgorithmConfig,
    max_completion_len: int,
    *,
    dapo_default_clip_low: float | None = 0.2,
    dapo_default_clip_high: float | None = 0.28,
    gspo_default_clip_low: float | None = 3e-4,
    gspo_default_clip_high: float | None = 4e-4,
) -> Algorithm:
    name = normalize_algorithm_name(algorithm.name)
    if name == "grpo":
        return GRPOAlgorithm()
    if name == "dr-grpo":
        return DrGRPOAlgorithm(
            normalize_rewards=algorithm.drgrpo_normalize_rewards,
            loss_reduction=algorithm.drgrpo_loss_reduction,
            max_tokens=algorithm.drgrpo_max_tokens or max_completion_len,
        )
    if name == "dapo":
        return DAPOAlgorithm(
            clip_low=(
                algorithm.clip_low
                if algorithm.clip_low is not None
                else dapo_default_clip_low
            ),
            clip_high=(
                algorithm.clip_high
                if algorithm.clip_high is not None
                else dapo_default_clip_high
            ),
            dynamic_sampling=algorithm.dapo_dynamic_sampling,
        )
    if name == "gspo":
        return GSPOAlgorithm(
            importance=algorithm.gspo_importance,
            clip_low=(
                algorithm.clip_low
                if algorithm.clip_low is not None
                else gspo_default_clip_low
            ),
            clip_high=(
                algorithm.clip_high
                if algorithm.clip_high is not None
                else gspo_default_clip_high
            ),
        )
    if name == "rloo":
        return RLOOAlgorithm()
    raise ValueError(f"Unknown algorithm {algorithm.name!r}.")


def _algorithm_from_config(config: TrainConfig) -> Algorithm:
    return _algorithm_from_algorithm_config(
        config.algorithm,
        max_completion_len=config.max_completion_len,
    )


def _trajectory_algorithm_from_config(config: TrainConfig) -> GiGPOAlgorithm:
    algorithm = config.algorithm
    name = normalize_algorithm_name(algorithm.name)
    if name != "gigpo":
        raise ValueError(f"Unknown trajectory algorithm {algorithm.name!r}.")
    return GiGPOAlgorithm(
        omega=algorithm.gigpo_omega,
        gamma=algorithm.gigpo_gamma,
        normalization=cast(Any, algorithm.gigpo_normalization),
        clip_low=algorithm.clip_low,
        clip_high=algorithm.clip_high,
    )


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


def _validated_output_path(raw: str, label: str = "output") -> Path:
    try:
        return validate_output_path(raw, label)
    except ConfigError as error:
        raise SystemExit(f"Invalid {label}: {error}") from error


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
        algorithm=GRPOAlgorithm(),
        use_checkpoint=use_checkpoint,
        compute_reference=beta != 0.0,
    )
    metrics = grpo_metrics_from_batch(
        model,
        batch,
        beta,
        pad_token_id,
        algorithm=GRPOAlgorithm(),
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
        config=LoRAConfig(
            rank=args.rank,
            scale=args.scale,
            dropout=args.dropout,
            grad_checkpoint=args.checkpoint_completion_forward,
        ),
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    pad_token_id = pad_token_id_from_tokenizer(tokenizer)
    sampling = _sampling_config(args)
    algorithm = getattr(args, "_algorithm_override", None) or _algorithm_from_args(args)
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
            use_checkpoint=args.checkpoint_completion_forward,
            micro_batch_size=getattr(args, "micro_batch_size", 0),
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
    output = _validated_output_path(args.output)
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


def _namespace_from_train_config(config: TrainConfig) -> argparse.Namespace:
    return argparse.Namespace(
        model=config.model_id,
        steps=config.steps,
        group_size=config.group_size,
        max_tokens=config.max_completion_len,
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        min_p=config.sampling.min_p,
        top_k=config.sampling.top_k,
        seed=config.seed,
        beta=config.kl_beta,
        learning_rate=config.optimizer.learning_rate,
        rank=config.rank,
        scale=config.scale,
        dropout=config.dropout,
        use_chat_template=config.use_chat_template,
        checkpoint_completion_forward=config.gradient_checkpointing,
        output=config.output,
        micro_batch_size=config.micro_batch_size,
        algorithm_config=config.algorithm,
        algorithm=config.algorithm.name,
        max_turns=config.max_turns,
        rollout_mode=config.rollout_mode,
        env_name=config.env_name,
    )


def _apply_train_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    replacements: dict[str, Any] = {}
    if args.model is not None:
        replacements["model_id"] = args.model
    if args.steps is not None:
        replacements["steps"] = args.steps
    if args.group_size is not None:
        replacements["group_size"] = args.group_size
    if args.max_tokens is not None:
        replacements["max_completion_len"] = args.max_tokens
    if args.beta is not None:
        replacements["kl_beta"] = args.beta
    if args.seed is not None:
        replacements["seed"] = args.seed
    if args.output is not None:
        replacements["output"] = args.output
    if args.algorithm is not None:
        replacements["algorithm"] = replace(
            config.algorithm,
            name=args.algorithm,
        )
    updated = replace(config, **replacements)
    updated.validate()
    return updated


def _phase_train(args: argparse.Namespace) -> int:
    try:
        config = _apply_train_overrides(TrainConfig.from_file(args.config), args)
    except ConfigError as error:
        raise SystemExit(f"Invalid config: {error}") from error

    available_memory_gb = args.available_memory_gb
    if available_memory_gb is None and config.iogpu_wired_limit_mb is not None:
        available_memory_gb = config.iogpu_wired_limit_mb / 1024.0
    if available_memory_gb is not None:
        estimate = memory_fit(config, available_memory_gb)
        print(
            f"memory_estimated_peak_gb: {estimate.estimated_peak_gb:.3f} "
            f"available_gb: {estimate.available_gb:.3f} "
            f"confidence: {estimate.display_label()} "
            f"fits: {estimate.fits}"
        )
        if not estimate.fits:
            print(f"memory_warning: {estimate.warning}")
            if estimate.suggestions:
                print("memory_suggestions: " + ", ".join(estimate.suggestions))
            if args.auto_fit:
                config = estimate.suggested_config
                print(
                    f"memory_auto_fit_peak_gb: {estimate.suggested_peak_gb:.3f} "
                    f"group_size: {config.group_size} "
                    f"max_completion_len: {config.max_completion_len} "
                    f"gradient_checkpointing: {config.gradient_checkpointing}"
                )
            elif not args.dry_run:
                raise SystemExit("Config is predicted to exceed available memory.")
    if args.dry_run:
        print("config_valid: true")
        print(f"algorithm: {normalize_algorithm_name(config.algorithm.name)}")
        return 0

    if normalize_algorithm_name(config.algorithm.name) == "gigpo":
        return _phase_agentic_train(config)

    namespace = _namespace_from_train_config(config)
    namespace._algorithm_override = _algorithm_from_config(config)
    return _phase1_gsm8k(namespace)


def _phase_agentic_train(config: TrainConfig) -> int:
    model, tokenizer, _ = load_policy_with_lora(
        model_id=config.model_id,
        config=LoRAConfig(
            rank=config.rank,
            scale=config.scale,
            dropout=config.dropout,
            grad_checkpoint=config.gradient_checkpointing,
        ),
    )
    optimizer = optim.Adam(learning_rate=config.optimizer.learning_rate)
    pad_token_id = pad_token_id_from_tokenizer(tokenizer)
    sampling = SamplingConfig(
        max_tokens=config.max_completion_len,
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        min_p=config.sampling.min_p,
        top_k=config.sampling.top_k,
    )
    algorithm = _trajectory_algorithm_from_config(config)
    env_factory = _env_factory_from_config(config)
    reward_history: list[float] = []
    kl_history: list[float] = []
    final_batch = None
    final_metrics = None
    print(f"algorithm: {algorithm.name}")
    print(f"env_name: {config.env_name}")
    print(f"rollout_mode: {config.rollout_mode}")

    for step in range(config.steps):
        task = f"{config.env_name}: task {step}"
        trajectories = generate_agentic_trajectories(
            model=model,
            tokenizer=tokenizer,
            env_factory=env_factory,
            tasks=(task,),
            group_size=config.group_size,
            sampling=sampling,
            seed=config.seed + step,
            rollout_mode=cast(RolloutMode, config.rollout_mode),
        )
        batch = batch_from_trajectories(
            model=model,
            trajectories=trajectories,
            group_size=config.group_size,
            pad_token_id=pad_token_id,
            algorithm=algorithm,
            use_checkpoint=config.gradient_checkpointing,
            compute_reference=config.kl_beta != 0.0,
        )
        metrics = optimizer_step_trajectory(
            model=model,
            optimizer=optimizer,
            batch=batch,
            beta=config.kl_beta,
            pad_token_id=pad_token_id,
            algorithm=algorithm,
            use_checkpoint=config.gradient_checkpointing,
            micro_batch_size=config.micro_batch_size,
        )
        reward_history.append(metrics.mean_reward)
        kl_history.append(metrics.kl)
        final_batch = batch
        final_metrics = metrics
        print(
            f"step={step + 1:02d} "
            f"reward={metrics.mean_reward:.4f} "
            f"kl={metrics.kl:.6f} "
            f"ratio={metrics.mean_ratio:.6f} "
            f"clip={metrics.clip_fraction:.6f} "
            f"loss={metrics.loss:.6f}",
            flush=True,
        )

    if final_batch is None or final_metrics is None:
        raise RuntimeError("No agentic batches were produced.")
    first_reward, last_reward = reward_trend(reward_history)
    print(f"reward_first_window: {first_reward:.6f}")
    print(f"reward_last_window: {last_reward:.6f}")
    print(f"kl_max: {max(kl_history):.6f}")

    output = _validated_output_path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(
        str(output),
        rewards=final_batch.rewards,
        step_advantages=final_batch.step_advantages,
        advantages=final_batch.advantages,
        old_policy_logprobs=final_batch.old_policy_logprobs,
        reference_logprobs=final_batch.reference_logprobs,
        action_mask=final_batch.action_mask,
        loss=mx.array(final_metrics.loss, dtype=mx.float32),
        kl=mx.array(final_metrics.kl, dtype=mx.float32),
    )
    print(f"reference_output: {output}")
    return 0


def _env_factory_from_config(config: TrainConfig) -> EnvFactory:
    env_name = config.env_name
    if env_name == "recurring-text":
        return lambda task, seed, group: RecurringStateTextEnv(
            task=str(task),
            max_turns=config.max_turns,
        )
    if env_name == "single-turn":
        return lambda task, seed, group: SingleTurnRewardEnv(
            prompt=str(task),
            reward_fn=lambda text: 1.0 if "finish" in text.lower() else 0.0,
        )
    raise ValueError(f"Unknown env_name {env_name!r}.")


def _phase2_equivalence(args: argparse.Namespace) -> int:
    model, tokenizer, _ = load_policy_with_lora(
        model_id=args.model,
        config=LoRAConfig(
            rank=args.rank,
            scale=args.scale,
            dropout=args.dropout,
            grad_checkpoint=args.checkpoint_completion_forward,
        ),
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
        output = _validated_output_path(args.output)
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
        choices=["grpo", "dr-grpo", "dapo", "gspo", "rloo"],
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
    gsm8k.add_argument("--dapo-dynamic-sampling", action="store_true")
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
    gsm8k.add_argument("--micro-batch-size", type=int, default=0)
    gsm8k.add_argument("--output", default="reference_outputs/phase1_reference.npz")
    gsm8k.set_defaults(func=_phase1_gsm8k)

    train = subparsers.add_parser(
        "train",
        help="Run config-driven GRPO-family training.",
    )
    train.add_argument("--config", required=True)
    train.add_argument("--model", default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--group-size", type=int, default=None)
    train.add_argument("--max-tokens", type=int, default=None)
    train.add_argument("--algorithm", choices=["grpo", "dr-grpo", "dapo", "gspo", "rloo", "gigpo"])
    train.add_argument("--beta", type=float, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--output", default=None)
    train.add_argument("--available-memory-gb", type=float, default=None)
    train.add_argument("--auto-fit", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=_phase_train)

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
