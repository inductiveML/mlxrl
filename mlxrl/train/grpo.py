"""One-step GRPO training over LoRA adapter parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxrl.algo.grpo import AlgorithmLossMetrics, GRPOAlgorithm, PolicyAlgorithm
from mlxrl.policy.logprobs import (
    CompletionLogprobs,
    adapters_disabled,
    completion_logprobs,
    prefix_cached_completion_logprobs,
)
from mlxrl.rollout.naive import Completion


@dataclass(frozen=True)
class GRPOBatch:
    """Reference data for one GRPO optimizer step."""

    prompt_token_ids: tuple[tuple[int, ...], ...]
    completion_token_ids: tuple[tuple[int, ...], ...]
    rewards: mx.array
    advantages: mx.array
    old_policy_logprobs: mx.array
    reference_logprobs: mx.array
    mask: mx.array


@dataclass(frozen=True)
class StepMetrics:
    """Python scalar diagnostics emitted after one optimizer step."""

    loss: float
    policy_gradient_loss: float
    kl: float
    mean_ratio: float
    clip_fraction: float
    mean_reward: float


def batch_from_rollouts(
    model: nn.Module,
    completions: Sequence[Completion],
    rewards: Sequence[float],
    group_size: int,
    pad_token_id: int,
    use_checkpoint: bool = False,
    algorithm: PolicyAlgorithm | None = None,
) -> GRPOBatch:
    """Load rollout old-policy logprobs, compute ref logprobs, and advantages."""

    if len(completions) != len(rewards):
        raise ValueError("completions and rewards must have the same length.")
    if not completions:
        raise ValueError("At least one completion is required.")
    del use_checkpoint

    prompt_token_ids = tuple(completion.prompt_tokens for completion in completions)
    completion_token_ids = tuple(completion.completion_tokens for completion in completions)
    old_policy = old_policy_logprobs_from_rollouts(completions)
    with adapters_disabled(model):
        reference = prefix_cached_completion_logprobs(
            model,
            prompt_token_ids,
            completion_token_ids,
            pad_token_id,
        )
        mx.eval(  # Logprob sync: freeze rollout/ref logprobs before adapter mutation.
            old_policy.logprobs,
            old_policy.mask,
            reference.logprobs,
            reference.mask,
        )
    active_algorithm = algorithm or GRPOAlgorithm()
    reward_array = mx.array(list(rewards), dtype=mx.float32)
    advantages = active_algorithm.advantages(reward_array, group_size=group_size)
    return GRPOBatch(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        rewards=reward_array,
        advantages=advantages,
        old_policy_logprobs=mx.stop_gradient(old_policy.logprobs),
        reference_logprobs=mx.stop_gradient(reference.logprobs),
        mask=old_policy.mask,
    )


def old_policy_logprobs_from_rollouts(
    completions: Sequence[Completion],
) -> CompletionLogprobs:
    """Pad rollout-captured old-policy logprobs into the training tensor shape."""

    if not completions:
        raise ValueError("At least one completion is required.")
    max_completion_len = max(len(completion.completion_tokens) for completion in completions)
    if max_completion_len == 0:
        raise ValueError("At least one completion token is required.")

    logprob_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    for completion in completions:
        token_count = len(completion.completion_tokens)
        if len(completion.old_policy_logprobs) != token_count:
            raise ValueError(
                "Each completion must carry one old-policy logprob per token."
            )
        pad_count = max_completion_len - token_count
        logprob_rows.append(
            list(completion.old_policy_logprobs) + [0.0] * pad_count
        )
        mask_rows.append([1.0] * token_count + [0.0] * pad_count)

    return CompletionLogprobs(
        logprobs=mx.array(logprob_rows, dtype=mx.float32),
        mask=mx.array(mask_rows, dtype=mx.float32),
    )


def grpo_metrics_from_batch(
    model: nn.Module,
    batch: GRPOBatch,
    beta: float,
    pad_token_id: int,
    use_checkpoint: bool = False,
    algorithm: PolicyAlgorithm | None = None,
) -> AlgorithmLossMetrics:
    """Recompute policy logprobs and evaluate GRPO metrics."""

    active_algorithm = algorithm or GRPOAlgorithm()
    current = completion_logprobs(
        model,
        batch.prompt_token_ids,
        batch.completion_token_ids,
        pad_token_id,
        use_checkpoint=use_checkpoint,
    )
    return active_algorithm.loss(
        policy_logprobs=current.logprobs,
        old_policy_logprobs=batch.old_policy_logprobs,
        reference_logprobs=batch.reference_logprobs,
        advantages=batch.advantages,
        mask=batch.mask,
        beta=beta,
    )


def optimizer_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: GRPOBatch,
    beta: float,
    pad_token_id: int,
    use_checkpoint: bool = False,
    algorithm: PolicyAlgorithm | None = None,
) -> StepMetrics:
    """Run value_and_grad over currently trainable adapter parameters once."""

    active_algorithm = algorithm or GRPOAlgorithm()

    def loss_fn(
        model: nn.Module,
    ) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array, mx.array]]:
        metrics = grpo_metrics_from_batch(
            model,
            batch,
            beta,
            pad_token_id,
            use_checkpoint=use_checkpoint,
            algorithm=active_algorithm,
        )
        return metrics.loss, (
            metrics.policy_gradient_loss,
            metrics.kl,
            metrics.mean_ratio,
            metrics.clip_fraction,
        )

    value_and_grad = nn.value_and_grad(model, loss_fn)
    (loss, (policy_gradient_loss, kl, mean_ratio, clip_fraction)), gradients = (
        value_and_grad(model)
    )
    mean_reward = mx.mean(batch.rewards)
    mx.eval(  # Optimizer pre-step sync: freeze gradients/diagnostics before weight mutation.
        gradients,
        loss,
        policy_gradient_loss,
        kl,
        mean_ratio,
        clip_fraction,
        mean_reward,
    )
    optimizer.update(model, gradients)
    mx.eval(  # Optimizer sync: materialize updated adapter weights and optimizer state.
        model.state,
        optimizer.state,
    )
    return StepMetrics(
        loss=float(loss.item()),
        policy_gradient_loss=float(policy_gradient_loss.item()),
        kl=float(kl.item()),
        mean_ratio=float(mean_ratio.item()),
        clip_fraction=float(clip_fraction.item()),
        mean_reward=float(mean_reward.item()),
    )


def reward_trend(values: Sequence[float], window: int = 5) -> tuple[float, float]:
    """Return first-window and last-window means for a short sanity run."""

    if not values:
        raise ValueError("At least one reward value is required.")
    window = max(1, min(window, len(values)))
    first = sum(values[:window]) / window
    last = sum(values[-window:]) / window
    return first, last
