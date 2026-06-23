"""One-step GRPO training over LoRA adapter parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxrl.algo.grpo import GRPOLossMetrics, group_normalize_rewards, grpo_loss
from mlxrl.policy.logprobs import completion_logprobs, dual_logprobs
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
    mean_reward: float


def batch_from_rollouts(
    model: nn.Module,
    completions: Sequence[Completion],
    rewards: Sequence[float],
    group_size: int,
    pad_token_id: int,
) -> GRPOBatch:
    """Compute old policy/ref logprobs and group-normalized advantages."""

    if len(completions) != len(rewards):
        raise ValueError("completions and rewards must have the same length.")
    if not completions:
        raise ValueError("At least one completion is required.")

    prompt_token_ids = tuple(completion.prompt_tokens for completion in completions)
    completion_token_ids = tuple(completion.completion_tokens for completion in completions)
    dual = dual_logprobs(model, prompt_token_ids, completion_token_ids, pad_token_id)
    reward_array = mx.array(list(rewards), dtype=mx.float32)
    advantages = group_normalize_rewards(reward_array, group_size=group_size)
    return GRPOBatch(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        rewards=reward_array,
        advantages=advantages,
        old_policy_logprobs=mx.stop_gradient(dual.policy),
        reference_logprobs=mx.stop_gradient(dual.reference),
        mask=dual.mask,
    )


def grpo_metrics_from_batch(
    model: nn.Module,
    batch: GRPOBatch,
    beta: float,
    pad_token_id: int,
) -> GRPOLossMetrics:
    """Recompute policy logprobs and evaluate GRPO metrics."""

    current = completion_logprobs(
        model,
        batch.prompt_token_ids,
        batch.completion_token_ids,
        pad_token_id,
    )
    return grpo_loss(
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
) -> StepMetrics:
    """Run value_and_grad over currently trainable adapter parameters once."""

    def loss_fn(model: nn.Module) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array]]:
        metrics = grpo_metrics_from_batch(model, batch, beta, pad_token_id)
        return metrics.loss, (
            metrics.policy_gradient_loss,
            metrics.kl,
            metrics.mean_ratio,
        )

    value_and_grad = nn.value_and_grad(model, loss_fn)
    (loss, (policy_gradient_loss, kl, mean_ratio)), gradients = value_and_grad(model)
    optimizer.update(model, gradients)
    mean_reward = mx.mean(batch.rewards)
    mx.eval(  # Optimizer sync: materialize updated adapter weights and scalar diagnostics.
        model.state,
        optimizer.state,
        loss,
        policy_gradient_loss,
        kl,
        mean_ratio,
        mean_reward,
    )
    return StepMetrics(
        loss=float(loss.item()),
        policy_gradient_loss=float(policy_gradient_loss.item()),
        kl=float(kl.item()),
        mean_ratio=float(mean_ratio.item()),
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
