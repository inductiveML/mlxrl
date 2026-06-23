"""Pure GRPO advantage and loss math."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class GRPOLossMetrics:
    """Scalar diagnostics for a GRPO loss batch."""

    loss: mx.array
    policy_gradient_loss: mx.array
    kl: mx.array
    mean_ratio: mx.array


def group_normalize_rewards(
    rewards: mx.array,
    group_size: int,
    eps: float = 1e-8,
) -> mx.array:
    """Normalize rewards within each prompt group."""

    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if rewards.size % group_size != 0:
        raise ValueError(
            f"Reward count {rewards.size} must be divisible by group_size={group_size}."
        )

    grouped = rewards.reshape((-1, group_size)).astype(mx.float32)
    mean = mx.mean(grouped, axis=1, keepdims=True)
    variance = mx.mean(mx.square(grouped - mean), axis=1, keepdims=True)
    advantages = (grouped - mean) / mx.sqrt(variance + eps)
    return advantages.reshape((-1,))


def approximate_kl(policy_logprobs: mx.array, reference_logprobs: mx.array) -> mx.array:
    """Compute the non-negative GRPO/TRL approximate per-token KL."""

    log_ratio = reference_logprobs - policy_logprobs
    return mx.exp(log_ratio) - log_ratio - 1.0


def masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    """Mean over valid completion tokens."""

    mask = mask.astype(mx.float32)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(values * mask) / denominator


def grpo_loss(
    policy_logprobs: mx.array,
    old_policy_logprobs: mx.array,
    reference_logprobs: mx.array,
    advantages: mx.array,
    mask: mx.array,
    beta: float,
) -> GRPOLossMetrics:
    """Compute unclipped GRPO loss from token logprobs and group advantages."""

    ratio = mx.exp(policy_logprobs - mx.stop_gradient(old_policy_logprobs))
    token_policy_gradient = -ratio * advantages[:, None]
    token_kl = approximate_kl(policy_logprobs, reference_logprobs)
    loss = masked_mean(token_policy_gradient + beta * token_kl, mask)
    policy_gradient_loss = masked_mean(token_policy_gradient, mask)
    kl = masked_mean(token_kl, mask)
    mean_ratio = masked_mean(ratio, mask)
    return GRPOLossMetrics(
        loss=loss,
        policy_gradient_loss=policy_gradient_loss,
        kl=kl,
        mean_ratio=mean_ratio,
    )

