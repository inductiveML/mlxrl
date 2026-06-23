"""Pure advantage and policy-loss math for GRPO-family algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx


@dataclass(frozen=True)
class AlgorithmLossMetrics:
    """Scalar diagnostics for one algorithm loss batch."""

    loss: mx.array
    policy_gradient_loss: mx.array
    kl: mx.array
    mean_ratio: mx.array
    clip_fraction: mx.array


GRPOLossMetrics = AlgorithmLossMetrics


class PolicyAlgorithm(Protocol):
    """Minimal interface for a GRPO-family loss variant."""

    @property
    def name(self) -> str:
        """Stable algorithm name for logs and outputs."""
        ...

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        """Compute per-completion advantages from scalar rewards."""
        ...

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        """Compute loss and diagnostics from token logprobs."""
        ...


@dataclass(frozen=True)
class GRPOAlgorithm:
    """Thin unclipped GRPO used by the Phase 1 reference path."""

    name: str = "grpo"

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return group_normalize_rewards(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        ratio = token_importance_ratio(policy_logprobs, old_policy_logprobs)
        token_policy_gradient = -ratio * advantages[:, None]
        return _token_loss_metrics(
            token_policy_gradient=token_policy_gradient,
            ratio=ratio,
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
            mask=mask,
            beta=beta,
        )


@dataclass(frozen=True)
class DrGRPOAlgorithm:
    """Dr. GRPO, with decoupled reward and length normalization knobs."""

    normalize_rewards: bool = False
    loss_reduction: str = "sequence_max_tokens"
    max_tokens: int | None = None
    name: str = "dr-grpo"

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        if self.normalize_rewards:
            return group_normalize_rewards(rewards, group_size)
        return group_center_rewards(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        ratio = token_importance_ratio(policy_logprobs, old_policy_logprobs)
        token_policy_gradient = -ratio * advantages[:, None]
        token_kl = approximate_kl(policy_logprobs, reference_logprobs)
        token_loss = token_policy_gradient + beta * token_kl
        if self.loss_reduction == "token_mean":
            loss = masked_mean(token_loss, mask)
            policy_gradient_loss = masked_mean(token_policy_gradient, mask)
        elif self.loss_reduction == "sequence_max_tokens":
            denominator = float(self.max_tokens or mask.shape[1])
            loss = mx.mean(mx.sum(token_loss * mask, axis=1) / denominator)
            policy_gradient_loss = mx.mean(
                mx.sum(token_policy_gradient * mask, axis=1) / denominator
            )
        else:
            raise ValueError(f"Unknown Dr. GRPO loss_reduction={self.loss_reduction!r}.")
        return AlgorithmLossMetrics(
            loss=loss,
            policy_gradient_loss=policy_gradient_loss,
            kl=masked_mean(token_kl, mask),
            mean_ratio=masked_mean(ratio, mask),
            clip_fraction=mx.array(0.0, dtype=mx.float32),
        )


@dataclass(frozen=True)
class DAPOAlgorithm:
    """DAPO's decoupled low/high token-ratio clipping."""

    clip_low: float | None = 0.2
    clip_high: float | None = 0.28
    name: str = "dapo"

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return group_normalize_rewards(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        ratio = token_importance_ratio(policy_logprobs, old_policy_logprobs)
        clipped_ratio = clip_ratio(ratio, self.clip_low, self.clip_high)
        surrogate = mx.minimum(
            ratio * advantages[:, None],
            clipped_ratio * advantages[:, None],
        )
        token_policy_gradient = -surrogate
        return _token_loss_metrics(
            token_policy_gradient=token_policy_gradient,
            ratio=ratio,
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
            mask=mask,
            beta=beta,
            clipped_ratio=clipped_ratio,
        )


@dataclass(frozen=True)
class GSPOAlgorithm:
    """GSPO with sequence-level or token-level importance weighting."""

    importance: str = "sequence"
    clip_low: float | None = 3e-4
    clip_high: float | None = 4e-4
    name: str = "gspo"

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return group_normalize_rewards(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        if self.importance == "token":
            ratio = token_importance_ratio(policy_logprobs, old_policy_logprobs)
            clipped_ratio = clip_ratio(ratio, self.clip_low, self.clip_high)
            surrogate = mx.minimum(
                ratio * advantages[:, None],
                clipped_ratio * advantages[:, None],
            )
            return _token_loss_metrics(
                token_policy_gradient=-surrogate,
                ratio=ratio,
                policy_logprobs=policy_logprobs,
                reference_logprobs=reference_logprobs,
                mask=mask,
                beta=beta,
                clipped_ratio=clipped_ratio,
            )
        if self.importance != "sequence":
            raise ValueError(f"Unknown GSPO importance={self.importance!r}.")

        sequence_ratio = sequence_importance_ratio(
            policy_logprobs,
            old_policy_logprobs,
            mask,
        )
        clipped_ratio = clip_ratio(sequence_ratio, self.clip_low, self.clip_high)
        surrogate = mx.minimum(sequence_ratio * advantages, clipped_ratio * advantages)
        policy_gradient_loss = -mx.mean(surrogate)
        token_kl = approximate_kl(policy_logprobs, reference_logprobs)
        kl = masked_mean(token_kl, mask)
        clip_fraction = mx.mean(clip_indicator(sequence_ratio, clipped_ratio))
        return AlgorithmLossMetrics(
            loss=policy_gradient_loss + beta * kl,
            policy_gradient_loss=policy_gradient_loss,
            kl=kl,
            mean_ratio=mx.mean(sequence_ratio),
            clip_fraction=clip_fraction,
        )


def group_normalize_rewards(
    rewards: mx.array,
    group_size: int,
    eps: float = 1e-8,
) -> mx.array:
    """Normalize rewards within each prompt group."""

    grouped = _grouped_rewards(rewards, group_size)
    mean = mx.mean(grouped, axis=1, keepdims=True)
    variance = mx.mean(mx.square(grouped - mean), axis=1, keepdims=True)
    advantages = (grouped - mean) / mx.sqrt(variance + eps)
    return advantages.reshape((-1,))


def group_center_rewards(rewards: mx.array, group_size: int) -> mx.array:
    """Center rewards within each prompt group without std normalization."""

    grouped = _grouped_rewards(rewards, group_size)
    mean = mx.mean(grouped, axis=1, keepdims=True)
    return (grouped - mean).reshape((-1,))


def _grouped_rewards(rewards: mx.array, group_size: int) -> mx.array:
    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if rewards.size % group_size != 0:
        raise ValueError(
            f"Reward count {rewards.size} must be divisible by group_size={group_size}."
        )
    return rewards.reshape((-1, group_size)).astype(mx.float32)


def approximate_kl(policy_logprobs: mx.array, reference_logprobs: mx.array) -> mx.array:
    """Compute the non-negative GRPO/TRL approximate per-token KL."""

    log_ratio = reference_logprobs - policy_logprobs
    return mx.exp(log_ratio) - log_ratio - 1.0


def token_importance_ratio(
    policy_logprobs: mx.array,
    old_policy_logprobs: mx.array,
) -> mx.array:
    """Token-level importance ratio against the rollout policy."""

    return mx.exp(policy_logprobs - mx.stop_gradient(old_policy_logprobs))


def sequence_importance_ratio(
    policy_logprobs: mx.array,
    old_policy_logprobs: mx.array,
    mask: mx.array,
) -> mx.array:
    """Length-normalized sequence-level importance ratio."""

    log_ratio = policy_logprobs - mx.stop_gradient(old_policy_logprobs)
    lengths = mx.maximum(mx.sum(mask.astype(mx.float32), axis=1), mx.array(1.0))
    return mx.exp(mx.sum(log_ratio * mask, axis=1) / lengths)


def clip_ratio(
    ratio: mx.array,
    clip_low: float | None,
    clip_high: float | None,
) -> mx.array:
    """Apply asymmetric ratio clipping."""

    clipped = ratio
    if clip_low is not None:
        clipped = mx.maximum(clipped, mx.array(1.0 - clip_low, dtype=ratio.dtype))
    if clip_high is not None:
        clipped = mx.minimum(clipped, mx.array(1.0 + clip_high, dtype=ratio.dtype))
    return clipped


def clip_indicator(ratio: mx.array, clipped_ratio: mx.array) -> mx.array:
    """Return a soft 0/1 diagnostic for whether clipping changed a ratio."""

    return mx.minimum(
        mx.ones_like(ratio).astype(mx.float32),
        mx.abs(ratio - clipped_ratio).astype(mx.float32) * 1e12,
    )


def masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    """Mean over valid completion tokens."""

    mask = mask.astype(mx.float32)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(values * mask) / denominator


def _token_loss_metrics(
    token_policy_gradient: mx.array,
    ratio: mx.array,
    policy_logprobs: mx.array,
    reference_logprobs: mx.array,
    mask: mx.array,
    beta: float,
    clipped_ratio: mx.array | None = None,
) -> AlgorithmLossMetrics:
    token_kl = approximate_kl(policy_logprobs, reference_logprobs)
    loss = masked_mean(token_policy_gradient + beta * token_kl, mask)
    if clipped_ratio is None:
        clip_fraction = mx.array(0.0, dtype=mx.float32)
    else:
        clip_fraction = masked_mean(clip_indicator(ratio, clipped_ratio), mask)
    return AlgorithmLossMetrics(
        loss=loss,
        policy_gradient_loss=masked_mean(token_policy_gradient, mask),
        kl=masked_mean(token_kl, mask),
        mean_ratio=masked_mean(ratio, mask),
        clip_fraction=clip_fraction,
    )


def grpo_loss(
    policy_logprobs: mx.array,
    old_policy_logprobs: mx.array,
    reference_logprobs: mx.array,
    advantages: mx.array,
    mask: mx.array,
    beta: float,
) -> GRPOLossMetrics:
    """Compute the Phase 1 unclipped GRPO loss."""

    return GRPOAlgorithm().loss(
        policy_logprobs=policy_logprobs,
        old_policy_logprobs=old_policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
        mask=mask,
        beta=beta,
    )


def algorithm_by_name(name: str) -> PolicyAlgorithm:
    """Resolve a default algorithm by CLI/import name."""

    normalized = name.lower().replace("_", "-")
    if normalized == "grpo":
        return GRPOAlgorithm()
    if normalized in {"dr-grpo", "drgrpo"}:
        return DrGRPOAlgorithm()
    if normalized == "dapo":
        return DAPOAlgorithm()
    if normalized == "gspo":
        return GSPOAlgorithm()
    raise ValueError(f"Unknown algorithm {name!r}.")
