"""Pure advantage and policy-loss math for GRPO-family algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlxrl.algorithm import AlgorithmLossMetrics, PolicyAlgorithm

GRPOLossMetrics = AlgorithmLossMetrics


@dataclass(frozen=True)
class GRPOAlgorithm:
    """Thin unclipped GRPO used by the Phase 1 reference path."""

    name: str = "grpo"

    @property
    def token_mean_reduction(self) -> bool:
        return True

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        return group_normalize_rewards(rewards, group_structure)

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        ratio = token_importance_ratio(policy_logprobs, old_policy_logprobs)
        token_policy_gradient = -ratio * advantages[:, None]
        return _token_loss_metrics(
            token_policy_gradient=token_policy_gradient,
            ratio=ratio,
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
            mask=completion_mask,
            beta=beta,
        )

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return self.compute_advantages(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        return self.compute_loss(
            policy_logprobs,
            old_policy_logprobs,
            reference_logprobs,
            advantages,
            mask,
            beta,
        )

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        del group_structure
        return batch


@dataclass(frozen=True)
class DrGRPOAlgorithm:
    """Dr. GRPO, with decoupled reward and length normalization knobs."""

    normalize_rewards: bool = False
    loss_reduction: str = "sequence_max_tokens"
    max_tokens: int | None = None
    name: str = "dr-grpo"

    @property
    def token_mean_reduction(self) -> bool:
        return self.loss_reduction == "token_mean"

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        if self.normalize_rewards:
            return group_normalize_rewards(rewards, group_structure)
        return group_center_rewards(rewards, group_structure)

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        mask = completion_mask
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

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return self.compute_advantages(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        return self.compute_loss(
            policy_logprobs,
            old_policy_logprobs,
            reference_logprobs,
            advantages,
            mask,
            beta,
        )

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        del group_structure
        return batch


@dataclass(frozen=True)
class DAPOAlgorithm:
    """DAPO's decoupled low/high token-ratio clipping."""

    clip_low: float | None = 0.2
    clip_high: float | None = 0.28
    dynamic_sampling: bool = True
    name: str = "dapo"

    @property
    def token_mean_reduction(self) -> bool:
        return True

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        return group_normalize_rewards(rewards, group_structure)

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
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
            mask=completion_mask,
            beta=beta,
            clipped_ratio=clipped_ratio,
        )

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        if not self.dynamic_sampling:
            return batch
        return filter_zero_advantage_groups(batch, group_structure)

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return self.compute_advantages(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        return self.compute_loss(
            policy_logprobs,
            old_policy_logprobs,
            reference_logprobs,
            advantages,
            mask,
            beta,
        )


@dataclass(frozen=True)
class GSPOAlgorithm:
    """GSPO with sequence-level or token-level importance weighting."""

    importance: str = "sequence"
    clip_low: float | None = 3e-4
    clip_high: float | None = 4e-4
    name: str = "gspo"

    @property
    def token_mean_reduction(self) -> bool:
        return self.importance == "token"

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        return group_normalize_rewards(rewards, group_structure)

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        mask = completion_mask
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

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return self.compute_advantages(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        return self.compute_loss(
            policy_logprobs,
            old_policy_logprobs,
            reference_logprobs,
            advantages,
            mask,
            beta,
        )

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        del group_structure
        return batch


@dataclass(frozen=True)
class RLOOAlgorithm:
    """REINFORCE Leave-One-Out with a per-sample group baseline."""

    name: str = "rloo"

    @property
    def token_mean_reduction(self) -> bool:
        return True

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        grouped = _grouped_rewards(rewards, group_structure)
        if group_structure < 2:
            raise ValueError("RLOO requires group_structure >= 2.")
        group_sum = mx.sum(grouped, axis=1, keepdims=True)
        baseline = (group_sum - grouped) / float(group_structure - 1)
        return (grouped - baseline).reshape((-1,))

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        del old_policy_logprobs
        token_policy_gradient = -policy_logprobs * advantages[:, None]
        ratio = mx.ones_like(policy_logprobs)
        return _token_loss_metrics(
            token_policy_gradient=token_policy_gradient,
            ratio=ratio,
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
            mask=completion_mask,
            beta=beta,
        )

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        del group_structure
        return batch

    def advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        return self.compute_advantages(rewards, group_size)

    def loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        return self.compute_loss(
            policy_logprobs,
            old_policy_logprobs,
            reference_logprobs,
            advantages,
            mask,
            beta,
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


def filter_zero_advantage_groups(batch: Any, group_size: int) -> Any:
    """Drop groups whose rewards/advantages are constant across all samples."""

    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    row_count = len(batch.completion_token_ids)
    if row_count % group_size != 0:
        raise ValueError(
            f"Batch row count {row_count} must be divisible by group_size={group_size}."
        )

    mx.eval(  # Batch-filter sync: inspect group rewards/advantages on the host.
        batch.rewards,
        batch.advantages,
    )
    rewards = [float(value) for value in batch.rewards.tolist()]
    advantages = [float(value) for value in batch.advantages.tolist()]
    keep_indices: list[int] = []
    for start in range(0, row_count, group_size):
        end = start + group_size
        reward_group = rewards[start:end]
        advantage_group = advantages[start:end]
        reward_span = max(reward_group) - min(reward_group)
        advantage_span = max(advantage_group) - min(advantage_group)
        if reward_span != 0.0 or advantage_span != 0.0:
            keep_indices.extend(range(start, end))
    if not keep_indices:
        raise ValueError("DAPO dynamic sampling dropped every group.")

    indices = mx.array(keep_indices, dtype=mx.int32)
    return type(batch)(
        prompt_token_ids=tuple(batch.prompt_token_ids[index] for index in keep_indices),
        completion_token_ids=tuple(
            batch.completion_token_ids[index] for index in keep_indices
        ),
        rewards=batch.rewards[indices],
        advantages=batch.advantages[indices],
        old_policy_logprobs=batch.old_policy_logprobs[indices],
        reference_logprobs=batch.reference_logprobs[indices],
        mask=batch.mask[indices],
    )


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
    if normalized == "rloo":
        return RLOOAlgorithm()
    raise ValueError(f"Unknown algorithm {name!r}.")
