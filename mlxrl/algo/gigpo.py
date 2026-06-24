"""GiGPO advantage and loss math for multi-turn trajectories."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlxrl.algo.grpo import _token_loss_metrics, clip_ratio, token_importance_ratio
from mlxrl.algorithm import AlgorithmLossMetrics
from mlxrl.trajectory import Trajectory

AdvantageNormalization = Literal["std", "center"]


@dataclass(frozen=True)
class StepRecord:
    """Flattened step metadata used for anchor-state grouping."""

    trajectory_index: int
    step_index: int
    state_id: Hashable
    return_to_go: float


@dataclass(frozen=True)
class GiGPOAlgorithm:
    """Group-in-Group Policy Optimization over multi-turn trajectories."""

    omega: float = 1.0
    gamma: float = 1.0
    normalization: AdvantageNormalization = "std"
    clip_low: float | None = None
    clip_high: float | None = None
    name: str = "gigpo"

    @property
    def token_mean_reduction(self) -> bool:
        return True

    def compute_step_advantages(
        self,
        trajectories: Sequence[Trajectory],
        group_structure: int,
    ) -> tuple[float, ...]:
        return compute_gigpo_step_advantages(
            trajectories=trajectories,
            group_size=group_structure,
            omega=self.omega,
            gamma=self.gamma,
            normalization=self.normalization,
        )

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        action_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        ratio = token_importance_ratio(policy_logprobs, old_policy_logprobs)
        if self.clip_low is None and self.clip_high is None:
            token_policy_gradient = -ratio * advantages
            clipped = None
        else:
            clipped = clip_ratio(ratio, self.clip_low, self.clip_high)
            surrogate = mx.minimum(ratio * advantages, clipped * advantages)
            token_policy_gradient = -surrogate
        return _token_loss_metrics(
            token_policy_gradient=token_policy_gradient,
            ratio=ratio,
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
            mask=action_mask,
            beta=beta,
            clipped_ratio=clipped,
        )

    def filter_batch(self, batch: object, group_structure: int) -> object:
        del group_structure
        return batch


def compute_gigpo_step_advantages(
    *,
    trajectories: Sequence[Trajectory],
    group_size: int,
    omega: float,
    gamma: float,
    normalization: AdvantageNormalization = "std",
) -> tuple[float, ...]:
    """Compute A^E + omega * A^S for every trajectory step."""

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if len(trajectories) % group_size != 0:
        raise ValueError("Trajectory count must be divisible by group_size.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    episode_returns = [trajectory.total_return for trajectory in trajectories]
    episode_advantages = _group_relative_advantages(
        episode_returns,
        group_size=group_size,
        normalization=normalization,
    )
    records = flatten_step_records(trajectories, gamma=gamma)
    micro = anchor_state_step_advantages(records, normalization=normalization)
    return tuple(
        episode_advantages[record.trajectory_index] + omega * micro[index]
        for index, record in enumerate(records)
    )


def flatten_step_records(
    trajectories: Sequence[Trajectory],
    gamma: float,
) -> tuple[StepRecord, ...]:
    """Flatten trajectory steps with discounted return-to-go attached."""

    records: list[StepRecord] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        returns_to_go = trajectory.discounted_returns_to_go(gamma)
        for step_index, (step, return_to_go) in enumerate(
            zip(trajectory.steps, returns_to_go, strict=True)
        ):
            records.append(
                StepRecord(
                    trajectory_index=trajectory_index,
                    step_index=step_index,
                    state_id=step.state_id,
                    return_to_go=return_to_go,
                )
            )
    return tuple(records)


def anchor_state_groups(records: Sequence[StepRecord]) -> dict[Hashable, tuple[int, ...]]:
    """Map each anchor state id to flattened step-record indices."""

    groups: dict[Hashable, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.state_id, []).append(index)
    return {state_id: tuple(indices) for state_id, indices in groups.items()}


def anchor_state_step_advantages(
    records: Sequence[StepRecord],
    normalization: AdvantageNormalization = "std",
) -> tuple[float, ...]:
    """Compute within-anchor step advantages; singleton anchors stay zero."""

    advantages = [0.0] * len(records)
    groups = anchor_state_groups(records)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        returns = [records[index].return_to_go for index in indices]
        relative = _relative_advantages(returns, normalization=normalization)
        for index, value in zip(indices, relative, strict=True):
            advantages[index] = value
    return tuple(advantages)


def _group_relative_advantages(
    values: Sequence[float],
    group_size: int,
    normalization: AdvantageNormalization,
) -> tuple[float, ...]:
    output: list[float] = []
    for start in range(0, len(values), group_size):
        output.extend(
            _relative_advantages(
                values[start : start + group_size],
                normalization=normalization,
            )
        )
    return tuple(output)


def _relative_advantages(
    values: Sequence[float],
    normalization: AdvantageNormalization,
) -> tuple[float, ...]:
    if normalization not in {"std", "center"}:
        raise ValueError("normalization must be 'std' or 'center'.")
    if not values:
        raise ValueError("At least one value is required.")
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if normalization == "center":
        return tuple(centered)
    variance = sum(value * value for value in centered) / len(centered)
    denominator = math.sqrt(variance + 1e-8)
    return tuple(value / denominator for value in centered)
