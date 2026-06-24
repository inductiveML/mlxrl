"""Shared algorithm protocol used by the training engine.

`mlxrl` deliberately supports critic-free, rollout-based policy-gradient
algorithms only. PPO needs a separate critic/value path, and DPO/ORPO are
offline preference objectives with no rollout phase, so they are outside this
library's one-policy-model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx


@dataclass(frozen=True)
class AlgorithmLossMetrics:
    """Scalar diagnostics for one algorithm loss batch."""

    loss: mx.array
    policy_gradient_loss: mx.array
    kl: mx.array
    mean_ratio: mx.array
    clip_fraction: mx.array


class Algorithm(Protocol):
    """Protocol implemented by rollout-based, critic-free RL objectives."""

    @property
    def name(self) -> str:
        """Stable algorithm name for logs and outputs."""
        ...

    @property
    def token_mean_reduction(self) -> bool:
        """Whether token-count weighting composes exact micro-batch gradients."""
        ...

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        """Compute per-completion advantages from scalar rewards."""
        ...

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        """Compute loss and diagnostics from completion-token logprobs."""
        ...

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        """Optionally filter a prepared batch before optimization."""
        return batch


class TrajectoryAlgorithm(Protocol):
    """Protocol for critic-free objectives over multi-turn trajectories."""

    @property
    def name(self) -> str:
        """Stable algorithm name for logs and outputs."""
        ...

    @property
    def token_mean_reduction(self) -> bool:
        """Whether token-count weighting composes exact micro-batch gradients."""
        ...

    def compute_step_advantages(
        self,
        trajectories: Any,
        group_structure: int,
    ) -> tuple[float, ...]:
        """Compute one advantage per trajectory step in trajectory-major order."""
        ...

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        action_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        """Compute loss and diagnostics from action-token logprobs."""
        ...

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        """Optionally filter a prepared trajectory batch before optimization."""
        return batch


PolicyAlgorithm = Algorithm
