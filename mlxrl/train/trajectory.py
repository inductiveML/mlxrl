"""One-step optimizer integration for trajectory-shaped RL batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from mlxrl.algorithm import AlgorithmLossMetrics, TrajectoryAlgorithm
from mlxrl.policy.model import enable_grad_checkpointing
from mlxrl.policy.trajectory_logprobs import (
    trajectory_action_logprobs,
    trajectory_dual_logprobs,
)
from mlxrl.train.grpo import StepMetrics
from mlxrl.trajectory import Trajectory


@dataclass(frozen=True)
class TrajectoryBatch:
    """Reference data for one trajectory optimizer step."""

    trajectories: tuple[Trajectory, ...]
    rewards: mx.array
    step_advantages: mx.array
    advantages: mx.array
    old_policy_logprobs: mx.array
    reference_logprobs: mx.array
    action_mask: mx.array
    reference_is_policy: bool = False


def batch_from_trajectories(
    model: nn.Module,
    trajectories: Sequence[Trajectory],
    group_size: int,
    pad_token_id: int,
    algorithm: TrajectoryAlgorithm,
    use_checkpoint: bool = False,
    compute_reference: bool = True,
) -> TrajectoryBatch:
    """Compute old/ref logprobs and trajectory-aware advantages."""

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    if use_checkpoint:
        enable_grad_checkpointing(model)
    trajectory_tuple = tuple(trajectories)
    dual = trajectory_dual_logprobs(
        model,
        trajectory_tuple,
        pad_token_id=pad_token_id,
        use_checkpoint=use_checkpoint,
        compute_reference=compute_reference,
    )
    mx.eval(  # Logprob sync: freeze old-policy/ref trajectory logprobs before mutation.
        dual.policy,
        dual.reference,
        dual.mask,
    )
    step_advantages = tuple(
        float(value)
        for value in algorithm.compute_step_advantages(
            trajectory_tuple,
            group_structure=group_size,
        )
    )
    token_advantages = _token_advantages_from_steps(trajectory_tuple, step_advantages)
    batch = TrajectoryBatch(
        trajectories=trajectory_tuple,
        rewards=mx.array(
            [trajectory.total_return for trajectory in trajectory_tuple],
            dtype=mx.float32,
        ),
        step_advantages=mx.array(step_advantages, dtype=mx.float32),
        advantages=mx.array(token_advantages, dtype=mx.float32),
        old_policy_logprobs=mx.stop_gradient(dual.policy),
        reference_logprobs=mx.stop_gradient(dual.reference),
        action_mask=dual.mask,
        reference_is_policy=not compute_reference,
    )
    return cast(TrajectoryBatch, algorithm.filter_batch(batch, group_structure=group_size))


def trajectory_metrics_from_batch(
    model: nn.Module,
    batch: TrajectoryBatch,
    beta: float,
    pad_token_id: int,
    algorithm: TrajectoryAlgorithm,
    use_checkpoint: bool = False,
) -> AlgorithmLossMetrics:
    """Recompute current policy action logprobs and evaluate trajectory loss."""

    if use_checkpoint:
        enable_grad_checkpointing(model)
    current = trajectory_action_logprobs(
        model,
        batch.trajectories,
        pad_token_id=pad_token_id,
        use_checkpoint=use_checkpoint,
    )
    reference_logprobs = (
        current.logprobs
        if beta == 0.0 and batch.reference_is_policy
        else batch.reference_logprobs
    )
    return algorithm.compute_loss(
        policy_logprobs=current.logprobs,
        old_policy_logprobs=batch.old_policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=batch.advantages,
        action_mask=batch.action_mask,
        beta=beta,
    )


def optimizer_step_trajectory(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: TrajectoryBatch,
    beta: float,
    pad_token_id: int,
    algorithm: TrajectoryAlgorithm,
    use_checkpoint: bool = False,
    micro_batch_size: int = 0,
) -> StepMetrics:
    """Run value_and_grad over trainable parameters for one trajectory batch."""

    if micro_batch_size < 0:
        raise ValueError("micro_batch_size must be non-negative.")
    num_trajectories = len(batch.trajectories)
    chunked = 0 < micro_batch_size < num_trajectories
    if chunked and not algorithm.token_mean_reduction:
        raise ValueError(
            "micro_batch_size currently supports token-mean trajectory losses only; "
            f"{algorithm.name} uses sequence-level reduction."
        )
    model.train()

    def loss_fn(
        model: nn.Module,
        sub: TrajectoryBatch,
    ) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array, mx.array]]:
        metrics = trajectory_metrics_from_batch(
            model,
            sub,
            beta,
            pad_token_id,
            use_checkpoint=use_checkpoint,
            algorithm=algorithm,
        )
        return metrics.loss, (
            metrics.policy_gradient_loss,
            metrics.kl,
            metrics.mean_ratio,
            metrics.clip_fraction,
        )

    gradients: Any | None
    if not chunked:
        (loss, (policy_gradient_loss, kl, mean_ratio, clip_fraction)), gradients = (
            nn.value_and_grad(model, lambda m: loss_fn(m, batch))(model)
        )
    else:
        total_tokens = _action_token_count(batch)
        if total_tokens <= 0:
            raise ValueError("Cannot micro-batch a trajectory batch with no action tokens.")
        gradients = None
        loss = policy_gradient_loss = kl = mean_ratio = clip_fraction = mx.array(
            0.0,
            dtype=mx.float32,
        )
        for start in range(0, num_trajectories, micro_batch_size):
            end = min(start + micro_batch_size, num_trajectories)
            sub = _slice_trajectory_batch(batch, start, end)
            weight = _action_token_count(sub) / total_tokens
            (sub_loss, sub_aux), sub_grad = nn.value_and_grad(
                model,
                lambda m, _s=sub: loss_fn(m, _s),
            )(model)
            sub_grad = tree_map(
                lambda gradient, _weight=weight: gradient * _weight,
                sub_grad,
            )
            gradients = (
                sub_grad
                if gradients is None
                else tree_map(lambda left, right: left + right, gradients, sub_grad)
            )
            loss = loss + sub_loss * weight
            policy_gradient_loss = policy_gradient_loss + sub_aux[0] * weight
            kl = kl + sub_aux[1] * weight
            mean_ratio = mean_ratio + sub_aux[2] * weight
            clip_fraction = clip_fraction + sub_aux[3] * weight
            mx.eval(  # Micro-batch sync: free this chunk's backward graph before next chunk.
                gradients,
                loss,
                policy_gradient_loss,
                kl,
                mean_ratio,
                clip_fraction,
            )
    if gradients is None:
        raise RuntimeError("Micro-batch gradient accumulation produced no gradients.")
    mean_reward = mx.mean(batch.rewards)
    mx.eval(  # Optimizer pre-step sync: freeze gradients/diagnostics before mutation.
        gradients,
        loss,
        policy_gradient_loss,
        kl,
        mean_ratio,
        clip_fraction,
        mean_reward,
    )
    optimizer.update(model, gradients)
    mx.eval(  # Optimizer sync: materialize updated weights and optimizer state.
        model.state,
        optimizer.state,
    )
    model.eval()
    return StepMetrics(
        loss=float(loss.item()),
        policy_gradient_loss=float(policy_gradient_loss.item()),
        kl=float(kl.item()),
        mean_ratio=float(mean_ratio.item()),
        clip_fraction=float(clip_fraction.item()),
        mean_reward=float(mean_reward.item()),
    )


def _token_advantages_from_steps(
    trajectories: Sequence[Trajectory],
    step_advantages: Sequence[float],
) -> list[list[float]]:
    max_action_tokens = max(trajectory.action_token_count for trajectory in trajectories)
    rows: list[list[float]] = []
    step_offset = 0
    for trajectory in trajectories:
        row: list[float] = []
        for step in trajectory.steps:
            row.extend([step_advantages[step_offset]] * len(step.action_tokens))
            step_offset += 1
        row.extend([0.0] * (max_action_tokens - len(row)))
        rows.append(row)
    if step_offset != len(step_advantages):
        raise ValueError("step_advantages count does not match trajectory steps.")
    return rows


def _slice_trajectory_batch(
    batch: TrajectoryBatch,
    start: int,
    end: int,
) -> TrajectoryBatch:
    trajectories = batch.trajectories[start:end]
    if not trajectories:
        raise ValueError("Cannot slice an empty trajectory micro-batch.")
    width = max(trajectory.action_token_count for trajectory in trajectories)
    step_start = sum(len(trajectory.steps) for trajectory in batch.trajectories[:start])
    step_end = step_start + sum(len(trajectory.steps) for trajectory in trajectories)
    return TrajectoryBatch(
        trajectories=trajectories,
        rewards=batch.rewards[start:end],
        step_advantages=batch.step_advantages[step_start:step_end],
        advantages=batch.advantages[start:end, :width],
        old_policy_logprobs=batch.old_policy_logprobs[start:end, :width],
        reference_logprobs=batch.reference_logprobs[start:end, :width],
        action_mask=batch.action_mask[start:end, :width],
        reference_is_policy=batch.reference_is_policy,
    )


def _action_token_count(batch: TrajectoryBatch) -> int:
    """Count valid action tokens from host-side trajectory metadata."""

    return sum(trajectory.action_token_count for trajectory in batch.trajectories)
