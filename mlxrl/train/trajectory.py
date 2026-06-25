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
from mlxrl.echo import ACTION, ECHO
from mlxrl.policy.model import enable_grad_checkpointing, temporary_eval_mode
from mlxrl.policy.trajectory_logprobs import (
    trajectory_action_logprobs,
    trajectory_dual_logprobs,
    trajectory_tagged_dual_logprobs,
    trajectory_tagged_logprobs,
)
from mlxrl.train.grpo import StepMetrics
from mlxrl.train.reference import reference_logprobs_for_loss
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
    echo_mask: mx.array | None = None
    target_ids: mx.array | None = None


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
    step_advantages = tuple(
        float(value)
        for value in algorithm.compute_step_advantages(
            trajectory_tuple,
            group_structure=group_size,
        )
    )
    rewards = mx.array(
        [trajectory.total_return for trajectory in trajectory_tuple],
        dtype=mx.float32,
    )
    if _has_tagged_roles(trajectory_tuple):
        with temporary_eval_mode(model):
            tagged = trajectory_tagged_dual_logprobs(
                model,
                trajectory_tuple,
                pad_token_id=pad_token_id,
                use_checkpoint=use_checkpoint,
                compute_reference=compute_reference,
            )
            if compute_reference:
                mx.eval(  # Logprob sync: freeze tagged old-policy/ref before mutation.
                    tagged.policy,
                    tagged.reference,
                    tagged.action_mask,
                    tagged.echo_mask,
                )
            else:
                mx.eval(  # Logprob sync: freeze tagged old-policy before mutation.
                    tagged.policy,
                    tagged.action_mask,
                    tagged.echo_mask,
                )
        batch = TrajectoryBatch(
            trajectories=trajectory_tuple,
            rewards=rewards,
            step_advantages=mx.array(step_advantages, dtype=mx.float32),
            advantages=mx.array(
                _target_advantages_from_roles(trajectory_tuple, step_advantages),
                dtype=mx.float32,
            ),
            old_policy_logprobs=mx.stop_gradient(tagged.policy),
            reference_logprobs=mx.stop_gradient(tagged.reference),
            action_mask=tagged.action_mask,
            reference_is_policy=not compute_reference,
            echo_mask=tagged.echo_mask,
            target_ids=tagged.target_ids,
        )
    else:
        with temporary_eval_mode(model):
            dual = trajectory_dual_logprobs(
                model,
                trajectory_tuple,
                pad_token_id=pad_token_id,
                use_checkpoint=use_checkpoint,
                compute_reference=compute_reference,
            )
            if compute_reference:
                mx.eval(  # Logprob sync: freeze old-policy/ref trajectory logprobs before mutation.
                    dual.policy,
                    dual.reference,
                    dual.mask,
                )
            else:
                mx.eval(  # Logprob sync: freeze old-policy trajectory logprobs before mutation.
                    dual.policy,
                    dual.mask,
                )
        batch = TrajectoryBatch(
            trajectories=trajectory_tuple,
            rewards=rewards,
            step_advantages=mx.array(step_advantages, dtype=mx.float32),
            advantages=mx.array(
                _token_advantages_from_steps(trajectory_tuple, step_advantages),
                dtype=mx.float32,
            ),
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
    echo_alpha: float = 0.0,
) -> AlgorithmLossMetrics:
    """Recompute current policy action logprobs and evaluate trajectory loss."""

    if use_checkpoint:
        enable_grad_checkpointing(model)
    if batch.echo_mask is not None:
        current_tagged = trajectory_tagged_logprobs(
            model,
            batch.trajectories,
            pad_token_id=pad_token_id,
            use_checkpoint=use_checkpoint,
        )
        policy_logprobs = current_tagged.logprobs
    else:
        current = trajectory_action_logprobs(
            model,
            batch.trajectories,
            pad_token_id=pad_token_id,
            use_checkpoint=use_checkpoint,
        )
        policy_logprobs = current.logprobs
    reference_logprobs = reference_logprobs_for_loss(
        policy_logprobs,
        batch.reference_logprobs,
        beta=beta,
        reference_is_policy=batch.reference_is_policy,
        batch_kind="Trajectory batch",
    )
    action_metrics = algorithm.compute_loss(
        policy_logprobs=policy_logprobs,
        old_policy_logprobs=batch.old_policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=batch.advantages,
        action_mask=batch.action_mask,
        beta=beta,
    )
    if batch.echo_mask is None:
        return action_metrics
    if batch.target_ids is None:
        raise ValueError("Tagged trajectory batches must carry target_ids.")
    echo_loss, echo_accuracy = _echo_loss_and_accuracy(
        policy_logprobs=policy_logprobs,
        predictions=current_tagged.predictions,
        target_ids=batch.target_ids,
        echo_mask=batch.echo_mask,
        alpha=echo_alpha,
    )
    return AlgorithmLossMetrics(
        loss=action_metrics.loss + echo_loss,
        policy_gradient_loss=action_metrics.policy_gradient_loss,
        kl=action_metrics.kl,
        mean_ratio=action_metrics.mean_ratio,
        clip_fraction=action_metrics.clip_fraction,
        loss_action=action_metrics.loss,
        loss_echo=echo_loss,
        echo_accuracy=echo_accuracy,
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
    echo_alpha: float = 0.0,
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
    ) -> tuple[
        mx.array,
        tuple[
            mx.array,
            mx.array,
            mx.array,
            mx.array,
            mx.array,
            mx.array,
            mx.array,
        ],
    ]:
        metrics = trajectory_metrics_from_batch(
            model,
            sub,
            beta,
            pad_token_id,
            use_checkpoint=use_checkpoint,
            algorithm=algorithm,
            echo_alpha=echo_alpha,
        )
        return metrics.loss, (
            metrics.policy_gradient_loss,
            metrics.kl,
            metrics.mean_ratio,
            metrics.clip_fraction,
            metrics.loss_action if metrics.loss_action is not None else metrics.loss,
            metrics.loss_echo
            if metrics.loss_echo is not None
            else mx.array(0.0, dtype=mx.float32),
            metrics.echo_accuracy
            if metrics.echo_accuracy is not None
            else mx.array(0.0, dtype=mx.float32),
        )

    gradients: Any | None
    if not chunked:
        (
            loss,
            (
                policy_gradient_loss,
                kl,
                mean_ratio,
                clip_fraction,
                loss_action,
                loss_echo,
                echo_accuracy,
            ),
        ), gradients = nn.value_and_grad(model, lambda m: loss_fn(m, batch))(model)
    else:
        total_action_tokens = _action_role_count(batch)
        total_echo_tokens = _echo_role_count(batch)
        if total_action_tokens <= 0 and total_echo_tokens <= 0:
            raise ValueError("Cannot micro-batch a trajectory batch with no trained tokens.")
        gradients = None
        loss = policy_gradient_loss = kl = mean_ratio = clip_fraction = mx.array(
            0.0,
            dtype=mx.float32,
        )
        loss_action = loss_echo = echo_accuracy = mx.array(0.0, dtype=mx.float32)
        for start in range(0, num_trajectories, micro_batch_size):
            end = min(start + micro_batch_size, num_trajectories)
            sub = _slice_trajectory_batch(batch, start, end)
            action_weight = (
                _action_role_count(sub) / total_action_tokens
                if total_action_tokens > 0
                else 0.0
            )
            echo_weight = (
                _echo_role_count(sub) / total_echo_tokens
                if total_echo_tokens > 0
                else 0.0
            )
            (sub_loss, sub_aux), sub_grad = nn.value_and_grad(
                model,
                lambda m, _s=sub, _aw=action_weight, _ew=echo_weight: _weighted_loss_fn(
                    loss_fn(m, _s),
                    action_weight=_aw,
                    echo_weight=_ew,
                ),
            )(model)
            gradients = (
                sub_grad
                if gradients is None
                else tree_map(lambda left, right: left + right, gradients, sub_grad)
            )
            loss = loss + sub_loss
            policy_gradient_loss = policy_gradient_loss + sub_aux[0] * action_weight
            kl = kl + sub_aux[1] * action_weight
            mean_ratio = mean_ratio + sub_aux[2] * action_weight
            clip_fraction = clip_fraction + sub_aux[3] * action_weight
            loss_action = loss_action + sub_aux[4] * action_weight
            loss_echo = loss_echo + sub_aux[5] * echo_weight
            echo_accuracy = echo_accuracy + sub_aux[6] * echo_weight
            mx.eval(  # Micro-batch sync: free this chunk's backward graph before next chunk.
                gradients,
                loss,
                policy_gradient_loss,
                kl,
                mean_ratio,
                clip_fraction,
                loss_action,
                loss_echo,
                echo_accuracy,
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
        loss_action,
        loss_echo,
        echo_accuracy,
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
        loss_action=float(loss_action.item()),
        loss_echo=float(loss_echo.item()),
        echo_accuracy=float(echo_accuracy.item()),
        echo_alpha=float(echo_alpha),
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
    width = (
        _target_width_from_trajectories(trajectories)
        if batch.echo_mask is not None
        else max(trajectory.action_token_count for trajectory in trajectories)
    )
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
        echo_mask=(
            batch.echo_mask[start:end, :width] if batch.echo_mask is not None else None
        ),
        target_ids=(
            batch.target_ids[start:end, :width] if batch.target_ids is not None else None
        ),
    )


def _action_token_count(batch: TrajectoryBatch) -> int:
    """Count valid action tokens from host-side trajectory metadata."""

    return sum(trajectory.action_token_count for trajectory in batch.trajectories)


def _has_tagged_roles(trajectories: Sequence[Trajectory]) -> bool:
    return any(
        trajectory.token_roles is not None or trajectory.token_advantages is not None
        for trajectory in trajectories
    )


def _target_width_from_trajectories(trajectories: Sequence[Trajectory]) -> int:
    return max(len(trajectory.full_token_ids) - 1 for trajectory in trajectories)


def _step_advantages_by_trajectory(
    trajectories: Sequence[Trajectory],
    step_advantages: Sequence[float],
) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    offset = 0
    for trajectory in trajectories:
        stop = offset + len(trajectory.steps)
        rows.append(tuple(float(value) for value in step_advantages[offset:stop]))
        offset = stop
    if offset != len(step_advantages):
        raise ValueError("step_advantages count does not match trajectory steps.")
    return rows


def _target_advantages_from_roles(
    trajectories: Sequence[Trajectory],
    step_advantages: Sequence[float],
) -> list[list[float]]:
    width = _target_width_from_trajectories(trajectories)
    step_rows = _step_advantages_by_trajectory(trajectories, step_advantages)
    output: list[list[float]] = []
    for trajectory, per_step in zip(trajectories, step_rows, strict=True):
        roles = trajectory.token_roles_or_default()
        if trajectory.token_advantages is not None:
            row = [float(value) for value in trajectory.token_advantages[1:]]
        else:
            row = [0.0] * (len(trajectory.full_token_ids) - 1)
            for span, advantage in zip(trajectory.action_spans, per_step, strict=True):
                for token_index in range(span.start, span.end):
                    row[token_index - 1] = float(advantage)
        row = [
            value if role == ACTION else 0.0
            for value, role in zip(row, roles[1:], strict=True)
        ]
        row.extend([0.0] * (width - len(row)))
        output.append(row)
    return output


def _action_role_count(batch: TrajectoryBatch) -> int:
    if batch.echo_mask is None:
        return _action_token_count(batch)
    return sum(
        1
        for trajectory in batch.trajectories
        for role in trajectory.token_roles_or_default()[1:]
        if role == ACTION
    )


def _echo_role_count(batch: TrajectoryBatch) -> int:
    if batch.echo_mask is None:
        return 0
    return sum(
        1
        for trajectory in batch.trajectories
        for role in trajectory.token_roles_or_default()[1:]
        if role == ECHO
    )


def _echo_loss_and_accuracy(
    *,
    policy_logprobs: mx.array,
    predictions: mx.array,
    target_ids: mx.array,
    echo_mask: mx.array,
    alpha: float,
) -> tuple[mx.array, mx.array]:
    if alpha < 0:
        raise ValueError("echo_alpha must be non-negative.")
    mask = echo_mask.astype(mx.float32)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    loss = -float(alpha) * mx.sum(policy_logprobs * mask) / denominator
    correct = mx.equal(predictions, target_ids).astype(mx.float32)
    accuracy = mx.sum(correct * mask) / denominator
    return loss, accuracy


def _weighted_loss_fn(
    result: tuple[
        mx.array,
        tuple[
            mx.array,
            mx.array,
            mx.array,
            mx.array,
            mx.array,
            mx.array,
            mx.array,
        ],
    ],
    *,
    action_weight: float,
    echo_weight: float,
) -> tuple[
    mx.array,
    tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
]:
    _, aux = result
    weighted_loss = aux[4] * action_weight + aux[5] * echo_weight
    return weighted_loss, aux
