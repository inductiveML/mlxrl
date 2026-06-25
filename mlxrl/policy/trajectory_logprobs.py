"""Full-forward logprob gathering for trajectory action tokens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlxrl.echo import ACTION, ECHO
from mlxrl.policy.logprobs import adapters_disabled, target_logprobs_from_logits
from mlxrl.trajectory import Trajectory


@dataclass(frozen=True)
class TrajectoryLogprobs:
    """Padded action-token logprobs and mask for trajectory batches."""

    logprobs: mx.array
    mask: mx.array


@dataclass(frozen=True)
class TrajectoryDualLogprobs:
    """Policy and reference action-token logprobs for trajectories."""

    policy: mx.array
    reference: mx.array
    mask: mx.array


@dataclass(frozen=True)
class TaggedTrajectoryLogprobs:
    """Full-forward logprobs and masks for tagged trajectory target tokens."""

    logprobs: mx.array
    target_ids: mx.array
    predictions: mx.array
    action_mask: mx.array
    echo_mask: mx.array


@dataclass(frozen=True)
class TaggedTrajectoryDualLogprobs:
    """Policy/reference logprobs for tagged trajectory target tokens."""

    policy: mx.array
    reference: mx.array
    target_ids: mx.array
    action_mask: mx.array
    echo_mask: mx.array


def trajectory_action_logprobs(
    model: nn.Module,
    trajectories: Sequence[Trajectory],
    pad_token_id: int = 0,
    use_checkpoint: bool = False,
) -> TrajectoryLogprobs:
    """Gather logprobs for action tokens by full-forward recompute."""

    del use_checkpoint
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    max_sequence_len = max(len(trajectory.full_token_ids) for trajectory in trajectories)
    max_action_len = max(trajectory.action_token_count for trajectory in trajectories)
    if max_action_len == 0:
        raise ValueError("At least one action token is required.")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    action_target_index_rows: list[list[int]] = []
    mask_rows: list[list[float]] = []
    target_width = max_sequence_len - 1
    for trajectory in trajectories:
        sequence = list(trajectory.full_token_ids)
        input_row = sequence[:-1] + [pad_token_id] * (target_width - (len(sequence) - 1))
        target_row = sequence[1:] + [pad_token_id] * (target_width - (len(sequence) - 1))
        indices: list[int] = []
        for span in trajectory.action_spans:
            indices.extend(range(span.start - 1, span.end - 1))
        input_rows.append(input_row)
        target_rows.append(target_row)
        action_target_index_rows.append(
            [indices[index] if index < len(indices) else 0 for index in range(max_action_len)]
        )
        mask_rows.append([1.0] * len(indices) + [0.0] * (max_action_len - len(indices)))

    input_ids = mx.array(input_rows, dtype=mx.int32)
    target_ids = mx.array(target_rows, dtype=mx.int32)
    logits = model(input_ids)
    target_logprobs = target_logprobs_from_logits(logits, target_ids)
    action_indices = mx.array(action_target_index_rows, dtype=mx.int32)
    mask = mx.array(mask_rows, dtype=mx.float32)
    action_logprobs = mx.take_along_axis(
        target_logprobs,
        action_indices,
        axis=1,
    )
    action_logprobs = action_logprobs * mask.astype(action_logprobs.dtype)

    return TrajectoryLogprobs(
        logprobs=action_logprobs,
        mask=mask,
    )


def trajectory_tagged_logprobs(
    model: nn.Module,
    trajectories: Sequence[Trajectory],
    pad_token_id: int = 0,
    use_checkpoint: bool = False,
) -> TaggedTrajectoryLogprobs:
    """Gather logprobs for ACTION and ECHO tags from one full forward."""

    del use_checkpoint
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    max_sequence_len = max(len(trajectory.full_token_ids) for trajectory in trajectories)
    if max_sequence_len < 2:
        raise ValueError("A trajectory needs at least two tokens for logprobs.")

    target_width = max_sequence_len - 1
    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    action_mask_rows: list[list[float]] = []
    echo_mask_rows: list[list[float]] = []
    for trajectory in trajectories:
        sequence = list(trajectory.full_token_ids)
        roles = trajectory.token_roles_or_default()
        target_roles = list(roles[1:])
        pad_count = target_width - (len(sequence) - 1)
        input_rows.append(sequence[:-1] + [pad_token_id] * pad_count)
        target_rows.append(sequence[1:] + [pad_token_id] * pad_count)
        action_mask_rows.append(
            [1.0 if role == ACTION else 0.0 for role in target_roles]
            + [0.0] * pad_count
        )
        echo_mask_rows.append(
            [1.0 if role == ECHO else 0.0 for role in target_roles]
            + [0.0] * pad_count
        )

    input_ids = mx.array(input_rows, dtype=mx.int32)
    target_ids = mx.array(target_rows, dtype=mx.int32)
    logits = model(input_ids)
    logprobs = target_logprobs_from_logits(logits, target_ids)
    predictions = mx.argmax(logits, axis=-1).astype(mx.int32)
    return TaggedTrajectoryLogprobs(
        logprobs=logprobs,
        target_ids=target_ids,
        predictions=predictions,
        action_mask=mx.array(action_mask_rows, dtype=mx.float32),
        echo_mask=mx.array(echo_mask_rows, dtype=mx.float32),
    )


def trajectory_tagged_dual_logprobs(
    model: nn.Module,
    trajectories: Sequence[Trajectory],
    pad_token_id: int = 0,
    use_checkpoint: bool = False,
    compute_reference: bool = True,
) -> TaggedTrajectoryDualLogprobs:
    """Compute tagged policy/reference logprobs using one model object."""

    policy = trajectory_tagged_logprobs(
        model,
        trajectories,
        pad_token_id=pad_token_id,
        use_checkpoint=use_checkpoint,
    )
    if not compute_reference:
        return TaggedTrajectoryDualLogprobs(
            policy=policy.logprobs,
            reference=mx.stop_gradient(policy.logprobs),
            target_ids=policy.target_ids,
            action_mask=policy.action_mask,
            echo_mask=policy.echo_mask,
        )
    with adapters_disabled(model):
        reference = trajectory_tagged_logprobs(
            model,
            trajectories,
            pad_token_id=pad_token_id,
            use_checkpoint=use_checkpoint,
        )
        mx.eval(  # Reference sync: materialize tagged ref before restoring adapters.
            reference.logprobs,
            reference.action_mask,
            reference.echo_mask,
        )
    return TaggedTrajectoryDualLogprobs(
        policy=policy.logprobs,
        reference=reference.logprobs,
        target_ids=policy.target_ids,
        action_mask=policy.action_mask,
        echo_mask=policy.echo_mask,
    )


def trajectory_dual_logprobs(
    model: nn.Module,
    trajectories: Sequence[Trajectory],
    pad_token_id: int = 0,
    use_checkpoint: bool = False,
    compute_reference: bool = True,
) -> TrajectoryDualLogprobs:
    """Compute trajectory action logprobs for policy and reference policy."""

    policy = trajectory_action_logprobs(
        model,
        trajectories,
        pad_token_id=pad_token_id,
        use_checkpoint=use_checkpoint,
    )
    if not compute_reference:
        return TrajectoryDualLogprobs(
            policy=policy.logprobs,
            reference=mx.stop_gradient(policy.logprobs),
            mask=policy.mask,
        )
    with adapters_disabled(model):
        reference = trajectory_action_logprobs(
            model,
            trajectories,
            pad_token_id=pad_token_id,
            use_checkpoint=use_checkpoint,
        )
        mx.eval(  # Reference sync: materialize full-forward ref before restoring adapters.
            reference.logprobs,
            reference.mask,
        )
    return TrajectoryDualLogprobs(
        policy=policy.logprobs,
        reference=reference.logprobs,
        mask=policy.mask,
    )
