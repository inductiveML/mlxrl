"""Full-forward logprob gathering for trajectory action tokens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlxrl.policy.logprobs import adapters_disabled
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
    action_target_indices: list[list[int]] = []
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
        action_target_indices.append(indices)
        mask_rows.append([1.0] * len(indices) + [0.0] * (max_action_len - len(indices)))

    input_ids = mx.array(input_rows, dtype=mx.int32)
    target_ids = mx.array(target_rows, dtype=mx.int32)
    logits = model(input_ids)
    all_logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_logprobs = mx.squeeze(
        mx.take_along_axis(all_logprobs, target_ids[..., None], axis=-1),
        axis=-1,
    )

    rows: list[mx.array] = []
    for row_index, indices in enumerate(action_target_indices):
        row_values = [target_logprobs[row_index, index] for index in indices]
        row = mx.stack(row_values, axis=0)
        if len(indices) < max_action_len:
            row = mx.concatenate(
                [
                    row,
                    mx.zeros((max_action_len - len(indices),), dtype=row.dtype),
                ],
                axis=0,
            )
        rows.append(row)

    return TrajectoryLogprobs(
        logprobs=mx.stack(rows, axis=0),
        mask=mx.array(mask_rows, dtype=mx.float32),
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
