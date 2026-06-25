"""Shared reference-policy safeguards for training losses."""

from __future__ import annotations

import mlx.core as mx


def reference_logprobs_for_loss(
    current_logprobs: mx.array,
    stored_reference_logprobs: mx.array,
    *,
    beta: float,
    reference_is_policy: bool,
    batch_kind: str,
) -> mx.array:
    """Return reference logprobs, rejecting skipped references when KL is active."""

    if not reference_is_policy:
        return stored_reference_logprobs
    if beta != 0.0:
        raise ValueError(
            f"{batch_kind} skipped reference logprobs, but beta={beta}. "
            "Set beta=0.0 or build the batch with compute_reference=True."
        )
    return current_logprobs
