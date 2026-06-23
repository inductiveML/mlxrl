"""Policy and reference logprob passes for generated completions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear


@dataclass(frozen=True)
class CompletionLogprobs:
    """Padded completion-token logprobs and their valid-token mask."""

    logprobs: mx.array
    mask: mx.array


@dataclass(frozen=True)
class DualLogprobs:
    """Policy and reference logprobs for the same completion tokens."""

    policy: mx.array
    reference: mx.array
    mask: mx.array


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily disable LoRA adapters by zeroing their scales on one model."""

    saved_scales: list[tuple[LoRALinear, float]] = []
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            saved_scales.append((module, float(module.scale)))
            module.scale = 0.0
    try:
        yield
    finally:
        for module, scale in saved_scales:
            module.scale = scale


def completion_logprobs(
    model: nn.Module,
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int = 0,
) -> CompletionLogprobs:
    """Gather logprobs assigned to completion tokens in a full forward pass."""

    if len(prompt_token_ids) != len(completion_token_ids):
        raise ValueError("prompt_token_ids and completion_token_ids must have the same length.")
    if not prompt_token_ids:
        raise ValueError("At least one sequence is required.")

    full_sequences = [
        tuple(prompt) + tuple(completion)
        for prompt, completion in zip(prompt_token_ids, completion_token_ids, strict=True)
    ]
    if any(len(sequence) < 2 for sequence in full_sequences):
        raise ValueError("Each prompt plus completion must contain at least two tokens.")

    max_sequence_len = max(len(sequence) for sequence in full_sequences)
    max_completion_len = max(len(completion) for completion in completion_token_ids)
    if max_completion_len == 0:
        raise ValueError("At least one completion token is required.")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    completion_mask_rows: list[list[float]] = []
    completion_starts: list[int] = []
    completion_lengths: list[int] = []

    for prompt, completion, sequence in zip(
        prompt_token_ids,
        completion_token_ids,
        full_sequences,
        strict=True,
    ):
        completion_start = len(prompt) - 1
        target_len = max_sequence_len - 1
        input_row = list(sequence[:-1])
        target_row = list(sequence[1:])
        input_row.extend([pad_token_id] * (target_len - len(input_row)))
        target_row.extend([pad_token_id] * (target_len - len(target_row)))

        mask_row = [0.0] * max_completion_len
        for index in range(len(completion)):
            mask_row[index] = 1.0

        input_rows.append(input_row)
        target_rows.append(target_row)
        completion_mask_rows.append(mask_row)
        completion_starts.append(completion_start)
        completion_lengths.append(len(completion))

    input_ids = mx.array(input_rows, dtype=mx.int32)
    target_ids = mx.array(target_rows, dtype=mx.int32)
    logits = model(input_ids)
    all_logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_logprobs = mx.take_along_axis(all_logprobs, target_ids[..., None], axis=-1)
    target_logprobs = mx.squeeze(target_logprobs, axis=-1)

    rows: list[mx.array] = []
    for row_index, (start, length) in enumerate(
        zip(completion_starts, completion_lengths, strict=True)
    ):
        row = target_logprobs[row_index, start : start + length]
        if length < max_completion_len:
            row = mx.concatenate(
                [
                    row,
                    mx.zeros((max_completion_len - length,), dtype=row.dtype),
                ],
                axis=0,
            )
        rows.append(row)

    return CompletionLogprobs(
        logprobs=mx.stack(rows, axis=0),
        mask=mx.array(completion_mask_rows, dtype=mx.float32),
    )


def dual_logprobs(
    model: nn.Module,
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int = 0,
) -> DualLogprobs:
    """Compute policy logprobs and reference logprobs using one model object."""

    policy = completion_logprobs(model, prompt_token_ids, completion_token_ids, pad_token_id)
    with adapters_disabled(model):
        reference = completion_logprobs(
            model,
            prompt_token_ids,
            completion_token_ids,
            pad_token_id,
        )
    return DualLogprobs(
        policy=policy.logprobs,
        reference=reference.logprobs,
        mask=policy.mask,
    )


def pad_token_id_from_tokenizer(tokenizer: Any) -> int:
    """Resolve a safe right-padding token id from a tokenizer."""

    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    return 0
