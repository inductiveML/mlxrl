"""Policy and reference logprob passes for generated completions.

Attribution: uses MLX-LM prompt-cache construction and LoRA adapter classes from
`mlx_lm.models.cache` and `mlx_lm.tuner.lora` (MIT, Copyright Apple Inc.).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.lora import LoRALinear

from mlxrl.rollout.optimized import clone_prompt_cache


@dataclass(frozen=True)
class CompletionLogprobs:
    """Padded completion-token logprobs and their valid-token mask."""

    logprobs: mx.array
    mask: mx.array


@dataclass(frozen=True)
class CompletionLogprobInputs:
    """Prepared padded tensors for a completion full-forward logprob pass."""

    input_ids: mx.array
    target_ids: mx.array
    gather_indices: mx.array
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
    use_checkpoint: bool = False,
) -> CompletionLogprobs:
    """Gather logprobs assigned to completion tokens in a full forward pass."""

    inputs = prepare_completion_logprob_inputs(
        prompt_token_ids,
        completion_token_ids,
        pad_token_id=pad_token_id,
    )
    return completion_logprobs_from_inputs(
        model,
        inputs,
        use_checkpoint=use_checkpoint,
    )


def prepare_completion_logprob_inputs(
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int = 0,
) -> CompletionLogprobInputs:
    """Build reusable padded tensors for completion logprob gathering."""

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
    gather_index_rows: list[list[int]] = []

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
        gather_index_rows.append(
            [
                completion_start + index if index < len(completion) else 0
                for index in range(max_completion_len)
            ]
        )

        input_rows.append(input_row)
        target_rows.append(target_row)
        completion_mask_rows.append(mask_row)

    input_ids = mx.array(input_rows, dtype=mx.int32)
    target_ids = mx.array(target_rows, dtype=mx.int32)
    gather_indices = mx.array(gather_index_rows, dtype=mx.int32)
    mask = mx.array(completion_mask_rows, dtype=mx.float32)
    return CompletionLogprobInputs(
        input_ids=input_ids,
        target_ids=target_ids,
        gather_indices=gather_indices,
        mask=mask,
    )


def completion_logprobs_from_inputs(
    model: nn.Module,
    inputs: CompletionLogprobInputs,
    use_checkpoint: bool = False,
) -> CompletionLogprobs:
    """Gather completion-token logprobs from prebuilt full-forward tensors."""

    logits = _completion_forward(model, inputs.input_ids, use_checkpoint=use_checkpoint)
    target_logprobs = target_logprobs_from_logits(logits, inputs.target_ids)
    completion_logprob_values = mx.take_along_axis(
        target_logprobs,
        inputs.gather_indices,
        axis=1,
    )
    completion_logprob_values = completion_logprob_values * inputs.mask.astype(
        completion_logprob_values.dtype
    )

    return CompletionLogprobs(
        logprobs=completion_logprob_values,
        mask=inputs.mask,
    )


def prefix_cached_completion_logprobs(
    model: nn.Module,
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int = 0,
) -> CompletionLogprobs:
    """Gather completion logprobs after one prefix prefill per unique prompt."""

    if len(prompt_token_ids) != len(completion_token_ids):
        raise ValueError("prompt_token_ids and completion_token_ids must have the same length.")
    if not prompt_token_ids:
        raise ValueError("At least one sequence is required.")

    max_completion_len = max(len(completion) for completion in completion_token_ids)
    if max_completion_len == 0:
        raise ValueError("At least one completion token is required.")

    completion_mask_rows = [
        [1.0] * len(completion) + [0.0] * (max_completion_len - len(completion))
        for completion in completion_token_ids
    ]
    output_rows: list[mx.array | None] = [None] * len(prompt_token_ids)
    prompt_groups: dict[tuple[int, ...], list[int]] = {}
    for index, prompt in enumerate(prompt_token_ids):
        if not prompt:
            raise ValueError("Prompt token sequences must be non-empty.")
        if not completion_token_ids[index]:
            raise ValueError("At least one completion token is required.")
        prompt_groups.setdefault(tuple(prompt), []).append(index)

    for prompt, row_indices in prompt_groups.items():
        prefix_cache = _prefill_prompt_prefix(model, prompt)
        batch_cache = _batch_cache_from_prefix(prefix_cache, len(row_indices))
        chunks = mx.array(
            [
                [prompt[-1]]
                + list(completion_token_ids[index][:-1])
                + [pad_token_id] * (max_completion_len - len(completion_token_ids[index]))
                for index in row_indices
            ],
            dtype=mx.int32,
        )
        targets = mx.array(
            [
                list(completion_token_ids[index])
                + [pad_token_id] * (max_completion_len - len(completion_token_ids[index]))
                for index in row_indices
            ],
            dtype=mx.int32,
        )
        logits = model(chunks, cache=batch_cache)
        grouped_logprobs = target_logprobs_from_logits(logits, targets)
        group_mask = mx.array(
            [
                [1.0] * len(completion_token_ids[index])
                + [0.0] * (max_completion_len - len(completion_token_ids[index]))
            for index in row_indices
            ],
            dtype=grouped_logprobs.dtype,
        )
        grouped_logprobs = grouped_logprobs * group_mask
        for group_row, original_index in enumerate(row_indices):
            output_rows[original_index] = grouped_logprobs[group_row]

    return CompletionLogprobs(
        logprobs=mx.stack([_require_row(row) for row in output_rows], axis=0),
        mask=mx.array(completion_mask_rows, dtype=mx.float32),
    )


def dual_logprobs(
    model: nn.Module,
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int = 0,
    use_checkpoint: bool = False,
    compute_reference: bool = True,
    prepared_inputs: CompletionLogprobInputs | None = None,
) -> DualLogprobs:
    """Compute policy logprobs and reference logprobs using one model object."""

    if prepared_inputs is None:
        policy = completion_logprobs(
            model,
            prompt_token_ids,
            completion_token_ids,
            pad_token_id,
            use_checkpoint=use_checkpoint,
        )
    else:
        policy = completion_logprobs_from_inputs(
            model,
            prepared_inputs,
            use_checkpoint=use_checkpoint,
        )
    if not compute_reference:
        return DualLogprobs(
            policy=policy.logprobs,
            reference=mx.stop_gradient(policy.logprobs),
            mask=policy.mask,
        )
    with adapters_disabled(model):
        reference = prefix_cached_completion_logprobs(
            model,
            prompt_token_ids,
            completion_token_ids,
            pad_token_id,
        )
        mx.eval(  # Reference sync: materialize logits before restoring LoRA scales.
            reference.logprobs,
            reference.mask,
        )
    return DualLogprobs(
        policy=policy.logprobs,
        reference=reference.logprobs,
        mask=policy.mask,
    )


def _prefill_prompt_prefix(model: nn.Module, prompt: Sequence[int]) -> list[Any]:
    cache = make_prompt_cache(model)
    if len(prompt) > 1:
        prefix = mx.array(list(prompt[:-1]), dtype=mx.int32)
        _ = model(prefix[None], cache=cache)
    return cache


def _batch_cache_from_prefix(prefix_cache: Sequence[Any], batch_size: int) -> list[Any]:
    row_caches = [clone_prompt_cache(prefix_cache) for _ in range(batch_size)]
    return [
        type(layer_cache).merge([row_cache[layer_index] for row_cache in row_caches])
        for layer_index, layer_cache in enumerate(prefix_cache)
    ]


def _require_row(row: mx.array | None) -> mx.array:
    if row is None:
        raise RuntimeError("Internal error: missing prefix-cached logprob row.")
    return row


def _completion_forward(
    model: nn.Module,
    input_ids: mx.array,
    use_checkpoint: bool,
) -> mx.array:
    """Forward completion tokens through the model.

    Gradient checkpointing is applied per transformer layer at model setup.
    The flag is retained for API compatibility with older call sites.
    """

    del use_checkpoint
    return model(input_ids)


def target_logprobs_from_logits(logits: mx.array, target_ids: mx.array) -> mx.array:
    """Gather target-token logprobs without materializing full log-softmax.

    This is algebraically equivalent to gathering from
    ``logits - logsumexp(logits)`` but avoids a large [batch, seq, vocab]
    intermediate on training forwards.
    """

    target_logits = mx.squeeze(
        mx.take_along_axis(logits, target_ids[..., None], axis=-1),
        axis=-1,
    )
    normalizer = mx.squeeze(mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)
    return target_logits - normalizer


def pad_token_id_from_tokenizer(tokenizer: Any) -> int:
    """Resolve a safe right-padding token id from a tokenizer."""

    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    return 0
