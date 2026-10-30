"""Sampling-distribution records for experimental sampler-aware losses.

Filters follow MLX-LM's make_sampler order: top-p, min-p, top-k, temperature.
The recorded distribution is normalized after those operations. Raw model
logprobs remain separate from these behavior-policy probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p


@dataclass(frozen=True)
class SamplerToken:
    """A sampled token's probability and the sampler's unrenormalized head."""

    logprob: float
    topk_ids: tuple[int, ...]
    topk_logprobs: tuple[float, ...]


@dataclass(frozen=True)
class SamplerBatch:
    """Padded, constant sampler probabilities aligned with completion tokens."""

    logprobs: mx.array
    topk_ids: mx.array
    topk_logprobs: mx.array

    def slice(self, start: int, end: int, width: int) -> SamplerBatch:
        return SamplerBatch(
            self.logprobs[start:end, :width],
            self.topk_ids[start:end, :width],
            self.topk_logprobs[start:end, :width],
        )


def sampler_logprobs(
    logprobs: mx.array,
    *,
    temperature: float,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
) -> mx.array:
    """Return the actual categorical law, including greedy point masses."""

    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if temperature == 0:
        ids = mx.argmax(logprobs, axis=-1, keepdims=True)
        return mx.where(mx.arange(logprobs.shape[-1]) == ids, 0.0, -float("inf"))
    # Keep filtering/scaling in the sampler's original dtype before normalizing.
    if 0 < top_p < 1.0:
        logprobs = apply_top_p(logprobs, top_p)
    if min_p != 0.0:
        logprobs = apply_min_p(logprobs, min_p, 1)
    if top_k > 0:
        logprobs = apply_top_k(logprobs, top_k)
    scaled = (logprobs * (1.0 / temperature)).astype(mx.float32)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def capture_sampler_tokens(
    logprobs: mx.array,
    tokens: mx.array,
    *,
    record_top_k: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> tuple[SamplerToken, ...]:
    """Capture a head per batch row without drawing any additional RNG keys."""

    if record_top_k < 1:
        raise ValueError("record_top_k must be positive.")
    q = sampler_logprobs(logprobs, temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k)
    k = min(record_top_k, q.shape[-1])
    ids = mx.argpartition(-q, kth=k - 1, axis=-1)[..., :k]
    head = mx.take_along_axis(q, ids, axis=-1)
    order = mx.argsort(-head, axis=-1)
    ids = mx.take_along_axis(ids, order, axis=-1)
    head = mx.take_along_axis(head, order, axis=-1)
    selected = mx.take_along_axis(q, tokens[..., None], axis=-1).squeeze(-1)
    mx.eval(ids, head, selected)  # Capture only O(batch * k) values on the host.
    return tuple(
        SamplerToken(float(lp), tuple(row_ids), tuple(row_lp))
        for lp, row_ids, row_lp in zip(
            cast(list[float], selected.tolist()),
            cast(list[list[int]], ids.tolist()),
            cast(list[list[float]], head.tolist()),
            strict=True,
        )
    )


def sampler_batch_from_traces(
    traces: Sequence[Sequence[SamplerToken] | None],
    completion_tokens: Sequence[Sequence[int]],
) -> SamplerBatch:
    """Validate captured records, then pad with zero-mass sampler heads."""

    if len(traces) != len(completion_tokens) or not traces:
        raise ValueError("Sampler traces must match the completion batch.")
    records = [token for trace in traces if trace is not None for token in trace]
    if not records:
        raise ValueError("Score centering requires captured sampler probabilities.")
    k = len(records[0].topk_ids)
    if k == 0:
        raise ValueError("Sampler heads must not be empty.")
    width = max(len(tokens) for tokens in completion_tokens)
    selected_rows, id_rows, head_rows = [], [], []
    for trace, tokens in zip(traces, completion_tokens, strict=True):
        if trace is None or len(trace) != len(tokens):
            raise ValueError("Score centering requires one sampler record per completion token.")
        for record, token in zip(trace, tokens, strict=True):
            if len(record.topk_ids) != k or len(record.topk_logprobs) != k:
                raise ValueError("Sampler head widths must match throughout the batch.")
            if len(set(record.topk_ids)) != k or any(i < 0 for i in record.topk_ids):
                raise ValueError("Sampler candidate IDs must be distinct and non-negative.")
            if not math.isfinite(record.logprob) or record.logprob > 1e-5:
                raise ValueError("The sampled token must have a finite log probability <= 0.")
            if any(math.isnan(p) or p > 1e-5 for p in record.topk_logprobs):
                raise ValueError("Sampler head log probabilities must be <= 0.")
            if sum(math.exp(p) for p in record.topk_logprobs) > 1.00001:
                raise ValueError("Sampler head probabilities must not exceed unit mass.")
            if token in record.topk_ids:
                head_lp = record.topk_logprobs[record.topk_ids.index(token)]
                if abs(head_lp - record.logprob) > 1e-5:
                    raise ValueError("Sampled and candidate sampler probabilities disagree.")
        pad = width - len(tokens)
        selected_rows.append([t.logprob for t in trace] + [0.0] * pad)
        id_rows.append([list(t.topk_ids) for t in trace] + [[0] * k for _ in range(pad)])
        head_rows.append(
            [list(t.topk_logprobs) for t in trace] + [[-float("inf")] * k for _ in range(pad)]
        )
    return SamplerBatch(
        mx.array(selected_rows, dtype=mx.float32),
        mx.array(id_rows, dtype=mx.int32),
        mx.array(head_rows, dtype=mx.float32),
    )
