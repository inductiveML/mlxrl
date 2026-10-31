"""Experimental CONF warm-start credit from the model's own frozen confidence.

Independently implemented from RAMP's CONF-5/6 mathematics. No critic, executor,
or correctness labels enter step scores. Optional task scores only identify the
mixed groups used to calibrate credit magnitude.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import mlx.core as mx

from mlxrl.config import ConfidenceConfig
from mlxrl.data.confidence import ConfidenceStep, TokenSpan

if TYPE_CHECKING:
    from mlxrl.train.grpo import GRPOBatch


def _validate_span(span: TokenSpan, start: int, end: int) -> None:
    if (
        type(span.start) is not int
        or type(span.end) is not int
        or not start <= span.start < span.end <= end
    ):
        raise ValueError("CONF spans must be nonempty, in bounds, and inside their step.")


def centered_confidence(
    logprobs: Sequence[float],
    steps: Sequence[ConfidenceStep],
) -> tuple[list[float], list[bool], int]:
    """Unit-scale credit, eligible mask, and scored-step count for one completion.

    Probabilities are products over result tokens. The baseline is weighted by
    eligible token counts of scored steps. Unscored steps carry exactly zero.
    Host arithmetic freezes all scores and scaling before differentiation.
    """

    n = len(logprobs)
    mask = [False] * n
    entries: list[tuple[list[int], float]] = []
    previous_end = 0
    for step in steps:
        _validate_span(step.span, previous_end, n)
        previous_end = step.span.end
        excluded = set()
        for span in (*step.results, *step.exclusions):
            _validate_span(span, step.span.start, step.span.end)
            excluded.update(range(span.start, span.end))
        eligible = [i for i in range(step.span.start, step.span.end) if i not in excluded]
        for i in eligible:
            mask[i] = True
        scores = []
        for result in step.results:
            values = logprobs[result.start : result.end]
            if any(math.isnan(x) or x > 0 for x in values):
                raise ValueError("CONF requires nonpositive raw model log-probabilities.")
            scores.append(math.exp(math.fsum(values)))
        if scores and eligible:
            entries.append((eligible, statistics.fmean(scores)))
    credit = [0.0] * n
    count = sum(len(tokens) for tokens, _ in entries)
    if count:
        mean = math.fsum(len(tokens) * score for tokens, score in entries) / count
        for tokens, score in entries:
            for token in tokens:
                credit[token] = score - mean
    return credit, mask, len(entries)


def _percentile99(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * 0.99
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


@dataclass(frozen=True)
class ConfidenceMetrics:
    update: int
    active: bool = False
    scale: float = 0.0
    share: float | None = None
    p99: float | None = None
    mixed_groups: int = 0
    scored_steps: int = 0
    credited_tokens: int = 0
    held: bool = False
    capped: bool = False


class ConfidenceWarmStart:
    """Prepare CONF once per optimizer update, outside the gradient graph.

    Call ``apply`` with one-based optimizer update numbers, including after the
    warm start. Save ``state_dict`` alongside model/optimizer/RNG state to resume.
    Each batch keeps its scalar GRPO advantages as well as optional token credits.
    """

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        config = config or ConfidenceConfig()
        config.validate()
        self.config = config
        self.last_update = 0
        self.previous_scale: float | None = None

    def active(self, update: int) -> bool:
        return self.config.enabled and 1 <= update <= self.config.warmup_updates

    def state_dict(self) -> dict[str, int | float | None]:
        return {"last_update": self.last_update, "previous_scale": self.previous_scale}

    def load_state_dict(self, state: dict[str, int | float | None]) -> None:
        if set(state) != {"last_update", "previous_scale"}:
            raise ValueError("Invalid CONF controller state fields.")
        update, scale = state["last_update"], state["previous_scale"]
        if type(update) is not int or update < 0:
            raise ValueError("Invalid CONF update number.")
        if scale is not None and (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale < 0
        ):
            raise ValueError("Invalid CONF previous scale.")
        self.last_update, self.previous_scale = update, scale

    def apply(
        self,
        batch: GRPOBatch,
        steps: Sequence[Sequence[ConfidenceStep]],
        *,
        update: int,
        group_size: int,
        task_scores: Sequence[float] | None = None,
    ) -> tuple[GRPOBatch, ConfidenceMetrics]:
        if type(update) is not int or update != self.last_update + 1:
            raise ValueError(
                "CONF requires consecutive one-based updates; restore state to resume."
            )
        if self.config.enabled and batch.token_advantages is not None:
            raise ValueError("CONF credit has already been attached to this batch.")
        if not self.active(update):
            self.last_update = update
            return batch, ConfidenceMetrics(update)
        if batch.confidence_logprobs is None:
            raise ValueError("CONF requires record_confidence=True when preparing the batch.")
        n, width = batch.mask.shape
        if type(group_size) is not int or group_size < 1 or n % group_size:
            raise ValueError("CONF requires complete prompt groups.")
        if len(steps) != n or batch.confidence_logprobs.shape != batch.mask.shape:
            raise ValueError("CONF spans/logprobs must align with the batch.")
        if batch.advantages.shape != (n,):
            raise ValueError("CONF requires scalar base advantages per completion.")
        adv = cast(list[float], batch.advantages.tolist())
        scores = (
            list(task_scores)
            if task_scores is not None
            else cast(list[float], batch.rewards.tolist())
        )
        if len(scores) != n or not all(math.isfinite(x) for x in (*adv, *scores)):
            raise ValueError("CONF advantages/task scores must be finite and aligned.")
        raw = cast(list[list[float]], batch.confidence_logprobs.tolist())
        credits, eligible, scored_rows = [], [], []
        scored = 0
        for row, units, tokens in zip(raw, steps, batch.completion_token_ids, strict=True):
            cr, mask, count = centered_confidence(row[: len(tokens)], units)
            credits.append(cr)
            eligible.append(mask)
            scored += count
            scored_rows.append(count > 0)
        mixed = [len(set(scores[i : i + group_size])) > 1 for i in range(0, n, group_size)]
        carrying = [any(credits[i]) for i in range(n)]
        mixed_groups = sum(
            flag and any(carrying[i * group_size : (i + 1) * group_size])
            for i, flag in enumerate(mixed)
        )
        numerator = denominator = 0.0
        magnitudes = []
        for i, (cr, mask) in enumerate(zip(credits, eligible, strict=True)):
            if mixed[i // group_size] and scored_rows[i] and any(mask):
                numerator += math.fsum(abs(x) for x in cr)
                denominator += sum(mask) * abs(adv[i])
                magnitudes.extend(abs(x) for x, yes in zip(cr, mask, strict=True) if yes)
        median = (
            statistics.median([abs(a) for i, a in enumerate(adv) if mixed[i // group_size]])
            if any(mixed)
            else 0.0
        )
        share1 = numerator / denominator if denominator else 0.0
        p991 = _percentile99(magnitudes) / median if median else 0.0
        scale = self.config.target_share / share1 if share1 else 0.0
        held = mixed_groups < self.config.min_mixed_groups and self.previous_scale is not None
        if held:
            assert self.previous_scale is not None
            scale = self.previous_scale
        # Reapply the measured cap even when holding a previous scale.
        capped = bool(p991 and scale * p991 > self.config.p99_cap)
        if capped:
            scale = self.config.p99_cap / p991
        if not math.isfinite(scale):
            raise ValueError("CONF produced a nonfinite scale.")
        rows = [
            [adv[i] + scale * x for x in cr] + [0.0] * (width - len(cr))
            for i, cr in enumerate(credits)
        ]
        token_advantages = mx.stop_gradient(mx.array(rows, dtype=mx.float32))
        if not bool(mx.all(mx.isfinite(token_advantages))):
            raise ValueError("CONF produced nonfinite token advantages.")
        self.last_update = update
        if share1 or held:
            self.previous_scale = scale
        metrics = ConfidenceMetrics(
            update,
            True,
            scale,
            scale * share1 if denominator else None,
            scale * p991 if median else None,
            mixed_groups,
            scored,
            sum(sum(x != 0 for x in cr) for cr in credits),
            held,
            capped,
        )
        # A zero contribution is an exact no-op, including loss dtype and shape.
        if not scale or not any(carrying):
            return batch, metrics
        return replace(batch, token_advantages=token_advantages), metrics
