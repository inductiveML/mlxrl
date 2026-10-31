"""Token spans for experimental CONF. Extractors never need ground-truth answers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenSpan:
    """Half-open indices into completion tokens (prompt tokens are excluded)."""

    start: int
    end: int


@dataclass(frozen=True)
class ConfidenceStep:
    """A reasoning segment, confidence result spans, and additional excluded tokens.

    Results are automatically excluded from credit. Multiple results are averaged
    in probability space. An empty result list makes a segment unscored.
    """

    span: TokenSpan
    results: tuple[TokenSpan, ...] = ()
    exclusions: tuple[TokenSpan, ...] = ()


class ConfidenceExtractor(Protocol):
    def __call__(
        self,
        tokenizer: Any,
        completion_tokens: Sequence[int],
    ) -> tuple[ConfidenceStep, ...]: ...


_RESULT = re.compile(r"=\s*([+\-\u2212]?\d+(?:\.\d+)?)\b(?![.\d]|\s*[+*/\u2212-])")
_OPERATOR = re.compile(r"[\w)]\s*[+*/\u2212-]\s*[+\-\u2212]?[\w(]")
_ANSWER = re.compile(r"\s*(?:ANSWER\s*:|FINAL\s*:|####|---|<answer>|</think>)", re.IGNORECASE)


def arithmetic_steps(
    tokenizer: Any,
    completion_tokens: Sequence[int],
) -> tuple[ConfidenceStep, ...]:
    """Read explicit ``step: b = a + 3 = 5`` lines, up to the answer block.

    Confidence reads the last written numeric result on each arithmetic line;
    every written result is excluded from credit. This intentionally small reader
    does not claim to parse arbitrary prose or validate arithmetic. Token alignment
    uses actual decoded prefixes; unstable byte/token boundaries fail closed.
    """

    ids = list(completion_tokens)
    if not ids:
        return ()
    prefixes = [""] + [str(tokenizer.decode(ids[:i])) for i in range(1, len(ids) + 1)]
    text = prefixes[-1]
    if any(not text.startswith(p) for p in prefixes):
        return ()
    offsets = [len(p) for p in prefixes]
    if any(a > b for a, b in pairwise(offsets)):
        return ()
    bounds = list(pairwise(offsets))
    steps = []
    pos = 0
    answer = _ANSWER.search(text)
    reasoning = text[: answer.start()] if answer else text
    for line in reasoning.splitlines(keepends=True):
        start, end = pos, pos + len(line)
        pos = end
        if _ANSWER.match(line):
            break
        if not re.match(r"\s*step\s*:", line, re.IGNORECASE):
            continue
        indices = [i for i, (a, b) in enumerate(bounds) if start <= a < b <= end]
        if not indices:
            continue
        segment = TokenSpan(indices[0], indices[-1] + 1)
        values = []
        valid = True
        for match in _RESULT.finditer(line):
            a, b = start + match.start(1), start + match.end(1)
            hits = [i for i, (x, y) in enumerate(bounds) if x < b and y > a]
            if not hits or hits[0] < segment.start or hits[-1] >= segment.end:
                valid = False
                break
            values.append(TokenSpan(hits[0], hits[-1] + 1))
        if valid:
            results = (values[-1],) if values and _OPERATOR.search(line) else ()
            steps.append(ConfidenceStep(segment, results, tuple(values)))
    return tuple(steps)


ARITHMETIC_CONFIDENCE_INSTRUCTION = (
    "\nWrite each reasoning calculation on its own line as "
    "'step: expression = numeric_result'. Finish with '<answer>final_number</answer>'."
)
