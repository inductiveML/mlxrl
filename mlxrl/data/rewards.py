"""Small reward registry and GSM8K-oriented reward functions."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeAlias

RewardFn: TypeAlias = Callable[..., float]

_REWARD_REGISTRY: dict[str, RewardFn] = {}
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
_ANSWER_CLOSE_RE = re.compile(r"</answer>", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")


def reward(name: str) -> Callable[[RewardFn], RewardFn]:
    """Register a reward function by name."""

    if not name:
        raise ValueError("Reward name must be non-empty.")

    def decorator(fn: RewardFn) -> RewardFn:
        if name in _REWARD_REGISTRY:
            raise ValueError(f"Reward already registered: {name}")
        _REWARD_REGISTRY[name] = fn
        return fn

    return decorator


def get_reward(name: str) -> RewardFn:
    """Fetch a registered reward function."""

    try:
        return _REWARD_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(list_rewards()) or "<none>"
        raise KeyError(f"Unknown reward '{name}'. Known rewards: {known}") from exc


def list_rewards() -> tuple[str, ...]:
    """List registered reward names."""

    return tuple(sorted(_REWARD_REGISTRY))


def extract_answer(text: str | None) -> str | None:
    """Extract an answer using <answer>, then GSM8K ####, then the last number."""

    if text is None:
        return None
    answer = _answer_tag_content(text)
    if answer is not None:
        return answer or None
    if "####" in text:
        answer = text.rsplit("####", maxsplit=1)[-1].strip()
        return answer or None
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    return None


def _normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    numbers = _NUMBER_RE.findall(value)
    if numbers:
        return numbers[-1].replace(",", "").lower()
    return value.strip().replace(",", "").lower()


@reward("accuracy")
def accuracy_reward(completion: str, answer: str | None = None, **_: object) -> float:
    """Return 1.0 when the extracted completion answer matches the expected answer."""

    predicted = _normalize_answer(extract_answer(completion))
    expected = _normalize_answer(extract_answer(answer))
    if predicted is None or expected is None:
        return 0.0
    return float(predicted == expected)


@reward("format")
def format_reward(completion: str, **_: object) -> float:
    """Return 1.0 when the completion contains a non-empty <answer>...</answer> block."""

    return float(extract_answer_from_tag(completion) is not None)


def extract_answer_from_tag(text: str | None) -> str | None:
    """Extract only the explicit <answer>...</answer> block."""

    if text is None:
        return None
    return _answer_tag_content(text) or None


def _answer_tag_content(text: str) -> str | None:
    # Fixed delimiters avoid backtracking or repeated body rescans on malformed output.
    opening = _ANSWER_OPEN_RE.search(text)
    if opening is None:
        return None
    closing = _ANSWER_CLOSE_RE.search(text, opening.end())
    if closing is None:
        return None
    return text[opening.end() : closing.start()].strip()
