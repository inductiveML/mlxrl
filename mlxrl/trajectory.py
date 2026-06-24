"""Trajectory data structures for multi-turn agentic RL."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionSpan:
    """Absolute token span for one model action inside a trajectory."""

    step_index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative.")
        if self.start < 1:
            raise ValueError("Action spans must start after at least one context token.")
        if self.end <= self.start:
            raise ValueError("Action span end must be greater than start.")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TrajectoryStep:
    """One environment state, model action, and resulting reward."""

    observation: str
    state_id: Hashable
    action_text: str
    action_tokens: tuple[int, ...]
    old_policy_logprobs: tuple[float, ...]
    reward: float
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    next_observation: str | None = None

    def __post_init__(self) -> None:
        if not self.action_tokens:
            raise ValueError("Each trajectory step must contain at least one action token.")
        if len(self.old_policy_logprobs) != len(self.action_tokens):
            raise ValueError("Each action token must have one rollout-time logprob.")


@dataclass(frozen=True)
class Trajectory:
    """A complete multi-turn rollout for one task/group sample."""

    task_index: int
    group_index: int
    task: str
    initial_observation: str
    full_token_ids: tuple[int, ...]
    action_spans: tuple[ActionSpan, ...]
    steps: tuple[TrajectoryStep, ...]
    done: bool
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative.")
        if self.group_index < 0:
            raise ValueError("group_index must be non-negative.")
        if len(self.action_spans) != len(self.steps):
            raise ValueError("Each trajectory step must have exactly one action span.")
        if len(self.full_token_ids) < 2:
            raise ValueError("A trajectory needs at least two tokens for logprobs.")
        token_count = len(self.full_token_ids)
        for span, step in zip(self.action_spans, self.steps, strict=True):
            if span.end > token_count:
                raise ValueError("Action span extends past the trajectory token sequence.")
            if span.length != len(step.action_tokens):
                raise ValueError("Action span length must match step action token count.")

    @property
    def total_return(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def action_token_count(self) -> int:
        return sum(span.length for span in self.action_spans)

    def action_token_ids(self) -> tuple[int, ...]:
        tokens: list[int] = []
        for span in self.action_spans:
            tokens.extend(self.full_token_ids[span.start : span.end])
        return tuple(tokens)

    def discounted_returns_to_go(self, gamma: float) -> tuple[float, ...]:
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        returns = [0.0] * len(self.steps)
        running = 0.0
        for index in range(len(self.steps) - 1, -1, -1):
            running = float(self.steps[index].reward) + gamma * running
            returns[index] = running
        return tuple(returns)


def trajectory_from_single_turn(
    *,
    task_index: int,
    group_index: int,
    task: str,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    completion_text: str,
    reward: float,
    state_id: Hashable,
    old_policy_logprobs: Sequence[float] | None = None,
    done: bool = True,
) -> Trajectory:
    """Represent a single-turn completion as a one-step trajectory."""

    prompt = tuple(int(token) for token in prompt_tokens)
    completion = tuple(int(token) for token in completion_tokens)
    logprobs = (
        tuple(float(value) for value in old_policy_logprobs)
        if old_policy_logprobs is not None
        else tuple(0.0 for _ in completion)
    )
    full = prompt + completion
    step = TrajectoryStep(
        observation=task,
        state_id=state_id,
        action_text=completion_text,
        action_tokens=completion,
        old_policy_logprobs=logprobs,
        reward=reward,
        done=done,
        next_observation=None,
    )
    return Trajectory(
        task_index=task_index,
        group_index=group_index,
        task=task,
        initial_observation=task,
        full_token_ids=full,
        action_spans=(
            ActionSpan(
                step_index=0,
                start=len(prompt),
                end=len(full),
            ),
        ),
        steps=(step,),
        done=done,
        truncated=not done,
    )
