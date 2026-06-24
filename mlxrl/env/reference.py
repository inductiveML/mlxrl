"""Small deterministic reference environments for tests and examples."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from mlxrl.env.base import StepResult


def default_action_parser(text: str) -> str:
    """Extract <action>...</action> when present, otherwise return stripped text."""

    match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


@dataclass
class RecurringStateTextEnv:
    """Tiny deterministic env with intentionally recurring states."""

    task: str = "reach goal"
    max_turns: int = 3
    state: str = "start"
    turns: int = 0
    terminal_observation: str = "done"
    history: list[str] = field(default_factory=list)

    def reset(self) -> str:
        self.state = "start"
        self.turns = 0
        self.history.clear()
        return self._observation()

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        self.turns += 1
        self.history.append(action)
        reward = 0.0
        done = False
        if action == "finish":
            self.state = "done"
            reward = 1.0
            done = True
        elif action == "loop":
            self.state = "start"
            reward = 0.0
        elif action == "advance":
            self.state = "middle"
            reward = 0.25
        else:
            self.state = "start"
            reward = -0.25

        if self.turns >= self.max_turns and not done:
            done = True
        observation = self.terminal_observation if self.state == "done" else self._observation()
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info={"turn": self.turns, "state": self.state},
        )

    def state_id(self, observation: str) -> Hashable:
        del observation
        return self.state

    def _observation(self) -> str:
        return f"task={self.task}; state={self.state}"


@dataclass
class SingleTurnRewardEnv:
    """Wrap a single prompt/reward function as a one-step environment."""

    prompt: str
    reward_fn: Callable[[str], float]
    max_turns: int = 1
    _done: bool = False
    _info: Mapping[str, Any] = field(default_factory=dict)

    def reset(self) -> str:
        self._done = False
        return self.prompt

    def step(self, action: str) -> StepResult:
        if self._done:
            raise RuntimeError("SingleTurnRewardEnv.step() called after termination.")
        self._done = True
        reward = float(self.reward_fn(action))
        return StepResult(
            observation="done",
            reward=reward,
            done=True,
            info={"single_turn": True, **dict(self._info)},
        )

    def state_id(self, observation: str) -> Hashable:
        return ("single-turn", observation)
