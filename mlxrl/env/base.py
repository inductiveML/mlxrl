"""Gym-style protocols for multi-turn text environments."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class StepResult:
    """Normalized result of one environment step."""

    observation: str
    reward: float
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    """Minimal text environment used by mlxrl's agentic rollout engine."""

    max_turns: int

    def reset(self) -> str:
        """Return the initial observation."""
        ...

    def step(self, action: str) -> StepResult | tuple[str, float, bool, Mapping[str, Any]]:
        """Apply one action and return observation, reward, done, info."""
        ...

    def state_id(self, observation: str) -> Hashable:
        """Return a hashable anchor-state id for grouping."""
        ...


@runtime_checkable
class BatchEnvironment(Protocol):
    """Optional vectorized stepping hook for a group of sibling env states."""

    def step_batch(
        self,
        actions: Sequence[str],
    ) -> Sequence[StepResult | tuple[str, float, bool, Mapping[str, Any]]]:
        """Apply one action per active group member."""
        ...


class ActionParser(Protocol):
    """Optional parser from generated text to environment action string."""

    def parse_action(self, text: str) -> str:
        """Extract the environment action from generated model text."""
        ...


EnvFactory = Callable[[Any, int, int], Environment]


def coerce_step_result(
    result: StepResult | tuple[str, float, bool, Mapping[str, Any]],
) -> StepResult:
    """Normalize tuple-style env results into StepResult."""

    if isinstance(result, StepResult):
        return result
    observation, reward, done, info = result
    return StepResult(
        observation=str(observation),
        reward=float(reward),
        done=bool(done),
        info=info,
    )
