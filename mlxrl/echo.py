"""ECHO token-role helpers and alpha scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MASKED = 0
ACTION = 1
ECHO = 2
VALID_TOKEN_ROLES = frozenset({MASKED, ACTION, ECHO})
EchoScheduleName = Literal["constant", "linear_taper_to_zero"]


@dataclass(frozen=True)
class EchoSchedule:
    """Schedule ECHO's SFT strength without changing the RL objective."""

    alpha: float = 0.0
    schedule: EchoScheduleName = "constant"
    taper_steps: int | None = None

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("ECHO alpha must be non-negative.")
        if self.schedule not in {"constant", "linear_taper_to_zero"}:
            raise ValueError("ECHO schedule must be 'constant' or 'linear_taper_to_zero'.")
        if (
            self.schedule == "linear_taper_to_zero"
            and (self.taper_steps is None or self.taper_steps < 1)
        ):
            raise ValueError("linear_taper_to_zero requires taper_steps >= 1.")

    def value(self, step: int) -> float:
        """Return alpha for a zero-based optimizer step."""

        if step < 0:
            raise ValueError("step must be non-negative.")
        if self.schedule == "constant" or self.alpha == 0.0:
            return self.alpha
        assert self.taper_steps is not None
        fraction = max(0.0, 1.0 - (float(step) / float(self.taper_steps)))
        return self.alpha * fraction
