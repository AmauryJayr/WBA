"""Protocol interface.

A protocol object holds the system parameters (n, f) and exposes:

  process(pid, value, rounds=None) -> generator
      The code run by an honest process. Each `yield outbox` is one synchronous
      step; the generator receives that step's inbox and finally `return`s the
      decided value. `rounds` restricts execution to the given round numbers
      (default: all), which lets the model checker run one round at a time.

  step_info(step) -> StepInfo
      What a given global step means (round number, phase name, coordinator).
      Honest code does not need it; adversaries use it to target coordinator
      phases, and the experiment harness uses it for reporting.

  VALUES
      The message alphabet honest processes use. Adversaries draw from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class StepInfo:
    round: int
    phase: str
    coordinator: Optional[int]  # pid that is king/queen in this phase, if any


class Protocol:
    name = "base"
    VALUES: Tuple[Any, ...] = (0, 1)

    def __init__(self, n: int, f: int):
        if not (0 <= f < n):
            raise ValueError("need 0 <= f < n")
        self.n = n
        self.f = f

    @classmethod
    def max_faults(cls, n: int) -> int:
        """Largest f the protocol is proven to tolerate for this n."""
        raise NotImplementedError

    def process(self, pid: int, value: Any):
        raise NotImplementedError

    def step_info(self, step: int) -> StepInfo:
        raise NotImplementedError

    @property
    def num_steps(self) -> int:
        raise NotImplementedError
