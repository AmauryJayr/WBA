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

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StepInfo:
    round: int
    phase: str
    coordinator: Optional[int]  # pid that is king/queen in this phase, if any


class Protocol:
    """Base class. Every protocol is weighted (Garg & Bridgman).

    weights  non-negative numbers, one per process (default: all 1)
    rho      the largest total *fraction* of weight that may be faulty
             (default f/n, which with uniform weights means "at most f faulty")

    Weights are scaled to integers and rho is a Fraction, so thresholds are
    exact. Uniform weights with rho = f/n give back the unweighted protocols
    exactly, e.g. "weight >= (1 - rho)W" is "count >= n - f".
    """
    name = "base"
    VALUES: Tuple[Any, ...] = (0, 1)

    def __init__(self, n: int, f: int, weights: Optional[Sequence] = None, rho=None,
                 order: Optional[Sequence[int]] = None):
        if not (0 <= f < n):
            raise ValueError("need 0 <= f < n")
        self.n = n
        self.f = f
        if weights is None:
            weights = [1] * n
        fw = [Fraction(x) for x in weights]
        if len(fw) != n or any(x < 0 for x in fw):
            raise ValueError("need n non-negative weights")
        scale = math.lcm(*(x.denominator for x in fw))
        self.w = [int(x * scale) for x in fw]
        self.W = sum(self.w)
        if self.W == 0:
            raise ValueError("total weight must be positive")
        self.rho = Fraction(f, n) if rho is None else Fraction(rho)
        if not (0 <= self.rho < 1):
            raise ValueError("need 0 <= rho < 1")
        self.senders = [j for j in range(n) if self.w[j] > 0]
        # Coordinator order. Default is heaviest first (stable, so uniform
        # weights keep pid order). The anchor alpha_rho is the length of the
        # shortest prefix weighing more than rho; it contains a correct process.
        # Any order is correct; heaviest-first minimises the prefix, while
        # most-reliable-first minimises the expected position of the first
        # correct coordinator (useful for the early-stopping variants).
        if order is None:
            order = sorted(range(n), key=lambda j: -self.w[j])
        self.order = list(order)
        if sorted(self.order) != list(range(n)):
            raise ValueError("order must be a permutation of the pids")
        self.by_weight = self.order  # name kept for older callers
        acc, self.anchor = 0, 0
        for j in self.order:
            acc += self.w[j]
            self.anchor += 1
            if self.gt(acc, self.rho):
                break

    # -- exact weight comparisons: s is in integer weight units ---------------
    def ge(self, s: int, frac) -> bool:
        """s >= frac * W"""
        frac = Fraction(frac)
        return s * frac.denominator >= frac.numerator * self.W

    def gt(self, s: int, frac) -> bool:
        """s > frac * W"""
        frac = Fraction(frac)
        return s * frac.denominator > frac.numerator * self.W

    def weight_of(self, inbox, value) -> int:
        return sum(self.w[j] for j in self.senders if inbox.get(j) == value)

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
