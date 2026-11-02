"""Queen algorithm (Berman & Garay, 1989), unweighted.

f+1 rounds, 2 phases each, constant-size messages, requires n > 4f.
The queen of round k is process k. Written to mirror Fig. 1 of Garg &
Bridgman's weighted paper with every weight equal to 1, so the weighted
version later only changes how votes are tallied and the thresholds:
    unweighted               weighted
    ones > n/2               s1 > 1/2
    count > n/2 + f          myweight > 3/4
"""

from __future__ import annotations

from ..network import broadcast
from .base import Protocol, StepInfo


class Queen(Protocol):
    name = "queen"
    VALUES = (0, 1)

    @classmethod
    def max_faults(cls, n: int) -> int:
        return (n - 1) // 4

    @property
    def num_steps(self) -> int:
        return 2 * (self.f + 1)

    def step_info(self, step: int) -> StepInfo:
        rnd, phase = divmod(step, 2)
        if phase == 0:
            return StepInfo(rnd, "vote", None)
        return StepInfo(rnd, "queen", rnd)

    def process(self, pid: int, value, rounds=None):
        n, f = self.n, self.f
        V = value
        for q in (range(f + 1) if rounds is None else rounds):
            # Phase 1: exchange preferences.
            inbox = yield broadcast(n, V)
            ones = sum(1 for j in range(n) if inbox.get(j) == 1)
            zeros = n - ones  # missing or malformed messages count as 0
            if ones > n / 2:
                myvalue, mycount = 1, ones
            else:
                myvalue, mycount = 0, zeros

            # Phase 2: the queen broadcasts her majority value.
            inbox = yield (broadcast(n, myvalue) if pid == q else {})
            queenvalue = inbox.get(q)
            if queenvalue not in (0, 1):
                queenvalue = 0  # default when the queen is silent or garbled

            V = myvalue if mycount > n / 2 + f else queenvalue
        return V
