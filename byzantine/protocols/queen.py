"""Queen algorithm (Berman & Garay, 1989), weighted (Garg & Bridgman, Fig. 1).

alpha_rho rounds (the anchor), 2 phases each, constant-size messages,
requires rho < 1/4. The queen of round k is the k-th heaviest process.

Thresholds are written in terms of rho so uniform weights give the
unweighted algorithm exactly:
    unweighted            weighted (here)        Garg & Bridgman Fig. 1
    ones > n/2            s1 > 1/2               s1 > 1/2
    count > n/2 + f       myweight > 1/2 + rho   myweight > 3/4
For rho < 1/4 both are correct. The paper's 3/4 is the rho -> 1/4 limit.
"""

from __future__ import annotations

from fractions import Fraction

from ..network import broadcast
from .base import Protocol, StepInfo

HALF = Fraction(1, 2)


class Queen(Protocol):
    name = "queen"
    VALUES = (0, 1)

    @classmethod
    def max_faults(cls, n: int) -> int:
        return (n - 1) // 4

    @property
    def num_steps(self) -> int:
        return 2 * self.anchor

    def step_info(self, step: int) -> StepInfo:
        rnd, phase = divmod(step, 2)
        if phase == 0:
            return StepInfo(rnd, "vote", None)
        return StepInfo(rnd, "queen", self.order[rnd])

    def process(self, pid: int, value, rounds=None):
        n = self.n
        speaks = self.w[pid] > 0
        V = value
        for q in (range(self.anchor) if rounds is None else rounds):
            queen = self.order[q]
            # Phase 1: exchange preferences.
            inbox = yield (broadcast(n, V) if speaks else {})
            s1 = self.weight_of(inbox, 1)
            s0 = self.W - s1  # missing or malformed messages count as 0
            if self.gt(s1, HALF):
                myvalue, myweight = 1, s1
            else:
                myvalue, myweight = 0, s0

            # Phase 2: the queen broadcasts her majority value.
            inbox = yield (broadcast(n, myvalue) if pid == queen else {})
            queenvalue = inbox.get(queen)
            if queenvalue not in (0, 1):
                queenvalue = 0  # default when the queen is silent or garbled

            V = myvalue if self.gt(myweight, HALF + self.rho) else queenvalue
        return V
