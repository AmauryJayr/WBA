"""Phase King algorithm (Berman, Garay & Perry, 1989), weighted (Garg & Bridgman, Fig. 2).

alpha_rho rounds (the anchor), 3 phases each, constant-size messages,
requires rho < 1/3. The king of round k is the k-th heaviest process.

Thresholds are written in terms of rho so uniform weights give the
unweighted algorithm exactly:
    unweighted          weighted (here)       Garg & Bridgman Fig. 2
    count >= n - f      s >= 1 - rho          s >= 2/3
    count > f           s > rho               s > 1/3
For rho < 1/3 both weighted forms are correct. The paper's constants are the
rho -> 1/3 limit. The rho form is the one that reduces to the tested baseline.
"""

from __future__ import annotations

from ..network import broadcast
from .base import Protocol, StepInfo

UNDECIDED = "u"


class King(Protocol):
    name = "king"
    VALUES = (0, 1, UNDECIDED)

    @classmethod
    def max_faults(cls, n: int) -> int:
        return (n - 1) // 3

    @property
    def num_steps(self) -> int:
        return 3 * self.anchor

    def step_info(self, step: int) -> StepInfo:
        rnd, phase = divmod(step, 3)
        return [StepInfo(rnd, "vote", None),
                StepInfo(rnd, "propose", None),
                StepInfo(rnd, "king", self.order[rnd])][phase]

    def process(self, pid: int, value, rounds=None):
        n, rho = self.n, self.rho
        speaks = self.w[pid] > 0
        V = value
        for k in (range(self.anchor) if rounds is None else rounds):
            king = self.order[k]
            # Phase 1: exchange preferences; adopt a value only with 1-rho support.
            inbox = yield (broadcast(n, V) if speaks else {})
            s0, s1 = self.weight_of(inbox, 0), self.weight_of(inbox, 1)
            if self.ge(s0, 1 - rho):
                V = 0
            elif self.ge(s1, 1 - rho):
                V = 1
            else:
                V = UNDECIDED

            # Phase 2: exchange the (possibly undecided) preference.
            inbox = yield (broadcast(n, V) if speaks else {})
            s0, s1 = self.weight_of(inbox, 0), self.weight_of(inbox, 1)
            su = self.W - s0 - s1  # undecided, missing and malformed
            myweight = 0
            if self.gt(s0, rho):
                V, myweight = 0, s0
            elif self.gt(s1, rho):
                V, myweight = 1, s1
            elif self.gt(su, rho):
                V, myweight = UNDECIDED, su
            # (with rho < 1/3 one of the three always exceeds rho)

            # Phase 3: the king breaks ties for processes without strong support.
            inbox = yield (broadcast(n, V) if pid == king else {})
            kingvalue = inbox.get(king)
            if V == UNDECIDED or not self.ge(myweight, 1 - rho):
                V = kingvalue if kingvalue in (0, 1) else 1
        return V
