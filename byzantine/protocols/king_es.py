"""Early-stopping weighted Phase King. Requires rho < 1/4.

Same three phases as King (king.py). Two changes:

  keep-own   in phase 3, keep your value if V != undecided and your phase-2
             weight is >= 1 - 2*rho, instead of >= 1 - rho
  decide     if your phase-2 weight for V is >= 1 - rho, decide V, run one more
             round so the others can decide too, then halt with V

Why it is correct (rho < 1/4):
  * Phase 1 non-undecided honest values all agree: two honest processes holding
    different values would need two (1-rho) quorums that overlap in more than
    rho of the weight, so in some honest process. Faulty weight is <= rho.
  * Honest king round: if an honest q keeps v, then q's phase-2 weight for v is
    >= 1-2rho, so at least 1-3rho of it is honest. That exceeds rho when
    rho < 1/4, so the honest king also sees more than rho for v and picks v.
    The other value has <= rho, and it is checked first only when it is 0.
    So every honest process ends the round with v. This is the step that
    forces rho < 1/4.
  * Decide: if p sees >= 1-rho for v in phase 2, every honest process sees
    >= 1-2rho > rho for v. So all set V = v with weight >= 1-2rho and keep it
    whatever the king says. Next round they all start with v and all decide.
  * Persistence: if all honest start a round with v, they all see >= 1-rho.
  * Without an early decision, alpha_rho rounds still contain an honest king.

Rounds: at most (position of the first honest king) + 2, capped at alpha_rho.
Ordering coordinators most-reliable-first makes the expected count small.
"""

from __future__ import annotations

from ..network import broadcast
from .king import UNDECIDED, King


class KingES(King):
    name = "king_es"

    @classmethod
    def max_faults(cls, n: int) -> int:
        return (n - 1) // 4

    def process(self, pid: int, value, rounds=None):
        n, rho = self.n, self.rho
        speaks = self.w[pid] > 0
        V = value
        decided = None
        for k in (range(self.anchor) if rounds is None else rounds):
            king = self.order[k]
            inbox = yield (broadcast(n, V) if speaks else {})
            s0, s1 = self.weight_of(inbox, 0), self.weight_of(inbox, 1)
            if self.ge(s0, 1 - rho):
                V = 0
            elif self.ge(s1, 1 - rho):
                V = 1
            else:
                V = UNDECIDED

            inbox = yield (broadcast(n, V) if speaks else {})
            s0, s1 = self.weight_of(inbox, 0), self.weight_of(inbox, 1)
            su = self.W - s0 - s1
            myweight = 0
            if self.gt(s0, rho):
                V, myweight = 0, s0
            elif self.gt(s1, rho):
                V, myweight = 1, s1
            elif self.gt(su, rho):
                V, myweight = UNDECIDED, su

            inbox = yield (broadcast(n, V) if pid == king else {})
            if decided is not None:
                return decided  # this was the extra round
            kingvalue = inbox.get(king)
            if V == UNDECIDED or not self.ge(myweight, 1 - 2 * rho):
                V = kingvalue if kingvalue in (0, 1) else 1
            elif self.ge(myweight, 1 - rho):
                decided = V
        return V if decided is None else decided
