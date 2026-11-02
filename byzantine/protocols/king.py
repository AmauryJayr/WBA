"""Phase King algorithm (Berman, Garay & Perry, 1989), unweighted.

f+1 rounds, 3 phases each, constant-size messages, requires n > 3f.
The king of round k is process k. Mirrors Fig. 2 of Garg & Bridgman with
every weight equal to 1:
    unweighted               weighted
    count >= n - f           s >= 2/3
    count > f                s > 1/3
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
        return 3 * (self.f + 1)

    def step_info(self, step: int) -> StepInfo:
        rnd, phase = divmod(step, 3)
        return [StepInfo(rnd, "vote", None),
                StepInfo(rnd, "propose", None),
                StepInfo(rnd, "king", rnd)][phase]

    def process(self, pid: int, value, rounds=None):
        n, f = self.n, self.f
        V = value
        for k in (range(f + 1) if rounds is None else rounds):
            # Phase 1: exchange preferences; adopt a value only with n-f support.
            inbox = yield broadcast(n, V)
            c0 = sum(1 for j in range(n) if inbox.get(j) == 0)
            c1 = sum(1 for j in range(n) if inbox.get(j) == 1)
            if c0 >= n - f:
                V = 0
            elif c1 >= n - f:
                V = 1
            else:
                V = UNDECIDED

            # Phase 2: exchange the (possibly undecided) preference.
            inbox = yield broadcast(n, V)
            c0 = sum(1 for j in range(n) if inbox.get(j) == 0)
            c1 = sum(1 for j in range(n) if inbox.get(j) == 1)
            cu = n - c0 - c1  # undecided, missing and malformed
            mycount = 0
            if c0 > f:
                V, mycount = 0, c0
            elif c1 > f:
                V, mycount = 1, c1
            elif cu > f:
                V, mycount = UNDECIDED, cu
            # (with n > 3f one of the three always exceeds f)

            # Phase 3: the king breaks ties for processes without strong support.
            inbox = yield (broadcast(n, V) if pid == k else {})
            kingvalue = inbox.get(k)
            if V == UNDECIDED or mycount < n - f:
                V = kingvalue if kingvalue in (0, 1) else 1
        return V
