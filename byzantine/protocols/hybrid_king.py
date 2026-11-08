"""Weighted Phase King for a hybrid fault model: crash faults + Byzantine faults.

Some processes can only crash. They behave correctly until they stop, and they
may stop partway through a broadcast. Others are Byzantine. Let rho_b bound
the Byzantine weight and rho_c the crash weight.

Thresholds (q = 1 - rho_b - rho_c is the weight every correct process hears
from at minimum):
    adopt (phase 1)      s >= q
    support (phase 2)    s > rho_b
    keep (phase 3)       myweight >= q
    rounds               anchor for rho_b + rho_c (some king never fails)

Correct when 3*rho_b + 2*rho_c < 1. (Derived here; check against the
hybrid-fault literature, e.g. Thambidurai & Park 1988, Garay & Perry 1992.)
  * Phase-1 values agree: two q-quorums overlap in weight >= 1 - 2rho_b - 2rho_c,
    which is > rho_b, so the overlap contains a non-Byzantine process. Crashed
    processes never send two different values, so it sent the same value to both.
  * Honest king round: if honest p keeps v, then p heard v with weight >= q, of
    which >= q - rho_b came from non-Byzantine senders. A sender that crashed
    mid-broadcast may have skipped the king, so the king hears
    >= q - rho_b - rho_c = 1 - 2rho_b - 2rho_c > rho_b for v. The other value
    has only Byzantine support (<= rho_b), so the king picks v.
  * Persistence: if all correct processes start with v, each hears >= q for v.
With rho_c = 0 this is exactly King.
"""

from __future__ import annotations

from fractions import Fraction

from ..network import broadcast
from .base import Protocol, StepInfo
from .king import UNDECIDED, King


class HybridKing(King):
    name = "hybrid_king"

    def __init__(self, n, f, weights=None, rho=None, order=None, rho_crash=0):
        self.rho_b = Fraction(f, n) if rho is None else Fraction(rho)
        self.rho_c = Fraction(rho_crash)
        # the anchor must cover every failure, crash or Byzantine
        Protocol.__init__(self, n, f, weights, self.rho_b + self.rho_c, order)

    def process(self, pid: int, value, rounds=None):
        n, rb = self.n, self.rho_b
        q = 1 - self.rho_b - self.rho_c
        speaks = self.w[pid] > 0
        V = value
        for k in (range(self.anchor) if rounds is None else rounds):
            king = self.order[k]
            inbox = yield (broadcast(n, V) if speaks else {})
            s0, s1 = self.weight_of(inbox, 0), self.weight_of(inbox, 1)
            V = 0 if self.ge(s0, q) else 1 if self.ge(s1, q) else UNDECIDED

            inbox = yield (broadcast(n, V) if speaks else {})
            s0, s1 = self.weight_of(inbox, 0), self.weight_of(inbox, 1)
            su = self.W - s0 - s1
            myweight = 0
            if self.gt(s0, rb):
                V, myweight = 0, s0
            elif self.gt(s1, rb):
                V, myweight = 1, s1
            elif self.gt(su, rb):
                V, myweight = UNDECIDED, su

            inbox = yield (broadcast(n, V) if pid == king else {})
            kingvalue = inbox.get(king)
            if V == UNDECIDED or not self.ge(myweight, q):
                V = kingvalue if kingvalue in (0, 1) else 1
        return V
