"""Weighted Byzantine agreement with signatures (Dolev & Strong 1983), rho < 1/2.

Every positive-weight process broadcasts its input with a signature chain.
That is interactive consistency, with n Dolev-Strong broadcasts in parallel:

  round r   accept value v for sender s if it arrives with a chain of r
            distinct valid signatures from positive-weight processes, starting
            with s. Relay each newly accepted value (at most 2 per sender) in
            round r+1, with your own signature appended.
  decide    after R rounds, sender s's value is the single value accepted for
            it (else none). Output the weighted majority of those values over
            the senders' weights (ties -> 0).

R = (the most faulty positive-weight processes, i.e. the most of the
lightest that fit in rho) + 1. A chain of R signatures therefore contains an
honest one, and the standard argument gives every honest process the same
accepted sets. Agreement follows. Validity needs honest weight > 1/2, so
rho < 1/2: signatures move the resilience bound from 1/3 to 1/2.
No early stopping: always R rounds, plus one round in which positive-weight
processes announce their decision for the zero-weight processes, which
do not relay.

Signatures are modelled with HMAC keys held by this protocol object. Honest
processes sign inside process(). An adversary can sign only through
signer_for(faulty), so it can't forge an honest signature (by convention: the
adversaries in this repo only use that signer).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from fractions import Fraction
from typing import Optional, Sequence

from ..network import broadcast
from .base import Protocol, StepInfo


class DolevStrong(Protocol):
    name = "dolev_strong"
    VALUES = (0, 1)

    def __init__(self, n: int, f: int, weights: Optional[Sequence] = None, rho=None,
                 order: Optional[Sequence[int]] = None):
        super().__init__(n, f, weights, rho, order)
        k, acc = 0, 0
        for x in sorted(self.w[j] for j in self.senders):
            if self.gt(acc + x, self.rho):
                break
            acc += x
            k += 1
        self.max_bad = k
        self.R = k + 1
        self._keys = {j: os.urandom(16) for j in self.senders}
        self._pos = set(self.senders)

    @classmethod
    def max_faults(cls, n: int) -> int:
        return (n - 1) // 2

    @property
    def num_steps(self) -> int:
        return self.R + 1  # R relay rounds + one decision round for zero-weight processes

    def step_info(self, step: int) -> StepInfo:
        return StepInfo(step, "relay", None)

    # -- signatures --------------------------------------------------------------
    def _sig(self, signer, s, v, prefix):
        return hmac.new(self._keys[signer], repr((s, v, prefix)).encode(), hashlib.sha256).digest()[:8]

    def signer_for(self, pids):
        """Signing capability for the given (faulty) processes only."""
        allowed = set(pids) & self._pos

        def sign(signer, s, v, chain):
            if signer not in allowed:
                raise PermissionError("cannot sign for that process")
            prefix = tuple(c[0] for c in chain)
            return chain + ((signer, self._sig(signer, s, v, prefix)),)
        return sign

    def valid(self, item, r: int) -> bool:
        try:
            s, v, chain = item
        except (TypeError, ValueError):
            return False
        if v not in (0, 1) or isinstance(v, bool) or not isinstance(chain, tuple) or len(chain) != r:
            return False
        signers = [c[0] for c in chain]
        if signers[0] != s or len(set(signers)) != r or not set(signers) <= self._pos:
            return False
        for i, (signer, sig) in enumerate(chain):
            if not hmac.compare_digest(sig, self._sig(signer, s, v, tuple(signers[:i]))):
                return False
        return True

    def process(self, pid: int, value, rounds=None):
        n, w = self.n, self.w
        signs = pid in self._pos
        own = self.signer_for([pid]) if signs else None
        accepted = {s: set() for s in self.senders}
        out = [(pid, value, own(pid, pid, value, ()))] if signs else []
        if signs:
            accepted[pid].add(value)
        for r in range(1, self.R + 1):
            inbox = yield (broadcast(n, tuple(out)) if out else {})
            out = []
            for src in sorted(inbox):
                msg = inbox[src]
                if not isinstance(msg, tuple):
                    continue
                for item in msg:
                    if not self.valid(item, r):
                        continue
                    s, v, chain = item
                    if v in accepted[s] or len(accepted[s]) >= 2:
                        continue
                    accepted[s].add(v)
                    if signs and pid not in [c[0] for c in chain]:
                        out.append((s, v, own(pid, s, v, chain)))
        tally = [0, 0]
        for s, vals in accepted.items():
            if len(vals) == 1:
                tally[next(iter(vals))] += w[s]
        decision = 1 if tally[1] > tally[0] else 0
        # Zero-weight processes do not relay, so their accepted sets can differ
        # (a value revealed only to them in the last rounds goes no further).
        # So positive-weight processes announce their decision, and zero-weight
        # processes adopt the value announced by weight > 1/2. The honest
        # positive-weight processes agree and hold > 1/2, since rho < 1/2.
        inbox = yield (broadcast(n, ("decide", decision)) if signs else {})
        if signs:
            return decision
        votes = [0, 0]
        for j in self.senders:
            m = inbox.get(j)
            if isinstance(m, tuple) and len(m) == 2 and m[0] == "decide" and m[1] in (0, 1):
                votes[m[1]] += w[j]
        return 1 if self.gt(votes[1], Fraction(1, 2)) else 0
