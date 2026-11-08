"""Plain-loop reference implementation of Gradecast consensus (and + king).

This is the original, unvectorised version of Gradecast.process: the per-leader
tallies are explicit Python loops over senders. tests/test_protocols.py runs it
against the numpy version in byzantine/protocols/gradecast.py and requires the
same steps, messages, bits and decisions. Slow; for tests only.
"""

from byzantine.network import broadcast
from byzantine.protocols.gradecast import Gradecast, GradecastKing, _entry


def _tally(P, inbox, bad, q):
    c = [0, 0]
    for j in P.senders:
        if j not in bad:
            x = _entry(inbox.get(j), q, P.n)
            if x is not None:
                c[x] += P.w[j]
    return c


def reference_process(P, pid, value):
    n, w = P.n, P.w
    speaks = w[pid] > 0
    v = value
    bad = set(P.initial_bad.get(pid, ()))
    exiting = False
    it = 0
    while True:
        it += 1
        inbox = yield (broadcast(n, v) if speaks else {})
        recv = tuple(None if q in bad else _entry(inbox.get(q), q, n) for q in range(n))
        inbox = yield (broadcast(n, recv) if speaks else {})
        support = [None] * n
        for q in P.senders:
            c = _tally(P, inbox, bad, q)
            if P.strong(c[0]):
                support[q] = 0
            elif P.strong(c[1]):
                support[q] = 1
        inbox = yield (broadcast(n, tuple(support)) if speaks else {})
        if exiting:
            return v
        s, s2 = [0, 0], [0, 0]
        for q in P.senders:
            c = _tally(P, inbox, bad, q)
            x = 0 if c[0] >= c[1] else 1
            if P.strong(c[x]):
                s[x] += w[q]
                s2[x] += w[q]
            elif P.weak(c[x]):
                s[x] += w[q]
                bad.add(q)
            else:
                bad.add(q)
        v = 1 if s[1] > s[0] else 0
        strong_exit = P.strong(s2[v])
        if P.KING:
            king = P.king_of(it)
            inbox = yield (broadcast(n, v) if pid == king else {})
            kv = inbox.get(king)
            if king in bad or kv not in (0, 1) or isinstance(kv, bool):
                kv = None
            if not P.strong(s[v]) and kv is not None:
                v = kv
        if strong_exit:
            if it >= P.cap:
                return v
            exiting = True
        elif it >= P.cap:
            return v


class RefGradecast(Gradecast):
    name = "gradecast"  # same name so adversaries treat it identically

    def process(self, pid, value, rounds=None):
        return reference_process(self, pid, value)


class RefGradecastKing(GradecastKing):
    name = "gc_king"

    def process(self, pid, value, rounds=None):
        return reference_process(self, pid, value)
