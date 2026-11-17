"""Gradecast-based Byzantine consensus (Ben-Or, Dolev & Hoch, "ByzConsensus"),
weighted in the style of Garg & Bridgman.

Each iteration is one Gradecast run by *every* positive-weight process at once
(3 synchronous steps):

    send     each leader sends its value v to everyone
    echo     everyone forwards, for every leader, the value received from it
    support  for leader q, if echoes for some value carry weight >= 1 - rho,
             support that value
    grade    support weight >= 1 - rho -> grade 2; > rho -> grade 1; else grade 0

Then (Fig. 2 of the paper):
    maj   = value with the largest total *leader weight* among grades >= 1
            (ties -> 0)
    v     = maj
    BAD  |= every leader received with grade <= 1    (ignored from now on)
    exit  if leaders with grade 2 for maj carry weight >= 1 - rho; then run one
          more iteration so the others can also exit, and decide v

Unweighted -> weighted, with uniform weights giving back the paper exactly:
    count >= n - t        weight >= 1 - rho
    count >= t + 1        weight >  rho
    n > 3t                rho < 1/3

Zero-weight processes send nothing and are ignored as leaders, but still
receive messages and decide. Weights are scaled to integers and rho is a
Fraction, so every threshold comparison is exact.

Message formats (anything else counts as "no message"):
    send     0 or 1
    echo     tuple of length n (entry q = value received from leader q, or None)
    support  tuple of length n (entry q = supported value for leader q, or None)
For echo and support, a bare 0/1 is shorthand for "this value for every leader",
so the generic scalar adversaries still mount meaningful attacks.
"""

from __future__ import annotations

import random
import zlib
from typing import Optional, Sequence

import numpy as np

from ..network import broadcast
from .base import Protocol, StepInfo

PHASES = ("send", "echo", "support")
NONE = -1  # "no value" inside the numpy encoding


def _entry(msg, q: int, n: int):
    if isinstance(msg, (tuple, np.ndarray)) and len(msg) == n:
        x = msg[q]
        if isinstance(x, np.integer):
            x = int(x)
    else:
        x = msg
    # only a plain 0/1 counts; anything else a Byzantine sender puts here
    # (strings, floats, bools, nested objects) is "no value"
    return x if type(x) is int and x in (0, 1) else None


def _row(msg, n: int) -> np.ndarray:
    """Decode an echo/support message into an int8 vector (0, 1 or NONE per leader)."""
    if isinstance(msg, np.ndarray):
        return msg if msg.shape == (n,) else np.full(n, NONE, dtype=np.int8)
    if isinstance(msg, tuple):
        if len(msg) != n:
            return np.full(n, NONE, dtype=np.int8)
        return np.array([x if (x in (0, 1) and not isinstance(x, bool)) else NONE for x in msg],
                        dtype=np.int8)
    if msg in (0, 1) and not isinstance(msg, bool):
        return np.full(n, msg, dtype=np.int8)  # shorthand: same value for every leader
    return np.full(n, NONE, dtype=np.int8)


def _well_formed_vector(msg, n: int) -> bool:
    """Exactly the echo/support messages _row accepts as meaningful: a length-n
    vector of 0/1/NONE, or the scalar shorthand 0/1. Anything else is malformed."""
    if isinstance(msg, np.ndarray):
        return msg.shape == (n,) and bool(np.isin(msg, (NONE, 0, 1)).all())
    if isinstance(msg, tuple):
        return len(msg) == n and all(type(x) is int and x in (0, 1) or x is None for x in msg)
    return type(msg) is int and msg in (0, 1)


class Gradecast(Protocol):
    name = "gradecast"
    VALUES = (0, 1)
    KING = False  # GradecastKing: a rotating king breaks ties (4th step)
    COIN = False  # GradecastCoin: a common coin breaks ties (4th step)

    def __init__(self, n: int, f: int, weights: Optional[Sequence] = None, rho=None,
                 order: Optional[Sequence[int]] = None):
        # Correctness is proven only for rho < 1/3. Larger values are allowed so
        # the boundary experiments can measure what happens past the bound.
        super().__init__(n, f, weights, rho, order)
        # Worst case: every faulty process causes one disagreement before it is
        # expelled. The most faulty positive-weight processes is the largest k
        # such that the k *lightest* of them weigh at most rho in total.
        # Compare the anchor, which counts the *heaviest*.
        self._den, self._num = self.rho.denominator, self.rho.numerator
        k, acc = 0, 0
        for x in sorted(self.w[j] for j in self.senders):
            if self.weak(acc + x):
                break
            acc += x
            k += 1
        self.max_bad = k
        self.cap = k + 1  # iterations (paper: t + 1)
        self._S = np.array(self.senders, dtype=np.int64)
        self._silent = np.array([self.w[j] == 0 for j in range(n)], dtype=bool)
        self._wS = np.array([self.w[j] for j in self.senders], dtype=object
                            if self.W * self._den > 2**62 else np.int64)
        # Repeated consensus (Ben-Or et al. Sec. 5.1): BAD sets carried between
        # instances. initial_bad[pid] seeds a process; final_bad[pid] is its set
        # at return.
        self.initial_bad: dict = {}
        self.final_bad: dict = {}
        # Garg & Bridgman Lemma 8 part 1: a positive-weight process that sends
        # nothing, or something malformed, is faulty. Recorded separately from
        # BAD so the protocol itself is unchanged; used by the weight update.
        self.detected: dict = {}
        self.coin_seed = 0

    def strong(self, s) -> bool:
        """s >= (1 - rho) * W"""
        return self.ge(int(s), 1 - self.rho)

    def weak(self, s) -> bool:
        """s > rho * W"""
        return self.gt(int(s), self.rho)

    # vectorised thresholds: strong c*den >= (den-num)*W ; weak c*den > num*W
    def _strong_v(self, c):
        return c * self._den >= (self._den - self._num) * self.W

    def _weak_v(self, c):
        return c * self._den > self._num * self.W

    @classmethod
    def max_faults(cls, n: int) -> int:
        return (n - 1) // 3

    @property
    def steps_per_iteration(self) -> int:
        return 4 if (self.KING or self.COIN) else 3

    @property
    def num_steps(self) -> int:
        return self.steps_per_iteration * (self.cap + 1)

    def king_of(self, it: int) -> int:
        """King of iteration `it` (1-based), cycling through the coordinator order."""
        return self.order[(it - 1) % self.n]

    def coin(self, it: int) -> int:
        """Common coin of iteration `it`: an oracle, revealed only in step 4.
        (The adversaries in this repo never look ahead at it.)"""
        return random.Random(zlib.crc32(repr((self.coin_seed, it)).encode())).randint(0, 1)

    def step_info(self, step: int) -> StepInfo:
        rnd, phase = divmod(step, self.steps_per_iteration)
        if phase == 3:
            if self.COIN:
                return StepInfo(rnd, "coin", None)
            return StepInfo(rnd, "king", self.king_of(rnd + 1))
        return StepInfo(rnd, PHASES[phase], None)

    # -- one gradecast for every leader, in parallel --------------------------
    def _tallies(self, inbox, bad):
        """Per-leader weight of 0s and 1s, over senders not in BAD."""
        n = self.n
        keep = [i for i, j in enumerate(self.senders) if j not in bad]
        if not keep:
            z = np.zeros(n, dtype=np.int64)
            return z, z
        M = np.stack([_row(inbox.get(self.senders[i]), n) for i in keep])
        w = self._wS[keep]
        return w @ (M == 0), w @ (M == 1)

    def _flag_missing(self, pid, bad, inbox, kind):
        """Part-1 detection for one step: every positive-weight process not in
        BAD must have sent a well-formed message. Only called before pid's own
        extra iteration, when every honest process is still running (Claim 7.5)."""
        n = self.n
        det = self.detected.setdefault(pid, set())
        for j in self.senders:
            if j in bad:
                continue
            m = inbox.get(j)
            if kind == "send":
                ok = _entry(m, j, n) is not None
            else:
                ok = _well_formed_vector(m, n)
            if not ok:
                det.add(j)

    def _finish(self, pid, v, bad):
        self.final_bad[pid] = set(bad)
        return v

    def process(self, pid: int, value, rounds=None):
        n, w = self.n, self.w
        speaks = w[pid] > 0
        v = value
        bad: set = set(self.initial_bad.get(pid, ()))
        exiting = False
        it = 0
        guard = 4 * n + 100  # only the coin variant can run long
        while True:
            it += 1
            # send (messages from BAD processes are ignored, paper line 1)
            inbox = yield (broadcast(n, v) if speaks else {})
            if not exiting:
                self._flag_missing(pid, bad, inbox, "send")
            recv = np.full(n, NONE, dtype=np.int8)
            for q in range(n):
                if q not in bad:
                    e = _entry(inbox.get(q), q, n)
                    if e is not None:
                        recv[q] = e
            # echo
            inbox = yield (broadcast(n, recv) if speaks else {})
            if not exiting:
                self._flag_missing(pid, bad, inbox, "echo")
            c0, c1 = self._tallies(inbox, bad)
            support = np.where(self._strong_v(c0), 0,
                               np.where(self._strong_v(c1), 1, NONE)).astype(np.int8)
            support[self._silent] = NONE  # zero-weight processes are not leaders
            # support
            inbox = yield (broadcast(n, support) if speaks else {})
            if exiting:
                return self._finish(pid, v, bad)
            self._flag_missing(pid, bad, inbox, "support")

            # grade every leader, then update (Fig. 2, lines 4-9). A leader in
            # BAD is still graded from the others' echoes and supports; only
            # its own messages are ignored. If every honest process has it in
            # BAD it gets grade 0 everywhere (Claim 1).
            c0, c1 = self._tallies(inbox, bad)
            S = self._S
            x = c1[S] > c0[S]                                 # leader's value (ties -> 0)
            cx = np.where(x, c1[S], c0[S])
            g2 = np.asarray(self._strong_v(cx), dtype=bool)
            g1 = np.asarray(self._weak_v(cx), dtype=bool) & ~g2
            s = [int(self._wS[(g2 | g1) & (x == b)].sum()) for b in (False, True)]  # grade >= 1
            s2 = [int(self._wS[g2 & (x == b)].sum()) for b in (False, True)]       # grade 2
            bad.update(int(q) for q in S[~g2])
            v = 1 if s[1] > s[0] else 0
            strong_exit = self.strong(s2[v])

            if self.KING or self.COIN:
                # tie-break step. Keep maj if values with grade >= 1 give it
                # weight >= 1 - rho; otherwise adopt the king's maj / the coin.
                if self.KING:
                    king = self.king_of(it)
                    inbox = yield (broadcast(n, v) if pid == king else {})
                    kv = inbox.get(king)
                    if king in bad or kv not in (0, 1) or isinstance(kv, bool):
                        kv = None
                else:
                    # a real common coin costs one all-to-all round of shares
                    yield (broadcast(n, ("share", it)) if speaks else {})
                    kv = self.coin(it)
                if not self.strong(s[v]) and kv is not None:
                    v = kv

            if strong_exit:
                if it >= self.cap and not self.COIN:
                    return self._finish(pid, v, bad)
                exiting = True  # one more iteration so everyone else exits too
            elif (it >= self.cap and not self.COIN) or it >= guard:
                return self._finish(pid, v, bad)


class GradecastKing(Gradecast):
    """Gradecast consensus plus a rotating king step: 4 steps per iteration.

    After grading, the iteration's king (coordinator order, heaviest first by
    default) broadcasts its maj. A process keeps its own maj if values with
    grade >= 1 give it weight >= 1 - rho, and adopts the king's value otherwise.

    Correct for rho < 1/3:
      * An honest king forces agreement. Suppose an honest q keeps x, so x has
        weight >= 1 - rho at q. The honest leaders on x then weigh at least
        1 - rho - f_x and those on the other value at most rho + f_x - F
        (gradecast keeps each faulty leader on one value; F is the total faulty
        weight, f_x the part on x). The king's margin for x is at least
        1 - 2rho - f_x >= 1 - 3rho > 0, so the king's maj is x too.
      * Exit stays safe. If p exits with grade-2 weight >= 1 - rho for x, every
        honest process grades those leaders >= 1, keeps x and ignores the king.
      * Persistence and expulsion work exactly as in Gradecast.
    Iterations: at most (position of the first honest king) + 2, capped at
    alpha_rho, which contains an honest king. The pure Gradecast cap counts the
    *lightest* processes; this one counts the *heaviest*.
    """
    name = "gc_king"
    KING = True

    def __init__(self, n, f, weights=None, rho=None, order=None):
        super().__init__(n, f, weights, rho, order)
        self.cap = self.anchor


class GradecastCoin(Gradecast):
    """Gradecast consensus plus a common coin (Rabin / Feldman-Micali style).

    Same as GradecastKing with the king's value replaced by a shared random bit.
    By the GradecastKing argument, if any honest process keeps x then every
    honest process that does not keep would need the tie-breaker to equal x.
    The coin equals x with probability 1/2, independent of the adversary, so
    each iteration ends in agreement with probability >= 1/2 and the expected
    number of iterations is O(1) whatever the weights or coordinator order.
    There is no deterministic cap; it runs until the exit rule fires (guarded).
    The coin is modelled as an oracle. A real one needs threshold signatures or
    secret sharing, and a *weighted* coin is an open design question.
    """
    name = "gc_coin"
    COIN = True
