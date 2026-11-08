"""Byzantine adversary strategies.

One adversary object controls *all* faulty processes (colluding adversary).
Every step it receives the honest outboxes for that step before deciding
(rushing), so it always knows each honest process's current preference.

Strategies range from benign to targeted, the usual ladder used in BA
evaluations:
  silent      faulty processes never send (crash at time 0)
  crash       run the honest protocol, then crash mid-broadcast at a random step
  random      every message is a random protocol value, garbage, or omitted (fuzzing)
  flip        always send the opposite of the current honest majority
  equivocate  send 0 to one fixed half of the honest processes and 1 to the other
  split       adaptive split-vote: re-partition the honest processes every step by
              their current preference and push each side toward its own value;
              as coordinator, tell each side something different
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Sequence

from .network import Outbox

GARBAGE = "garbage"


class Adversary:
    name = "base"

    def setup(self, protocol, faulty: List[int], honest: List[int],
              inputs: Sequence[Any], rng: random.Random) -> None:
        self.protocol = protocol
        self.n = protocol.n
        self.faulty = faulty
        self.honest = honest
        self.inputs = inputs
        self.rng = rng

    def act(self, step: int, honest_out: Dict[int, Outbox]) -> Dict[int, Outbox]:
        raise NotImplementedError

    def observe(self, step: int, inboxes) -> None:
        """Called after delivery; stateful adversaries may use it."""

    # -- helpers -----------------------------------------------------------
    def honest_values(self, honest_out: Dict[int, Outbox]) -> Dict[int, Any]:
        """Value each honest process sent this step (to anyone), if any."""
        vals = {}
        for p, ob in honest_out.items():
            if ob:
                m = next(iter(ob.values()))
                # vector messages (Gradecast echo/support) are not a single vote
                vals[p] = m if isinstance(m, (int, str)) else None
        return vals

    def is_coordinator_step(self, step: int) -> bool:
        return self.protocol.step_info(step).coordinator is not None

    def coordinator(self, step: int):
        return self.protocol.step_info(step).coordinator

    def speakers(self, step: int) -> List[int]:
        """Faulty processes that are expected to send in this step."""
        if self.is_coordinator_step(step):
            c = self.coordinator(step)
            return [c] if c in self.faulty else []
        return self.faulty


class Silent(Adversary):
    name = "silent"

    def act(self, step, honest_out):
        return {}


class Crash(Adversary):
    """Faulty processes behave honestly, then each crashes at a random step,
    delivering its last broadcast to only a random subset of processes."""
    name = "crash"

    def setup(self, protocol, faulty, honest, inputs, rng):
        super().setup(protocol, faulty, honest, inputs, rng)
        self.gens = {p: protocol.process(p, inputs[p]) for p in faulty}
        self.pending = {p: next(g) for p, g in self.gens.items()}
        self.crash_at = {p: rng.randrange(protocol.num_steps + 1) for p in faulty}

    def act(self, step, honest_out):
        out = {}
        for p in self.faulty:
            ob = self.pending.get(p) or {}
            if step < self.crash_at[p]:
                out[p] = ob
            elif step == self.crash_at[p]:
                out[p] = {d: m for d, m in ob.items() if self.rng.random() < 0.5}
        return out

    def observe(self, step, inboxes):
        for p in list(self.gens):
            try:
                self.pending[p] = self.gens[p].send(inboxes[p])
            except StopIteration:
                del self.gens[p]
                self.pending[p] = {}


class RandomFuzz(Adversary):
    name = "random"

    def act(self, step, honest_out):
        choices = list(self.protocol.VALUES) + [None, GARBAGE]
        out = {}
        for p in self.speakers(step):
            ob = {}
            for d in range(self.n):
                m = self.rng.choice(choices)
                if m is not None:
                    ob[d] = m
            out[p] = ob
        return out


class Flip(Adversary):
    name = "flip"

    def act(self, step, honest_out):
        cnt = Counter(v for v in self.honest_values(honest_out).values() if v in (0, 1))
        maj = 1 if cnt[1] > cnt[0] else 0
        return {p: {d: 1 - maj for d in range(self.n)} for p in self.speakers(step)}


class Equivocate(Adversary):
    name = "equivocate"

    def setup(self, protocol, faulty, honest, inputs, rng):
        super().setup(protocol, faulty, honest, inputs, rng)
        half = len(honest) // 2
        self.side = {p: (0 if i < half else 1) for i, p in enumerate(honest)}

    def act(self, step, honest_out):
        return {p: {d: self.side.get(d, 0) for d in range(self.n)} for p in self.speakers(step)}


class SplitVote(Adversary):
    """Adaptive, rushing split-vote attack.

    Every step, sort the honest processes by the value they just sent and cut
    them into two equal camps; camp 0 contains the 0-leaning processes, camp 1
    the 1-leaning ones. Faulty voters push each camp toward its own value,
    which keeps honest support for either value as close to the decision
    thresholds as possible. A faulty coordinator tells camp 0 "0" and camp 1
    "1", so processes that defer to it end the round disagreeing.
    """
    name = "split"

    def setup(self, protocol, faulty, honest, inputs, rng):
        super().setup(protocol, faulty, honest, inputs, rng)
        self.camp = {p: i % 2 for i, p in enumerate(honest)}

    def act(self, step, honest_out):
        vals = self.honest_values(honest_out)
        if vals:
            order = sorted(self.honest, key=lambda p: (_lean(vals.get(p)), p))
            half = len(order) // 2
            self.camp = {p: (0 if i < half else 1) for i, p in enumerate(order)}
        return {p: {d: self.camp.get(d, 0) for d in range(self.n)} for p in self.speakers(step)}


def _lean(v) -> float:
    return {0: 0.0, 1: 1.0}.get(v, 0.5)


class Stagger(SplitVote):
    """Delay attack on Gradecast consensus; falls back to `split` elsewhere.

    Gradecast lets each faulty process cause disagreement at most once, so the
    best an adversary can do is spend one faulty process z per iteration. To do
    that, z's gradecast must give one honest camp X grade 1 and the other camp
    Y grade 0. Then X counts z's value toward maj and Y does not. All other live
    faulty processes gradecast consistently, with values chosen so the weighted
    tally sits exactly where z's weight tips it. That makes X and Y end the
    iteration with different values.

    Mechanics for z's gradecast (rushing, so the honest values are known):
      send     z sends its value only to a set S1 of honest weight just below 1 - rho
      echo     all live faulty echo z's value to camp A only (weight <= rho), so A
               crosses 1 - rho and supports it, and no other honest process does
      support  all live faulty support z's value to camp X only, so X sees weight
               > rho (grade 1), and Y sees only A's support <= rho (grade 0)
    X is chosen to split the honest weight in half, so the next iteration
    starts balanced again. When no split is possible, faulty processes act
    consistently and the honest processes finish.

    S1 and A are chosen by a subset-sum heuristic (heaviest first, skipping
    overshoots). In every run we checked it reaches the bound of F + 3
    iterations, or the cap, where F is the number of positive-weight faulty
    processes. It is a strong heuristic attack, not a proven optimum.
    It does not attack the coin in GradecastCoin.
    """
    name = "stagger"

    def setup(self, protocol, faulty, honest, inputs, rng):
        super().setup(protocol, faulty, honest, inputs, rng)
        self.gc = getattr(protocol, "name", "") in ("gradecast", "gc_king", "gc_coin")
        if self.gc:
            p = protocol
            self.w = p.w
            gone = getattr(p, "expelled_all", set())  # in every honest BAD already
            self.live = [z for z in faulty if p.w[z] > 0 and z not in gone]
            self.hpos = [h for h in honest if p.w[h] > 0]
            self.plan = None

    def act(self, step, honest_out):
        if not self.gc:
            return super().act(step, honest_out)
        info = self.protocol.step_info(step)
        if info.phase == "king":
            return self._king(info.coordinator)
        if info.phase not in ("send", "echo", "support"):
            return {}  # e.g. the common-coin step: nothing useful to say
        phase = ("send", "echo", "support").index(info.phase)
        if phase == 0:
            self.plan = self._plan(honest_out)
        return self._send(phase)

    def _king(self, king):
        """A faulty king tells the two camps opposite values (GradecastKing)."""
        if king not in self.faulty:
            return {}
        X = self.plan.get("X") if self.plan else None
        if X is None:
            X = set(self.honest[: len(self.honest) // 2])
        zval = self.plan.get("zval", 0) if self.plan else 0
        return {king: {d: (zval if d in X else 1 - zval) for d in range(self.n)}}

    # -- planning ------------------------------------------------------------
    def _plan(self, honest_out):
        p, w = self.protocol, self.w
        vals = {h: ob.get(h) for h, ob in honest_out.items() if ob}
        hv = [0, 0]
        for h in self.hpos:
            if vals.get(h) in (0, 1):
                hv[vals[h]] += w[h]
        plan = dict(z=None, others={o: 0 for o in self.live})
        if not self.live:
            return plan
        z = max(self.live, key=lambda o: (w[o], -o))
        others = [o for o in self.live if o != z]
        fw = sum(w[o] for o in self.live)

        def fill(ok):
            """Largest honest weight keeping ok(total): heaviest first, skipping
            any process that would overshoot (a subset-sum heuristic)."""
            S, tot = [], 0
            for h in sorted(self.hpos, key=lambda h: (-w[h], h)):
                if ok(tot + w[h]):
                    S.append(h)
                    tot += w[h]
            return S, tot

        # camp A: honest weight a with a <= rho*W and a + fw > rho*W
        A, a = fill(lambda t: not p.weak(t))
        if not p.weak(a + fw):
            return plan
        # S1: honest weight in [ (1-rho)W - fw, (1-rho)W )
        S1, s = fill(lambda t: not p.strong(t))
        if not p.strong(s + fw):
            return plan

        # choose z's value and the others' consistent values so the tally
        # D = weight(1) - weight(0) lands where z's weight decides maj
        for zval in (1, 0):
            lo, hi = ((-w[z], 0) if zval == 1 else (0, w[z]))  # need lo < D <= hi
            target = (lo + hi) / 2
            D = hv[1] - hv[0]
            chosen = {}
            for o in sorted(others, key=lambda o: -w[o]):
                up, down = D + w[o], D - w[o]
                if abs(up - target) <= abs(down - target):
                    chosen[o], D = 1, up
                else:
                    chosen[o], D = 0, down
            if lo < D <= hi:
                break
        else:
            return plan

        # camp X: about half the honest weight (zero-weight honest split evenly)
        X, xs, ys = set(), [0, 0], [0, 0]  # [weight, count] per camp
        for h in sorted(self.honest, key=lambda h: (-w[h], h)):
            if xs <= ys:
                X.add(h)
                xs = [xs[0] + w[h], xs[1] + 1]
            else:
                ys = [ys[0] + w[h], ys[1] + 1]
        plan.update(z=z, zval=zval, others=chosen, S1=set(S1), A=set(A), X=X)
        return plan

    # -- messages --------------------------------------------------------------
    def _send(self, phase):
        plan, n = self.plan, self.n
        out = {}
        z = plan["z"]
        if phase == 0:
            for o, x in plan["others"].items():
                out[o] = {d: x for d in range(n)}
            if z is not None:
                out[z] = {d: plan["zval"] for d in plan["S1"]}
        elif z is not None:
            vec = tuple(plan["zval"] if q == z else None for q in range(n))
            targets = plan["A"] if phase == 1 else plan["X"]
            for o in self.live:
                out[o] = {d: vec for d in targets}
            if phase == 2:
                self.live.remove(z)  # every honest process now has z in BAD
        return out


ADVERSARIES = {cls.name: cls for cls in
               (Silent, Crash, RandomFuzz, Flip, Equivocate, SplitVote, Stagger)}


class SigAttack(SplitVote):
    """Attack on Dolev-Strong using valid faulty signatures; `split` elsewhere.

    - the first faulty signer equivocates in round 1: 0 to half the honest
      processes, 1 to the rest
    - the second reveals a value as late as it can: in round r = number of
      faulty signers, with a chain signed by r faulty processes, to a single
      honest process. That process must relay it in round r+1 <= R or the
      honest processes end up with different accepted sets.
    """
    name = "sig"

    def setup(self, protocol, faulty, honest, inputs, rng):
        super().setup(protocol, faulty, honest, inputs, rng)
        self.ds = getattr(protocol, "name", "") == "dolev_strong"
        if self.ds:
            self.signers = [z for z in faulty if z in protocol._pos]
            self.sign = protocol.signer_for(self.signers)

    def act(self, step, honest_out):
        if not self.ds:
            return super().act(step, honest_out)
        r, out, n = step + 1, {}, self.n
        hs, w = self.honest, self.protocol.w
        if self.signers and r == 1:
            z = self.signers[0]
            out[z] = {d: ((z, i % 2, self.sign(z, z, i % 2, ())),) for i, d in enumerate(hs)}
            # the other faulty senders vote consistently, steering the tally
            # D = weight(1) - weight(0) into (-w[z2], 0]: everyone decides 0,
            # except the single process that receives z2's late 1.
            if len(self.signers) >= 2:
                z2 = self.signers[1]
                D = sum((1 if self.inputs[h] == 1 else -1) * w[h] for h in hs if w[h] > 0)
                target = -w[z2] / 2
                for y in sorted(self.signers[2:], key=lambda y: -w[y]):
                    x = 1 if abs(D + w[y] - target) <= abs(D - w[y] - target) else 0
                    D += w[y] if x else -w[y]
                    out[y] = {d: ((y, x, self.sign(y, y, x, ())),) for d in range(n)}
        if len(self.signers) >= 2 and r == len(self.signers):
            z = self.signers[1]
            chain = ()
            for signer in [z] + [y for y in self.signers if y != z][: r - 1]:
                chain = self.sign(signer, z, 1, chain)
            last = chain[-1][0]
            out.setdefault(last, {})[hs[0]] = ((z, 1, chain),)
        return out


class Hybrid(Adversary):
    """Hybrid faults: processes in `crash_set` only crash (Crash); the other
    faulty processes are Byzantine (SplitVote). run_trial sets crash_set."""
    name = "hybrid"
    crash_set: frozenset = frozenset()

    def setup(self, protocol, faulty, honest, inputs, rng):
        super().setup(protocol, faulty, honest, inputs, rng)
        cs = [p for p in faulty if p in self.crash_set]
        bz = [p for p in faulty if p not in self.crash_set]
        self.parts = []
        if cs:
            a = Crash()
            a.setup(protocol, cs, honest, inputs, rng)
            self.parts.append(a)
        if bz:
            a = SplitVote()
            a.setup(protocol, bz, honest, inputs, rng)
            self.parts.append(a)

    def act(self, step, honest_out):
        out = {}
        for a in self.parts:
            out.update(a.act(step, honest_out))
        return out

    def observe(self, step, inboxes):
        for a in self.parts:
            a.observe(step, inboxes)


ADVERSARIES.update({cls.name: cls for cls in (SigAttack, Hybrid)})
