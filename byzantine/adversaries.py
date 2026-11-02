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
                vals[p] = next(iter(ob.values()))
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


ADVERSARIES = {cls.name: cls for cls in
               (Silent, Crash, RandomFuzz, Flip, Equivocate, SplitVote)}
