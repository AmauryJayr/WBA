"""Lock-step synchronous network simulator for Byzantine agreement protocols.

Model (the standard one used in the BA literature, e.g. Lynch ch. 6, Garg ch. 15):
  * n processes, complete graph, reliable FIFO channels, known message bound
    -> execution proceeds in synchronous *steps* (one message exchange each).
  * up to f processes are Byzantine and are controlled by a single adversary.
  * the adversary is *rushing* and *omniscient*: in every step it sees every
    message the honest processes send in that step (and their internal
    preference, via those messages) before choosing what the faulty
    processes send. This is the strongest adversary that is still
    message-passing, so a protocol that survives it survives any weaker one.

Protocols are written as Python generators (see protocols/base.py). Each
`yield outbox` is one synchronous step; the value sent back in is the inbox
for that step: {sender_pid: message}. A missing key means "no message".
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Outbox = Dict[int, Any]  # dest pid -> message
Inbox = Dict[int, Any]   # sender pid -> message


def broadcast(n: int, msg: Any) -> Outbox:
    """Send `msg` to every process, including the sender itself."""
    return {j: msg for j in range(n)}


def msg_bits(msg: Any) -> int:
    """Rough size of a message in bits, for message-complexity reporting."""
    if msg is None:
        return 0
    if isinstance(msg, (bool, int)) or isinstance(msg, str) and len(msg) == 1:
        return 2  # {0, 1, undecided} fits in 2 bits
    if isinstance(msg, (tuple, list, set, frozenset)):
        return sum(msg_bits(m) for m in msg) + 8
    return 8 * sys.getsizeof(msg)


@dataclass
class RunResult:
    n: int
    f: int
    faulty: List[int]
    inputs: Dict[int, Any]          # honest pid -> input value
    decisions: Dict[int, Any]       # honest pid -> decided value (missing = no decision)
    steps: int                      # synchronous message steps until all honest decided
    honest_msgs: int                # messages sent by honest processes (self-sends excluded)
    faulty_msgs: int
    honest_bits: int
    decide_step: Dict[int, int] = field(default_factory=dict)

    @property
    def terminated(self) -> bool:
        return len(self.decisions) == len(self.inputs)

    @property
    def agreement(self) -> bool:
        return len(set(self.decisions.values())) <= 1

    @property
    def validity(self) -> bool:
        """If all honest processes propose v, every honest decision is v."""
        vals = set(self.inputs.values())
        if len(vals) != 1:
            return True
        (v,) = vals
        return all(d == v for d in self.decisions.values())

    @property
    def correct(self) -> bool:
        return self.terminated and self.agreement and self.validity


class Simulation:
    """Runs one execution of `protocol` against `adversary`."""

    def __init__(self, protocol, adversary, inputs: Sequence[Any], faulty: Sequence[int],
                 seed: Optional[int] = None, max_steps: int = 100_000):
        self.protocol = protocol
        self.adversary = adversary
        self.n = protocol.n
        self.f = protocol.f
        self.faulty = sorted(set(faulty))
        self.honest = [p for p in range(self.n) if p not in self.faulty]
        self.inputs = list(inputs)
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        assert len(self.inputs) == self.n

    def run(self) -> RunResult:
        n = self.n
        gens = {p: self.protocol.process(p, self.inputs[p]) for p in self.honest}
        outboxes: Dict[int, Outbox] = {}
        decisions: Dict[int, Any] = {}
        decide_step: Dict[int, int] = {}

        for p, g in gens.items():
            outboxes[p] = next(g)

        self.adversary.setup(self.protocol, self.faulty, self.honest, self.inputs, self.rng)

        step = 0
        honest_msgs = faulty_msgs = honest_bits = 0
        while gens and step < self.max_steps:
            # Rushing adversary sees this step's honest traffic first.
            byz_out = self.adversary.act(step, outboxes)

            inboxes: Dict[int, Inbox] = {p: {} for p in range(n)}
            for src, ob in outboxes.items():
                for dst, m in ob.items():
                    inboxes[dst][src] = m
                    if dst != src:
                        honest_msgs += 1
                        honest_bits += msg_bits(m)
            for src, ob in byz_out.items():
                assert src in self.faulty, "adversary may only speak for faulty processes"
                for dst, m in ob.items():
                    inboxes[dst][src] = m
                    if dst != src:
                        faulty_msgs += 1

            self.adversary.observe(step, inboxes)

            step += 1
            new_out: Dict[int, Outbox] = {}
            for p in list(gens):
                try:
                    new_out[p] = gens[p].send(inboxes[p])
                except StopIteration as stop:
                    decisions[p] = stop.value
                    decide_step[p] = step
                    del gens[p]
            outboxes = new_out

        return RunResult(
            n=n, f=self.f, faulty=self.faulty,
            inputs={p: self.inputs[p] for p in self.honest},
            decisions=decisions, steps=step,
            honest_msgs=honest_msgs, faulty_msgs=faulty_msgs, honest_bits=honest_bits,
            decide_step=decide_step,
        )
