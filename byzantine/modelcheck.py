"""Exhaustive small-scope model checker.

For small n, enumerate *every* behaviour of the Byzantine processes and every
honest input vector, and check agreement and validity on all of them. Random
and scripted adversaries can only show that the attacks we thought of fail;
this shows that no attack at all works (for that n and f), and when f is
past the proven bound it finds a concrete counterexample.

Reductions that keep the search exact:
  * Between rounds an honest process's whole state is its preference V, so the
    search is a breadth-first search over honest preference vectors, round by
    round, and duplicate vectors are merged.
  * Honest processes only *count* the values they receive, so in an all-to-all
    phase it is enough to choose, per honest receiver, the multiset of values
    it gets from the f faulty processes. Silence and malformed messages count
    the same as one of the protocol values (0 for Queen, undecided for King),
    so the protocol alphabet covers them.
  * In a coordinator phase only a faulty coordinator can deviate; it picks one
    value per receiver.
The honest code that runs is the real `Protocol.process` generator, one round
at a time, so the checker checks the implementation, not a model of it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .network import broadcast
from .protocols import PROTOCOLS


@dataclass
class CheckResult:
    protocol: str
    n: int
    f: int
    faulty: Tuple[int, ...]
    states_explored: int
    round_executions: int
    ok: bool
    counterexample: Optional[dict] = None


def _phase_choices(proto, honest, faulty, step):
    """All distinguishable faulty behaviours for one step, as lists of
    {faulty_pid: outbox} dicts."""
    info = proto.step_info(step)
    if info.coordinator is not None:
        if info.coordinator not in faulty:
            return [{}]
        per_rcv = [proto.VALUES] * len(honest)
        return [{info.coordinator: dict(zip(honest, vals))}
                for vals in itertools.product(*per_rcv)]
    multisets = list(itertools.combinations_with_replacement(proto.VALUES, len(faulty)))
    out = []
    for combo in itertools.product(multisets, repeat=len(honest)):
        obs = {z: {} for z in faulty}
        for r, ms in zip(honest, combo):
            for z, v in zip(faulty, ms):
                obs[z][r] = v
        out.append(obs)
    return out


def _run_round(proto, honest, vec, k, choices):
    """Execute round k from honest preferences `vec` with the scripted faulty
    messages `choices` (one entry per step of the round)."""
    gens = {p: proto.process(p, v, rounds=[k]) for p, v in zip(honest, vec)}
    outs = {p: next(g) for p, g in gens.items()}
    result = {}
    for byz in choices:
        inboxes = {p: {} for p in range(proto.n)}
        for src, ob in list(outs.items()) + list(byz.items()):
            for dst, m in ob.items():
                inboxes[dst][src] = m
        outs = {}
        for p, g in gens.items():
            if p in result:
                continue
            try:
                outs[p] = g.send(inboxes[p])
            except StopIteration as s:
                result[p] = s.value
    return tuple(result[p] for p in honest)


def check(protocol: str, n: int, f: int, faulty=None, max_round_execs: int = 5_000_000) -> CheckResult:
    proto = PROTOCOLS[protocol](n, f)
    faulty = tuple(sorted(faulty if faulty is not None else range(f)))
    honest = [p for p in range(n) if p not in faulty]
    steps_per_round = proto.num_steps // (f + 1)

    # frontier: (vec, unanimous_input_or_None) -> trace that reached it
    frontier: Dict[tuple, list] = {}
    for vec in itertools.product((0, 1), repeat=len(honest)):
        orig = vec[0] if len(set(vec)) == 1 else None
        frontier[(vec, orig)] = [("input", vec)]

    explored = len(frontier)
    execs = 0
    for k in range(f + 1):
        per_step = [_phase_choices(proto, honest, faulty, k * steps_per_round + s)
                    for s in range(steps_per_round)]
        combos = 1
        for c in per_step:
            combos *= len(c)
        if combos * len(frontier) + execs > max_round_execs:
            raise RuntimeError(f"search too large: {combos} adversary choices x {len(frontier)} states")
        nxt: Dict[tuple, list] = {}
        for (vec, orig), trace in frontier.items():
            for choice in itertools.product(*per_step):
                execs += 1
                new = _run_round(proto, honest, vec, k, choice)
                key = (new, orig)
                if key not in nxt:
                    nxt[key] = trace + [("round", k, choice, new)]
        frontier = nxt
        explored += len(frontier)

    for (vec, orig), trace in frontier.items():
        bad_agree = len(set(vec)) > 1
        bad_valid = orig is not None and any(v != orig for v in vec)
        if bad_agree or bad_valid:
            return CheckResult(protocol, n, f, faulty, explored, execs, False, {
                "violates": "agreement" if bad_agree else "validity",
                "honest": honest, "final": dict(zip(honest, vec)), "trace": trace})
    return CheckResult(protocol, n, f, faulty, explored, execs, True)


def format_counterexample(res: CheckResult) -> str:
    cx = res.counterexample
    lines = [f"{res.protocol} n={res.n} f={res.f} faulty={list(res.faulty)}: "
             f"violates {cx['violates']}"]
    for entry in cx["trace"]:
        if entry[0] == "input":
            lines.append(f"  inputs        {dict(zip(cx['honest'], entry[1]))}")
        else:
            _, k, choice, new = entry
            lines.append(f"  round {k}:")
            for s, byz in enumerate(choice):
                if byz:
                    lines.append(f"    step {s}: faulty send {byz}")
            lines.append(f"    -> preferences {dict(zip(cx['honest'], new))}")
    return "\n".join(lines)
