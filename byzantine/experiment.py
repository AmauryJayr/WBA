"""Single-trial runner and parameter sweeps."""

from __future__ import annotations

import itertools
import random
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Iterable, List

from .adversaries import ADVERSARIES
from .network import Simulation
from .protocols import PROTOCOLS

INPUT_MODES = ("all0", "all1", "split", "random")
PLACEMENTS = ("first", "last", "random")


@dataclass(frozen=True)
class TrialConfig:
    protocol: str
    n: int
    f: int                      # faults the protocol is configured to tolerate
    actual_faults: int          # faulty processes actually present
    adversary: str
    inputs: str = "random"
    placement: str = "first"
    seed: int = 0


def make_inputs(mode: str, n: int, rng: random.Random):
    if mode == "all0":
        return [0] * n
    if mode == "all1":
        return [1] * n
    if mode == "split":
        return [i % 2 for i in range(n)]
    if mode == "random":
        return [rng.randint(0, 1) for _ in range(n)]
    raise ValueError(mode)


def choose_faulty(placement: str, n: int, k: int, rng: random.Random) -> List[int]:
    """'first': faulty processes are the coordinators of the first k rounds.
    'last': round 0 has an honest coordinator, the following k are faulty
    (stresses persistence of agreement). 'random': uniform subset."""
    if placement == "first":
        return list(range(k))
    if placement == "last":
        return list(range(1, k + 1))
    if placement == "random":
        return sorted(rng.sample(range(n), k))
    raise ValueError(placement)


def run_trial(cfg: TrialConfig) -> dict:
    rng = random.Random(cfg.seed)
    proto = PROTOCOLS[cfg.protocol](cfg.n, cfg.f)
    adv = ADVERSARIES[cfg.adversary]()
    inputs = make_inputs(cfg.inputs, cfg.n, rng)
    faulty = choose_faulty(cfg.placement, cfg.n, cfg.actual_faults, rng)
    t0 = time.perf_counter()
    res = Simulation(proto, adv, inputs, faulty, seed=cfg.seed).run()
    elapsed = time.perf_counter() - t0
    row = asdict(cfg)
    row.update(
        within_bound=cfg.actual_faults <= cfg.f <= PROTOCOLS[cfg.protocol].max_faults(cfg.n),
        terminated=res.terminated,
        agreement=res.agreement,
        validity=res.validity,
        correct=res.correct,
        steps=res.steps,
        honest_msgs=res.honest_msgs,
        faulty_msgs=res.faulty_msgs,
        honest_bits=res.honest_bits,
        decided=sorted(set(map(str, res.decisions.values()))),
        wall_ms=elapsed * 1000,
    )
    return row


def sweep(protocols: Iterable[str], ns: Iterable[int], adversaries: Iterable[str],
          trials: int, inputs: Iterable[str] = INPUT_MODES,
          placements: Iterable[str] = PLACEMENTS, beyond_bound: int = 0,
          base_seed: int = 0):
    """Yield one result row per trial.

    For each n, f ranges over 0..max_faults(n) + beyond_bound. Rows with f past
    the proven bound probe whether the adversary can actually break the protocol.
    """
    for proto, n in itertools.product(protocols, ns):
        fmax = min(PROTOCOLS[proto].max_faults(n) + beyond_bound, n - 1)
        for f, adv, inp, pl in itertools.product(range(fmax + 1), adversaries, inputs, placements):
            for t in range(trials):
                seed = zlib.crc32(repr((base_seed, proto, n, f, adv, inp, pl, t)).encode())
                yield run_trial(TrialConfig(proto, n, f, f, adv, inp, pl, seed))
