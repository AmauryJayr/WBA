"""Everything side by side on the Garg & Bridgman Fig. 4 setup.

Each protocol runs under its own fault model and its own best weights:

  protocol       model / bound                       weights
  queen          Byzantine weight < 1/4              optimal ratio for 1/4
  king           Byzantine weight < 1/3              optimal ratio for 1/3
  king_es        Byzantine weight < 1/4              optimal ratio for 1/4
  gradecast      Byzantine weight < 1/3              optimal ratio for 1/3
  gc_king        Byzantine weight < 1/3              optimal ratio for 1/3
  gc_coin        Byzantine weight < 1/3 + coin       optimal ratio for 1/3
  hybrid_king    A crash-only, B Byzantine:          best ratio and best split
                 3 rho_b + 2 rho_c < 1               of the crash/Byzantine budget
  dolev_strong   signatures: Byzantine weight < 1/2  optimal ratio for 1/2

Each protocol also faces the strongest attack we have against it: stagger for
the Gradecast family, sig for Dolev-Strong, hybrid (A crash, B split) for
Hybrid King, and split for the rest. The benign case is silent faults.
"""

from __future__ import annotations

import random
import zlib
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.stats import binom

from .experiment import TrialConfig, run_trial
from .improvements import EPS, anchor, int_weights, run_sequence
from .protocols import PROTOCOLS
from .weight_assignment import optimal_ratio

ORDER = ["queen", "king", "king_es", "gradecast", "gc_king", "gc_coin", "hybrid_king", "dolev_strong"]
LABEL = {"queen": "Queen", "king": "King", "king_es": "Early-stopping King",
         "gradecast": "Gradecast", "gc_king": "Gradecast + king", "gc_coin": "Gradecast + coin* (coin not attacked)",
         "hybrid_king": "Hybrid King (A crash-only)*", "dolev_strong": "Dolev–Strong (signatures)*"}
BOUND = {"queen": Fraction(1, 4), "king_es": Fraction(1, 4), "king": Fraction(1, 3),
         "gradecast": Fraction(1, 3), "gc_king": Fraction(1, 3), "gc_coin": Fraction(1, 3),
         "dolev_strong": Fraction(1, 2)}
ATTACK = {"gradecast": "stagger", "gc_king": "stagger", "gc_coin": "stagger",
          "dolev_strong": "sig", "hybrid_king": "hybrid"}


@lru_cache(maxsize=None)
def best_config(proto: str, na: int, nb: int, fa: float, fb: float):
    """(weights, rho, rho_crash, p_unsafe) maximising P(guarantee holds)."""
    if proto != "hybrid_king":
        t = BOUND[proto]
        # among equally safe ratios, take the one with the fewest King rounds
        r, p = optimal_ratio(na, nb, fa, fb, t, prefer=lambda rr: (anchor(na, nb, rr, t - EPS), rr[1] == 0))
        ratio = (1, 0) if r is None else (r.numerator, r.denominator)
        return int_weights(na, nb, ratio), t - EPS, None, p
    return _best_hybrid(na, nb, fa, fb)


def _best_hybrid(na, nb, fa, fb):
    """Grid over r = wA/wB (and only-A); for each r, the best split: allow
    crash weight up to a* A-processes, Byzantine weight up to the most
    B-processes b* with 3 b* wB < W - 2 a* r wA."""
    pa = binom.pmf(np.arange(na + 1), na, fa)
    pb = binom.pmf(np.arange(nb + 1), nb, fb)
    ca, cb = np.cumsum(pa), np.cumsum(pb)
    best = (-1.0, None)
    grid = sorted({Fraction(x).limit_denominator(1000) for x in np.logspace(-2, 3, 400)})
    for r in grid + [None]:
        wa, wb = ((1, 0) if r is None else (r.numerator, r.denominator))
        W = na * wa + nb * wb
        for a_star in range(na + 1):
            budget = W - 2 * a_star * wa          # need 3*b*wb < budget
            if budget <= 0:
                continue
            b_star = nb if wb == 0 else min(nb, (budget - 1) // (3 * wb))
            if 3 * b_star * wb >= budget:
                continue
            p_safe = ca[a_star] * (cb[b_star] if nb else 1.0)
            if p_safe > best[0]:
                best = (p_safe, (wa, wb, a_star, b_star, W))
    wa, wb, a_star, b_star, W = best[1]
    w = tuple([wa] * na + [wb] * nb)
    rho_c = Fraction(a_star * wa, W)
    rho_b = Fraction(b_star * wb, W)
    return w, rho_b, rho_c, 1 - best[0]


def draw_faulty(na, nb, fa, fb, key):
    rng = random.Random(zlib.crc32(repr(key).encode()))
    while True:
        faulty = [i for i in range(na + nb) if rng.random() < (fa if i < na else fb)]
        if len(faulty) < na + nb:
            return faulty


def is_safe(proto, w, rho, rho_c, faulty, na):
    W = sum(w)
    if proto == "hybrid_king":
        crash = sum(w[i] for i in faulty if i < na)
        byz = sum(w[i] for i in faulty if i >= na)
        return Fraction(crash, W) <= rho_c and Fraction(byz, W) <= rho
    return Fraction(sum(w[i] for i in faulty), W) <= rho


def simulate(args):
    proto, na, nb, fa, fb, mode, draw = args
    w, rho, rho_c, _ = best_config(proto, na, nb, fa, fb)
    faulty = draw_faulty(na, nb, fa, fb, (nb, draw))  # same draws for every protocol
    row = dict(protocol=proto, nb=nb, mode=mode, draw=draw, faulty=len(faulty))
    row["safe"] = is_safe(proto, w, rho, rho_c, faulty, na)
    if not row["safe"]:
        return row
    adv = ATTACK.get(proto, "split") if mode == "attack" else "silent"
    if proto == "hybrid_king" and mode == "silent":
        adv = "silent"
    r = run_trial(TrialConfig(proto, na + nb, 0, len(faulty), adv, "split", seed=draw,
                              weights=w, rho=str(rho), faulty=tuple(faulty),
                              crash=tuple(i for i in faulty if i < na),
                              rho_crash=None if rho_c is None else str(rho_c)))
    row.update(correct=r["correct"], steps=r["steps"], bits=r["honest_bits"], msgs=r["honest_msgs"])
    return row


def exact_steps(proto, na, nb, fa, fb):
    """Deterministic step counts (no early stopping): King/Queen/Hybrid, Dolev-Strong."""
    w, rho, rho_c, _ = best_config(proto, na, nb, fa, fb)
    kw = {} if rho_c is None else {"rho_crash": rho_c}
    P = PROTOCOLS[proto](na + nb, 0, weights=w, rho=rho, **kw)
    return P.num_steps


def sequence(args):
    proto, carry, na, nb, fa, fb, scheme_ratio, ell, draw = args
    w = int_weights(na, nb, scheme_ratio)
    rho = (Fraction(1, 4) if proto == "king_es" else Fraction(1, 3)) - EPS
    W = sum(w)
    k = 0
    while True:  # first safe draw for this protocol
        faulty = draw_faulty(na, nb, fa, fb, ("seq", nb, draw, k))
        if Fraction(sum(w[i] for i in faulty), W) <= rho:
            break
        k += 1
    rows = run_sequence(proto, w, str(rho), faulty, "stagger", ell, draw, carry)
    for r in rows:
        r.update(protocol=proto, carry=carry, draw=draw, faulty=len(faulty))
    return rows


BOUND_TEXT = {"queen": "Byz < 1/4", "king": "Byz < 1/3", "king_es": "Byz < 1/4",
              "gradecast": "Byz < 1/3", "gc_king": "Byz < 1/3", "gc_coin": "Byz < 1/3",
              "hybrid_king": "3 Byz + 2 crash < 1", "dolev_strong": "Byz < 1/2"}


def worst_steps(proto, na, nb, fa, fb):
    """Proven worst-case step count for the protocol's chosen weights.
    King/Queen/Hybrid: phases x anchor. Early-stopping King: <= 3 x anchor.
    Gradecast: 3 x cap. Gradecast + king: 4 x anchor. Dolev-Strong: R + 1.
    Gradecast + coin: no deterministic bound (expected O(1)) -> NaN."""
    if proto == "gc_coin":
        return float("nan")
    w, rho, rho_c, _ = best_config(proto, na, nb, fa, fb)
    kw = {} if rho_c is None else {"rho_crash": rho_c}
    P = PROTOCOLS[proto](na + nb, 0, weights=w, rho=rho, **kw)
    if proto in ("gradecast", "gc_king"):
        return P.steps_per_iteration * P.cap
    if proto == "king_es":
        return 3 * P.anchor
    return P.num_steps
