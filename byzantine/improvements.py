"""Which heuristics help, measured on the Garg & Bridgman Fig. 4 setup.

|A| = na processes fail with probability fa, |B| = nb with probability fb.

Weights (exact):
  paper schemes, log-odds, the optimal ratio wA/wB, and the optimal *sparse*
  assignment (weight only on A plus the best k <= |B| processes of B). For each
  we report P(unsafe) at threshold 1/3 and the King round count alpha.
Resilience (exact): P(failed weight >= t) for t = 1/4 (Queen, early-stopping
  King), 1/3 (King, Gradecast, Gradecast+king), 1/2 (what signatures would allow).
Protocols (simulated): king, king_es, gradecast, gc_king with the paper's best
  weights (inverse failure probability). Failures are drawn from the model, and
  only draws inside each protocol's own bound count. Adversaries: stagger
  (worst case we have) and silent. Coordinators go heaviest first, which here
  is also most reliable first (A).
"""

from __future__ import annotations

import random
import zlib
from fractions import Fraction

import numpy as np

from .experiment import TrialConfig, run_trial
from .protocols import PROTOCOLS
from .weight_assignment import (EXTRA_SCHEMES, SCHEMES, optimal_ratio, p_unsafe_ratio)

EPS = Fraction(1, 10**6)
PROTO_RHO = {"king": Fraction(1, 3) - EPS, "gradecast": Fraction(1, 3) - EPS,
             "gc_king": Fraction(1, 3) - EPS, "king_es": Fraction(1, 4) - EPS,
             "queen": Fraction(1, 4) - EPS}


def ratio_of(scheme, fa, fb):
    f = SCHEMES.get(scheme) or EXTRA_SCHEMES[scheme]
    wa, wb = f(Fraction(fa), Fraction(fb))
    return (1, 0) if wb == 0 else ((wa / wb).numerator, (wa / wb).denominator)


def int_weights(na, nb, ratio):
    """Integer weights for ratio (p, q) = wA/wB; (1, 0) means only A."""
    p, q = ratio
    return tuple([1] * na + [0] * nb) if q == 0 else tuple([p] * na + [q] * nb)


def anchor(na, nb, ratio, rho=Fraction(1, 3) - EPS):
    """alpha_rho for the given ratio, heaviest first."""
    w = int_weights(na, nb, ratio)
    W = sum(w)
    acc = 0
    for k, x in enumerate(sorted(w, reverse=True), 1):
        acc += x
        if acc > rho * W:
            return k
    return len(w)


def weight_rows(na, nb, fa, fb):
    rows = []
    named = list(SCHEMES) + list(EXTRA_SCHEMES)
    ratios = [ratio_of(s, fa, fb) for s in named]
    for t in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
        probs = p_unsafe_ratio(na, nb, float(fa), float(fb), ratios, t)
        for s, r, pr in zip(named, ratios, probs):
            rows.append(dict(nb=nb, scheme=s, threshold=str(t), p_unsafe=float(pr),
                             ratio=(None if r[1] == 0 else r[0] / r[1]),
                             senders=na + (0 if r[1] == 0 else nb),
                             anchor=anchor(na, nb, r, t - EPS)))
        r, pr = optimal_ratio(na, nb, float(fa), float(fb), t,
                              prefer=lambda rr: (anchor(na, nb, rr, t - EPS), rr[1] == 0))
        rr = (1, 0) if r is None else (r.numerator, r.denominator)
        rows.append(dict(nb=nb, scheme="optimal", threshold=str(t), p_unsafe=pr,
                         ratio=None if r is None else float(r), senders=na + (0 if r is None else nb),
                         anchor=anchor(na, nb, rr, t - EPS)))
    return rows


def sparse_rows(df_opt):
    """Best sparse assignment: weight only A and k <= |B| processes of B.

    Processes in B are interchangeable, so weighting k of them is exactly the
    |B| = k system. The best sparse assignment is therefore the best optimal
    curve over k <= |B|.
    """
    rows = []
    for t, g in df_opt.groupby("threshold"):
        g = g.sort_values("nb")
        best = None
        for _, r in g.iterrows():
            if best is None or r.p_unsafe < best.p_unsafe - 1e-15:
                best = r
            rows.append(dict(nb=r.nb, scheme="optimal_sparse", threshold=t,
                             p_unsafe=best.p_unsafe, ratio=best.ratio,
                             senders=best.senders, anchor=best.anchor, k_b=int(best.nb)))
    return rows


def simulate(args):
    proto, scheme, na, nb, fa, fb, adversary, draw = args
    rho = PROTO_RHO[proto]
    if scheme == "optimal":  # optimal for this protocol's own bound (1/4 or 1/3)
        r, _ = optimal_ratio(na, nb, float(fa), float(fb), rho + EPS,
                             prefer=lambda rr: (anchor(na, nb, rr, rho), rr[1] == 0))
        ratio = (1, 0) if r is None else (r.numerator, r.denominator)
    else:
        ratio = ratio_of(scheme, fa, fb)
    w = int_weights(na, nb, ratio)
    W = sum(w)
    rng = random.Random(zlib.crc32(repr((scheme, nb, draw)).encode()))
    # same failure draws for every protocol and adversary (common random numbers)
    while True:
        faulty = [i for i in range(na + nb) if rng.random() < (float(fa) if i < na else float(fb))]
        if len(faulty) < na + nb:
            break
    fw = Fraction(sum(w[i] for i in faulty), W)
    base = dict(protocol=proto, scheme=scheme, nb=nb, adversary=adversary, draw=draw,
                faulty=len(faulty), failed_weight=float(fw), safe=bool(fw <= rho))
    if not base["safe"]:
        return base  # outside this protocol's guarantee: not run
    row = run_trial(TrialConfig(proto, na + nb, 0, len(faulty), adversary, "split", seed=draw,
                                weights=w, rho=str(rho), faulty=tuple(faulty)))
    base.update(correct=row["correct"], steps=row["steps"], honest_msgs=row["honest_msgs"],
                honest_bits=row["honest_bits"])
    return base


def run_sequence(proto: str, w, rho, faulty, adversary: str, ell: int, seed: int,
                 carry: bool = True):
    """Run `ell` consensus instances back to back with the same faulty set.

    Synchronized starts (Ben-Or et al. Sec. 5.1): instance k+1 starts once
    instance k has ended everywhere. Gradecast-family protocols carry each
    process's BAD set into the next instance when `carry` is set, so a faulty
    process can disrupt at most once over the whole sequence. Other protocols
    keep no state. Inputs are fresh random bits each instance.
    Returns one row per instance.
    """
    from .adversaries import ADVERSARIES
    from .experiment import make_inputs
    from .network import Simulation

    n = len(w)
    rng = random.Random(seed)
    honest = [p for p in range(n) if p not in set(faulty)]
    bad = {p: set() for p in honest}
    rows, total = [], 0
    for k in range(ell):
        P = PROTOCOLS[proto](n, 0, weights=w, rho=rho)
        if hasattr(P, "initial_bad") and carry:
            P.initial_bad = {p: set(bad[p]) for p in honest}
            P.expelled_all = set.intersection(*bad.values()) if bad else set()
        if hasattr(P, "coin_seed"):
            P.coin_seed = seed * 1000 + k
        inputs = make_inputs("random", n, rng)
        res = Simulation(P, ADVERSARIES[adversary](), inputs, faulty, seed=seed * 1000 + k).run()
        if hasattr(P, "final_bad") and carry:
            bad = {p: set(P.final_bad.get(p, bad[p])) for p in honest}
        total += res.steps
        rows.append(dict(instance=k + 1, steps=res.steps, cum_steps=total,
                         correct=res.correct, honest_bits=res.honest_bits))
    return rows
