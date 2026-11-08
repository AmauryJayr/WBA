"""Weighted King vs weighted Gradecast on the setup of Garg & Bridgman's Fig. 4.

Setup: |A| = 6 processes failing with probability fa, |B| failing with fb, each
failing independently. There are four weight schemes (see weight_assignment.py).
Both protocols are configured for rho just below 1/3, so both are guaranteed
correct exactly when the failed weight is < 1/3. Panel (a) of the figure is
therefore one curve per scheme, shared by both protocols.

They differ in time to decide. All values are synchronous steps, conditioned
on a safe run (failed weight < 1/3):

  king_steps       3 * alpha_rho. Always, whatever fails (exact).
  gc_bound_steps   expected 3 * min(F + 3, cap). F is the number of failed
                   positive-weight processes, cap the iteration limit. This is
                   the proven worst case; the expectation is exact over the
                   binomial failure model.
  gc_sim_*         Gradecast simulated with random failures drawn from the
                   model, against the `stagger` adversary (spends one faulty
                   process per iteration) and against silent faults.
"""

from __future__ import annotations

import random
import zlib
from fractions import Fraction

import numpy as np
from scipy.stats import binom

from .experiment import TrialConfig, run_trial
from .protocols import Gradecast, King
from .weight_assignment import SCHEMES

RHO = Fraction(1, 3) - Fraction(1, 10**6)  # "just below 1/3": the most both tolerate


def scheme_weights(scheme: str, na: int, nb: int, fa: str, fb: str):
    wa, wb = SCHEMES[scheme](Fraction(fa), Fraction(fb))
    return [wa] * na + [wb] * nb, wa, wb


def analytic(scheme: str, na: int, nb: int, fa: str, fb: str) -> dict:
    w, wa, wb = scheme_weights(scheme, na, nb, fa, fb)
    n = na + nb
    king = King(n, 0, w, RHO)
    gc = Gradecast(n, 0, w, RHO)
    total = na * wa + nb * wb
    pa = binom.pmf(np.arange(na + 1), na, float(fa))
    pb = binom.pmf(np.arange(nb + 1), nb, float(fb))
    p_safe = e_iters = 0.0
    for a in range(na + 1):
        for b in range(nb + 1):
            if 3 * (a * wa + b * wb) >= total:  # unsafe: no guarantee for either
                continue
            pr = pa[a] * pb[b]
            F = a * (wa > 0) + b * (wb > 0)
            p_safe += pr
            e_iters += pr * min(F + 3, gc.cap)
    return dict(scheme=scheme, nb=nb, n=n, anchor=king.anchor, gc_cap=gc.cap,
                p_safe=p_safe, king_steps=3 * king.anchor,
                gc_bound_steps=3 * e_iters / p_safe)


def sample_faulty(rng: random.Random, w, na: int, fa: float, fb: float):
    """Draw failures from the model, rejecting unsafe draws (failed weight >= 1/3)."""
    W = sum(w)
    while True:
        faulty = [i for i in range(len(w)) if rng.random() < (fa if i < na else fb)]
        if 3 * sum(w[i] for i in faulty) < W and len(faulty) < len(w):
            return faulty


def simulate(args) -> dict:
    scheme, na, nb, fa, fb, adversary, draw = args
    w, _, _ = scheme_weights(scheme, na, nb, fa, fb)
    rng = random.Random(zlib.crc32(repr((scheme, nb, adversary, draw)).encode()))
    faulty = sample_faulty(rng, w, na, float(fa), float(fb))
    lcm = np.lcm.reduce([x.denominator for x in w])
    wi = tuple(int(x * lcm) for x in w)
    row = run_trial(TrialConfig("gradecast", na + nb, 0, len(faulty), adversary, "split",
                                seed=draw, weights=wi, rho=str(RHO), faulty=tuple(faulty)))
    return dict(scheme=scheme, nb=nb, adversary=adversary, draw=draw,
                faulty=len(faulty), faulty_pos=sum(1 for i in faulty if wi[i] > 0),
                correct=row["correct"], steps=row["steps"], honest_msgs=row["honest_msgs"])
