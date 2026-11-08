"""Weight-assignment analysis from Garg & Bridgman, Section VI (Fig. 4).

Two groups of processes: every process in A fails independently with
probability fa, every process in B with probability fb. For a weight
assignment, compute exactly the probability that the failed processes hold
at least 1/3 of the total weight. At that point no WBA protocol is guaranteed
to work (Lemma 1), so this is the probability that the system is unsafe.

Every process in a group gets the same weight, so the failed weight depends only
on the failure counts (ka, kb), which are binomial. The computation is an exact
double sum. Weights are kept as integer ratios, so the comparison with 1/3 is exact.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.stats import binom

# Each scheme maps (fa, fb) -> unnormalised per-process weights (wA, wB).
SCHEMES = {
    "only_a": lambda fa, fb: (Fraction(1), Fraction(0)),
    "uniform": lambda fa, fb: (Fraction(1), Fraction(1)),
    "inverse_failure": lambda fa, fb: (1 / Fraction(fa), 1 / Fraction(fb)),
    "not_failing": lambda fa, fb: (1 - Fraction(fa), 1 - Fraction(fb)),
}

LABELS = {
    "only_a": "Only A non-zero",
    "uniform": "Uniform",
    "inverse_failure": "Proportional to the inverse probability of failure",
    "not_failing": "Proportional to the probability of not failing",
    "log_odds": "Log-odds, log((1-p)/p)",
}


def log_odds(fa, fb):
    """Nitzan-Paroush weights, rounded to a close rational so thresholds stay exact."""
    lo = lambda p: Fraction(math.log((1 - p) / p)).limit_denominator(10**6)
    return lo(float(fa)), lo(float(fb))


EXTRA_SCHEMES = {"log_odds": log_odds}


def p_unsafe_ratio(na: int, nb: int, fa: float, fb: float, ratios, threshold=Fraction(1, 3)):
    """Exact P(failed weight >= threshold) for many weight ratios r = wA/wB at once.

    Each ratio is a pair (p, q) meaning r = p/q, and (1, 0) means "only A".
    Unsafe <=> v(a r + b) >= u(na r + nb) with threshold = u/v, which is
    p (v a - na u) >= q (u nb - v b): integer arithmetic, so ties are exact.
    """
    u, v = Fraction(threshold).numerator, Fraction(threshold).denominator
    P = np.array([x[0] for x in ratios], dtype=np.int64)[:, None, None]
    Q = np.array([x[1] for x in ratios], dtype=np.int64)[:, None, None]
    a = np.arange(na + 1)[None, :, None]
    b = np.arange(nb + 1)[None, None, :]
    unsafe = P * (v * a - na * u) >= Q * (u * nb - v * b)
    pa = binom.pmf(np.arange(na + 1), na, fa)
    pb = binom.pmf(np.arange(nb + 1), nb, fb)
    return np.einsum("rab,a,b->r", unsafe.astype(float), pa, pb)


def optimal_ratio(na: int, nb: int, fa: float, fb: float, threshold=Fraction(1, 3),
                  prefer=None):
    """A ratio wA/wB minimising P(unsafe), found exactly (two-class weights only).

    P is piecewise constant in r and changes only where some (a, b) failure
    count sits exactly on the threshold. Checking r = 0, r = infinity, every
    breakpoint and every midpoint between neighbouring breakpoints covers
    every possible value.
    """
    u, v = Fraction(threshold).numerator, Fraction(threshold).denominator
    bps = set()
    for a in range(na + 1):
        den = v * a - na * u
        if den == 0:
            continue
        for b in range(nb + 1):
            r = Fraction(u * nb - v * b, den)
            if r > 0:
                bps.add(r)
    bps = sorted(bps)
    cands = [Fraction(0)] + bps + [(x + y) / 2 for x, y in zip(bps, bps[1:])]
    if bps:
        cands.append(bps[-1] + 1)
    ratios = [(c.numerator, c.denominator) for c in cands] + [(1, 0)]
    probs = p_unsafe_ratio(na, nb, fa, fb, ratios, threshold)
    # P is flat over whole ranges of r, so the minimiser is usually not unique.
    # `prefer` (a key on the (p, q) ratio) picks among the tied minimisers,
    # e.g. the one with the fewest King rounds. Without it, the first is used.
    best = float(probs.min())
    tied = [i for i in range(len(ratios)) if probs[i] <= best + 1e-12]
    i = min(tied, key=lambda i: prefer(ratios[i])) if prefer else tied[0]
    r = None if ratios[i][1] == 0 else Fraction(*ratios[i])  # None = only A
    return r, float(probs[i])


def p_unsafe(scheme: str, na: int, nb: int, fa: str, fb: str, ties: str = "exact") -> float:
    """P(failed weight >= 1/3 of the total) for one weight scheme.

    fa and fb are decimal strings (e.g. "0.1") so the weights are exact rationals.

    ties="exact": a failed weight of exactly 1/3 counts as unsafe. This is correct,
        because Lemma 1 rules out WBA for rho >= 1/3.
    ties="float": normalise the weights to floats and sum them one process at a
        time, as a direct simulation would. When the true weight is exactly 1/3,
        the rounding error decides which side it falls on. This reproduces the
        scattered points in the paper's Fig. 4.
    """
    wa, wb = SCHEMES[scheme](Fraction(fa), Fraction(fb))
    total = na * wa + nb * wb
    ka = np.arange(na + 1)
    kb = np.arange(nb + 1)
    pa = binom.pmf(ka, na, float(fa))
    pb = binom.pmf(kb, nb, float(fb))
    if ties == "exact":
        # failed weight >= total/3  <=>  3*(ka*wa + kb*wb) >= total, in exact arithmetic
        unsafe = np.array([[3 * (a * wa + b * wb) >= total for b in kb] for a in ka])
    elif ties == "float":
        sum_a = np.concatenate([[0.0], np.cumsum(np.full(na, float(wa / total)))])
        sum_b = np.concatenate([[0.0], np.cumsum(np.full(nb, float(wb / total)))])
        unsafe = (sum_a[:, None] + sum_b[None, :]) >= 1 / 3
    else:
        raise ValueError(f"ties must be 'exact' or 'float', not {ties!r}")
    return float(pa @ unsafe @ pb)


def fig4(na: int = 6, fa: str = "0.1", fb: str = "0.3", nb_max: int = 400, step: int = 3,
         ties: str = "exact"):
    """Rows for Fig. 4. As in the paper, only points where na + nb is divisible by 3 are used."""
    rows = []
    for nb in range(0, nb_max + 1):
        if (na + nb) % step:
            continue
        for s in SCHEMES:
            rows.append(dict(scheme=s, na=na, nb=nb, fa=float(fa), fb=float(fb), ties=ties,
                             p_unsafe=p_unsafe(s, na, nb, fa, fb, ties)))
    return rows
