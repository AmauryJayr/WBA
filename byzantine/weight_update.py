"""Repeated weighted consensus with weight updates (appendix experiment).

Model. The Fig. 4 groups (|A| = na, |B| = nb), but processes fail *over time*:
process i fails at instance T_i ~ Geometric(q_A or q_B) and stays faulty
(Byzantine) from then on. Every instance runs one binary consensus with the
current weights. The failure times depend only on the trial seed, so every
method faces exactly the same failures (common random numbers).

A method keeps its guarantee at instance k while the current faulty weight is
below its bound (1/4 for Queen, 1/3 for King and Gradecast). The first instance
where that fails ends the trial ("guarantee lost").

Methods
  none_queen   Queen, weights never change (bound 1/4).              exact
  none_king    King or Gradecast, weights never change (bound 1/3).  exact
  garg_queen   Garg & Bridgman Fig. 3: Queen + weight update. Detection is
               Lemma 8: (1) a process that sends nothing or a malformed
               message is faulty; (2) a queen that contradicts a > 1/2 + rho
               majority is faulty. Bound 1/4.                        simulated
  garg_king    The same update on King, with (1) and the analogue of (2) (a
               king that contradicts a >= 1 - rho phase-2 majority). Bound
               1/3. Not in the paper; isolates detection from the bound.
  gc_bad       Ours: weighted Gradecast consensus + the same update. Detection
               is (1) plus the BAD sets (every leader graded below 2).
               Bound 1/3.                                            simulated

The update (both methods), after each instance:
  learning   every process broadcasts its detected set;
  agreement  one binary weighted BA per candidate j (all in parallel),
             input 1 iff the processes reporting j weigh more than rho;
  update     set w[j] = 0 for every j agreed on (the total shrinks, which
             is the renormalisation).
We do not simulate the N agreement instances; we apply their worst-case
outcome, which follows from validity alone:
  * if the honest reporters of j weigh more than rho, every honest input is 1,
    so j is removed whatever the adversary does;
  * otherwise the adversary can make the inputs mixed (faulty reports) and then
    choose the outcome, so we assume j is kept.
A correct process is never reported by an honest process (detection is sound),
and faulty reports weigh at most rho, so every honest input for it is 0 and
validity keeps it. Hence correct processes are never removed.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from .adversaries import ADVERSARIES
from .experiment import make_inputs
from .network import Simulation
from .protocols import Gradecast, King, Queen

EPS = Fraction(1, 10**6)
BOUND = {"none_queen": Fraction(1, 4), "garg_queen": Fraction(1, 4),
         "none_king": Fraction(1, 3), "garg_king": Fraction(1, 3), "gc_bad": Fraction(1, 3)}
SIMULATED = {"garg_queen": Queen, "garg_king": King, "gc_bad": Gradecast}
ATTACK = {"garg_queen": "split", "garg_king": "split", "gc_bad": "stagger"}


def failure_times(n_a: int, n_b: int, qa: float, qb: float, seed: int):
    """Instance (1-based) at which each process becomes faulty."""
    rng = random.Random(seed)
    out = []
    for i in range(n_a + n_b):
        q = qa if i < n_a else qb
        u = 1.0 - rng.random()  # (0, 1]
        out.append(1 + int(math.log(u) / math.log(1.0 - q)))
    return out


def trial(args) -> list[dict]:
    """One trial of one method. Returns one row per instance reached."""
    method, adv, na, nb, qa, qb, weights, ell, seed = args
    w = list(weights)
    n = len(w)
    T = failure_times(na, nb, qa, qb, seed)
    bound = BOUND[method]
    rho = bound - EPS
    cls = SIMULATED.get(method)
    if adv == "attack":
        adv = ATTACK.get(method, "split")
    detected: dict[int, set] = {}  # honest pid -> cumulative detected set
    rows = []
    rng = random.Random(seed * 7919 + 1)
    for k in range(1, ell + 1):
        faulty = [i for i in range(n) if T[i] <= k]
        fset = set(faulty)
        W = sum(w)
        fw = sum(w[i] for i in faulty)
        alive = fw * bound.denominator < bound.numerator * W
        row = dict(method=method, adversary=adv, seed=seed, instance=k, alive=alive,
                   faulty=len(faulty), faulty_share=fw / W,
                   removed=sum(1 for i in range(n) if weights[i] > 0 and w[i] == 0))
        if not alive:
            rows.append(row)
            break
        if cls is None:
            rows.append(row)
            continue
        honest = [p for p in range(n) if p not in fset]
        P = cls(n, 0, weights=w, rho=rho)
        if cls is Gradecast:  # carry BAD sets between instances (Ben-Or et al. Sec. 5.1)
            P.initial_bad = {p: set(detected.get(p, ())) for p in honest}
            P.expelled_all = set.intersection(*(P.initial_bad[p] for p in honest)) if honest else set()
        res = Simulation(P, ADVERSARIES[adv](), make_inputs("random", n, rng), faulty,
                         seed=seed * 1000 + k).run()
        row.update(correct=res.correct, steps=res.steps)
        found = {p: set(P.detected.get(p, ())) | set(getattr(P, "final_bad", {}).get(p, ()))
                 for p in honest}
        for p in honest:
            detected[p] = set(detected.get(p, set())) | set(found.get(p, ()))
        # learning + agreement (worst case, see module docstring) + update
        removed_now = 0
        for j in range(n):
            if w[j] == 0:
                continue
            H = sum(w[p] for p in honest if j in detected.get(p, ()))
            if H * rho.denominator > rho.numerator * W:
                assert j in fset, "a correct process was removed"
                w[j] = 0
                removed_now += 1
        row["removed_now"] = removed_now
        rows.append(row)
    return rows


def survival_exact(na, nb, qa, qb, weights, bound, ell):
    """P(guarantee still holds at instance k) with no update, k = 1..ell.

    Faulty weight at k = wA * Bin(na, pA(k)) + wB * Bin(nb, pB(k)), with
    p(k) = 1 - (1 - q)^k. The faulty set only grows, so 'holds at k' is the
    same as 'held at every instance up to k'."""
    from scipy.stats import binom
    import numpy as np
    wa, wb = weights[0], weights[-1] if nb else 0
    W = na * wa + nb * wb
    out = []
    for k in range(1, ell + 1):
        pa, pb = 1 - (1 - qa) ** k, 1 - (1 - qb) ** k
        PA = binom.pmf(np.arange(na + 1), na, pa)
        PB = binom.pmf(np.arange(nb + 1), nb, pb)
        s = 0.0
        for a in range(na + 1):
            for b in range(nb + 1):
                if (a * wa + b * wb) * bound.denominator < bound.numerator * W:
                    s += PA[a] * PB[b]
        out.append(s)
    return out
