"""Closed-form weight formula vs. the exact optimal ratio (two-group setup).

Run from the repo root:  python paper/scripts/ideal_ratio.py  -> paper/data/ideal_ratio.txt
Uses the same formula as run_experiments.py cmd_idealratio and the repo's exact
optimal_ratio / p_unsafe_ratio.
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.getcwd())
from byzantine.weight_assignment import optimal_ratio, p_unsafe_ratio  # noqa: E402

NA, FA, FB, TAU = 6, 0.1, 0.3, Fraction(1, 3)


def formula(p, t):
    return max(0.0, (t - p) / (p * (1 - p)))


def pair(r):
    f = Fraction(r).limit_denominator(1000)
    return f.numerator, f.denominator


r_star = formula(FA, 1 / 3) / formula(FB, 1 / 3)
lines = [f"formula weights tau=1/3: wA={formula(FA, 1/3):.4f} wB={formula(FB, 1/3):.4f} ratio={r_star:.3f}",
         f"formula weights tau=1/4: wA={formula(FA, 0.25):.4f} wB={formula(FB, 0.25):.4f} (B gets 0 -> only A)",
         "nb  inverse(3:1)  formula  exact_opt  exact_ratio"]
for nb in (24, 30, 60, 90, 120, 201, 300, 399):
    inv, form = p_unsafe_ratio(NA, nb, FA, FB, [(3, 1), pair(r_star)], TAU)
    r, p = optimal_ratio(NA, nb, FA, FB, TAU)
    lines.append(f"{nb:3d}  {inv:.4f}  {form:.4f}  {p:.4f}  {'only A' if r is None else f'{float(r):.2f}'}")
os.makedirs("paper/data", exist_ok=True)
open("paper/data/ideal_ratio.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
