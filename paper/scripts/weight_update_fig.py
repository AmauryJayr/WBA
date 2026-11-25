"""Figure for Section V-F: P(guarantee still holds) per instance, with and without weight updates.

Run from the repo root:  python paper/scripts/weight_update_fig.py -> paper/figures/weight_update.pdf
Reads results/weight_update.csv (the repo's run). paper/scripts/weight_update.py re-ran the same
jobs with the same seeds and got identical numbers (paper/data/weight_update.txt); this script
re-checks the summary against that file before plotting.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.getcwd())
from byzantine.improvements import int_weights  # noqa: E402
from byzantine.weight_update import BOUND, survival_exact  # noqa: E402

NA, NB, QA, QB, ELL, TRIALS = 6, 30, 0.005, 0.015, 100, 200
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
df = pd.read_csv("results/weight_update.csv")
ks = list(range(1, ELL + 1))
w = int_weights(NA, NB, (3, 1))
exact = {m: survival_exact(NA, NB, QA, QB, w, BOUND[m], ELL) for m in ("none_queen", "none_king")}


def surv(g):
    lost = g[~g.alive.astype(bool)].groupby("seed").instance.min()
    return [1 - (lost <= k).sum() / TRIALS for k in ks], lost


# re-check against the re-run summary
ref = {}
for line in open("paper/data/weight_update.txt"):
    p = line.split()
    if len(p) >= 3 and p[1] in ("garg_queen", "garg_king", "gc_bad"):
        ref[(p[0], p[1])] = float(p[2])
for (adv, m), v in ref.items():
    s, _ = surv(df[(df.method == m) & (df.adversary == adv)])
    assert abs(s[-1] - v) < 1e-9, (adv, m, s[-1], v)

STYLE = {
    "none_queen": dict(color="#9a9a9a", ls=":", lw=1.1, label="Queen, no update"),
    "none_king": dict(color="#9a9a9a", ls="--", lw=1.1, label="King or WGC, no update"),
    "garg_queen": dict(color="#222222", ls=":", lw=1.3, label="Queen + Garg–Bridgman update"),
    "garg_king": dict(color="#222222", ls="-", lw=1.1, label="King + Garg–Bridgman update"),
    "gc_bad": dict(color="#1f4e79", ls="-", lw=1.8, label="WGC + BAD-set update"),
}
PANELS = [("silent", "(a) Silent"), ("equivocate", "(b) Equivocate"),
          ("split", "(c) Split vote"), ("flip", "(d) Same wrong value to all")]
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.6), sharey=True)
for ax, (adv, title) in zip(axes, PANELS):
    for m in ("none_queen", "none_king"):
        ax.plot(ks, exact[m], **STYLE[m])
    for m in ("garg_queen", "garg_king", "gc_bad"):
        s, _ = surv(df[(df.method == m) & (df.adversary == adv)])
        ax.plot(ks, s, **STYLE[m])
    ax.set_title(title, loc="left")
    ax.set_xlim(1, ELL)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Agreement instance")
axes[0].set_ylabel("P(guarantee still holds)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(bottom=0.38, wspace=0.12)
os.makedirs("paper/figures", exist_ok=True)
fig.savefig("paper/figures/weight_update.pdf")
print("paper/figures/weight_update.pdf")
