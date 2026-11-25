"""Vector figures and the numbers quoted in the paper.

Everything is computed with the repo's own code:
  byzantine.weight_assignment   exact P(faulty weight >= tau)   (Garg & Bridgman Fig. 4)
  byzantine.fig4_compare        alpha_rho, cap_rho, E[min(F+3, cap_rho)]  (Fig. 5 + ours)
  results/fig4_compare_sim.csv  simulated weighted Gradecast runs (run_experiments.py fig4compare)
  paper/data/messages.csv       paper/scripts/messages.py
  paper/data/rare_sim.csv       paper/scripts/rare_sim.py (Fig. 2(c): WGC with rare faults)

Run from the repo root:  python paper/scripts/make_figures.py
Writes paper/figures/*.pdf and paper/data/{analytic.csv,numbers.txt}.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.getcwd())
from byzantine.fig4_compare import RHO, analytic  # noqa: E402
from byzantine.weight_assignment import SCHEMES, p_unsafe, p_unsafe_ratio  # noqa: E402

NA, FA, FB, NB_MAX = 6, "0.1", "0.3", 400
FIG, DATA = "paper/figures", "paper/data"
SIM = sys.argv[1] if len(sys.argv) > 1 else "results/fig4_compare_sim.csv"
RARE_SIM = "paper/data/rare_sim.csv"  # paper/scripts/rare_sim.py
RARE_FA, RARE_FB = "0.005", "0.015"  # per-instance rates of the weight-update test

# per-scheme weight ratio wA/wB as (p, q); (1, 0) = only A
RATIO = {"only_a": (1, 0), "uniform": (1, 1), "inverse_failure": (3, 1), "not_failing": (9, 7)}
LABEL = {"only_a": "Only A non-zero", "uniform": "Uniform (= unweighted)",
         "inverse_failure": "Inverse prob. of failure", "not_failing": "Prob. of not failing"}
STYLE = {
    "only_a": dict(color="#222222", ls="--", lw=1.2, marker=None),
    "uniform": dict(color="#8c8c8c", ls="none", marker="x", ms=2.6, mew=0.7),
    "inverse_failure": dict(color="#1f4e79", ls="none", marker="o", ms=2.6, mfc="none", mew=0.7),
    "not_failing": dict(color="#222222", ls="none", marker=".", ms=2.6),
}
# the weight formula of Section V (w_i ~ (tau - p_i) / (p_i (1 - p_i)), tau = 1/3), same ratio as
# paper/scripts/ideal_ratio.py; at tau = 1/4 it gives B weight 0, i.e. the only-A curve
def formula(p, tau):
    return max(0.0, (tau - p) / (p * (1 - p)))


FORMULA_RATIO = Fraction(formula(0.1, 1 / 3) / formula(0.3, 1 / 3)).limit_denominator(1000)
FORMULA_STYLE = dict(color="#c0392b", ls="-", lw=1.1, marker=None)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    nbs = [nb for nb in range(NB_MAX + 1) if (NA + nb) % 3 == 0]
    out = []

    # ---- analytic table ----------------------------------------------------
    rows = []
    for s in SCHEMES:
        for nb in nbs:
            a = analytic(s, NA, nb, FA, FB)
            a["p_lost_third"] = p_unsafe(s, NA, nb, FA, FB)
            a["p_lost_quarter"] = float(p_unsafe_ratio(NA, nb, float(FA), float(FB), [RATIO[s]],
                                                       Fraction(1, 4))[0])
            # cross-check the two exact routines at tau = 1/3
            chk = float(p_unsafe_ratio(NA, nb, float(FA), float(FB), [RATIO[s]], Fraction(1, 3))[0])
            assert abs(chk - a["p_lost_third"]) < 1e-12, (s, nb, chk, a["p_lost_third"])
            a["gc_expected"] = a["gc_bound_steps"] / 3
            rows.append(a)
    an = pd.DataFrame(rows)
    an.to_csv(f"{DATA}/analytic.csv", index=False)

    out.append(f"rho used for King/Gradecast = 1/3 - 1e-6 = {float(RHO):.9f}")
    out.append("\n== |B| = 399 (n = 405) ==")
    sel = an[an.nb == 399].set_index("scheme")
    out.append(sel[["n", "p_lost_third", "p_lost_quarter", "anchor", "gc_cap", "gc_expected"]]
               .to_string(float_format=lambda x: f"{x:.4f}"))
    out.append("\n== selected |B| ==")
    out.append(an[an.nb.isin([0, 6, 12, 30, 60, 90, 120, 201, 300, 399])]
               .pivot(index="nb", columns="scheme",
                      values=["p_lost_third", "p_lost_quarter", "anchor", "gc_cap", "gc_expected"])
               .to_string(float_format=lambda x: f"{x:.4f}"))
    # maxima of the P(lost) curves at 1/3
    for s in SCHEMES:
        g = an[an.scheme == s]
        i = g.p_lost_third.idxmax()
        out.append(f"max P(lost, 1/3) {s}: {g.p_lost_third[i]:.4f} at nb={g.nb[i]}")
    # where Gradecast's expected rounds drop below King's (inverse weights)
    g = an[an.scheme == "inverse_failure"]
    below = g[g.gc_expected < g.anchor]
    out.append(f"inverse: first nb with E[GC] < alpha: {below.nb.min() if len(below) else None}")
    g = an[an.scheme == "not_failing"]
    out.append("not_failing: nb where cap > alpha: "
               f"{list(g[g.gc_cap > g.anchor].nb)[:5]}... count={int((g.gc_cap > g.anchor).sum())}")

    # ---- weight formula ----------------------------------------------------
    # tau = 1/3 (King, WGC): ratio FORMULA_RATIO; tau = 1/4 (Queen): B gets weight 0 = only A.
    # Rounds use the repo's analytic() with the formula weights registered for the call only.
    def formula_analytic(nb):
        SCHEMES["formula"] = lambda fa, fb: (Fraction(FORMULA_RATIO.numerator),
                                             Fraction(FORMULA_RATIO.denominator))
        try:
            return analytic("formula", NA, nb, FA, FB)
        finally:
            del SCHEMES["formula"]
    assert formula(0.3, 0.25) == 0
    form = pd.DataFrame([formula_analytic(nb) for nb in nbs])
    form["p_lost_third"] = [float(p_unsafe_ratio(NA, nb, float(FA), float(FB),
                                                 [(FORMULA_RATIO.numerator, FORMULA_RATIO.denominator)],
                                                 Fraction(1, 3))[0]) for nb in nbs]
    form["p_lost_quarter"] = [float(p_unsafe_ratio(NA, nb, float(FA), float(FB), [(1, 0)],
                                                   Fraction(1, 4))[0]) for nb in nbs]
    form["gc_expected"] = form.gc_bound_steps / 3
    out.append(f"\n== weight formula, ratio {FORMULA_RATIO} = {float(FORMULA_RATIO):.3f} at tau = 1/3; "
               "B weight 0 at tau = 1/4 ==")
    out.append(form[form.nb.isin([0, 6, 12, 24, 30, 60, 90, 120, 201, 261, 300, 399])]
               [["nb", "p_lost_third", "p_lost_quarter", "anchor", "gc_cap", "gc_expected"]]
               .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    i = form.p_lost_third.idxmax()
    out.append(f"max P(lost, 1/3) formula: {form.p_lost_third[i]:.4f} at nb={form.nb[i]}")
    out.append(f"formula: largest nb with alpha <= 8: {form[form.anchor <= 8].nb.max()}")

    # ---- Fig: failure probability -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), sharey=False)
    for ax, col, title in ((axes[0], "p_lost_third", r"(a) King and weighted Gradecast: $P(\text{faulty weight} \geq 1/3)$"),
                           (axes[1], "p_lost_quarter", r"(b) Queen: $P(\text{faulty weight} \geq 1/4)$")):
        for s in SCHEMES:
            g = an[an.scheme == s]
            ax.plot(g.nb, g[col], label=LABEL[s], **STYLE[s])
        ax.plot(form.nb, form[col], label="Our formula", zorder=1.5,
                **{**FORMULA_STYLE, "lw": 1.1 if col == "p_lost_third" else 2.4})
        ax.set_title(title, loc="left")
        ax.set_xlim(0, NB_MAX)
        ax.set_ylim(0, 0.26 if col == "p_lost_third" else 1.0)
        ax.set_xlabel(r"Number of processes in set $B$")
    axes[0].set_ylabel("Probability guarantee is lost")
    axes[0].legend(loc="upper right", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{FIG}/failure_probability.pdf")
    plt.close(fig)

    # ---- Fig: King rounds (Garg & Bridgman Fig. 5) -------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for s in SCHEMES:
        g = an[an.scheme == s]
        ax.plot(g.nb, g.anchor, label=LABEL[s], **STYLE[s])
    ax.plot(form.nb, form.anchor, label="Our formula", zorder=1.5, **FORMULA_STYLE)
    ax.set_xlim(0, NB_MAX)
    ax.set_ylim(0, 145)
    ax.set_xlabel(r"Number of processes in set $B$")
    ax.set_ylabel(r"King rounds $\alpha_\rho$")
    ax.legend(loc="upper left", handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/king_rounds.pdf")
    plt.close(fig)

    # ---- Fig: King vs Gradecast --------------------------------------------
    sim = pd.read_csv(SIM)
    sim = sim[sim.adversary == "stagger"].assign(rounds=lambda d: d.steps / 3)
    agg = sim.groupby(["scheme", "nb"]).rounds.agg(["mean", "min", "max"]).reset_index()
    # (c): same as (b) but with the rare per-instance fault rates of the weight-update test
    rare_an = pd.DataFrame([analytic("inverse_failure", NA, nb, RARE_FA, RARE_FB) for nb in nbs])
    rare_an["gc_expected"] = rare_an.gc_bound_steps / 3
    rare = pd.read_csv(RARE_SIM).assign(rounds=lambda d: d.steps / 3)
    assert rare.correct.all()
    rare_agg = rare.groupby("nb").rounds.agg(["mean", "min", "max"]).reset_index()
    panels = (
        (an[an.scheme == "uniform"], agg[agg.scheme == "uniform"], "(a) Uniform weights"),
        (an[an.scheme == "inverse_failure"], agg[agg.scheme == "inverse_failure"],
         "(b) Inverse-failure weights"),
        (rare_an, rare_agg, "(c) Inverse-failure, rare faults"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for ax, (g, m, t) in zip(axes, panels):
        ax.plot(g.nb, g.anchor, color="#222222", lw=1.2, label=r"King (always $\alpha_\rho$)")
        ax.plot(g.nb, g.gc_cap, color="#8c8c8c", lw=1.2, ls="--",
                label=r"WGC worst case $\mathrm{cap}_\rho$")
        ax.plot(g.nb, g.gc_expected, color="#1f4e79", lw=1.2,
                label=r"WGC $E[\min(F{+}3,\mathrm{cap}_\rho)]$")
        ax.errorbar(m.nb, m["mean"], yerr=[m["mean"] - m["min"], m["max"] - m["mean"]],
                    ls="none", marker="o", ms=2.8, mfc="white", mec="#1f4e79", mew=0.8,
                    ecolor="#1f4e79", elinewidth=0.6, capsize=1.5,
                    label="WGC simulated, stagger")
        ax.set_title(t, loc="left")
        ax.set_xlim(0, NB_MAX)
        ax.set_ylim(0, 145)
        ax.set_xlabel(r"Number of processes in set $B$")
    axes[0].set_ylabel("Rounds (3 exchanges each)")
    axes[0].legend(loc="upper left", handletextpad=0.3, fontsize=6.3, labelspacing=0.25,
                   borderaxespad=0.2)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{FIG}/king_vs_gradecast.pdf")
    plt.close(fig)

    out.append("\n== Fig. 2(c): rare faults, inverse weights ==")
    out.append(rare_an[rare_an.nb.isin([0, 30, 60, 120, 201, 300, 399])]
               [["nb", "anchor", "gc_cap", "gc_expected"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out.append(f"simulated (stagger): runs {len(rare)}, all correct {bool(rare.correct.all())}")
    rcap = rare_an.set_index("nb").gc_cap
    rb = [min(x.faulty_pos + 3, rcap[x.nb]) for x in rare.itertuples()]
    out.append(f"runs exceeding min(F+3, cap): {int((rare.rounds > rb).sum())}; "
               f"hitting it exactly: {int((rare.rounds == rb).sum())}/{len(rare)}")
    out.append(rare.groupby("nb").agg(mean=("rounds", "mean"), min=("rounds", "min"),
                                      max=("rounds", "max"), F=("faulty_pos", "mean"))
               .round(2).to_string())

    # simulation summary
    out.append("\n== simulated weighted Gradecast (stagger and silent) ==")
    allsim = pd.read_csv(SIM).assign(rounds=lambda d: d.steps / 3)
    out.append(f"trials: {len(allsim)}, all correct: {bool(allsim.correct.all())}")
    ex = an.set_index(["scheme", "nb"])
    allsim["cap"] = [ex.loc[(r.scheme, r.nb), "gc_cap"] for r in allsim.itertuples()]
    allsim["bound"] = [min(r.faulty_pos + 3, r.cap) for r in allsim.itertuples()]
    out.append(f"runs exceeding min(F+3, cap): {int((allsim.rounds > allsim.bound).sum())}")
    st = allsim[allsim.adversary == "stagger"]
    out.append(f"stagger runs hitting min(F+3, cap) exactly: {int((st.rounds == st.bound).sum())}/{len(st)}")
    tight = st[(st.rounds == st.faulty_pos + 3) & (st.faulty_pos + 3 < st.cap)]
    out.append(f"stagger runs ending at exactly F+3 < cap: {len(tight)}/{len(st)}; by scheme: "
               + ", ".join(f"{k} {v}" for k, v in tight.groupby("scheme").size().items()))
    out.append(allsim.groupby(["scheme", "adversary", "nb"]).agg(
        mean=("rounds", "mean"), min=("rounds", "min"), max=("rounds", "max"),
        F=("faulty_pos", "mean")).round(2).to_string())

    # ---- Fig: message complexity -------------------------------------------
    msg = pd.read_csv(f"{DATA}/messages.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    st = {"queen": ("#8c8c8c", "s", "Queen"), "king": ("#222222", "^", "King"),
          "gradecast": ("#1f4e79", "o", "W. Gradecast")}
    for p, (c, mk, lab) in st.items():
        for scen, ls, mfc in (("fault-free", "-", c), ("attacked", "--", "white")):
            g = msg[(msg.protocol == p) & (msg.scenario == scen)]
            ax.plot(g.n, g.honest_bits, color=c, ls=ls, marker=mk, ms=3, mfc=mfc, mew=0.7, lw=0.9,
                    label=f"{lab}, {'no faults' if scen == 'fault-free' else 'attack, $t$ faults'}")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of processes $N$ (uniform weights, largest $t$)")
    ax.set_ylabel("Bits sent by honest processes")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.36)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, handletextpad=0.3, columnspacing=0.8, fontsize=6.3)
    fig.savefig(f"{FIG}/messages.pdf")
    plt.close(fig)
    out.append("\n== messages ==")
    out.append(msg.pivot_table(index="n", columns=["protocol", "scenario"],
                               values=["steps", "honest_msgs", "honest_bits"]).to_string())
    big = msg[msg.n == 49].set_index(["protocol", "scenario"])
    out.append(f"n=49 bits ratio GC/King fault-free: "
               f"{big.honest_bits[('gradecast', 'fault-free')] / big.honest_bits[('king', 'fault-free')]:.1f}; "
               f"attacked: {big.honest_bits[('gradecast', 'attacked')] / big.honest_bits[('king', 'attacked')]:.1f}")

    # ---- rare faults: same setup, per-instance rates of the weight-update test ----
    RA, RB = RARE_FA, RARE_FB
    rr = Fraction(formula(float(RA), 1 / 3) / formula(float(RB), 1 / 3)).limit_denominator(1000)
    out.append(f"\n== rare faults: fA={RA}, fB={RB}; formula ratio {rr} = {float(rr):.3f} ==")
    SCHEMES["formula"] = lambda fa, fb: (Fraction(rr.numerator), Fraction(rr.denominator))
    try:
        for nb in (60, 201, 399):
            for s in list(SCHEMES):
                ratio = (rr.numerator, rr.denominator) if s == "formula" else RATIO[s]
                a = analytic(s, NA, nb, RA, RB)
                pl = float(p_unsafe_ratio(NA, nb, float(RA), float(RB), [ratio], Fraction(1, 3))[0])
                out.append(f"nb={nb:3d} {s:16s} king_alpha={a['anchor']:4d} gc_cap={a['gc_cap']:4d} "
                           f"gc_expected={a['gc_bound_steps'] / 3:6.2f} p_lost_third={pl:.2e}")
    finally:
        del SCHEMES["formula"]
    # how far King gets by moving weight onto A (WGC's expected bound barely changes)
    out.append("\n== heavier A, |B| = 399: King rounds vs. safety ==")
    for fa, fb in ((RA, RB), (FA, FB)):
        for r in (1, 3, 10, 30, 60, 100):
            SCHEMES["ratio"] = lambda a, b, r=r: (Fraction(r), Fraction(1))
            try:
                a = analytic("ratio", NA, 399, fa, fb)
            finally:
                del SCHEMES["ratio"]
            pl = float(p_unsafe_ratio(NA, 399, float(fa), float(fb), [(r, 1)], Fraction(1, 3))[0])
            out.append(f"fA={fa} fB={fb} {r:3d}:1 king_alpha={a['anchor']:4d} gc_cap={a['gc_cap']:4d} "
                       f"gc_expected={a['gc_bound_steps'] / 3:6.2f} p_lost_third={pl:.2e}")

    text = "\n".join(out)
    open(f"{DATA}/numbers.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
