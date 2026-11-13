"""Experiment driver: parameter sweeps -> CSV -> figures.

Examples
  python run_experiments.py sweep   --n 4 7 10 13 16 25 31 49 64 --trials 20
  python run_experiments.py boundary --n 4 5 6 7 8 9 10 12 --trials 50
  python run_experiments.py modelcheck
  python run_experiments.py plot
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

import pandas as pd

from byzantine.adversaries import ADVERSARIES
from byzantine.experiment import INPUT_MODES, PLACEMENTS, sweep
from byzantine.modelcheck import check, format_counterexample
from byzantine.protocols import PROTOCOLS

RESULTS = "results"
COLORS = {"queen": "#2a78d6", "king": "#eb6834"}  # categorical slots 1-2


def cmd_sweep(args):
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    rows = list(sweep(args.protocols, args.n, args.adversaries, args.trials,
                      args.inputs, args.placements, beyond_bound=args.beyond))
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS, args.out)
    df.to_csv(path, index=False)
    print(f"{len(df)} trials in {time.time() - t0:.1f}s -> {path}")
    summary(df)


def summary(df: pd.DataFrame):
    g = df.groupby(["protocol", "within_bound", "adversary"])
    tab = g.agg(trials=("correct", "size"),
                agreement=("agreement", "mean"),
                validity=("validity", "mean"),
                correct=("correct", "mean"))
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))


def cmd_boundary(args):
    """Run each protocol with f one or two past its proven bound."""
    args.beyond = 2
    args.out = "boundary.csv"
    cmd_sweep(args)


def cmd_modelcheck(args):
    os.makedirs(RESULTS, exist_ok=True)
    configs = [("queen", 5, 1), ("queen", 4, 1), ("queen", 3, 1),
               ("king", 4, 1), ("king", 3, 1), ("king", 2, 1)]
    rows = []
    for proto, n, f in configs:
        for faulty in itertools.combinations(range(n), f):
            t0 = time.time()
            r = check(proto, n, f, faulty)
            rows.append(dict(protocol=proto, n=n, f=f, faulty=list(faulty),
                             within_bound=f <= PROTOCOLS[proto].max_faults(n),
                             ok=r.ok, states=r.states_explored,
                             round_executions=r.round_executions,
                             seconds=time.time() - t0,
                             violates=r.counterexample["violates"] if r.counterexample else ""))
            print(f"{proto:5s} n={n} f={f} faulty={list(faulty)}: "
                  f"{'OK' if r.ok else 'VIOLATION'} ({r.round_executions} round executions)")
            if not r.ok:
                print(format_counterexample(r))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "modelcheck.csv"), index=False)


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 10,
                         "legend.frameon": False})
    df = pd.read_csv(os.path.join(RESULTS, "sweep.csv"))
    df = df[df.within_bound]
    fig_dir = os.path.join(RESULTS, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. message complexity at maximum tolerated f, vs n
    at_max = df[df.apply(lambda r: r.f == PROTOCOLS[r.protocol].max_faults(r.n), axis=1)]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for proto, g in at_max.groupby("protocol"):
        m = g.groupby("n").honest_msgs.mean()
        ax.plot(m.index, m.values, marker="o", ms=5, lw=2, color=COLORS[proto], label=proto.title())
        ax.annotate(proto.title(), (m.index[-1], m.values[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color="#52514e")
    ax.set_xlabel("n (processes)")
    ax.set_ylabel("messages sent by honest processes")
    ax.set_title("Message complexity at maximum tolerated f", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "messages_vs_n.png"), dpi=200)

    # 2. synchronous steps vs f
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for proto, g in df.groupby("protocol"):
        s = g.groupby("f").steps.mean()
        ax.plot(s.index, s.values, marker="o", ms=5, lw=2, color=COLORS[proto], label=proto.title())
    ax.set_xlabel("f (tolerated faults)")
    ax.set_ylabel("synchronous message steps")
    ax.set_title("Latency: (f+1) rounds x phases per round", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "steps_vs_f.png"), dpi=200)

    # 3. boundary: failure rate past the proven bound, per adversary
    bpath = os.path.join(RESULTS, "boundary.csv")
    if os.path.exists(bpath):
        b = pd.read_csv(bpath)
        b = b[~b.within_bound]
        fail = 1 - b.groupby(["adversary", "protocol"]).correct.mean().unstack()
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        x = range(len(fail.index))
        w = 0.38
        for i, proto in enumerate(fail.columns):
            ax.bar([xi + (i - 0.5) * (w + 0.02) for xi in x], fail[proto], width=w,
                   color=COLORS[proto], label=proto.title())
        ax.set_xticks(list(x), fail.index)
        ax.set_ylabel("fraction of runs violating BA")
        ax.set_title("Beyond the bound (f > max tolerated)", loc="left")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "boundary_failures.png"), dpi=200)
    print(f"figures written to {fig_dir}")


def cmd_fig4(args):
    """Recreate Fig. 4 of Garg & Bridgman: P(failed weight >= 1/3) vs |B|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from byzantine.weight_assignment import LABELS, fig4

    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    df = pd.DataFrame(fig4(na=args.na, fa=args.fa, fb=args.fb, nb_max=args.nb_max,
                           ties=args.ties))
    suffix = "" if args.ties == "exact" else "_float"
    csv = os.path.join(RESULTS, f"fig4_weight_assignment{suffix}.csv")
    df.to_csv(csv, index=False)

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 10,
                         "legend.frameon": False})
    # categorical slots 1-4 in fixed order; markers mirror the paper's (secondary encoding)
    style = {
        "only_a": dict(color="#2a78d6", ls="--", lw=2, marker=None),
        "uniform": dict(color="#eb6834", ls="none", marker="x", ms=4, mew=1.2),
        "inverse_failure": dict(color="#1baf7a", ls="none", marker="o", ms=4,
                                mfc="none", mew=1.2),
        "not_failing": dict(color="#eda100", ls="none", marker="*", ms=5),
    }
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for scheme, st in style.items():
        g = df[df.scheme == scheme]
        ax.plot(g.nb, g.p_unsafe, label=LABELS[scheme], **st)
    ax.set_xlim(0, args.nb_max)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("Number of processes in set B")
    ax.set_ylabel("P(weight of failed processes ≥ 1/3)")
    how = "exact" if args.ties == "exact" else "float sums, as in the paper"
    ax.set_title(f"|A| = {args.na}, fa = {args.fa}, fb = {args.fb}  ({how})", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(RESULTS, "figures", f"fig4_weight_assignment{suffix}.png")
    fig.savefig(out, dpi=200)
    print(f"{csv}\n{out}")
    print(df.pivot(index="nb", columns="scheme", values="p_unsafe")
            .iloc[[0, 1, 2, 5, 10, 33, 66, 100, -1]].to_string(float_format=lambda x: f"{x:.4f}"))


def cmd_fig4compare(args):
    """Fig. 4 setup: weighted King vs weighted Gradecast (resilience + time to decide)."""
    import multiprocessing as mp

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from byzantine.fig4_compare import analytic, simulate
    from byzantine.weight_assignment import LABELS, SCHEMES, fig4

    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    nbs = [nb for nb in range(args.nb_max + 1) if (args.na + nb) % 3 == 0]
    p4 = pd.DataFrame(fig4(na=args.na, fa=args.fa, fb=args.fb, nb_max=args.nb_max))
    an = pd.DataFrame([analytic(s, args.na, nb, args.fa, args.fb) for s in SCHEMES for nb in nbs])
    an.to_csv(os.path.join(RESULTS, "fig4_compare_analytic.csv"), index=False)

    sim_csv = os.path.join(RESULTS, "fig4_compare_sim.csv")
    if args.plot_only and os.path.exists(sim_csv):
        sim = pd.read_csv(sim_csv)
    else:
        jobs = [(s, args.na, nb, args.fa, args.fb, adv, d)
                for s in SCHEMES for nb in args.sim_nb for adv in ("stagger", "silent")
                for d in range(args.draws)]
        # biggest systems first so the pool stays busy
        jobs.sort(key=lambda j: -j[2])
        t0 = time.time()
        with mp.Pool(args.jobs) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(simulate, jobs, chunksize=1), 1):
                rows.append(r)
                if i % 50 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} trials, {time.time() - t0:.0f}s", flush=True)
        sim = pd.DataFrame(rows)
        sim.to_csv(sim_csv, index=False)
    print(f"simulated trials: {len(sim)}, all correct: {bool(sim.correct.all())}")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 9,
                         "legend.frameon": False})
    scheme_style = {
        "only_a": dict(color="#2a78d6", ls="--", lw=2, marker=None),
        "uniform": dict(color="#eb6834", ls="none", marker="x", ms=3.5, mew=1.1),
        "inverse_failure": dict(color="#1baf7a", ls="none", marker="o", ms=3.5, mfc="none", mew=1.1),
        "not_failing": dict(color="#eda100", ls="none", marker="*", ms=4.5),
    }
    KING, GC = "#2a78d6", "#eb6834"

    fig = plt.figure(figsize=(9, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    for s, st in scheme_style.items():
        g = p4[p4.scheme == s]
        ax.plot(g.nb, g.p_unsafe, label=LABELS[s], **st)
    ax.set_xlim(0, args.nb_max)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("Number of processes in set B")
    ax.set_ylabel("P(failed weight ≥ 1/3)")
    ax.set_title("(a) Fig. 4: probability the guarantee is lost. Weighted King and weighted "
                 "Gradecast share this curve\n(both need failed weight < 1/3); Uniform is also "
                 "unweighted King and Ben-Or–Dolev–Hoch Gradecast", loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    agg = (sim.groupby(["scheme", "nb", "adversary"]).steps
              .agg(["mean", "min", "max"]).reset_index())
    for i, s in enumerate(SCHEMES):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        a = an[an.scheme == s]
        ax.plot(a.nb, a.king_steps, color=KING, lw=2, label="Weighted King (always 3α)")
        ax.plot(a.nb, a.gc_bound_steps, color=GC, lw=1.5, ls="--",
                label="Gradecast worst-case bound, averaged over failures")
        for adv, mk in (("stagger", dict(marker="o", mfc=GC)), ("silent", dict(marker="o", mfc="white"))):
            g = agg[(agg.scheme == s) & (agg.adversary == adv)]
            ax.errorbar(g.nb, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]],
                        ls="none", color=GC, ms=4.5, mew=1.2, elinewidth=0.8, capsize=2, **mk,
                        label=("Gradecast simulated, stagger attack" if adv == "stagger"
                               else "Gradecast simulated, silent faults"))
        ax.set_title(f"({'bcde'[i]}) {LABELS[s]}", loc="left", fontsize=9)
        ax.set_xlim(0, args.nb_max)
        ax.set_ylim(0, 12 if s == "only_a" else None)
        if i // 2 == 1:
            ax.set_xlabel("Number of processes in set B")
        if i % 2 == 0:
            ax.set_ylabel("synchronous steps to decide")
        if s == "uniform":  # top-left of this panel is empty
            ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Weighted King vs weighted Gradecast, |A| = {args.na}, fa = {args.fa}, fb = {args.fb} "
                 f"(safe runs only; simulated points: mean, bars min–max over {args.draws} failure draws)",
                 fontsize=9, x=0.07, ha="left", y=0.935)
    out = os.path.join(RESULTS, "figures", "fig4_king_vs_gradecast.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)
    print(an[an.nb.isin([0, 30, 60, 99, 201, 399])]
            .pivot(index="nb", columns="scheme", values=["king_steps", "gc_bound_steps"])
            .round(1).to_string())
    print(agg.round(1).to_string())


def cmd_improve(args):
    """Try every heuristic on the Fig. 4 setup; see byzantine/improvements.py."""
    import multiprocessing as mp

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from byzantine.improvements import simulate, sparse_rows, weight_rows

    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    nbs = [nb for nb in range(args.nb_max + 1) if (args.na + nb) % 3 == 0]
    t0 = time.time()
    wr = pd.DataFrame([r for nb in nbs for r in weight_rows(args.na, nb, args.fa, args.fb)])
    wr = pd.concat([wr, pd.DataFrame(sparse_rows(wr[wr.scheme == "optimal"]))], ignore_index=True)
    wr.to_csv(os.path.join(RESULTS, "improve_weights.csv"), index=False)
    print(f"weights: {len(wr)} rows in {time.time() - t0:.0f}s")

    sim_csv = os.path.join(RESULTS, "improve_protocols.csv")
    if args.plot_only and os.path.exists(sim_csv):
        sim = pd.read_csv(sim_csv)
    else:
        jobs = [(p, s, args.na, nb, args.fa, args.fb, adv, d)
                for p in ("king", "king_es", "gradecast", "gc_king")
                for s in ("inverse_failure", "optimal", "uniform") for nb in args.sim_nb
                for adv in ("stagger", "silent") for d in range(args.draws)]
        jobs.sort(key=lambda j: -j[3])
        t0 = time.time()
        rows = []
        with mp.Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(simulate, jobs, chunksize=1), 1):
                rows.append(r)
                if i % 100 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} trials, {time.time() - t0:.0f}s", flush=True)
        sim = pd.DataFrame(rows)
        sim.to_csv(sim_csv, index=False)
    ran = sim[sim.safe]
    print(f"protocol trials run: {len(ran)} (skipped {len(sim) - len(ran)} outside the bound); "
          f"all correct: {bool(ran.correct.all())}")

    # ---- figure -------------------------------------------------------------
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 9, "legend.frameon": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axs[0]
    w13 = wr[wr.threshold == "1/3"]
    wstyle = [("uniform", "Uniform (unweighted)", dict(color="#8a8984", lw=1.2, ls=":")),
              ("inverse_failure", "Inverse prob. of failure (paper's best)", dict(color="#2a78d6", lw=2)),
              ("log_odds", "Log-odds", dict(color="#eda100", lw=2, ls="--")),
              ("optimal", "Optimal ratio wA/wB (exact)", dict(color="#1baf7a", lw=2)),
              ("optimal_sparse", "Optimal, weight on only k ≤ |B| of B", dict(color="#eb6834", lw=1.5, ls="--"))]
    for s, lab, st in wstyle:
        g = w13[w13.scheme == s].sort_values("nb")
        ax.plot(g.nb, g.p_unsafe, label=lab, **st)
    ax.set_ylim(0, 0.25)
    ax.set_xlim(0, args.nb_max)
    ax.set_xlabel("Number of processes in set B")
    ax.set_ylabel("P(failed weight ≥ 1/3)")
    ax.set_title("(a) Weights: probability the guarantee is lost", loc="left")
    ax.legend(fontsize=7.5, loc="upper right")

    PSTY = {"king": ("Weighted King", "#2a78d6"), "king_es": ("Early-stopping King (ρ<1/4)", "#1baf7a"),
            "gradecast": ("Gradecast", "#eb6834"), "gc_king": ("Gradecast + king", "#eda100")}
    inv = ran[ran.scheme == "inverse_failure"]
    agg = inv.groupby(["protocol", "adversary", "nb"]).agg(
        steps=("steps", "mean"), bits=("honest_bits", "mean")).reset_index()
    for k, (col, ylab, title, log) in enumerate([
            ("steps", "synchronous steps to decide (mean)",
             "(b) Time, inverse weights\n(runs inside each bound only; early-stopping\n"
             "King is outside its 1/4 bound 39–51% of the time)", False),
            ("bits", "bits sent by honest processes (mean)", "(c) Communication, inverse weights", True)]):
        ax = axs[1 + k]
        for p, (lab, c) in PSTY.items():
            for adv, ls, mfc in (("stagger", "-", c), ("silent", ":", "white")):
                g = agg[(agg.protocol == p) & (agg.adversary == adv)].sort_values("nb")
                ax.plot(g.nb, g[col], ls=ls, color=c, marker="o", ms=4, mfc=mfc, lw=1.6,
                        label=f"{lab}, {'stagger attack' if adv == 'stagger' else 'silent faults'}")
        ax.set_xlabel("Number of processes in set B")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = os.path.join(RESULTS, "figures", "improvements.png")
    fig.savefig(out, dpi=200)
    print(out)

    # ---- tables -------------------------------------------------------------
    pick = [0, 30, 60, 120, 201, 399]
    for t in ("1/4", "1/3", "1/2"):
        tab = wr[(wr.threshold == t) & wr.nb.isin(pick)].pivot(index="nb", columns="scheme", values="p_unsafe")
        print(f"\nP(failed weight >= {t})\n" + tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\noptimal ratio wA/wB and King anchor at threshold 1/3")
    o = wr[(wr.threshold == "1/3") & wr.nb.isin(pick) & wr.scheme.isin(["optimal", "inverse_failure", "optimal_sparse"])]
    print(o.pivot(index="nb", columns="scheme", values=["ratio", "anchor", "senders"]).to_string())
    print("\nprotocols (mean over safe draws): steps / bits / fraction of draws inside the bound")
    allp = sim.groupby(["scheme", "protocol", "adversary", "nb"]).agg(
        safe=("safe", "mean")).reset_index()
    g = ran.groupby(["scheme", "protocol", "adversary", "nb"]).agg(
        steps=("steps", "mean"), bits=("honest_bits", "mean")).reset_index()
    print(g.merge(allp).pivot_table(index=["scheme", "adversary", "nb"], columns="protocol",
                                    values="steps").round(1).to_string())
    print(g.merge(allp).pivot_table(index=["scheme", "nb"], columns="protocol",
                                    values="safe").round(2).to_string())


def cmd_compare(args):
    """Every protocol and model side by side on the Fig. 4 setup (byzantine/compare.py)."""
    import multiprocessing as mp

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from byzantine import compare as C
    from byzantine.weight_assignment import p_unsafe_ratio

    na, fa, fb = args.na, float(args.fa), float(args.fb)
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    nbs = [nb for nb in range(args.nb_max + 1) if (na + nb) % 3 == 0]

    # ---- exact: resilience and deterministic step counts -------------------
    t0 = time.time()
    ex = []
    for nb in nbs:
        for p in C.ORDER:
            _, _, _, pu = C.best_config(p, na, nb, fa, fb)
            row = dict(protocol=p, nb=nb, p_unsafe=pu)
            if p in ("king", "queen", "hybrid_king", "dolev_strong"):
                row["steps"] = C.exact_steps(p, na, nb, fa, fb)
            ex.append(row)
        ex.append(dict(protocol="paper_inverse", nb=nb,
                       p_unsafe=float(p_unsafe_ratio(na, nb, fa, fb, [(3, 1)])[0])))
        ex.append(dict(protocol="unweighted", nb=nb,
                       p_unsafe=float(p_unsafe_ratio(na, nb, fa, fb, [(1, 1)])[0])))
    ex = pd.DataFrame(ex)
    ex.to_csv(os.path.join(RESULTS, "compare_exact.csv"), index=False)
    print(f"exact part: {time.time() - t0:.0f}s")

    sim_csv = os.path.join(RESULTS, "compare_sim.csv")
    seq_csv = os.path.join(RESULTS, "compare_sequence.csv")
    if args.plot_only and os.path.exists(sim_csv) and os.path.exists(seq_csv):
        sim, seq = pd.read_csv(sim_csv), pd.read_csv(seq_csv)
    else:
        jobs = []
        for p in C.ORDER:
            for nb in args.sim_nb:
                for mode in ("attack", "silent"):
                    if p == "dolev_strong" and nb > 30:
                        continue  # signature chains: bits explode; steps are exact anyway
                    if p == "gradecast" and mode == "attack" and nb > 120:
                        continue  # O(F) iterations of O(n^3): minutes per trial
                    jobs += [(p, na, nb, fa, fb, mode, d) for d in range(args.draws)]
        jobs.sort(key=lambda j: -j[2])
        seq_jobs = [(p, carry, na, args.seq_nb, fa, fb, (3, 1), args.ell, d)
                    for p, carry in [("king", False), ("king_es", False), ("gradecast", False),
                                     ("gradecast", True), ("gc_king", True), ("gc_coin", True)]
                    for d in range(args.seq_draws)]
        t0 = time.time()
        rows, srows = [], []
        with mp.Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(C.simulate, jobs, chunksize=1), 1):
                rows.append(r)
                if i % 200 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} trials, {time.time() - t0:.0f}s", flush=True)
            for rs in pool.imap_unordered(C.sequence, seq_jobs, chunksize=1):
                srows += rs
        sim, seq = pd.DataFrame(rows), pd.DataFrame(srows)
        sim.to_csv(sim_csv, index=False)
        seq.to_csv(seq_csv, index=False)
    ran = sim[sim.safe]
    print(f"simulated: {len(ran)} runs inside their bound, all correct: {bool(ran.correct.all())}; "
          f"sequences: {len(seq)} instances, all correct: {bool(seq.correct.all())}")

    # ---- overview figure -------------------------------------------------------
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 9, "legend.frameon": False})
    COL = dict(zip(C.ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                             "#e87ba4", "#008300", "#6250d6", "#e34948"]))
    MK = dict(zip(C.ORDER, ["v", "s", "^", "o", "D", "P", "X", "*"]))
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axs[0, 0]
    groups = [("queen", "Byzantine < 1/4: Queen, Early-stopping King"),
              ("king", "Byzantine < 1/3: King, Gradecast, + king, + coin"),
              ("hybrid_king", "A crash-only: Hybrid King (3ρb + 2ρc < 1)"),
              ("dolev_strong", "Signatures, < 1/2: Dolev–Strong")]
    for p, lab in groups:
        g = ex[ex.protocol == p].sort_values("nb")
        ax.plot(g.nb, g.p_unsafe, color=COL[p], lw=2, label=lab)
    for p, lab, ls in (("paper_inverse", "ref: < 1/3 with paper's inverse weights", "--"),
                       ("unweighted", "ref: < 1/3 unweighted", ":")):
        g = ex[ex.protocol == p].sort_values("nb")
        ax.plot(g.nb, g.p_unsafe, color="#8a8984", lw=1.2, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1)
    ax.set_xlim(0, args.nb_max)
    ax.set_xlabel("Number of processes in set B")
    ax.set_ylabel("P(guarantee lost)  (log; signatures drop below 1e-7)")
    ax.set_title("(a) Resilience, each model with its optimal weights (exact)", loc="left")
    ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.42))

    def steps_panel(ax, mode, title):
        agg = ran[ran["mode"] == mode].groupby(["protocol", "nb"]).agg(
            steps=("steps", "mean"), bits=("bits", "mean")).reset_index()
        for p in C.ORDER:
            if p in ("king", "queen", "hybrid_king", "dolev_strong"):
                g = ex[ex.protocol == p].sort_values("nb")
                ax.plot(g.nb, g.steps, color=COL[p], lw=1.8, label=C.LABEL[p])
            else:
                g = agg[agg.protocol == p].sort_values("nb")
                ax.plot(g.nb, g.steps, color=COL[p], lw=1.8, marker=MK[p], ms=5, label=C.LABEL[p])
        ax.set_yscale("log")
        ax.set_xlim(0, args.nb_max)
        ax.set_xlabel("Number of processes in set B")
        ax.set_ylabel("synchronous steps to decide (mean)")
        ax.set_title(title, loc="left")
        return agg

    agg_att = steps_panel(axs[0, 1], "attack",
                          "(b) Mean steps over random fault draws; faulty processes run the\n"
                          "strongest attack we implemented (Gradecast: |B| ≤ 90 only)")
    axs[0, 1].legend(fontsize=7.5, loc="upper left", ncol=2)

    ax = axs[1, 0]
    for p in C.ORDER:
        g = agg_att[agg_att.protocol == p].sort_values("nb")
        ax.plot(g.nb, g.bits, color=COL[p], lw=1.8, marker=MK[p], ms=5, label=C.LABEL[p])
    ax.set_yscale("log")
    ax.set_xlim(0, args.nb_max)
    ax.set_xlabel("Number of processes in set B")
    ax.set_ylabel("bits sent by honest processes (mean)")
    ax.set_title("(c) Communication under attack", loc="left")

    ax = axs[1, 1]
    seq["label"] = seq.apply(lambda r: C.LABEL[r.protocol] + (" (memory)" if r.carry else ""), axis=1)
    SEQCOL = {"king": COL["king"], "king_es": COL["king_es"], "gradecast": COL["gradecast"],
              "gc_king": COL["gc_king"], "gc_coin": COL["gc_coin"]}
    for (p, carry), g in seq.groupby(["protocol", "carry"]):
        m = g.groupby("instance").cum_steps.mean()
        ax.plot(m.index, m.values, color=SEQCOL[p], lw=2, ls="-" if carry or p.startswith("king") else ":",
                marker=MK[p], ms=4, label=g.label.iloc[0])
    ax.set_xlabel("consensus instance")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("cumulative steps (mean)")
    ax.set_title(f"(d) Repeated consensus, |B| = {args.seq_nb}, inverse weights, stagger attack\n"
                 "(Early-stopping King: only fault draws inside its 1/4 bound)", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"All protocols on the Fig. 4 setup (|A| = {na}, fa = {args.fa}, fb = {args.fb}). "
                 "* = extra assumption: oracle coin, crash-only A, or unforgeable signatures",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "figures", "compare_overview.png")
    fig.savefig(out, dpi=200)
    print(out)

    # ---- scorecard ----------------------------------------------------------------
    nb0 = args.score_nb
    agg_sil = ran[ran["mode"] == "silent"].groupby(["protocol", "nb"]).steps.mean()
    att = agg_att.set_index(["protocol", "nb"])
    rows, cells = [], []
    for p in C.ORDER:
        e = ex[(ex.protocol == p) & (ex.nb == nb0)].iloc[0]
        st_att = e.steps if p in ("king", "queen", "hybrid_king", "dolev_strong") else \
            att.steps.get((p, nb0), float("nan"))
        st_sil = e.steps if p in ("king", "queen", "hybrid_king", "dolev_strong") else \
            agg_sil.get((p, nb0), float("nan"))
        bits = att.bits.get((p, nb0), float("nan"))
        rows.append(dict(protocol=C.LABEL[p], bound=C.BOUND_TEXT[p], p_unsafe=e.p_unsafe,
                         steps_attack=st_att, steps_silent=st_sil,
                         steps_worst=C.worst_steps(p, na, nb0, fa, fb), kbits_attack=bits / 1e3))
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(RESULTS, f"compare_scorecard_nb{nb0}.csv"), index=False)
    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.axis("off")
    cols = ["tolerates", "P(guarantee lost)", "mean steps,\nattack", "mean steps,\nsilent faults",
            "worst-case steps\n(proven bound)", "kbits, attack"]
    keys = ["bound", "p_unsafe", "steps_attack", "steps_silent", "steps_worst", "kbits_attack"]
    fmt = [str, lambda x: f"{x:.4f}" if x >= 1e-3 else f"{x:.1e}", lambda x: f"{x:.0f}",
           lambda x: f"{x:.0f}", lambda x: "unbounded" if x != x else f"{x:.0f}",
           lambda x: "n/a" if x != x else f"{x:,.0f}"]
    text = [[f(r[k]) for k, f in zip(keys, fmt)] for _, r in sc.iterrows()]
    tb = ax.table(cellText=text, rowLabels=list(sc.protocol), colLabels=cols, loc="center",
                  cellLoc="right")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.45)
    for j, k in enumerate(keys):
        if k == "bound":
            continue
        best = sc[k].min()
        for i, v in enumerate(sc[k]):
            if v == v and abs(v - best) < 1e-12:
                tb[i + 1, j].set_text_props(fontweight="bold")
                tb[i + 1, j].set_facecolor("#d9f2e8")
    ax.set_title(f"Scorecard at |B| = {nb0}: lower is better; best in bold. Each protocol uses its own "
                 "optimal weights (for the 1/4 protocols that means trusting only A).\n"
                 "* = extra assumption (oracle coin, not attacked here; crash-only A; unforgeable "
                 "signatures). Mean columns: 15 random fault draws.", loc="left", fontsize=9)
    fig.tight_layout()
    out2 = os.path.join(RESULTS, "figures", f"compare_scorecard_nb{nb0}.png")
    fig.savefig(out2, dpi=200)
    print(out2)
    print(sc.to_string(float_format=lambda x: f"{x:.4g}"))
    fin = seq[seq.instance == seq.instance.max()].groupby(["protocol", "carry"]).cum_steps.mean()
    print("\ncumulative steps after", args.ell, "instances:\n" + fin.round(0).to_string())


def cmd_fig5(args):
    """Garg & Bridgman Fig. 5 (rounds King needs) + weighted Gradecast.

    One "round" = one loop iteration (3 message exchanges for King and for
    Gradecast alike), as in the paper. Uniform weights = the unweighted
    algorithm (King: f+1 phases; Gradecast: Ben-Or-Dolev-Hoch ByzConsensus).
      King                       alpha_rho, always (no early stopping)
      Gradecast, worst case      cap_rho
      Gradecast, expected        E[min(F+3, cap_rho)] over the failure model
                                 (fa, fb), safe runs only (failed weight < 1/3)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from byzantine.fig4_compare import analytic
    from byzantine.weight_assignment import LABELS, SCHEMES

    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    nbs = [nb for nb in range(args.nb_max + 1) if (args.na + nb) % 3 == 0]
    an = pd.DataFrame([analytic(s, args.na, nb, args.fa, args.fb) for s in SCHEMES for nb in nbs])
    an["king_rounds"] = an.anchor
    an["gc_worst_rounds"] = an.gc_cap
    an["gc_expected_rounds"] = an.gc_bound_steps / 3
    an.to_csv(os.path.join(RESULTS, "fig5.csv"), index=False)
    sim_csv = os.path.join(RESULTS, "fig4_compare_sim.csv")
    sim = pd.read_csv(sim_csv) if os.path.exists(sim_csv) else None

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 10, "legend.frameon": False})
    style = {  # same encodings as the paper / our Fig. 4
        "only_a": dict(color="#2a78d6", ls="--", lw=2, marker=None),
        "uniform": dict(color="#eb6834", ls="none", marker="x", ms=4, mew=1.1),
        "inverse_failure": dict(color="#1baf7a", ls="none", marker="o", ms=4, mfc="none", mew=1.1),
        "not_failing": dict(color="#eda100", ls="none", marker="*", ms=5),
    }
    label = dict(LABELS, uniform="Uniform (= unweighted algorithm)")
    ymax = 145

    # Figure 1: three panels, same axes as the paper's Fig. 5
    panels = [("king_rounds", "(a) Weighted King: rounds required (paper's Fig. 5)"),
              ("gc_worst_rounds", "(b) Weighted Gradecast: rounds required, worst case"),
              ("gc_expected_rounds", f"(c) Weighted Gradecast: expected rounds\n"
                                     f"(early stopping, fa = {args.fa}, fb = {args.fb})")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        for s in SCHEMES:
            g = an[an.scheme == s]
            ax.plot(g.nb, g[col], label=label[s], **style[s])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, args.nb_max)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Number of processes in set B")
    axes[0].set_ylabel("Number of rounds")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Rounds to decide vs |B|  (|A| = {args.na}; one round = one loop iteration = "
                 f"3 message exchanges, for both algorithms)", x=0.07, ha="left", fontsize=10)
    fig.tight_layout()
    out1 = os.path.join(RESULTS, "figures", "fig5_king_and_gradecast.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")

    # Figure 2: one panel per weighting, King vs Gradecast head to head
    KING, GCW, GCE = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, s in zip(axes.flat, SCHEMES):
        g = an[an.scheme == s]
        ax.plot(g.nb, g.king_rounds, color=KING, lw=2, label="Weighted King (always α_ρ)")
        ax.plot(g.nb, g.gc_worst_rounds, color=GCW, lw=1.8, ls="--",
                label="Weighted Gradecast, worst case (cap_ρ)")
        ax.plot(g.nb, g.gc_expected_rounds, color=GCE, lw=1.8,
                label="Weighted Gradecast, expected (early stopping)")
        if sim is not None:
            m = sim[(sim.scheme == s) & (sim.adversary == "stagger")].groupby("nb").steps.mean() / 3
            ax.plot(m.index, m.values, ls="none", marker="o", ms=5, mfc="white", mec=GCE, mew=1.3,
                    label="Gradecast simulated vs stagger attack (mean)")
        ax.set_title(label[s], loc="left", fontsize=10)
        ax.set_xlim(0, args.nb_max)
        ax.set_ylim(0, 12 if s == "only_a" else ymax)
    for ax in axes[1]:
        ax.set_xlabel("Number of processes in set B")
    for ax in axes[:, 0]:
        ax.set_ylabel("Number of rounds")
    axes[0, 1].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Weighted King vs weighted Gradecast, rounds to decide (|A| = {args.na}, "
                 f"fa = {args.fa}, fb = {args.fb}; expected values over safe runs)",
                 x=0.07, ha="left", fontsize=10)
    fig.tight_layout()
    out2 = os.path.join(RESULTS, "figures", "fig5_king_vs_gradecast_by_weighting.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    print(out1, out2, sep="\n")
    print(an[an.nb.isin([0, 30, 90, 198, 399])]
            .pivot(index="nb", columns="scheme",
                   values=["king_rounds", "gc_worst_rounds", "gc_expected_rounds"])
            .round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sweep", cmd_sweep), ("boundary", cmd_boundary)):
        p = sub.add_parser(name)
        p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
        p.add_argument("--n", nargs="+", type=int, default=[4, 7, 10, 13, 16, 25, 31, 49])
        p.add_argument("--adversaries", nargs="+", default=list(ADVERSARIES))
        p.add_argument("--inputs", nargs="+", default=list(INPUT_MODES))
        p.add_argument("--placements", nargs="+", default=list(PLACEMENTS))
        p.add_argument("--trials", type=int, default=10)
        p.add_argument("--beyond", type=int, default=0)
        p.add_argument("--out", default="sweep.csv")
        p.set_defaults(fn=fn)
    sub.add_parser("modelcheck").set_defaults(fn=cmd_modelcheck)
    sub.add_parser("plot").set_defaults(fn=cmd_plot)
    p = sub.add_parser("fig4")
    p.add_argument("--na", type=int, default=6)
    p.add_argument("--fa", default="0.1")
    p.add_argument("--fb", default="0.3")
    p.add_argument("--nb-max", type=int, default=400)
    p.add_argument("--ties", choices=["exact", "float"], default="exact",
                   help="exact: weight == 1/3 counts as unsafe; float: reproduce the paper's rounding")
    p.set_defaults(fn=cmd_fig4)
    p = sub.add_parser("fig4compare")
    p.add_argument("--na", type=int, default=6)
    p.add_argument("--fa", default="0.1")
    p.add_argument("--fb", default="0.3")
    p.add_argument("--nb-max", type=int, default=400)
    p.add_argument("--sim-nb", nargs="+", type=int, default=[0, 6, 12, 18, 30, 42, 60])
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--plot-only", action="store_true", help="reuse results/fig4_compare_sim.csv")
    p.set_defaults(fn=cmd_fig4compare)
    p = sub.add_parser("fig5")
    p.add_argument("--na", type=int, default=6)
    p.add_argument("--fa", default="0.1")
    p.add_argument("--fb", default="0.3")
    p.add_argument("--nb-max", type=int, default=400)
    p.set_defaults(fn=cmd_fig5)
    p = sub.add_parser("improve")
    p.add_argument("--na", type=int, default=6)
    p.add_argument("--fa", default="0.1")
    p.add_argument("--fb", default="0.3")
    p.add_argument("--nb-max", type=int, default=400)
    p.add_argument("--sim-nb", nargs="+", type=int, default=[0, 12, 30, 60, 90])
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--plot-only", action="store_true", help="reuse results/improve_protocols.csv")
    p.set_defaults(fn=cmd_improve)
    p = sub.add_parser("compare")
    p.add_argument("--na", type=int, default=6)
    p.add_argument("--fa", default="0.1")
    p.add_argument("--fb", default="0.3")
    p.add_argument("--nb-max", type=int, default=400)
    p.add_argument("--sim-nb", nargs="+", type=int, default=[0, 12, 30, 60, 90, 150, 240, 399])
    p.add_argument("--draws", type=int, default=15)
    p.add_argument("--seq-nb", type=int, default=60)
    p.add_argument("--ell", type=int, default=20)
    p.add_argument("--seq-draws", type=int, default=8)
    p.add_argument("--score-nb", type=int, default=90)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--plot-only", action="store_true")
    p.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
