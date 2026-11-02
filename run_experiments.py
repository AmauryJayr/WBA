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
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
