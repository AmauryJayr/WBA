"""Simulated WGC runs with rare faults (Fig. 2(c)).

Same setup and the same simulate() as `run_experiments.py fig4compare`, but with the
per-instance failure rates of the weight-update test: fA = 0.005, fB = 0.015.
Inverse-failure weights (3:1), stagger attack, 20 failure draws per |B|.

Run from the repo root:  python paper/scripts/rare_sim.py [--jobs 7]
-> paper/data/rare_sim.csv
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.getcwd())
from byzantine.fig4_compare import simulate  # noqa: E402

NA, FA, FB = 6, "0.005", "0.015"
NBS = [0, 6, 12, 18, 30, 42, 60, 90, 120, 201, 300, 399]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--nb", type=int, nargs="+", default=NBS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="paper/data/rare_sim.csv")
    a = ap.parse_args()
    jobs = [("inverse_failure", NA, nb, FA, FB, "stagger", d) for nb in a.nb for d in range(a.draws)]
    jobs.sort(key=lambda j: -j[2])  # biggest first
    t0 = time.time()
    rows = []
    with mp.Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(simulate, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["nb", "draw"])
    df.to_csv(a.out, index=False)
    print(f"all correct: {bool(df.correct.all())}")
    print(df.assign(rounds=df.steps / 3).groupby("nb").agg(
        mean=("rounds", "mean"), min=("rounds", "min"), max=("rounds", "max"),
        F=("faulty_pos", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
