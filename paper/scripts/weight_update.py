"""Repeated consensus with weight updates: re-run of `run_experiments.py weightupdate`
without the plot. Same jobs, same parameters and the same summary numbers.

Run from the repo root:
    python paper/scripts/weight_update.py [--trials 200] [--jobs 7]
-> paper/data/weight_update.txt (summary) and paper/data/weight_update_rows.csv.gz (raw rows)
"""
import argparse
import csv
import gzip
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
from byzantine.improvements import int_weights  # noqa: E402
from byzantine.weight_update import BOUND, survival_exact, trial  # noqa: E402

ADVS = ["silent", "equivocate", "split", "flip"]
METHODS = ["garg_queen", "garg_king", "gc_bad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--na", type=int, default=6)
    ap.add_argument("--nb", type=int, default=30)
    ap.add_argument("--qa", type=float, default=0.005)
    ap.add_argument("--qb", type=float, default=0.015)
    ap.add_argument("--ell", type=int, default=100)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="paper/data/weight_update.txt")
    a = ap.parse_args()

    w = int_weights(a.na, a.nb, (3, 1))  # inverse failure probability, as in the repo
    jobs = [(m, adv, a.na, a.nb, a.qa, a.qb, w, a.ell, s)
            for m in METHODS for adv in ADVS for s in range(a.trials)]
    jobs.sort(key=lambda j: j[0] != "gc_bad")
    t0 = time.time()
    rows = []
    with mp.Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(trial, jobs, chunksize=1), 1):
            rows.extend(r)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} trials, {time.time() - t0:.0f}s", flush=True)

    keys = sorted({k for r in rows for k in r})
    with gzip.open(a.out.replace(".txt", "_rows.csv.gz"), "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    sim = [r for r in rows if r.get("correct") is not None]
    out = [f"setup: |A|={a.na}, |B|={a.nb}, weights 3:1 (inverse failure), per-instance failure "
           f"probability qA={a.qa}, qB={a.qb}, {a.ell} instances, {a.trials} trials per cell",
           f"simulated instances: {len(sim)}, all correct: {all(r['correct'] for r in sim)}",
           f"wall time: {time.time() - t0:.0f}s"]
    for m in ("none_queen", "none_king"):
        ex = survival_exact(a.na, a.nb, a.qa, a.qb, w, BOUND[m], a.ell)
        med = next((k for k, p in enumerate(ex, 1) if p < 0.5), None)
        out.append(f"{m:10s} (exact)  survive_end={ex[-1]:.3f}  median_lost_at={med}")
    out.append("adversary   method      survive_end  median_lost_at  mean_removed  mean_steps")
    for adv in ADVS:
        for m in METHODS:
            g = [r for r in rows if r["method"] == m and r["adversary"] == adv]
            lost = {}
            for r in g:
                if not r["alive"]:
                    lost[r["seed"]] = min(lost.get(r["seed"], 10**9), r["instance"])
            surv = 1 - len(lost) / a.trials
            med = statistics.median(lost.values()) if len(lost) > a.trials / 2 else None
            rem = {}
            for r in g:
                rem[r["seed"]] = max(rem.get(r["seed"], 0), r["removed"])
            steps = [r["steps"] for r in g if r.get("steps") is not None]
            out.append(f"{adv:11s} {m:11s} {surv:11.3f}  {str(med):14s}  "
                       f"{statistics.mean(rem.values()):12.2f}  {statistics.mean(steps):10.1f}")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
