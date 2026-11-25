"""Message and bit complexity of weighted Queen, King and Gradecast consensus.

Uniform weights (= the unweighted algorithms), each protocol at its largest
tolerated t for the given n. Two scenarios per (protocol, n):
  fault-free   no faulty process (Gradecast stops early: 2 iterations)
  attacked     t faulty processes; `split` for Queen/King (their message counts
               do not depend on the adversary), `stagger` for Gradecast (forces
               its worst case of t + 1 iterations)
Counts are for messages sent by honest processes to other processes, as
measured by the simulator (byzantine/network.py).

Run from the repo root:  python paper/scripts/messages.py
Writes paper/data/messages.csv.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from byzantine.experiment import TrialConfig, run_trial  # noqa: E402
from byzantine.protocols import PROTOCOLS  # noqa: E402

NS = [4, 7, 10, 13, 16, 19, 25, 31, 40, 49]
ATTACK = {"queen": "split", "king": "split", "gradecast": "stagger"}


def main():
    rows = []
    for proto in ("queen", "king", "gradecast"):
        for n in NS:
            t = PROTOCOLS[proto].max_faults(n)
            for scen, F, adv in (("fault-free", 0, "silent"), ("attacked", t, ATTACK[proto])):
                r = run_trial(TrialConfig(proto, n, t, F, adv, "split", "first", seed=1))
                assert r["correct"], r
                rows.append(dict(protocol=proto, n=n, t=t, scenario=scen, faulty=F,
                                 steps=r["steps"], honest_msgs=r["honest_msgs"],
                                 honest_bits=r["honest_bits"]))
                print(rows[-1], flush=True)
    os.makedirs("paper/data", exist_ok=True)
    pd.DataFrame(rows).to_csv("paper/data/messages.csv", index=False)


if __name__ == "__main__":
    main()
