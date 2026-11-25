"""The 13-process example of Section III: three processes of weight 0.25, ten of
weight 0.025, rho = 0.3, all ten light processes faulty.

Run from the repo root:  python paper/scripts/example13.py  -> paper/data/example13.txt
"""

import os
import sys

sys.path.insert(0, os.getcwd())
from byzantine.experiment import TrialConfig, run_trial  # noqa: E402
from byzantine.protocols import Gradecast, King  # noqa: E402

W = (10,) * 3 + (1,) * 10  # 0.25 and 0.025 after normalisation
RHO = "3/10"
lines = [f"alpha_rho = {King(13, 0, W, RHO).anchor}, cap_rho = {Gradecast(13, 0, W, RHO).cap}"]
for proto in ("king", "gradecast"):
    for adv in ("silent", "split", "stagger"):
        r = run_trial(TrialConfig(proto, 13, 0, 10, adv, "split", seed=1, weights=W, rho=RHO,
                                  faulty=tuple(range(3, 13))))
        lines.append(f"{proto:9s} {adv:8s} correct={r['correct']} rounds={r['steps'] // 3}")
os.makedirs("paper/data", exist_ok=True)
open("paper/data/example13.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
