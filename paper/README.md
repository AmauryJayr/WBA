# Term paper: Weighted Byzantine Agreement on Gradecast

- `paper.tex`, `references.bib` → `paper.pdf` (IEEEtran conference, two columns).
  Build: `tectonic -X compile paper.tex` (or `latexmk -pdf paper.tex`; needs `newtx`, `algorithmicx`).
- `scripts/messages.py` → `data/messages.csv` (Queen / King / Gradecast message and bit counts, ~15 s).
- `scripts/make_figures.py` → `figures/*.pdf`, `data/analytic.csv`, `data/numbers.txt`.
  It uses the repo's own analysis code (`byzantine/weight_assignment.py`, `byzantine/fig4_compare.py`)
  and the simulated runs in `results/fig4_compare_sim.csv`
  (`python run_experiments.py fig4compare --sim-nb 0 6 12 18 30 42 60 90 120 --draws 20`).
- `scripts/example13.py` → `data/example13.txt` (the 13-process example in Section III).
- `scripts/rare_sim.py` → `data/rare_sim.csv` (240 simulated WGC runs with rare faults for Fig. 2(c), ~21 min on 7 cores; run before `make_figures.py`).
- `scripts/ideal_ratio.py` → `data/ideal_ratio.txt` (the weight formula vs. the exact optimal ratio, Section V).
- `scripts/weight_update.py` → `data/weight_update.txt` (weight updates over 100 instances, Section V-F; ~40 min on 8 cores).
- `scripts/weight_update_fig.py` → `figures/weight_update.pdf` (Fig. 5, from `results/weight_update.csv`; checks it against `data/weight_update.txt` first).
- `data/sweep_summary.txt`: correctness sweep quoted in Section IV
  (`python run_experiments.py sweep --protocols queen king gradecast --n 4 7 10 13 16 25 31 49
  --adversaries silent crash random flip equivocate split stagger --trials 2`).

Run all scripts from the repository root. Every number in the paper is in `data/numbers.txt`,
`data/messages.csv`, `data/example13.txt`, `data/ideal_ratio.txt`, `data/weight_update.txt`, `data/rare_sim.csv` or `data/sweep_summary.txt`.
