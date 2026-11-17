# Weighted Byzantine Agreement — simulation & evaluation framework

EE 382N term project (due Nov 19). This stage contains the **unweighted baselines**
(Queen, Phase King). It also has the test and evaluation harness that the weighted
Queen/King (Garg & Bridgman) and a weighted Gradecast protocol will reuse.

## Layout

```
byzantine/
  network.py        lock-step synchronous simulator, rushing omniscient adversary, metrics
  protocols/        all weighted (weights=, rho=); uniform weights = the unweighted originals
    base.py         exact integer-weight thresholds, anchor α_ρ, heaviest-first coordinators
    queen.py        Berman–Garay Queen       ρ < 1/4, α_ρ rounds x 2 phases
    king.py         Berman–Garay–Perry King  ρ < 1/3, α_ρ rounds x 3 phases
    king_es.py      early-stopping King: ρ < 1/4, ≤ (first honest king) + 2 rounds
    hybrid_king.py  King with crash-only + Byzantine faults: 3ρ_b + 2ρ_c < 1
    gradecast.py    Ben-Or–Dolev–Hoch ByzConsensus, weighted: ρ < 1/3,
                    ≤ min(F+3, cap) iterations x 3 steps (early stopping);
                    numpy-vectorised; BAD sets can carry across instances.
                    Variants: gc_king (rotating king step), gc_coin (oracle common coin)
    dolev_strong.py signatures (HMAC-modelled): ρ < 1/2, R + 1 rounds
  adversaries.py    silent, crash, random (fuzz), flip, equivocate, split (adaptive),
                    stagger (spends one faulty process per Gradecast iteration),
                    sig (equivocation + late reveal with valid faulty signatures),
                    hybrid (crash-only set + split for the rest)
  weight_assignment.py  exact P(failed weight ≥ 1/3) for Garg–Bridgman Fig. 4
  fig4_compare.py   weighted King vs weighted Gradecast on the Fig. 4 setup
  experiment.py     one trial = (protocol, n, f, adversary, inputs, fault placement, seed)
  modelcheck.py     exhaustive search over ALL adversary behaviours for small n
tests/              python -m unittest discover -s tests -v
run_experiments.py  sweep | boundary | modelcheck | plot | fig4 | fig4compare | fig5 | improve | compare | idealratio | weightupdate
                    ->  results/*.csv, results/figures/
```

## Evaluation method

1. **Property checking across many runs.** Every trial checks agreement, validity and
   termination. Each sweep covers n, f, six adversary strategies, four input patterns
   (all-0, all-1, alternating split, random) and three fault placements (faulty
   coordinators first, honest coordinator first, random).
2. **Exhaustive small-scope model checking.** For n ≤ 5, every Byzantine behaviour
   is enumerated, and the real protocol code runs one round at a time. This shows
   the protocols are correct for those n, and it finds counterexamples once f passes
   the bound (Queen at n = 4f, King at n = 3f).
3. **Beyond-the-bound runs.** The protocols run with f past their proven bound, to
   measure how often each adversary actually breaks them.
4. **Cost metrics.** Synchronous steps, number of messages and bits (sent by honest
   processes only), and wall-clock time.

## Results so far

| Experiment | Outcome |
|---|---|
| `sweep` (n ≤ 49, 23,760 trials) | 100% agreement/validity/termination for both protocols, every adversary |
| `boundary` (f past the bound) | equivocate/split attacks break ~60% of runs; crash ~10–20% |
| `modelcheck` | Queen n=5,f=1 and King n=4,f=1 safe under all behaviours; Queen n=4,f=1 and King n=3,f=1 have counterexamples |
| messages | Queen (f+1)(n²−1), King (f+1)(2n²−n−1); exact counts asserted in the tests |

## Weighted thresholds

ρ is the largest fraction of total weight that may be faulty. Thresholds are written in
terms of ρ, so uniform weights with ρ = f/n give the unweighted algorithms exactly. The
paper's constants are the limits as ρ → 1/4 (Queen) and ρ → 1/3 (King).

| | unweighted | weighted (code) | Garg & Bridgman |
|---|---|---|---|
| Queen majority | `ones > n/2` | `s1 > 1/2` | `s1 > 1/2` |
| Queen keep-own | `count > n/2 + f` | `myweight > 1/2 + ρ` | `myweight > 3/4` |
| King adopt | `count >= n - f` | `s >= 1 - ρ` | `s >= 2/3` |
| King support | `count > f` | `s > ρ` | `s > 1/3` |
| Gradecast strong | `count >= n - t` | `s >= 1 - ρ` | |
| Gradecast weak | `count >= t + 1` | `s > ρ` | |
| rounds | `f + 1` | anchor α_ρ: fewest heaviest processes with weight > ρ | |

Weighted Gradecast's iteration cap is set by the *lightest* processes, not the heaviest:
each faulty process can cause one disagreement before it is expelled. The cap is the
number of lightest positive-weight processes that fit in ρ, plus 1. Processes in BAD and
zero-weight processes contribute no weight.

## Weighted results (Fig. 4 setup: |A| = 6, fa = 0.1, fb = 0.3)

- `fig4`: exact recreation of Fig. 4. The paper's scattered points come from floating-point
  ties at exactly 1/3. `--ties float` reproduces them.
- `fig4compare`: weighted King and weighted Gradecast lose their guarantee in exactly the
  same runs (failed weight ≥ 1/3), so they share Fig. 4. They differ in time to decide.
  See `results/figures/fig4_king_vs_gradecast.png`.
- `improve`: tries the heuristics. It finds the exact optimal weight ratio wA/wB, and it
  compares early-stopping King (`king_es`, ρ < 1/4) and Gradecast + rotating king
  (`gc_king`, ρ < 1/3) against King and Gradecast. Outputs: `results/improve_*.csv` and
  `results/figures/improvements.png`.
- `compare`: all 8 protocols on the Fig. 4 setup. Each uses its own optimal weights and
  faces its strongest attack. Covers resilience, time, communication and repeated
  consensus. Outputs: `results/compare_*.csv`, `results/figures/compare_overview.png` and
  `compare_scorecard_nb90.png`. About 15 minutes; `--plot-only` replots.
