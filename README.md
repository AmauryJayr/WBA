# Weighted Byzantine Agreement — simulation & evaluation framework

EE 382N term project (due Nov 19). This stage contains the **unweighted baselines**
(Queen, Phase King). It also has the test and evaluation harness that the weighted
Queen/King (Garg & Bridgman) and a weighted Gradecast protocol will reuse.

## Layout

```
byzantine/
  network.py        lock-step synchronous simulator, rushing omniscient adversary, metrics
  protocols/
    queen.py        Berman–Garay Queen       n > 4f, f+1 rounds x 2 phases
    king.py         Berman–Garay–Perry King  n > 3f, f+1 rounds x 3 phases
  adversaries.py    silent, crash, random (fuzz), flip, equivocate, split (adaptive)
  experiment.py     one trial = (protocol, n, f, adversary, inputs, fault placement, seed)
  modelcheck.py     exhaustive search over ALL adversary behaviours for small n
tests/              python -m unittest discover -s tests -v
run_experiments.py  sweep | boundary | modelcheck | plot  ->  results/*.csv, results/figures/
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

## Where the weighted versions plug in

Both protocols are written to mirror Fig. 1 and Fig. 2 of the weighted paper, with every
weight equal to 1. To make them weighted, keep the code structure and change only the
vote counting and the thresholds:

| | unweighted | weighted |
|---|---|---|
| Queen majority | `ones > n/2` | `s1 > 1/2` |
| Queen keep-own | `count > n/2 + f` | `myweight > 3/4` |
| King adopt | `count >= n - f` | `s >= 2/3` |
| King support | `count > f` | `s > 1/3` |
| rounds | `f + 1` | anchor ρ (fewest heaviest processes with total weight > α) |

For the weighted Gradecast protocol, do the same with Ben-Or/Dolev/Hoch's thresholds:
`n - t` becomes weight ≥ 1 − α and `t + 1` becomes weight > α. A process in BAD stops
contributing any weight.
