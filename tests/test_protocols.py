"""Correctness tests. Run with:  python -m unittest discover -s tests -v"""

import itertools
import random
import unittest
from fractions import Fraction

from byzantine.adversaries import ADVERSARIES
from byzantine.experiment import INPUT_MODES, PLACEMENTS, TrialConfig, run_trial
from byzantine.modelcheck import check
from byzantine.protocols import PROTOCOLS


class WithinBound(unittest.TestCase):
    """Agreement, validity and termination under every adversary, input pattern
    and fault placement, for every n up to 16 and every f the protocol tolerates."""

    def test_all_adversaries(self):
        for proto, n in itertools.product(PROTOCOLS, range(1, 17)):
            for f in range(PROTOCOLS[proto].max_faults(n) + 1):
                for adv, inp, pl, seed in itertools.product(ADVERSARIES, INPUT_MODES, PLACEMENTS, range(3)):
                    row = run_trial(TrialConfig(proto, n, f, f, adv, inp, pl, seed))
                    with self.subTest(**{k: row[k] for k in ("protocol", "n", "f", "adversary", "inputs", "placement", "seed")}):
                        self.assertTrue(row["terminated"])
                        self.assertTrue(row["agreement"], row)
                        self.assertTrue(row["validity"], row)

    def test_round_and_message_counts(self):
        for n, f in [(9, 2), (13, 3)]:
            q = run_trial(TrialConfig("queen", n, f, 0, "silent", "random"))
            self.assertEqual(q["steps"], 2 * (f + 1))
            # phase 1: all-to-all, phase 2: queen to everyone else
            self.assertEqual(q["honest_msgs"], (f + 1) * (n * (n - 1) + (n - 1)))
            k = run_trial(TrialConfig("king", n, f, 0, "silent", "random"))
            self.assertEqual(k["steps"], 3 * (f + 1))
            self.assertEqual(k["honest_msgs"], (f + 1) * (2 * n * (n - 1) + (n - 1)))


BOUND = {"queen": Fraction(1, 4), "king": Fraction(1, 3), "king_es": Fraction(1, 4),
         "gradecast": Fraction(1, 3), "gc_king": Fraction(1, 3), "gc_coin": Fraction(1, 3),
         "hybrid_king": Fraction(1, 3), "dolev_strong": Fraction(1, 2)}


def random_weighted_system(rng, proto):
    """Random integer weights (some zero) and a random faulty set whose weight
    is strictly below the protocol's bound; rho = that exact faulty fraction."""
    n = rng.randint(4, 12)
    w = [rng.choice([0, 1, 1, 2, 3, 5, 8]) for _ in range(n)]
    if sum(w) == 0:
        w[0] = 1
    W = sum(w)
    order = list(range(n))
    rng.shuffle(order)
    faulty, fw = [], 0
    for p in order:
        if Fraction(fw + w[p], W) < BOUND[proto] and rng.random() < 0.7:
            faulty.append(p)
            fw += w[p]
    if len(faulty) == n:  # keep at least one honest process
        faulty.pop()
    return n, tuple(w), tuple(sorted(faulty)), str(Fraction(fw, W))


class Weighted(unittest.TestCase):
    def test_uniform_weights_reduce_to_unweighted(self):
        for n in range(1, 30):
            for proto in PROTOCOLS:
                for f in range(PROTOCOLS[proto].max_faults(n) + 1):
                    p = PROTOCOLS[proto](n, f)
                    self.assertEqual(p.anchor, f + 1)
                    self.assertEqual(p.by_weight[: f + 1], list(range(f + 1)))
                    if proto == "gradecast":
                        self.assertEqual(p.cap, f + 1)

    def test_more_than_a_third_faulty_by_count(self):
        # 10 processes, 4 faulty light ones (weight 1) and 6 heavy honest ones
        # (weight 3): 40% faulty by count, but only 4/22 < 1/4 of the weight.
        w = (1, 1, 1, 1, 3, 3, 3, 3, 3, 3)
        for proto, adv, inp, seed in itertools.product(PROTOCOLS, ADVERSARIES, INPUT_MODES, range(3)):
            row = run_trial(TrialConfig(proto, 10, 3, 4, adv, inp, seed=seed,
                                        weights=w, rho="4/22", faulty=(0, 1, 2, 3)))
            with self.subTest(protocol=proto, adversary=adv, inputs=inp, seed=seed):
                self.assertTrue(row["correct"], row)

    def test_random_weighted_systems(self):
        rng = random.Random(382)
        for proto in PROTOCOLS:
            for trial in range(40):
                n, w, faulty, rho = random_weighted_system(rng, proto)
                for adv, inp in itertools.product(ADVERSARIES, INPUT_MODES):
                    row = run_trial(TrialConfig(proto, n, 0, len(faulty), adv, inp, seed=trial,
                                                weights=w, rho=rho, faulty=faulty))
                    with self.subTest(protocol=proto, w=w, faulty=faulty, adversary=adv, inputs=inp):
                        self.assertTrue(row["correct"], row)

    def test_stagger_reaches_the_bound_on_gradecast(self):
        # Unweighted: the paper's bound of t+1 iterations is reached.
        for n, f in [(10, 3), (13, 4), (31, 10)]:
            row = run_trial(TrialConfig("gradecast", n, f, f, "stagger", "split", "random", 1))
            self.assertTrue(row["correct"])
            self.assertEqual(row["steps"], 3 * (f + 1))
        # Weighted (Fig. 4 setup, optimal weights): the proven bound is
        # min(F + 3, cap) iterations (F = positive-weight faulty processes). No
        # run may exceed it; the (heuristic) attack must reach it in most runs.
        hits = runs = 0
        from byzantine.compare import best_config, draw_faulty, is_safe
        from byzantine.protocols import Gradecast
        for nb in (12, 30):
            w, rho, rc, _ = best_config("gradecast", 6, nb, 0.1, 0.3)
            cap = Gradecast(6 + nb, 0, w, rho).cap
            for draw in range(4):
                faulty = draw_faulty(6, nb, 0.1, 0.3, (nb, draw))
                if not is_safe("gradecast", w, rho, rc, faulty, 6):
                    continue
                F = sum(1 for i in faulty if w[i] > 0)
                row = run_trial(TrialConfig("gradecast", 6 + nb, 0, len(faulty), "stagger", "split",
                                            seed=draw, weights=w, rho=str(rho), faulty=tuple(faulty)))
                with self.subTest(nb=nb, draw=draw):
                    self.assertTrue(row["correct"])
                    self.assertLessEqual(row["steps"], 3 * min(F + 3, cap))
                runs += 1
                hits += row["steps"] == 3 * min(F + 3, cap)
        self.assertGreater(hits, runs / 2)

    def test_vectorised_gradecast_matches_loop_reference(self):
        from byzantine import experiment
        from byzantine.protocols import PROTOCOLS
        from tests.reference_gradecast import RefGradecast, RefGradecastKing

        def run(proto_cls, cfg):
            saved = PROTOCOLS[cfg.protocol]
            PROTOCOLS[cfg.protocol] = proto_cls
            try:
                r = experiment.run_trial(cfg)
            finally:
                PROTOCOLS[cfg.protocol] = saved
            return r["steps"], r["honest_msgs"], r["honest_bits"], r["decided"], r["correct"]

        rng = random.Random(11)
        cfgs = [TrialConfig(p, n, f, f, adv, inp, pl, s)
                for p in ("gradecast", "gc_king") for n, f in [(4, 1), (10, 3), (13, 4)]
                for adv in ADVERSARIES for inp in ("split", "random") for pl in ("first", "random")
                for s in range(1)]
        for p in ("gradecast", "gc_king"):
            for t in range(15):
                n, w, faulty, rho = random_weighted_system(rng, p)
                cfgs += [TrialConfig(p, n, 0, len(faulty), adv, "split", seed=t, weights=w,
                                     rho=rho, faulty=faulty) for adv in ADVERSARIES]
        for cfg in cfgs:
            ref = {"gradecast": RefGradecast, "gc_king": RefGradecastKing}[cfg.protocol]
            with self.subTest(cfg=cfg):
                self.assertEqual(run(PROTOCOLS[cfg.protocol], cfg), run(ref, cfg))


class ModelExtensions(unittest.TestCase):
    def test_signatures_tolerate_almost_half(self):
        # 4 of 9 faulty (44%): impossible without signatures.
        for adv, inp, s in itertools.product(["sig", "split", "silent", "random"], INPUT_MODES, range(3)):
            row = run_trial(TrialConfig("dolev_strong", 9, 4, 4, adv, inp, "random", s))
            with self.subTest(adversary=adv, inputs=inp, seed=s):
                self.assertTrue(row["correct"], row)

    def test_hybrid_crash_and_byzantine(self):
        # n = 11: 2 crash-only + 2 Byzantine. 3*2 + 2*2 = 10 < 11, but plain
        # King would need n > 3*4 = 12.
        for inp, s in itertools.product(INPUT_MODES, range(10)):
            for faulty, crash in [((0, 1, 2, 3), (0, 1)), ((5, 6, 9, 10), (9, 10))]:
                row = run_trial(TrialConfig("hybrid_king", 11, 2, 4, "hybrid", inp, seed=s,
                                            rho="2/11", rho_crash="2/11", faulty=faulty, crash=crash))
                with self.subTest(inputs=inp, seed=s, faulty=faulty):
                    self.assertTrue(row["correct"], row)

    def test_repeated_consensus_with_memory(self):
        from byzantine.improvements import run_sequence
        rng = random.Random(5)
        for proto in ("gradecast", "gc_king", "gc_coin"):
            for trial in range(5):
                n, w, faulty, rho = random_weighted_system(rng, proto)
                rows = run_sequence(proto, w, rho, list(faulty), "stagger", 6, trial)
                with self.subTest(protocol=proto, w=w, faulty=faulty):
                    self.assertTrue(all(r["correct"] for r in rows), rows)


class WeightUpdate(unittest.TestCase):
    """Detection used by the weight update never accuses an honest process."""

    def test_detection_is_sound(self):
        from byzantine.experiment import make_inputs
        from byzantine.network import Simulation
        rng = random.Random(11)
        for proto in ("queen", "king", "gradecast"):
            for trial in range(30):
                n, w, faulty, rho = random_weighted_system(rng, proto)
                for adv in ADVERSARIES:
                    P = PROTOCOLS[proto](n, 0, weights=w, rho=rho)
                    res = Simulation(P, ADVERSARIES[adv](), make_inputs("random", n, rng),
                                     list(faulty), seed=trial).run()
                    found = {p: set(P.detected.get(p, ())) | set(getattr(P, "final_bad", {}).get(p, ()))
                             for p in range(n)}
                    with self.subTest(protocol=proto, w=w, faulty=faulty, adversary=adv):
                        self.assertTrue(res.correct)
                        for p, s in found.items():
                            if p not in faulty:
                                self.assertLessEqual(set(s), set(faulty))

    def test_update_keeps_honest_and_stays_correct(self):
        from byzantine.improvements import int_weights
        from byzantine.weight_update import trial
        w = int_weights(6, 12, (3, 1))
        for method in ("garg_queen", "garg_king", "gc_bad"):
            for adv in ("silent", "random", "attack"):
                for seed in range(3):
                    rows = trial((method, adv, 6, 12, 0.02, 0.05, w, 25, seed))  # asserts inside
                    with self.subTest(method=method, adversary=adv, seed=seed):
                        self.assertTrue(all(r.get("correct", True) for r in rows if r["alive"]))


class ModelCheck(unittest.TestCase):
    """Exhaustive search over all adversary behaviours for small systems."""

    def test_queen_n5_f1_every_placement(self):
        for z in range(5):
            self.assertTrue(check("queen", 5, 1, [z]).ok)

    def test_king_n4_f1_every_placement(self):
        for z in range(4):
            self.assertTrue(check("king", 4, 1, [z]).ok)

    def test_queen_bound_is_tight(self):
        # n = 4f: some adversary breaks the Queen algorithm.
        self.assertFalse(check("queen", 4, 1).ok)

    def test_king_bound_is_tight(self):
        # n = 3f: some adversary breaks the King algorithm.
        self.assertFalse(check("king", 3, 1).ok)


if __name__ == "__main__":
    unittest.main()
