"""Correctness tests. Run with:  python -m unittest discover -s tests -v"""

import itertools
import unittest

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
