import math
import unittest

from benchmark.backends.simulated import SimulatedBackend
from benchmark.scoring import score_choices
from benchmark.scoring.normalization import softmax


class ScoringTests(unittest.TestCase):
    def test_softmax_known_ratio(self):
        for actual, expected in zip(softmax([0, math.log(3)]), (0.25, 0.75)):
            self.assertAlmostEqual(actual, expected)

    def test_large_scores_and_shift_invariance(self):
        expected = softmax([0, 1, 2])
        for scores in ([10000, 10001, 10002], [-10000, -9999, -9998]):
            self.assertEqual(softmax(scores), expected)
        self.assertEqual(softmax([1e308, -1e308]), (1, 0))
        self.assertEqual(softmax([1e308, 1e308]), (0.5, 0.5))

    def test_softmax_invalid_scores(self):
        for scores in ([], [1], [math.nan, 1], [math.inf, 1], [-math.inf, 1]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                softmax(scores)

    def test_simulated_lifecycle_and_no_tokenization_claim(self):
        backend = SimulatedBackend()
        with self.assertRaises(RuntimeError):
            backend.score("prompt", ("1", "2"))
        backend.load()
        result = score_choices(backend, "prompt", ("1", "2", "3"))
        self.assertEqual(result.raw_scores, (0, 1, 2))
        self.assertEqual(result.predicted_index, 2)
        self.assertEqual(result.method, "simulated_scores")
        self.assertEqual(result.tokenization_diagnostic, "not_applicable_simulation")
        self.assertIsNone(result.original_label_probability_mass)
        self.assertAlmostEqual(sum(result.probabilities), 1)
        backend.close()
        self.assertFalse(backend.loaded)

    def test_score_count_mismatch(self):
        class WrongSize(SimulatedBackend):
            def score(self, prompt, labels):
                return (1, 2, 3)

        with self.assertRaisesRegex(ValueError, "quantidade"):
            score_choices(WrongSize(), "prompt", ("1", "2"))

    def test_tied_scores_choose_first_candidate(self):
        class Tied(SimulatedBackend):
            def score(self, prompt, labels):
                return (7, 7, 7)

        result = score_choices(Tied(), "prompt", ("1", "2", "3"))
        self.assertEqual(result.predicted_index, 0)
        self.assertEqual(result.tie_break, "lowest_index")
