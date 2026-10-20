import math
import unittest

from benchmark.metrics import accuracy, normalized_entropy, predicted_index, summarize


class MetricsTests(unittest.TestCase):
    def test_uniform_distributions(self):
        for size in (2, 3, 4, 7, 100):
            with self.subTest(size=size):
                self.assertAlmostEqual(normalized_entropy([1 / size] * size), 1)

    def test_deterministic_distributions_include_zeros(self):
        for probabilities in ([1, 0], [0, 0, 1], [0, 1, 0, 0]):
            with self.subTest(probabilities=probabilities):
                self.assertEqual(normalized_entropy(probabilities), 0)

    def test_independent_reference_values(self):
        # H(1/2, 1/4, 1/8, 1/8) = 7/4 bits; log2(4) = 2.
        self.assertAlmostEqual(normalized_entropy([0.5, 0.25, 0.125, 0.125]), 0.875)
        self.assertAlmostEqual(normalized_entropy([0.25, 0.75]), 0.8112781244591328)

    def test_invalid_probabilities(self):
        invalid = (
            [], [1], [-0.1, 1.1], [0.2, 0.2], [0, 0], [1, 1],
            [math.nan, 0], [math.inf, 0], [-math.inf, 1],
        )
        for probabilities in invalid:
            for metric in (normalized_entropy, predicted_index):
                with self.subTest(probabilities=probabilities, metric=metric.__name__):
                    with self.assertRaises(ValueError):
                        metric(probabilities)

    def test_argmax_preserves_order_and_uses_lowest_index_on_ties(self):
        self.assertEqual(predicted_index([0.1, 0.7, 0.2]), 1)
        self.assertEqual(predicted_index([0.1, 0.45, 0.45]), 1)
        self.assertEqual(predicted_index([0.25] * 4), 0)

    def test_accuracy(self):
        self.assertEqual(accuracy([False, False]), 0)
        self.assertEqual(accuracy([True, True]), 1)
        self.assertEqual(accuracy([True, False, True, False]), 0.5)
        for invalid in ([], [None], [2], [0, 1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                accuracy(invalid)

    def test_empty_summary_has_no_invented_metrics(self):
        summary = summarize([])
        self.assertEqual(summary["total_count"], 0)
        self.assertIsNone(summary["accuracy"])
        self.assertIsNone(summary["mean_normalized_entropy"])
