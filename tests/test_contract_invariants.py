from dataclasses import replace
import unittest

from benchmark.metrics import summarize
from benchmark.schemas import EvaluationRecord, Question, RunMetadata, ScoringResult


class ContractInvariantTests(unittest.TestCase):
    def setUp(self):
        self.scoring = ScoringResult(("1", "2"), (0.0, 0.0), (0.5, 0.5), 0)
        self.record = EvaluationRecord(
            Question("q1", "Escolha?", ("Primeira", "Segunda"), 0),
            "Prompt", "synthetic_multiple_choice_v1", self.scoring, True, 1.0,
        )
        self.metadata = RunMetadata(
            run_id="run", started_at="2026-10-20T12:00:00+00:00",
            finished_at="2026-10-20T12:00:01+00:00", config_version=1,
            code={"package_version": "0.1.0", "commit": None, "dirty": None},
            dataset={"name": "synthetic", "revision": "v1"},
            model={"name": "controlled_scores", "revision": "v1"},
            environment={
                "python": "3.12.2", "system": "Test", "release": "1", "machine": "test",
                "dependencies": {"PyYAML": "6.0.2", "llm-benchmark": None},
            },
            question_ids=("q1",), valid_count=1, failed_count=0, status="completed",
        )

    def test_scores_probabilities_and_prediction_must_agree(self):
        for changes in (
            {"probabilities": (0.9, 0.1)}, {"raw_scores": (10, 0)},
            {"predicted_index": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.scoring, **changes)

    def test_scoring_rejects_mutable_sequences_and_invalid_values(self):
        for changes in (
            {"labels": ["1", "2"]}, {"raw_scores": [0, 0]},
            {"probabilities": [0.5, 0.5]}, {"labels": (" ", "2")},
            {"labels": (1, "2")}, {"raw_scores": (False, 0)},
            {"raw_scores": ("0", 0)}, {"raw_scores": (10**1000, 0)},
            {"probabilities": (True, False)}, {"probabilities": ("0.5", 0.5)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.scoring, **changes)

    def test_scoring_cannot_claim_unsupported_method_or_diagnostics(self):
        for key, value in (
            ("method", "label_next_token"), ("score_unit", "original_probabilities"),
            ("normalization", "unknown"), ("tie_break", "highest_index"),
            ("tokenization_diagnostic", "validated"), ("original_label_probability_mass", 0.9),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(self.scoring, **{key: value})

    def test_record_entropy_must_match_probabilities(self):
        for entropy in (0.0, 0.8, True, "1", float("nan")):
            with self.subTest(entropy=entropy), self.assertRaises(ValueError):
                replace(self.record, normalized_entropy=entropy)
        # Tolerância somente numérica, sem modificar o valor armazenado.
        accepted = replace(self.record, normalized_entropy=1.0 - 1e-13)
        self.assertEqual(accepted.normalized_entropy, 1.0 - 1e-13)

    def test_record_requires_valid_nested_types_and_error_shape(self):
        for changes in (
            {"question": None}, {"scoring": {}}, {"prompt": None},
            {"prompt_template": " "},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.record, **changes)
        failed = replace(self.record, scoring=None, correct=None, normalized_entropy=None, error="failed")
        for changes in ({"error": " "}, {"error": 1}, {"correct": False}, {"scoring": self.scoring}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(failed, **changes)

    def test_metadata_status_requires_matching_counts_and_close_error(self):
        for changes in (
            {"valid_count": 0, "failed_count": 1},
            {"status": "failed"}, {"status": "completed_with_errors"},
            {"close_error": "close failed"}, {"close_error": ""}, {"config_version": True},
            {"question_ids": []}, {"question_ids": ("",)}, {"question_ids": (" ",)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.metadata, **changes)
        replace(self.metadata, close_error="close failed", status="completed_with_errors")
        replace(self.metadata, valid_count=0, failed_count=1, status="failed", close_error="close failed")

    def test_metadata_requires_timezone_and_chronological_timestamps(self):
        for changes in (
            {"started_at": "start"}, {"finished_at": "end"}, {"started_at": 1},
            {"started_at": "2026-10-20T12:00:00"}, {"finished_at": "2026-10-20T11:59:59+00:00"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.metadata, **changes)
        replace(self.metadata, started_at="2026-10-20T09:00:00-03:00")

    def test_metadata_requires_explicit_code_provenance(self):
        for code in (
            {}, {"package_version": "0.1.0", "commit": None},
            {**self.metadata.code, "package_version": ""},
            {**self.metadata.code, "commit": "not-a-commit", "dirty": True},
            {**self.metadata.code, "commit": None, "dirty": True},
            {**self.metadata.code, "commit": "a" * 40, "dirty": 1},
        ):
            with self.subTest(code=code), self.assertRaises(ValueError):
                replace(self.metadata, code=code)
        replace(self.metadata, code={**self.metadata.code, "commit": "a" * 40, "dirty": True})

    def test_metadata_requires_environment_and_dependency_fields(self):
        for environment in (
            {}, {**self.metadata.environment, "python": ""},
            {**self.metadata.environment, "machine": None},
            {**self.metadata.environment, "dependencies": {}},
            {**self.metadata.environment, "dependencies": {"PyYAML": "", "llm-benchmark": None}},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                replace(self.metadata, environment=environment)
        for identity in ({}, {"name": " ", "revision": "v1"}, {"name": "synthetic", "revision": 1}):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                replace(self.metadata, dataset=identity)

    def test_bundle_validates_summary_types_values_and_ids(self):
        summary = summarize([self.record])
        self.metadata.validate_records([self.record], summary)
        for changed in (
            {**summary, "accuracy": 0.0}, {**summary, "accuracy": float("nan")},
            {**summary, "valid_count": True}, {**summary, "mean_normalized_entropy": 0.0},
            {**summary, "extra": 0}, {},
        ):
            with self.subTest(summary=changed), self.assertRaises(ValueError):
                self.metadata.validate_records([self.record], changed)
        with self.assertRaises(ValueError):
            self.metadata.validate_records([], summary)
