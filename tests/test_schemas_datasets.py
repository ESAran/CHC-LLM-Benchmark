from dataclasses import replace
import unittest

from benchmark.datasets import SYNTHETIC_ROWS, adapt_rows, load_synthetic
from benchmark.prompts import build_prompt
from benchmark.schemas import EvaluationRecord, Question, RunMetadata, ScoringResult


class SchemaDatasetTests(unittest.TestCase):
    def setUp(self):
        self.question = Question("q1", "Escolha?", ("Primeira", "Segunda"), 1)
        self.scoring = ScoringResult(("1", "2"), (0, 0), (0.5, 0.5), 0)

    def test_question_validation(self):
        for changes in (
            {"id": ""}, {"text": " "}, {"choices": ("única",)},
            {"choices": ("", "b")}, {"choices": ["a", "b"]},
            {"correct_index": -1}, {"correct_index": 2},
            {"correct_index": True}, {"correct_index": 1.0}, {"subject": 3},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.question, **changes)

    def test_scoring_contract(self):
        for changes in (
            {"labels": ("1", "1")}, {"raw_scores": (0,)},
            {"raw_scores": (0, float("nan"))}, {"probabilities": (0.3, 0.3)},
            {"predicted_index": 1}, {"predicted_index": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.scoring, **changes)

    def test_evaluation_contract(self):
        record = EvaluationRecord(self.question, "Prompt", "v1", self.scoring, False, 1)
        for changes in (
            {"correct": True}, {"scoring": None}, {"prompt": ""},
            {"normalized_entropy": float("nan")}, {"normalized_entropy": 1.1},
            {"error": "falha"}, {"prompt_template": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(record, **changes)
        failed = EvaluationRecord(self.question, "Prompt", "v1", None, None, None, "falha")
        self.assertIsNone(failed.correct)

    def test_adapter_preserves_ids_choices_and_answer(self):
        questions = adapt_rows(SYNTHETIC_ROWS)
        for row, question in zip(SYNTHETIC_ROWS, questions, strict=True):
            self.assertEqual(question.id, row["id"])
            self.assertEqual(question.choices, row["choices"])
            self.assertEqual(question.correct_index, row["correct_index"])
        self.assertEqual([len(q.choices) for q in questions], [2, 3, 4])

    def test_duplicates_empty_and_bad_rows_fail(self):
        invalid_rows = (
            [], [SYNTHETIC_ROWS[0], SYNTHETIC_ROWS[0]],
            [{**SYNTHETIC_ROWS[0], "correct_index": 8}],
            [{**SYNTHETIC_ROWS[0], "choices": "ab"}],
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                adapt_rows(rows)

    def test_selection_is_repeatable_and_prefix_matches_limit(self):
        full = load_synthetic(3, 42)
        self.assertEqual(full, load_synthetic(3, 42))
        self.assertEqual(full[:2], load_synthetic(2, 42))
        self.assertEqual(len({q.id for q in full}), 3)
        for limit in (0, 4, True):
            with self.assertRaises(ValueError):
                load_synthetic(limit, 42)

    def test_prompt_exact_text_and_order(self):
        prompt, labels = build_prompt(self.question)
        self.assertEqual(labels, ("1", "2"))
        self.assertEqual(prompt, (
            "Selecione uma alternativa e responda somente com seu identificador.\n\n"
            "Escolha?\n1. Primeira\n2. Segunda\n\nResposta:"
        ))

    def test_metadata_requires_provenance_and_consistent_counts(self):
        metadata = RunMetadata(
            run_id="run", started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00", config_version=1,
            code={"package_version": "0.1.0", "commit": None, "dirty": None},
            dataset={"name": "synthetic", "revision": "v1"},
            model={"name": "controlled_scores", "revision": "v1"},
            environment={
                "python": "3.12.2", "system": "Test", "release": "1", "machine": "test",
                "dependencies": {"PyYAML": "6.0.2", "llm-benchmark": None},
            },
            question_ids=("q1",), valid_count=1, failed_count=0,
            status="completed",
        )
        for changes in (
            {"run_id": ""}, {"dataset": {}}, {"model": {}}, {"valid_count": 2},
            {"failed_count": -1}, {"question_ids": ("q1", "q1")},
            {"simulation": False}, {"status": "unknown"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(metadata, **changes)
