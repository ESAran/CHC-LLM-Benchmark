from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from benchmark.backends.simulated import SimulatedBackend
from benchmark.cli import main
from benchmark.config import RunConfig, load_config
from benchmark.metrics import normalized_entropy
from benchmark.runner import run
from benchmark.storage import save_run


class RunnerStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = RunConfig("test", 3, 42, str(self.root / "outputs"))

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def records(self, directory):
        return [
            json.loads(line)
            for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_end_to_end_results_metadata_and_precision(self):
        class Tracked(SimulatedBackend):
            loads = 0
            closes = 0

            def load(self):
                self.loads += 1
                super().load()

            def close(self):
                self.closes += 1
                super().close()

        backend = Tracked()
        directory = run(self.config, backend)
        self.assertEqual((backend.loads, backend.closes, backend.loaded), (1, 1, False))
        self.assertEqual(
            {file.name for file in directory.iterdir()},
            {"config.yaml", "results.jsonl", "metadata.json", "summary.json"},
        )
        self.assertEqual(load_config(directory / "config.yaml"), self.config)
        records = self.records(directory)
        self.assertEqual(len(records), 3)
        summary = self.read_json(directory / "summary.json")
        self.assertEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["valid_count"], 3)
        for record in records:
            question, scoring = record["question"], record["scoring"]
            self.assertIn(question["text"], record["prompt"])
            self.assertEqual(len(scoring["probabilities"]), len(question["choices"]))
            self.assertEqual(
                record["normalized_entropy"], normalized_entropy(scoring["probabilities"])
            )
            self.assertEqual(
                record["correct"], scoring["predicted_index"] == question["correct_index"]
            )
            self.assertIsNone(record["error"])
        metadata = self.read_json(directory / "metadata.json")
        self.assertTrue(metadata["simulation"])
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["question_ids"], [r["question"]["id"] for r in records])
        self.assertEqual(metadata["dataset"]["revision"], "v1")
        self.assertEqual(metadata["dataset"]["split"], "synthetic")
        self.assertIn("commit", metadata["code"])
        self.assertIn("dirty", metadata["code"])
        self.assertIn("PyYAML", metadata["environment"]["dependencies"])
        self.assertLessEqual(metadata["started_at"], metadata["finished_at"])

    def test_repeatability_and_runs_never_overwrite(self):
        first = run(self.config)
        original = {path.name: path.read_bytes() for path in first.iterdir()}
        second = run(self.config)
        self.assertNotEqual(first, second)
        self.assertEqual(self.records(first), self.records(second))
        self.assertEqual(self.read_json(first / "summary.json"), self.read_json(second / "summary.json"))
        self.assertEqual(original, {path.name: path.read_bytes() for path in first.iterdir()})
        # Falha antes de usar os argumentos restantes se o diretório já existe.
        with self.assertRaises(FileExistsError):
            save_run(first, {}, None, [], {})
        self.assertEqual(original, {path.name: path.read_bytes() for path in first.iterdir()})

    def test_partial_failure_is_persisted_and_denominator_is_explicit(self):
        class PartialFailure(SimulatedBackend):
            def score(self, prompt, labels):
                if len(labels) == 3:
                    raise RuntimeError("falha controlada")
                return super().score(prompt, labels)

        backend = PartialFailure()
        directory = run(self.config, backend)
        summary = self.read_json(directory / "summary.json")
        self.assertEqual(summary["total_count"], 3)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["valid_count"], 2)
        self.assertEqual(summary["accuracy"], 1)
        failures = [record for record in self.records(directory) if record["error"]]
        self.assertEqual(len(failures), 1)
        self.assertIn("falha controlada", failures[0]["error"])
        for field in ("correct", "scoring", "normalized_entropy"):
            self.assertIsNone(failures[0][field])
        self.assertFalse(backend.loaded)
        self.assertEqual(
            self.read_json(directory / "metadata.json")["status"], "completed_with_errors"
        )

    def test_all_failures_do_not_produce_fake_zero_accuracy(self):
        class InvalidScores(SimulatedBackend):
            def score(self, prompt, labels):
                return tuple(float("nan") for _ in labels)

        directory = run(self.config, InvalidScores())
        summary = self.read_json(directory / "summary.json")
        self.assertEqual(summary["failed_count"], 3)
        self.assertEqual(len(self.records(directory)), 3)
        self.assertIsNone(summary["accuracy"])
        self.assertIsNone(summary["mean_normalized_entropy"])
        self.assertEqual(self.read_json(directory / "metadata.json")["status"], "failed")

    def test_backend_contract_rejects_incompatible_capabilities_and_identity(self):
        for field, value in (("capabilities", ("next_token",)), ("name", "other")):
            backend = SimulatedBackend()
            setattr(backend, field, value)
            with self.assertRaises(ValueError):
                run(self.config, backend)
            self.assertFalse(backend.loaded)

    def test_cleanup_on_setup_failure(self):
        class FailedLoad(SimulatedBackend):
            def load(self):
                super().load()
                raise RuntimeError("load failed")

        backend = FailedLoad()
        with self.assertRaises(RuntimeError):
            run(self.config, backend)
        self.assertFalse(backend.loaded)

    def test_cli_exit_codes(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config.to_dict()), encoding="utf-8")
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--config", str(config_path)]), 0)
        self.assertIn("Simulação", output.getvalue())
        with patch.object(SimulatedBackend, "score", side_effect=RuntimeError("failure")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(config_path)]), 1)
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["--config", str(self.root / "missing.yaml")]), 2)
        self.assertIn("Erro:", errors.getvalue())

    def test_close_failure_preserves_all_results_and_records_run_error(self):
        with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed")):
            directory = run(self.config)
        metadata = self.read_json(directory / "metadata.json")
        self.assertEqual(metadata["status"], "completed_with_errors")
        self.assertEqual(metadata["close_error"], "RuntimeError: close failed")
        self.assertEqual(metadata["valid_count"], 3)
        self.assertEqual(len(self.records(directory)), 3)
        self.assertEqual(self.read_json(directory / "summary.json")["accuracy"], 2 / 3)

    def test_item_and_close_errors_are_both_preserved(self):
        with patch.object(SimulatedBackend, "score", side_effect=ValueError("score failed")):
            with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed")):
                directory = run(self.config)
        self.assertTrue(all("score failed" in record["error"] for record in self.records(directory)))
        metadata = self.read_json(directory / "metadata.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("close failed", metadata["close_error"])
        self.assertEqual(metadata["failed_count"], 3)

    def test_load_error_is_not_masked_by_close_error(self):
        primary = RuntimeError("load failed first")
        with patch.object(SimulatedBackend, "load", side_effect=primary):
            with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed second")):
                with self.assertRaises(RuntimeError) as caught:
                    run(self.config)
        self.assertIs(caught.exception, primary)
        self.assertIn("close failed second", " ".join(primary.__notes__))
        self.assertFalse((self.root / "outputs").exists())

    def test_interrupt_is_not_masked_by_close_error(self):
        primary = KeyboardInterrupt("interrupted")
        with patch.object(SimulatedBackend, "score", side_effect=primary):
            with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed")):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    run(self.config)
        self.assertIs(caught.exception, primary)
        self.assertIn("close failed", " ".join(primary.__notes__))

    def test_write_failure_surfaces_previous_item_and_close_errors(self):
        primary = OSError("write failed")
        with patch.object(SimulatedBackend, "score", side_effect=RuntimeError("score failed")):
            with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed")):
                with patch("benchmark.runner.save_run", side_effect=primary):
                    with self.assertRaises(OSError) as caught:
                        run(self.config)
        self.assertIs(caught.exception, primary)
        self.assertIn("close failed", " ".join(primary.__notes__))
        self.assertIn("score failed", " ".join(primary.__notes__))

    def test_cli_reports_close_error_without_claiming_success(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config.to_dict()), encoding="utf-8")
        with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed")):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(["--config", str(config_path)]), 1)
        self.assertIn("close failed", errors.getvalue())

    def test_cli_reports_primary_and_secondary_initialization_errors(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config.to_dict()), encoding="utf-8")
        with patch.object(SimulatedBackend, "load", side_effect=RuntimeError("load failed")):
            with patch.object(SimulatedBackend, "close", side_effect=RuntimeError("close failed")):
                with redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(main(["--config", str(config_path)]), 2)
        self.assertIn("load failed", errors.getvalue())
        self.assertIn("close failed", errors.getvalue())
