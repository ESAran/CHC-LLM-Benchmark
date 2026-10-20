from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.config import RunConfig
from benchmark.runner import run
from benchmark.storage import save_run


class StorageFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = RunConfig("storage-test", 3, 42, self.temporary.name)

    def payload(self):
        # Obtém um conjunto coerente; nenhum arquivo é criado por este helper.
        with patch("benchmark.runner.save_run") as save:
            run(self.config)
        return save.call_args.args

    def test_metadata_is_published_only_after_other_files_are_closed(self):
        payload = self.payload()
        directory = payload[0]
        original_replace = Path.replace

        def publish(path, target):
            self.assertFalse(target.exists())
            self.assertEqual(path.name, "metadata.json.tmp")
            self.assertEqual(len((directory / "results.jsonl").read_text().splitlines()), 3)
            self.assertEqual(json.loads((directory / "summary.json").read_text())["valid_count"], 3)
            self.assertTrue((directory / "config.yaml").is_file())
            self.assertEqual(json.loads(path.read_text())["status"], "completed")
            return original_replace(path, target)

        with patch.object(Path, "replace", new=publish):
            save_run(*payload)
        self.assertEqual({p.name for p in directory.iterdir()},
                         {"metadata.json", "config.yaml", "results.jsonl", "summary.json"})

    def test_open_failure_at_each_file_never_publishes_final_metadata(self):
        original_open = Path.open
        for name in ("config.yaml", "results.jsonl", "summary.json", "metadata.json.tmp"):
            with self.subTest(name=name):
                payload = self.payload()
                failure = OSError(f"open failed: {name}")

                def broken_open(path, *args, **kwargs):
                    if path.name == name:
                        raise failure
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, "open", new=broken_open):
                    with self.assertRaises(OSError) as caught:
                        save_run(*payload)
                self.assertIs(caught.exception, failure)
                self.assertFalse((payload[0] / "metadata.json").exists())
                self.assertIn("incompleta", " ".join(failure.__notes__))

    def test_partial_write_flush_and_close_failures_leave_incomplete_directory(self):
        original_open = Path.open
        for name in ("config.yaml", "results.jsonl", "summary.json", "metadata.json.tmp"):
            for phase in ("write", "flush", "close"):
                with self.subTest(name=name, phase=phase):
                    payload = self.payload()

                    class BrokenStream:
                        def __init__(self, stream):
                            self.stream = stream

                        def __enter__(self):
                            return self

                        def __exit__(self, *exc):
                            self.stream.close()
                            if phase == "close":
                                raise OSError("controlled close failure")

                        def write(self, text):
                            if phase == "write":
                                self.stream.write(text[:10])
                                raise OSError("controlled partial write")
                            return self.stream.write(text)

                        def flush(self):
                            if phase == "flush":
                                raise OSError("controlled flush failure")
                            return self.stream.flush()

                        def fileno(self):
                            return self.stream.fileno()

                    def broken_open(path, *args, **kwargs):
                        stream = original_open(path, *args, **kwargs)
                        return BrokenStream(stream) if path.name == name else stream

                    with patch.object(Path, "open", new=broken_open):
                        with self.assertRaises(OSError):
                            save_run(*payload)
                    self.assertFalse((payload[0] / "metadata.json").exists())
                    self.assertTrue((payload[0] / name).exists())

    def test_fsync_failure_never_publishes_final_metadata(self):
        for index in range(4):
            with self.subTest(file_index=index):
                payload = self.payload()
                with patch("benchmark.storage.os.fsync",
                           side_effect=[None] * index + [OSError("sync failed")]):
                    with self.assertRaises(OSError):
                        save_run(*payload)
                self.assertFalse((payload[0] / "metadata.json").exists())

    def test_publication_failure_preserves_partial_files_without_final_metadata(self):
        payload = self.payload()
        with patch.object(Path, "replace", side_effect=OSError("rename failed")):
            with self.assertRaises(OSError):
                save_run(*payload)
        self.assertTrue((payload[0] / "results.jsonl").exists())
        self.assertTrue((payload[0] / "metadata.json.tmp").exists())
        self.assertFalse((payload[0] / "metadata.json").exists())

    def test_inconsistent_bundle_is_rejected_before_creating_directory(self):
        for problem in ("order", "summary", "counts", "identity", "template", "directory", "mutable"):
            with self.subTest(problem=problem):
                directory, config, metadata, records, summary = self.payload()
                if problem == "order":
                    records = list(reversed(records))
                elif problem == "summary":
                    summary = {**summary, "accuracy": 0.0}
                elif problem == "counts":
                    metadata = replace(metadata, valid_count=2, failed_count=1,
                                       status="completed_with_errors")
                elif problem == "identity":
                    metadata = replace(metadata, model={**metadata.model, "revision": "other"})
                elif problem == "template":
                    records[0] = replace(records[0], prompt_template="other")
                elif problem == "directory":
                    metadata = replace(metadata, run_id="other")
                else:
                    metadata.environment.clear()
                with self.assertRaises(ValueError):
                    save_run(directory, config, metadata, records, summary)
                self.assertFalse(directory.exists())
