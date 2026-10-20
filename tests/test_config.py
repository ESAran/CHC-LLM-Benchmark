from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from benchmark.config import RunConfig, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = RunConfig("test", 3, 42, "outputs")
        self.data = self.config.to_dict()

    def test_example_and_round_trip(self):
        self.assertEqual(RunConfig.from_dict(self.data), self.config)
        example = Path(__file__).resolve().parents[1] / "configs" / "synthetic.yaml"
        self.assertEqual(load_config(example).limit, 3)

    def test_unknown_missing_and_later_stage_settings_rejected(self):
        invalid = [None, [], {**self.data, "unknown": 1}, {**self.data, "version": True}]
        missing = deepcopy(self.data)
        del missing["scoring"]
        invalid.append(missing)
        for section, key, value in (
            ("dataset", "name", "mmlu"), ("dataset", "split", "test"),
            ("model", "backend", "mlx"), ("model", "revision", "unknown"),
            ("scoring", "method", "label_next_token"),
            ("prompt", "template", "unknown"), ("dataset", "unexpected", 1),
            ("dataset", "limit", 4), ("dataset", "limit", True),
            ("dataset", "seed", 1.2), ("output", "directory", ""),
        ):
            changed = deepcopy(self.data)
            changed[section][key] = value
            invalid.append(changed)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                RunConfig.from_dict(data)

    def test_duplicate_yaml_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.yaml"
            path.write_text("version: 1\nversion: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "únicos"):
                load_config(path)
