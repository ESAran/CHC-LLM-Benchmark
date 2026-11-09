"""Integração MMLU offline: dados autorais, arquivos locais e modelo double."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.backends.mlx import MLXBackend, REVISION as MODEL_REVISION
from benchmark.config import load_config, RunConfig
from benchmark.mmlu import MMLUSource, adapt_mmlu_rows, canonical_hash, file_hash
from benchmark.mmlu_selection import REVISION, SUBJECTS, load_frozen_mmlu
from benchmark.prompts import build_prompt, CURRENT_TEMPLATE
from benchmark.runner import run
from test_mlx_backend import Runtime, Tokenizer, Model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_audit', ROOT/'diagnostics/mmlu_pilot_validation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class MMLURunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / REVISION
        self.snapshot.mkdir()
        self.tok, self.mx = Tokenizer(), Runtime()
        groups, records, origins, self.rows = [], [], [], {}
        for subject in SUBJECTS:
            for split, count in (('dev', 5), ('test', 20)):
                path = self.snapshot/subject/f'{split}-00000-of-00001.parquet'
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(f'fixture {subject}/{split}'.encode())
                source = MMLUSource(REVISION, subject, split, file_hash(path), path.stat().st_size)
                rows = [{'question': f'Original {subject} {split} question {i}?',
                         'subject': subject, 'choices': ['a ', ' b', 'c\nline', 'd'], 'answer': 2}
                        for i in range(count)]
                if subject == 'high_school_biology' and split == 'dev':
                    rows[2].update(choices=['same', 'other', 'same', 'correct'], answer=3)
                self.rows[str(path)] = rows
                items = adapt_mmlu_rows(rows, source)
                groups.append({**asdict(source), 'source_file': source.filename,
                    'population_count': count, 'selected_indices': list(range(count)),
                    'selected_ids': [i.question.id for i in items]})
                for item in items:
                    user, _ = build_prompt(item.question, CURRENT_TEMPLATE)
                    prompt = self.tok.apply_chat_template([{'role': 'user', 'content': user}])
                    ids = self.tok.encode(prompt)
                    origins.append(item.provenance())
                    records.append({**item.provenance(), 'prompt_sha256': canonical_hash(prompt),
                                    'input_tokens': len(ids), 'input_ids_sha256': canonical_hash(ids)})
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'groups': groups, 'selected_records': records,
                                             'selection_sha256': canonical_hash(origins)}))
        self.config = replace(load_config(ROOT/'configs/mmlu_dev.yaml'),
            selection_manifest=str(self.manifest), dataset_snapshot_path=str(self.snapshot),
            snapshot_path=str(self.root/MODEL_REVISION), output_directory=str(self.root/'outputs'))
        model_path = Path(self.config.snapshot_path)
        model_path.mkdir()
        (model_path/'tokenizer_config.json').write_text(json.dumps({'chat_template': self.tok.chat_template}))
        self.backend = MLXBackend(str(model_path))
        self.model = Model(self.mx)
        for patcher in (
            patch('benchmark.mmlu_selection.MANIFEST_SHA256', file_hash(self.manifest)),
            patch('benchmark.mmlu.read_parquet_rows', side_effect=lambda path: self.rows[str(path)]),
            patch.object(self.backend, '_verify_snapshot', return_value={}),
            patch('benchmark.backends.mlx._load_runtime', return_value=(self.mx, lambda *a, **kw: (self.model, self.tok))),
            patch('benchmark.backends.mlx.version', return_value='offline-test'),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.backend.close)

    def load(self, subjects=SUBJECTS, split='dev'):
        return load_frozen_mmlu(self.manifest, self.snapshot, subjects, split)

    def test_configs_roundtrip_and_subject_split_scope(self):
        for name, count in (('dev', 15), ('test', 60)):
            config = load_config(ROOT/f'configs/mmlu_{name}.yaml')
            self.assertEqual(config.limit, count)
            self.assertEqual(RunConfig.from_dict(config.to_dict()), config)
        config = replace(self.config, subjects=(SUBJECTS[1],), limit=5)
        self.assertEqual(RunConfig.from_dict(config.to_dict()), config)
        items, meta = self.load((SUBJECTS[1],), 'test')
        self.assertEqual(len(items), 20)
        self.assertEqual([i.original_index for i in items], list(range(20)))
        self.assertEqual(meta['question_provenance'][0]['selection_index'], 30)

    def test_config_rejects_protocol_or_selection_drift(self):
        for values in ({'seed': 43}, {'dataset_revision': 'main'}, {'selection_id': 'new'},
                       {'limit': 14}, {'max_prompt_tokens': 1000}, {'dataset_split': 'validation'},
                       {'subjects': SUBJECTS[::-1]}, {'subjects': (SUBJECTS[0],)*2},
                       {'subjects': ()}, {'subjects': ('unknown',)}, {'backend': 'simulated'},
                       {'selection_manifest': ''}, {'dataset_snapshot_path': None}):
            with self.subTest(values=values), self.assertRaises(ValueError): replace(self.config, **values)
        data = self.config.to_dict();data['dataset']['indices'] = [0]
        with self.assertRaises(ValueError): RunConfig.from_dict(data)

    def test_manifest_tampering_fails_before_parquet_or_backend(self):
        self.manifest.write_text(self.manifest.read_text()+' ')
        with patch('benchmark.mmlu_selection.load_mmlu_parquet') as reader, patch.object(self.backend, 'load') as load:
            with self.assertRaisesRegex(ValueError, 'Manifesto'): run(self.config, self.backend)
        reader.assert_not_called();load.assert_not_called()
        self.assertFalse(Path(self.config.output_directory).exists())

    def test_parquet_corruption_fails_before_backend_load(self):
        # Corrompe explicitamente arquivo dev, preservando o tamanho.
        path = self.snapshot/SUBJECTS[0]/'dev-00000-of-00001.parquet'
        path.write_bytes(b'x'*path.stat().st_size)
        with patch.object(self.backend, 'load') as load:
            with self.assertRaisesRegex(ValueError, 'hash'): run(self.config, self.backend)
        load.assert_not_called()

    def test_missing_snapshot_and_file_fail_without_network(self):
        with self.assertRaises(ValueError):
            load_frozen_mmlu(self.manifest, self.root/'missing', SUBJECTS, 'dev')
        (self.snapshot/SUBJECTS[0]/'dev-00000-of-00001.parquet').unlink()
        with self.assertRaises(FileNotFoundError): self.load()

    def test_row_mutation_rejected_even_with_reader_double(self):
        key = str(self.snapshot/SUBJECTS[0]/'dev-00000-of-00001.parquet')
        self.rows[key][0]['answer'] = 1
        with self.assertRaisesRegex(ValueError, 'linha aprovada'): self.load()

    def test_loader_reads_only_requested_files_and_preserves_anomaly(self):
        with patch('benchmark.mmlu.read_parquet_rows', side_effect=lambda p: self.rows[str(p)]) as reader:
            items, meta = self.load((SUBJECTS[2],))
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(items[2].question.choices, ('same', 'other', 'same', 'correct'))
        self.assertEqual(items[2].question.correct_index, 3)
        self.assertEqual(meta['question_provenance'][2]['anomalies'], ['repeated_choice_text'])
        self.assertEqual([p['execution_index'] for p in meta['question_provenance']], list(range(5)))

    def test_runner_persists_complete_provenance_and_passes_audit(self):
        directory = run(self.config, self.backend)
        evidence = audit.audit_run(directory)
        self.assertTrue(evidence['technical_validation_passed'])
        self.assertEqual(evidence['summary']['valid_count'], 15)
        self.assertEqual(len(self.model.calls), 15)
        self.assertEqual(len(evidence['records']), 15)
        self.assertEqual(set(evidence['by_subject']), set(SUBJECTS))
        self.assertEqual(evidence['records'][12]['anomalies'], ['repeated_choice_text'])

    def test_zero_accuracy_does_not_fail_technical_gate(self):
        class WrongModel(Model):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                result.data[0][-1][32:36] = [10., 0., 0., 0.]
                return result
        self.model = WrongModel(self.mx)
        evidence = audit.audit_run(run(self.config, self.backend))
        self.assertEqual(evidence['summary']['accuracy'], 0.)
        self.assertTrue(evidence['technical_validation_passed'])

    def test_prompt_drift_prevents_forward_and_gate(self):
        with patch.object(self.backend, 'prepare_prompt', return_value='changed'):
            directory = run(self.config, self.backend)
        self.assertEqual(self.model.calls, [])
        rows = [json.loads(x) for x in (directory/'results.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['provenance'] and r['error'] for r in rows))
        with self.assertRaisesRegex(ValueError, 'técnica falhou'): audit.audit_run(directory)

    def test_token_drift_fails_without_retry(self):
        encode = self.tok.encode
        def changed(prompt, add_special_tokens=False):
            ids = encode(prompt, add_special_tokens)
            return [65] + ids[1:]
        self.tok.encode = changed
        directory = run(self.config, self.backend)
        self.assertEqual(len(self.model.calls), 15)
        rows = [json.loads(x) for x in (directory/'results.jsonl').read_text().splitlines()]
        self.assertTrue(all('Tokens de entrada' in r['error'] for r in rows))
        with self.assertRaises(ValueError): audit.audit_run(directory)

    def test_close_error_blocks_gate_but_preserves_records(self):
        close = self.backend.close
        def fail(): close(); raise RuntimeError('secondary close')
        with patch.object(self.backend, 'close', side_effect=fail):
            directory = run(self.config, self.backend)
        self.assertEqual(len((directory/'results.jsonl').read_text().splitlines()), 15)
        with self.assertRaisesRegex(ValueError, 'técnica falhou'): audit.audit_run(directory)

    def test_persisted_provenance_tampering_and_missing_metadata_rejected(self):
        directory = run(self.config, self.backend)
        path = directory/'results.jsonl'
        data = [json.loads(x) for x in path.read_text().splitlines()]
        data[0]['provenance']['original_index'] = 9
        path.write_text(''.join(json.dumps(x)+'\n' for x in data))
        with self.assertRaisesRegex(ValueError, 'Procedência'): audit.audit_run(directory)
        (directory/'metadata.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incompleta'): audit.audit_run(directory)

    def test_provenance_contract_rejects_bad_indices_hash_and_identity(self):
        directory = run(self.config, self.backend)
        data = json.loads((directory/'results.jsonl').read_text().splitlines()[0])
        record = audit.read_record(data)
        for field, value in (('original_index', True), ('execution_index', -1),
                             ('source_sha256', 'bad'), ('id', 'other'), ('subject', 'other')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(record, provenance={**record.provenance, field: value})
