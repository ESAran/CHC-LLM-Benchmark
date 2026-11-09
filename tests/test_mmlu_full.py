"""Expansão integral com fixtures offline; nenhuma inferência real/rede."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmark.analysis import analyze_test, macro_accuracy
from benchmark.config import RunConfig, load_config
from benchmark.mmlu import file_hash
from benchmark.mmlu_selection import FULL_SELECTION_ID, SUBJECTS, load_frozen_mmlu, validate_scope
from benchmark.runner import run
from diagnostics.mmlu_full_validation import compare_pilot
from diagnostics.mmlu_full_preflight import contextual_tokens, prepare_full
import test_mmlu_runner as pilot_tests
from test_analysis import record
from test_mmlu import Tokenizer as CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]


def full_record(r):
    return replace(r, provenance={**r.provenance, 'selection_id': FULL_SELECTION_ID})


class FullSelectionTests(unittest.TestCase):
    def test_config_exact_counts_seed_null_and_roundtrip(self):
        c = load_config(ROOT/'configs/mmlu_full_test.yaml')
        self.assertEqual(c.limit,1092)
        self.assertIsNone(c.seed)
        self.assertEqual(RunConfig.from_dict(c.to_dict()),c)
        for change in ({'limit':1091},{'seed':42},{'dataset_split':'dev'},
                       {'subjects':SUBJECTS[:2]},{'selection_id':'other'},{'max_prompt_tokens':2048}):
            with self.subTest(change=change),self.assertRaises(ValueError):replace(c,**change)
        self.assertEqual(validate_scope(SUBJECTS,'test'),60)
        self.assertEqual(validate_scope(SUBJECTS,'dev'),15)

    def fixture(self):
        helper=pilot_tests.MMLURunnerTests('test_configs_roundtrip_and_subject_split_scope')
        helper.setUp();self.addCleanup(helper.doCleanups)
        manifest=json.loads(helper.manifest.read_text())
        manifest['groups']=[g for g in manifest['groups'] if g['split']=='test']
        manifest['selected_records']=[r for r in manifest['selected_records'] if r['split']=='test']
        manifest.update(selection_id=FULL_SELECTION_ID,status='preflight_passed_no_model_evaluation',
                        over_limit_ids=[],contextual_error_ids=[])
        helper.manifest.write_text(json.dumps(manifest))
        for p in (patch('benchmark.mmlu_selection.FULL_MANIFEST_SHA256',file_hash(helper.manifest)),
                  patch('benchmark.mmlu_selection.FULL_COUNTS',dict.fromkeys(SUBJECTS,20))):
            p.start();self.addCleanup(p.stop)
        helper.config=replace(helper.config,selection_id=FULL_SELECTION_ID,dataset_split='test',limit=60,seed=None)
        return helper,manifest

    def test_full_runner_preserves_all_indices_and_no_retries(self):
        h,_=self.fixture()
        path=run(h.config,h.backend)
        records=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(h.model.calls),60)
        self.assertTrue(all(r['error'] is None for r in records))
        self.assertEqual([r['provenance']['execution_index'] for r in records],list(range(60)))
        for subject in SUBJECTS:
            rows=[r for r in records if r['question']['subject']==subject]
            self.assertEqual([r['provenance']['original_index'] for r in rows],list(range(20)))
            self.assertTrue(all(r['provenance']['selection_id']==FULL_SELECTION_ID for r in rows))

    def test_full_loader_rejects_missing_indices_and_failed_preflight(self):
        h,manifest=self.fixture()
        for change in ('indices','preflight'):
            m=json.loads(json.dumps(manifest))
            if change=='indices':m['groups'][0]['selected_indices']=list(range(19))
            else:m['status']='preflight_blocked'
            h.manifest.write_text(json.dumps(m))
            with patch('benchmark.mmlu_selection.FULL_MANIFEST_SHA256',file_hash(h.manifest)), self.assertRaises(ValueError):
                load_frozen_mmlu(h.manifest,h.snapshot,SUBJECTS,'test',FULL_SELECTION_ID)

    def test_full_loader_rejects_original_pilot_manifest(self):
        c=load_config(ROOT/'configs/mmlu_full_test.yaml')
        with self.assertRaisesRegex(ValueError,'Manifesto'):
            load_frozen_mmlu(ROOT/'docs/reports/MMLU_SELECTION_MANIFEST.json',Path(c.dataset_snapshot_path),
                             SUBJECTS,'test',FULL_SELECTION_ID)

    def test_preflight_preserves_overlength_rows_and_reports_contextual_errors(self):
        h,_=self.fixture()
        class Tok(CharacterTokenizer):
            all_special_ids=[]
            def decode(self,ids):return bytes(ids).decode()
        plan=json.loads((ROOT/'configs/mmlu_selection.json').read_text())
        # Reutiliza fontes autorais; o leitor de teste não acessa Arrow/Hub.
        m=json.loads(h.manifest.read_text())
        sources=[]
        from benchmark.mmlu import MMLUSource
        for g in m['groups']:
            sources.append(MMLUSource(**{k:g[k] for k in ('revision','subject','split','sha256','size_bytes')}))
        plan['max_prompt_tokens']=1
        with patch('diagnostics.mmlu_full_preflight.validate_plan',return_value=sources), \
             patch('diagnostics.mmlu_full_preflight.COUNTS',dict.fromkeys(SUBJECTS,20)):
            items,manifest=prepare_full(plan,h.snapshot,Tok())
        self.assertEqual(len(items),60)
        self.assertEqual(len(manifest['over_limit_ids']),60)
        self.assertEqual(manifest['status'],'preflight_blocked')
        self.assertEqual(manifest['groups'][0]['selected_indices'],list(range(20)))
        class Bad(Tok):
            def encode(self,text,**kwargs):
                ids=super().encode(text,**kwargs)
                return [99]+ids if text.endswith('A') else ids
        self.assertTrue(contextual_tokens(items[0].question,Bad())['contextual_errors'])


class FullAnalysisTests(unittest.TestCase):
    def test_macro_uses_equal_subject_weights_and_requires_all_denominators(self):
        rows=[record(0),record(0,SUBJECTS[1],correct=False),record(1,SUBJECTS[1],correct=False),
              record(0,SUBJECTS[2])]
        a=analyze_test(rows)
        self.assertEqual(a['overall']['accuracy'],.5)
        self.assertAlmostEqual(macro_accuracy(a)['value'],2/3)
        self.assertIsNone(macro_accuracy(analyze_test(rows[:1]))['value'])
        self.assertIsNone(macro_accuracy(analyze_test([],subjects=()))['value'])

    def test_analysis_requires_explicit_selection_and_no_mixing(self):
        p=record(0);f=full_record(record(1))
        self.assertEqual(analyze_test([f],selection_id=FULL_SELECTION_ID)['overall']['valid_count'],1)
        for rows,selection in (([f],'mmlu-3a-v1'),([p,f],FULL_SELECTION_ID),([f],'bad')):
            with self.assertRaises(ValueError):analyze_test(rows,selection_id=selection)

    def test_overlap_exact_and_changed_prompt_prediction_mass_and_missing(self):
        p=record(0);f=full_record(p)
        result=compare_pilot([p],[f])
        self.assertTrue(result['passed'])
        self.assertEqual(result['max_absolute_differences']['raw_scores'],0)
        changed=replace(f.scoring,original_label_probability_mass=.7,
                        original_label_probabilities=tuple(v*.7 for v in f.scoring.probabilities))
        for records in ([],[replace(f,prompt='changed')],[replace(f,scoring=changed)],
                        [full_record(record(0,scores=(0.,3.,0.,0.)))],
                        [full_record(record(0,failed=True))]):
            self.assertFalse(compare_pilot([p],records)['passed'])
        with self.assertRaises(ValueError):compare_pilot([p],[f,f])

    def test_overlap_float_tolerance_and_raw_logits_exact(self):
        p=record(0);f=full_record(p)
        scoring=replace(f.scoring,original_label_probability_mass=.8+1e-13,
                        original_label_probabilities=tuple(v*(.8+1e-13) for v in f.scoring.probabilities))
        self.assertTrue(compare_pilot([p],[replace(f,scoring=scoring)])['passed'])
        scoring=replace(f.scoring,raw_scores=tuple(v+1e-13 for v in f.scoring.raw_scores))
        self.assertFalse(compare_pilot([p],[replace(f,scoring=scoring)])['passed'])


class FullArtifactTests(unittest.TestCase):
    def test_archive_roundtrip_preserves_bytes_and_is_deterministic(self):
        import tempfile
        import tarfile
        from diagnostics.mmlu_full_report import archive_run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run_path=root/'run-id';run_path.mkdir()
            names=('config.yaml','metadata.json','results.jsonl','summary.json')
            for name in names:(run_path/name).write_bytes(('original '+name+'\n').encode())
            a=archive_run(run_path,root/'a.tar.gz');b=archive_run(run_path,root/'b.tar.gz')
            self.assertEqual(a['sha256'],b['sha256'])
            with tarfile.open(root/'a.tar.gz','r:gz') as archive:
                self.assertEqual(archive.getnames(),[f'run-id/{n}' for n in names])
                for name in names:
                    self.assertEqual(archive.extractfile(f'run-id/{name}').read(),(run_path/name).read_bytes())
            with self.assertRaises(FileExistsError):archive_run(run_path,root/'a.tar.gz')

    def test_report_blocks_consolidation_on_overlap_divergence(self):
        import tempfile
        from diagnostics.mmlu_full_report import generate
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit=root/'audit.json'
            audit.write_text(json.dumps({'overlap_comparison':{'passed':False}}))
            with self.assertRaisesRegex(ValueError,'Divergência'):
                generate(audit,root/'resources',root/'out','2026-11-09')
            self.assertFalse((root/'out').exists())
