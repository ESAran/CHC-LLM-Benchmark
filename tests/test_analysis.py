"""Análise offline de fixtures autorais; sem modelo real, rede ou gráficos obrigatórios."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark.analysis import analyze_test, describe, entropy_distribution, load_approved_test
from benchmark.analysis_report import build_plot_data
from benchmark.config import RunConfig, load_config
from benchmark.metrics import normalized_entropy, summarize
from benchmark.mmlu import canonical_hash, file_hash
from benchmark.mmlu_selection import DATASET, REVISION, SUBJECTS, SELECTION_ID
from benchmark.numerics import softmax
from benchmark.prompts import CURRENT_TEMPLATE
from benchmark.results import read_run
from benchmark.runner import run
from benchmark.schemas import Question, ScoringResult, EvaluationRecord, TokenizationEvidence

ROOT = Path(__file__).resolve().parents[1]


def record(index, subject=SUBJECTS[0], *, correct=True, scores=(3., 0., 0., 0.), split='test', failed=False):
    predicted = max(range(4), key=lambda i: scores[i])
    gold = predicted if correct else (predicted+1)%4
    q = Question(f'{DATASET}@{REVISION}/{subject}/{split}/{index}', f'Question {index}?',
                 tuple('abcd'), gold, subject)
    probabilities = softmax(scores)
    scoring = ScoringResult(tuple('ABCD'), tuple(scores), probabilities, predicted,
        method='label_next_token', score_unit='logits', tokenization_diagnostic='contextual_single_token_validated',
        original_label_probability_mass=.8, original_label_probabilities=tuple(p*.8 for p in probabilities),
        tokenization=TokenizationEvidence((0,), (0,1,2,3), 0, 4, 'bfloat16'))
    provenance = {'id': q.id, 'dataset': DATASET, 'revision': REVISION, 'subject': subject,
        'split': split, 'original_index': index, 'selection_index': index, 'execution_index': index,
        'source_file': f'{subject}/{split}.parquet', 'source_sha256': 'a'*64,
        'row_sha256': canonical_hash({'question': q.text, 'subject': subject, 'choices': q.choices, 'answer': gold}),
        'selection_id': SELECTION_ID, 'anomalies': []}
    return EvaluationRecord(q, 'persisted prompt', CURRENT_TEMPLATE, None if failed else scoring,
        None if failed else correct, None if failed else normalized_entropy(probabilities),
        error='technical failure' if failed else None, provenance=provenance)


class AnalysisTests(unittest.TestCase):
    def test_describe_empty_singleton_and_reference_quartiles(self):
        self.assertEqual(describe([])['count'], 0)
        self.assertTrue(all(v is None for k,v in describe([]).items() if k!='count'))
        self.assertEqual(describe([.4])['std_population'], 0.)
        d = describe([0., .25, .5, .75, 1.])
        self.assertEqual((d['mean'],d['q1'],d['median'],d['q3']), (.5,.25,.5,.75))
        self.assertAlmostEqual(d['std_population'], (0.125)**.5)
        for values in ([float('nan')],[float('inf')],[True]):
            with self.assertRaises(ValueError):describe(values)

    def test_histogram_boundaries_and_empty(self):
        h = entropy_distribution([0., .1, .9, 1.])
        self.assertEqual(h['counts'], [1,1,0,0,0,0,0,0,0,2])
        self.assertEqual(h['denominator'], 4)
        self.assertEqual(sum(entropy_distribution([])['counts']), 0)
        for values in ([-.1],[1.1],[float('nan')]):
            with self.assertRaises(ValueError):entropy_distribution(values)

    def test_empty_groups_have_null_metrics_and_zero_denominators(self):
        a = analyze_test([])
        for group in [a['overall'], *a['by_subject'].values()]:
            self.assertEqual(group['accuracy_denominator'],0)
            self.assertIsNone(group['accuracy'])
            self.assertIsNone(group['entropy']['mean'])
            self.assertIsNone(group['high_concentration_errors']['fraction_of_incorrect'])
            self.assertEqual(group['prediction_frequencies']['counts'],dict.fromkeys('ABCD',0))
        self.assertEqual(a['examples'], [])

    def test_accuracy_and_outcome_denominators_with_failure(self):
        rows = [record(0), record(1,correct=False), record(2,failed=True)]
        g = analyze_test(rows)['overall']
        self.assertEqual((g['total_count'],g['valid_count'],g['failed_count']), (3,2,1))
        self.assertEqual((g['correct_count'],g['incorrect_count'],g['accuracy_denominator']), (1,1,2))
        self.assertEqual(g['accuracy'], .5)
        self.assertEqual(g['gold_frequencies']['denominator'], 3)
        self.assertEqual(g['prediction_frequencies']['denominator'], 2)
        self.assertEqual(g['entropy_by_outcome']['correct']['count'],1)
        self.assertEqual(g['entropy_by_outcome']['incorrect']['count'],1)

    def test_grouped_accuracy_uses_counts_not_unweighted_subject_means(self):
        a = analyze_test([record(0),record(1,SUBJECTS[1],correct=False),record(2,SUBJECTS[1],correct=False)])
        self.assertEqual(a['overall']['accuracy'],1/3)
        self.assertEqual(a['by_subject'][SUBJECTS[0]]['accuracy'],1.)
        self.assertEqual(a['by_subject'][SUBJECTS[1]]['accuracy'],0.)
        self.assertIsNone(a['by_subject'][SUBJECTS[2]]['accuracy'])

    def test_dev_mixed_duplicate_and_wrong_selection_rejected(self):
        bad = record(3)
        for rows in ([record(0,split='dev')], [record(0),record(1,split='dev')],
                     [bad,bad], [replace(bad,provenance={**bad.provenance,'selection_id':'other'})]):
            with self.assertRaises(ValueError):analyze_test(rows)

    def test_concentration_inclusive_and_does_not_filter_accuracy(self):
        wrong = record(1,correct=False,scores=(7.,0.,0.,0.))
        correct = record(0,scores=(7.,0.,0.,0.))
        low = record(2,correct=False,scores=(0.,0.,0.,0.))
        threshold=max(wrong.scoring.probabilities)
        a=analyze_test([correct,wrong,low],high_concentration_threshold=threshold)
        self.assertEqual(a['overall']['high_concentration_errors']['ids'],[wrong.question.id])
        self.assertEqual(a['overall']['high_concentration_errors']['fraction_of_incorrect'],.5)
        self.assertEqual(a['overall']['accuracy'],1/3)
        for value in (-.1,1.1,True,float('nan')):
            with self.assertRaises(ValueError):analyze_test([],high_concentration_threshold=value)

    def test_frequencies_include_all_labels_and_match_denominators(self):
        g=analyze_test([record(0),record(1,correct=False),record(2,scores=(0.,0.,0.,4.))])['overall']
        self.assertEqual(g['gold_frequencies']['counts'],{'A':1,'B':1,'C':0,'D':1})
        self.assertEqual(g['prediction_frequencies']['counts'],{'A':2,'B':0,'C':0,'D':1})
        self.assertEqual(sum(g['prediction_frequencies']['counts'].values()),g['valid_count'])

    def test_mass_uses_original_mass_not_conditional_sum(self):
        g=analyze_test([record(0),record(1)])['overall']
        self.assertEqual(g['original_label_mass']['mean'],.8)
        self.assertEqual(g['original_label_mass']['min'],.8)
        self.assertEqual(g['original_label_mass']['std_population'],0.)

    def test_examples_symmetry_extremes_ties_and_no_mutation(self):
        rows=[record(i,correct=correct,scores=scores) for i,correct,scores in
              [(0,True,(9.,0.,0.,0.)),(1,True,(0.,0.,0.,0.)),
               (2,False,(9.,0.,0.,0.)),(3,False,(0.,0.,0.,0.)),(4,False,(0.,0.,0.,0.))]]
        before=[asdict(r) for r in rows]
        a=analyze_test(rows)
        self.assertEqual([e['id'] for e in a['examples']],[rows[i].question.id for i in range(4)])
        self.assertEqual([asdict(r) for r in rows],before)
        self.assertEqual(a,analyze_test(rows))
        one=analyze_test([rows[0]])['examples']
        self.assertEqual(len(one),1)
        self.assertEqual(one[0]['roles'],['min_entropy','max_entropy'])

    def test_plot_values_are_exact_projections_and_omit_only_failures(self):
        rows=[record(0),record(1,correct=False),record(2,failed=True)]
        a=analyze_test(rows);p=build_plot_data(a)
        self.assertEqual(p['entropy_by_outcome']['correct'],[rows[0].normalized_entropy])
        self.assertEqual(p['entropy_by_outcome']['incorrect'],[rows[1].normalized_entropy])
        self.assertEqual(len(p['entropy_by_subject'][SUBJECTS[0]]),2)
        self.assertEqual(p['label_frequencies']['overall']['gold_frequencies']['denominator'],3)
        self.assertEqual(p['accuracy_by_subject'][0]['accuracy'],.5)


class ResultReaderTests(unittest.TestCase):
    def test_read_run_round_trip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            path=run(RunConfig('offline-reader',3,42,temp))
            hashes={p.name:file_hash(p) for p in path.iterdir()}
            config,metadata,records,summary=read_run(path,hashes)
            self.assertEqual(summary['accuracy'],2/3)
            self.assertEqual(len(records),3)
            self.assertEqual(metadata.question_ids,tuple(r.question.id for r in records))
            (path/'results.jsonl').write_text('[]\n')
            with self.assertRaisesRegex(ValueError,'diverge'):read_run(path,hashes)

    def test_missing_or_partial_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=run(RunConfig('offline-reader',3,42,temp))
            hashes={p.name:file_hash(p) for p in path.iterdir()}
            (path/'metadata.json').rename(path/'metadata.json.tmp')
            with self.assertRaisesRegex(ValueError,'incompleta'):read_run(path,hashes)


class ApprovedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.records=tuple(record(i,s) for s in SUBJECTS for i in range(20))
        self.config=load_config(ROOT/'configs/mmlu_test.yaml')
        self.summary=summarize(self.records)
        self.meta=SimpleNamespace(status='completed',simulation=False,run_id='test-run',
                                  model={},code={'commit':'fixture'})
        self.evidence=self.root/'evidence.json';self.manifest=self.root/'manifest.json'
        self.evidence.write_text(json.dumps({'runs':[
            {'phase':'dev','run_directory':'must-not-be-read','metadata':{'run_id':'dev-run'},'summary':{'total_count':15}},
            {'phase':'test','run_directory':'test-run','artifact_sha256':{},'summary':self.summary}]}))
        self.manifest.write_text(json.dumps({'selected_records':[
            {**r.provenance,'prompt_sha256':canonical_hash(r.prompt),
             'input_ids_sha256':canonical_hash(r.scoring.tokenization.input_ids)} for r in self.records]}))
        for p in (patch('benchmark.analysis.PILOT_EVIDENCE_SHA256',file_hash(self.evidence)),
                  patch('benchmark.analysis.MANIFEST_SHA256',file_hash(self.manifest))):
            p.start();self.addCleanup(p.stop)
        self.reader=patch('benchmark.analysis.read_run',return_value=(self.config,self.meta,self.records,self.summary)).start()
        self.addCleanup(patch.stopall)

    def test_only_test_artifacts_are_loaded_and_dev_is_separate(self):
        records,source=load_approved_test(self.evidence,self.manifest,root=self.root)
        self.assertEqual(len(records),60)
        self.reader.assert_called_once_with(self.root/'test-run',{})
        self.assertEqual(source['dev_excluded'][0]['count'],15)
        self.assertTrue(all(r.provenance['split']=='test' for r in records))

    def test_approved_evidence_change_rejected_before_loading(self):
        self.evidence.write_text(self.evidence.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'aprovada'):load_approved_test(self.evidence,self.manifest)
        self.reader.assert_not_called()

    def test_wrong_count_split_or_selection_rejected(self):
        for config in (replace(self.config,subjects=(SUBJECTS[0],),limit=20),
                       replace(self.config,dataset_split='dev',limit=15)):
            self.reader.return_value=(config,self.meta,self.records,self.summary)
            with self.assertRaisesRegex(ValueError,'60 test'):load_approved_test(self.evidence,self.manifest)

    def test_missing_or_reordered_records_rejected(self):
        for records in (self.records[:-1],self.records[::-1]):
            self.reader.return_value=(self.config,self.meta,records,self.summary)
            with self.assertRaisesRegex(ValueError,'IDs'):load_approved_test(self.evidence,self.manifest)
