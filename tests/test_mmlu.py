"""Fixtures autorais e doubles; nenhum teste acessa o Hub, MLX ou pesos."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmark.mmlu import (MMLUSource, adapt_mmlu_rows, canonical_hash,
                            duplicate_report, load_mmlu_parquet, select_indices)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mmlu_diagnostic', ROOT/'diagnostics/mmlu_dataset.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def row(subject='high_school_biology', index=0):
    return {'question':f'  Original question {index}?\n', 'subject':subject,
            'choices':['first ', ' second', 'third\nline', 'fourth'], 'answer':2}


class Tokenizer:
    chat_template = 'offline-chat'
    def encode(self, text, add_special_tokens=False, truncation=False):
        assert not add_special_tokens and not truncation
        return list(text.encode('utf-8'))
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_dict=True):
        assert add_generation_prompt
        if tokenize:
            assert return_dict is False
        text = 'system\nuser\n'+messages[0]['content']+'\nassistant\n'
        return self.encode(text) if tokenize else text


class MMLUTests(unittest.TestCase):
    def setUp(self):
        self.source=MMLUSource('a'*40,'high_school_biology','test','b'*64,10)

    def test_preserves_exact_text_choices_subject_answer_and_position(self):
        rows=[row(index=i) for i in range(3)]
        items=adapt_mmlu_rows(rows,self.source)
        for i,item in enumerate(items):
            self.assertEqual(item.question.text,rows[i]['question'])
            self.assertEqual(item.question.choices,tuple(rows[i]['choices']))
            self.assertEqual(item.question.correct_index,2)
            self.assertEqual(item.original_index,i)
            self.assertEqual(item.question.subject,self.source.subject)
            self.assertEqual(item.question.id,self.source.question_id(i))
            self.assertEqual(item.row_sha256,canonical_hash(rows[i]))
            self.assertEqual(item.provenance()['source_sha256'],self.source.sha256)

    def test_ids_distinguish_revision_subject_split_and_position(self):
        sources=[self.source,replace(self.source,revision='c'*40),
                 replace(self.source,subject='high_school_psychology'),replace(self.source,split='dev')]
        ids=[s.question_id(0) for s in sources]+[self.source.question_id(1)]
        self.assertEqual(len(set(ids)),5)

    def test_required_fields_types_options_and_answer_range(self):
        invalid=[None,{}, {'extra':1,**row()}, {k:v for k,v in row().items() if k!='answer'}]
        for key,values in {'question':[' ',None,7],'subject':['other',None],
                           'choices':[None,'ABCD',['a']*3,['a']*5,['a','','c','d']],
                           'answer':[-1,4,True,1.0,'A',None]}.items():
            invalid += [{**row(),key:value} for value in values]
        for value in invalid:
            with self.subTest(value=value),self.assertRaises(ValueError):adapt_mmlu_rows([value],self.source)
        with self.assertRaises(ValueError):adapt_mmlu_rows([],self.source)

    def test_provenance_invariants_and_revision_pinning(self):
        for changes in ({'revision':'main'},{'sha256':'bad'},{'subject':'../secret'},
                        {'split':'validation'},{'size_bytes':True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(self.source,**changes)
        item=adapt_mmlu_rows([row()],self.source)[0]
        for changes in ({'original_index':True},{'original_index':1},{'row_sha256':'0'*64},
                        {'question':replace(item.question,subject='other')}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(item,**changes)

    def test_selection_repeatable_unique_original_order_and_seed_sensitive(self):
        first=select_indices(100,20,42,self.source)
        self.assertEqual(first,select_indices(100,20,42,self.source))
        self.assertEqual(tuple(sorted(set(first))),first)
        self.assertEqual(len(first),20)
        self.assertNotEqual(first,select_indices(100,20,43,self.source))
        self.assertEqual(select_indices(5,5,42,replace(self.source,split='dev')),tuple(range(5)))

    def test_rank_has_fixed_cross_runtime_reference(self):
        # Construção independente por bytes: sem acesso a respostas/texto.
        values=[]
        for index in range(30):
            encoded=json.dumps(['sha256_index_rank_v1','cais/mmlu','a'*40,
                                'high_school_biology','test',42,index],separators=(',',':')).encode()
            values.append((hashlib.sha256(encoded).hexdigest(),index))
        expected=tuple(sorted(i for _,i in sorted(values)[:5]))
        self.assertEqual(select_indices(30,5,42,self.source),expected)

    def test_selection_does_not_depend_on_answers_or_content(self):
        original=[row(index=i) for i in range(30)]
        changed=[{**r,'answer':(r['answer']+1)%4,'question':'changed '+r['question']} for r in original]
        a=adapt_mmlu_rows(original,self.source);b=adapt_mmlu_rows(changed,self.source)
        ia=select_indices(len(a),20,42,self.source);ib=select_indices(len(b),20,42,self.source)
        self.assertEqual([a[i].question.id for i in ia],[b[i].question.id for i in ib])
        self.assertNotEqual(a[0].row_sha256,b[0].row_sha256)

    def test_invalid_selection_sizes_fail_without_replacement(self):
        for values in ((0,1,42),(5,6,42),(5,0,42),(True,1,42),(5,1,True),(5,1.0,42)):
            with self.subTest(values=values),self.assertRaises(ValueError):select_indices(*values,self.source)

    def test_duplicate_content_cross_split_conflicts_and_repeated_choices_reported(self):
        items=adapt_mmlu_rows([row(),{**row(),'answer':0}],self.source)
        dev=adapt_mmlu_rows([row()],replace(self.source,split='dev'))
        repeat=adapt_mmlu_rows([{**row(index=9),'choices':['x','x','z','w']}],replace(self.source,revision='c'*40))
        report=duplicate_report(items+dev+repeat)
        self.assertEqual(len(report['same_question_text'][0]),3)
        self.assertTrue(report['same_question_and_choices'][0]['conflicting_answers'])
        self.assertEqual(report['repeated_choices_within_question'],[repeat[0].question.id])
        self.assertEqual(len(items+dev+repeat),4)  # Nenhuma deduplicação silenciosa.
        with self.assertRaises(ValueError):duplicate_report(items+items)

    def test_source_hash_checked_before_parquet_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data.parquet';path.write_bytes(b'offline fixture')
            source=replace(self.source,size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with patch('benchmark.mmlu.read_parquet_rows',return_value=[row()]) as read:
                items=load_mmlu_parquet(path,source);self.assertEqual(len(items),1);read.assert_called_once_with(path)
            path.write_bytes(b'changed fixture')
            with patch('benchmark.mmlu.read_parquet_rows') as read:
                with self.assertRaises(ValueError):load_mmlu_parquet(path,source)
                read.assert_not_called()

    def test_token_limit_includes_chat_and_never_truncates(self):
        q=adapt_mmlu_rows([row()],self.source)[0].question;tok=Tokenizer()
        measured=diag.token_budget(q,tok,1024)
        count=measured['input_tokens']
        self.assertFalse(diag.token_budget(q,tok,count)['over_limit'])
        self.assertTrue(diag.token_budget(q,tok,count-1)['over_limit'])
        self.assertEqual(q.text,row()['question'])
        long=replace(q,text='x'*2000)
        self.assertTrue(diag.token_budget(long,tok,1024)['over_limit'])
        self.assertEqual(long.text,'x'*2000)

    def test_prepare_keeps_over_limit_items_and_all_groups(self):
        plan=json.loads((ROOT/'configs/mmlu_selection.json').read_text())
        sources=diag.validate_plan(plan)
        def load(path,source):
            n=5 if source.split=='dev' else 30
            return adapt_mmlu_rows([{**row(source.subject,i),'question':'x'*2000+str(i)} for i in range(n)],source)
        paths={(s.subject,s.split):Path('offline') for s in sources}
        with patch.object(diag,'load_mmlu_parquet',side_effect=load):
            selected,manifest=diag.prepare(plan,paths,Tokenizer())
            again,second=diag.prepare(plan,paths,Tokenizer())
        self.assertEqual(len(selected),75)
        self.assertEqual(len(manifest['over_limit_ids']),75)
        self.assertEqual(manifest['selection_sha256'],second['selection_sha256'])
        self.assertEqual(selected,again)
        self.assertEqual([len(g['selected_indices']) for g in manifest['groups']],[5,20]*3)

    def test_tokenization_disagreement_is_not_accepted(self):
        class DivergentTokenizer(Tokenizer):
            def apply_chat_template(self, *args, **kwargs):
                result = super().apply_chat_template(*args, **kwargs)
                return result + [99] if kwargs.get('tokenize') else result
        q = adapt_mmlu_rows([row()], self.source)[0].question
        with self.assertRaisesRegex(ValueError, 'diverge'):
            diag.token_budget(q, DivergentTokenizer(), 1024)

    def test_plan_rejects_unexpected_scope_and_duplicate_sources(self):
        plan=json.loads((ROOT/'configs/mmlu_selection.json').read_text())
        for changes in ({'dataset':'other'},{'revision':'main'},{'seed':True},{'max_prompt_tokens':2048},
                        {'counts':{'dev':5,'validation':20}},{'files':plan['files'][:-1]},
                        {'files':[plan['files'][0]]*6},{'unknown':1}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):diag.validate_plan({**plan,**changes})

    def test_hub_requests_only_six_pinned_files_and_defaults_to_offline(self):
        plan = json.loads((ROOT/'configs/mmlu_selection.json').read_text())
        sources = diag.validate_plan(plan)
        for download in (False, True):
            fetch = Mock(return_value='offline.parquet')
            with patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(hf_hub_download=fetch)}):
                paths = diag.source_paths(plan, sources, Path('offline-cache'), download=download)
            self.assertEqual(len(paths), 6)
            self.assertEqual(fetch.call_count, 6)
            for call, source in zip(fetch.call_args_list, sources):
                self.assertEqual(call.kwargs, {
                    'repo_id': 'cais/mmlu', 'repo_type': 'dataset', 'revision': plan['revision'],
                    'filename': source.filename, 'cache_dir': 'offline-cache',
                    'local_files_only': not download,
                })
