"""Testes offline: não importam MLX/NumPy, não acessam Metal nem baixam pesos."""
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark.backends.mlx import MLXBackend, MODEL, REVISION, APPROVED_SHA256
from benchmark.config import RunConfig, load_config
from benchmark.datasets import load_controlled
from benchmark.prompts import build_prompt, CURRENT_TEMPLATE
from benchmark.runner import run
from benchmark.scoring import score_choices
from benchmark.schemas import NextTokenOutput, ScoringResult, TokenizationEvidence
from benchmark.numerics import softmax


class Array:
    def __init__(self, data, shape, runtime, dtype="bfloat16"):
        self.data, self.shape, self.runtime, self.dtype = data, shape, runtime, dtype

    def __getitem__(self, key):
        batch, position, columns = key
        assert columns == slice(None)
        return Array(self.data[batch][position], (self.shape[-1],), self.runtime, self.dtype)

    def astype(self, dtype):
        self.runtime.promotions.append(dtype)
        return Array(self.data, self.shape, self.runtime, dtype)

    def tolist(self):
        assert self.dtype == "float32"
        return self.data


class Runtime:
    float32 = "float32"

    def __init__(self):
        self.promotions = []
        self.cache_limit = 123

    def array(self, data): return data
    def eval(self, array): pass
    def clear_cache(self): pass
    def reset_peak_memory(self): pass
    def get_peak_memory(self): return 1000
    def get_active_memory(self): return 0
    def set_cache_limit(self, value):
        old, self.cache_limit = self.cache_limit, value
        return old


class Tokenizer:
    chat_template = "test-template"
    all_special_ids = [127]
    failure = None

    def encode(self, prompt, add_special_tokens=False):
        assert add_special_tokens is False
        if prompt[-1:] in "ABCD" and prompt[:-1].endswith("assistant\n"):
            ids = self.encode(prompt[:-1])
            token = 32 + "ABCD".index(prompt[-1])
            if self.failure == "boundary": return ids[:-1] + [token]
            if self.failure == "multi": return ids + [token, 66]
            if self.failure == "special": return ids + [127]
            if self.failure == "duplicate": return ids + [32]
            if self.failure == "range": return ids + [200]
            return ids + [token]
        return [64 + ord(c) % 32 for c in prompt]

    def decode(self, ids):
        return chr(ord("A") + ids[0] - 32) if 32 <= ids[0] < 36 else "?"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert add_generation_prompt is True
        text = "system\nuser\n" + messages[0]['content'] + "\nassistant\n"
        return self.encode(text) if tokenize else text


class Model:
    def __init__(self, runtime):
        self.runtime, self.calls = runtime, []
        self.nonfinite = False
        self.bad_shape = False
        self.evaluated = False

    def eval(self): self.evaluated = True

    def __call__(self, inputs, **kwargs):
        assert kwargs == {}, "Não pode usar KV cache, sampling ou penalidades."
        self.calls.append(inputs)
        last = [0.0] * 128
        last[32:36] = [0.0, 1.0, 2.0, 3.0]
        if self.nonfinite: last[0] = math.nan  # verifica TODO o vocabulário
        earlier = [0.0] * 128
        earlier[32] = 50.0
        length = len(inputs[0])
        return Array([[earlier] * (length - 1) + [last]],
                     (1, length - int(self.bad_shape), 128), self.runtime)


class MLXBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / REVISION
        self.snapshot.mkdir()
        (self.snapshot / "tokenizer_config.json").write_text(json.dumps({'chat_template':'test-template'}))
        self.mx, self.tokenizer = Runtime(), Tokenizer()
        self.model = Model(self.mx)
        self.backend = MLXBackend(str(self.snapshot))
        self.load = unittest.mock.Mock(return_value=(self.model,self.tokenizer))
        for patcher in (patch.object(self.backend, '_verify_snapshot', return_value={}),
                        patch('benchmark.backends.mlx._load_runtime', return_value=(self.mx,self.load)),
                        patch('benchmark.backends.mlx.version', return_value='offline-test')):
            patcher.start(); self.addCleanup(patcher.stop)
        self.addCleanup(self.backend.close)

    def config(self):
        return RunConfig("mlx-test",16,42,str(self.root/'outputs'),backend="mlx",
                         dataset_revision="controlled-v1",prompt_template=CURRENT_TEMPLATE,
                         snapshot_path=str(self.snapshot))

    def test_lifecycle_local_load_template_and_no_cache(self):
        with self.assertRaises(RuntimeError): self.backend.prepare_prompt('question')
        self.backend.load()
        self.load.assert_called_once_with(str(self.snapshot),trust_remote_code=False,
                       tokenizer_config={'local_files_only':True,'trust_remote_code':False})
        self.assertTrue(self.model.evaluated)
        prompt = self.backend.prepare_prompt('arbitrary user content')
        output = self.backend.next_token_logits(prompt,tuple('ABCD'))
        self.assertEqual(output.vocabulary_logits[32:36],(0,1,2,3))
        self.assertEqual(output.tokenization.position,len(output.tokenization.input_ids)-1)
        self.assertEqual(self.mx.promotions,['float32'])
        self.assertEqual(len(self.model.calls),1)
        self.backend.close(); self.backend.close()
        self.assertEqual(self.mx.cache_limit,123)
        with self.assertRaises(RuntimeError): self.backend.next_token_logits(prompt,tuple('ABCD'))

    def test_original_mass_and_conditional_probabilities(self):
        self.backend.load()
        result = score_choices(self.backend,self.backend.prepare_prompt('question'),tuple('ABCD'),'label_next_token')
        self.assertEqual(result.predicted_index,3)
        self.assertEqual(result.probabilities,softmax((0.,1.,2.,3.)))
        denominator = 124 + sum(math.exp(i) for i in range(4))
        expected = tuple(math.exp(i)/denominator for i in range(4))
        for a,b in zip(result.original_label_probabilities,expected): self.assertAlmostEqual(a,b,places=14)
        self.assertAlmostEqual(result.original_label_probability_mass,sum(expected),places=14)
        self.assertLess(result.original_label_probability_mass,1)

    def test_boundary_multitoken_special_duplicate_and_range_rejected(self):
        self.backend.load(); prompt = self.backend.prepare_prompt('question')
        for failure in ('boundary','multi','special','duplicate','range'):
            self.tokenizer.failure = failure
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.backend.next_token_logits(prompt,tuple('ABCD'))
        self.assertEqual(self.model.calls,[])

    def test_invalid_labels_empty_long_prompt_and_shape(self):
        self.backend.load()
        for prompt,labels in [('',tuple('ABCD')),('x'*1025,tuple('ABCD')),('x',tuple(' ABC')),('x',('A','B'))]:
            with self.subTest(prompt=prompt[:5]), self.assertRaises(ValueError):
                self.backend.next_token_logits(prompt,labels)
        self.model.bad_shape=True
        with self.assertRaisesRegex(ValueError,'Shape'):
            self.backend.next_token_logits(self.backend.prepare_prompt('q'),tuple('ABCD'))

    def test_nonfinite_vocabulary_rejected(self):
        self.backend.load(); self.model.nonfinite=True
        with self.assertRaisesRegex(ValueError,'finitos'):
            self.backend.next_token_logits(self.backend.prepare_prompt('q'),tuple('ABCD'))

    def test_changed_chat_template_rejected_and_cleanup(self):
        self.tokenizer.chat_template='changed'
        with self.assertRaisesRegex(ValueError,'template'):self.backend.load()
        self.backend.close()
        self.assertIsNone(self.backend._model)
        self.assertEqual(self.mx.cache_limit,123)

    def test_runner_persists_real_identity_prompt_mass_and_errors_as_answers(self):
        directory=run(self.config(),self.backend)
        meta=json.loads((directory/'metadata.json').read_text())
        rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
        summary=json.loads((directory/'summary.json').read_text())
        self.assertEqual((meta['simulation'],meta['status']),(False,'completed'))
        self.assertEqual(meta['model']['forward_mode'],'full_no_kv_cache')
        self.assertEqual(summary['accuracy'],0.25)  # Erros do modelo continuam resultados válidos.
        self.assertEqual(summary['valid_count'],16)
        self.assertTrue(all(x['prompt'].endswith('assistant\n') for x in rows))
        self.assertTrue(all(x['scoring']['original_label_probability_mass']>0 for x in rows))
        self.assertEqual(load_config(directory/'config.yaml'),self.config())
        self.assertIn('source_sha256',meta['code'])

    def test_runner_configuration_selects_mlx(self):
        with patch('benchmark.backends.mlx.MLXBackend',return_value=self.backend) as factory:
            run(self.config())
        factory.assert_called_once_with(str(self.snapshot),max_prompt_tokens=1024)

    def test_tokenization_failure_is_persisted_without_fake_metrics(self):
        self.tokenizer.failure='boundary'
        directory=run(self.config(),self.backend)
        summary=json.loads((directory/'summary.json').read_text())
        self.assertEqual(summary['failed_count'],16)
        self.assertIsNone(summary['accuracy'])
        self.assertEqual(self.model.calls,[])

    def test_close_failure_preserves_real_results(self):
        close=self.backend.close
        def fail(): close(); raise RuntimeError('close failed')
        with patch.object(self.backend,'close',side_effect=fail):directory=run(self.config(),self.backend)
        meta=json.loads((directory/'metadata.json').read_text())
        self.assertEqual(meta['status'],'completed_with_errors')
        self.assertEqual(meta['valid_count'],16)
        self.assertIn('close failed',meta['close_error'])

    def test_runner_load_failure_preserves_primary_and_releases_model(self):
        self.tokenizer.chat_template = 'changed'
        close = self.backend.close
        def fail():
            close()
            raise RuntimeError('secondary close error')
        with patch.object(self.backend, 'close', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'template') as caught:
                run(self.config(), self.backend)
        self.assertIn('secondary close error', ' '.join(caught.exception.__notes__))
        self.assertIsNone(self.backend._model)
        self.assertFalse((self.root / 'outputs').exists())

    def test_render_tokenization_mismatch_is_rejected(self):
        self.backend.load()
        original = self.tokenizer.apply_chat_template
        def mismatch(messages, tokenize=False, add_generation_prompt=True):
            value = original(messages, tokenize, add_generation_prompt)
            return value + [66] if tokenize else value
        with patch.object(self.tokenizer, 'apply_chat_template', side_effect=mismatch):
            with self.assertRaisesRegex(ValueError, 'divergem'):
                self.backend.prepare_prompt('q')
        self.assertEqual(self.model.calls, [])

    def test_inconsistent_real_contracts_rejected(self):
        self.backend.load()
        result=score_choices(self.backend,self.backend.prepare_prompt('q'),tuple('ABCD'),'label_next_token')
        for changes in ({'tokenization':None},{'original_label_probability_mass':0.9},
                        {'original_label_probabilities':(0.,0.,0.,1.)},{'labels':tuple('DCBA')},
                        {'method':'simulated_scores'},{'score_unit':'synthetic_logits'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(result,**changes)
        for changes in ({'position':0},{'candidate_token_ids':(32,32,34,35)},
                        {'forward_mode':'cached'},{'candidate_token_ids':(32,33,34,999)}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(result.tokenization,**changes)

    def test_snapshot_missing_and_corruption_fail_before_runtime(self):
        real=MLXBackend(str(self.snapshot))
        with self.assertRaises((FileNotFoundError,ValueError)):real.load()
        invalid=MLXBackend(str(self.root/'missing'))
        with self.assertRaises(ValueError):invalid.load()
        with patch('benchmark.backends.mlx.APPROVED_SHA256',{'tokenizer_config.json':'0'*64}):
            with self.assertRaisesRegex(ValueError,'diverge'):real._verify_snapshot()


class MLXConfigPromptTests(unittest.TestCase):
    def test_prompt_exact_and_fixture_balanced(self):
        questions=load_controlled(16,42)
        self.assertEqual(len({q.id for q in questions}),16)
        self.assertEqual([sum(q.correct_index==i for q in questions) for i in range(4)],[4]*4)
        q=next(q for q in questions if q.id=='controlled-triangle-0')
        prompt,labels=build_prompt(q,CURRENT_TEMPLATE)
        self.assertEqual(prompt,'How many sides does a triangle have?\nA. 3\nB. 2\nC. 4\nD. 5\nReply with exactly one letter. Do not add spaces or explanations.')
        self.assertEqual(labels,tuple('ABCD'))

    def test_mlx_config_roundtrip_and_reject_unapproved_options(self):
        c=RunConfig('real',16,42,'outputs',backend='mlx',dataset_revision='controlled-v1',
                    prompt_template=CURRENT_TEMPLATE,snapshot_path='local/'+REVISION)
        self.assertEqual(RunConfig.from_dict(c.to_dict()),c)
        for section,key,value in [('model','revision','main'),('model','name','other'),
                                  ('model','kv_cache',True),('model','max_prompt_tokens',True),
                                  ('prompt','template','unknown'),('scoring','method','sequence_loglikelihood')]:
            d=c.to_dict();d[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):RunConfig.from_dict(d)

    def test_optional_runtime_not_imported_by_config(self):
        self.assertNotIn('mlx.core',sys.modules)
        self.assertNotIn('mlx_lm',sys.modules)
