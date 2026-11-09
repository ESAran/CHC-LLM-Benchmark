"""Seleção integral e preflight offline; nunca carrega pesos ou executa o modelo."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys

from diagnostics.mmlu_dataset import load_local_tokenizer, token_budget, validate_plan
from benchmark.mmlu import canonical_hash, duplicate_report, file_hash, load_mmlu_parquet
from benchmark.mmlu_selection import MANIFEST_SHA256, REVISION, SUBJECTS
from benchmark.prompts import CURRENT_TEMPLATE, build_prompt

SELECTION_ID = 'mmlu-full-test-v1'
COUNTS = dict(zip(SUBJECTS, (545, 237, 310), strict=True))


def contextual_tokens(question, tokenizer):
    user, labels = build_prompt(question, CURRENT_TEMPLATE)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': user}],
                                          tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    candidates, errors = [], []
    for label in labels:
        full = tokenizer.encode(prompt + label, add_special_tokens=False, truncation=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            errors.append(f'{label}: contextual boundary or non-unit continuation')
            continue
        token = full[-1]
        candidates.append(token)
        if token in tokenizer.all_special_ids or tokenizer.decode([token]) != label:
            errors.append(f'{label}: special token or decoding mismatch')
    if len(set(candidates)) != 4:
        errors.append('four distinct candidate tokens required')
    return {'candidate_token_ids': candidates, 'logit_position': len(ids)-1,
            'contextual_errors': errors}


def prepare_full(plan, snapshot, tokenizer):
    sources = [s for s in validate_plan(plan) if s.split == 'test']
    if plan['revision'] != REVISION or tuple(s.subject for s in sources) != SUBJECTS:
        raise ValueError('Revisão/disciplinas divergem do escopo aprovado.')
    if snapshot.name != REVISION:
        raise ValueError('Snapshot deve identificar a revisão fixa.')
    selected, groups = [], []
    for source in sources:
        items = load_mmlu_parquet(snapshot / source.filename, source)
        if len(items) != COUNTS[source.subject]:
            raise ValueError('Contagem da origem diverge da população autorizada.')
        selected.extend(items)
        groups.append({**asdict(source), 'source_file': source.filename,
                       'population_count': len(items), 'selected_indices': list(range(len(items))),
                       'selected_ids': [i.question.id for i in items]})
    # Todos os índices são fixados antes de qualquer inspeção de conteúdo/tokens.
    records = [{**item.provenance(),
                **token_budget(item.question, tokenizer, plan['max_prompt_tokens']),
                **contextual_tokens(item.question, tokenizer)} for item in selected]
    over = [r['id'] for r in records if r['over_limit']]
    contextual = [r['id'] for r in records if r['contextual_errors']]
    return selected, {
        'version': 1, 'selection_id': SELECTION_ID,
        'status': 'preflight_blocked' if over or contextual else 'preflight_passed_no_model_evaluation',
        'dataset': plan['dataset'], 'revision': REVISION, 'split': 'test',
        'selection_method': 'all_original_indices_in_source_order_v1',
        'seed': None, 'seed_explanation': 'no_sampling', 'index_base': 0,
        'groups': groups, 'selected_count': len(selected), 'selected_records': records,
        'selection_sha256': canonical_hash([i.provenance() for i in selected]),
        'duplicates': duplicate_report(selected), 'over_limit_ids': over,
        'contextual_error_ids': contextual, 'max_prompt_tokens': plan['max_prompt_tokens'],
        'prompt_template': CURRENT_TEMPLATE, 'tokenizer': plan['tokenizer'],
        'chat_template': tokenizer.chat_template,
        'token_policy': 'no_truncation_no_filter_no_replacement',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tokenizer-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use uma pasta nova; preservar preparações anteriores.')
    plan_path = Path('configs/mmlu_selection.json')
    plan = json.loads(plan_path.read_text())
    old_path = Path('docs/reports/MMLU_SELECTION_MANIFEST.json')
    if file_hash(old_path) != MANIFEST_SHA256:
        raise ValueError('Manifesto original de 3A foi alterado.')
    tokenizer = load_local_tokenizer(plan, args.tokenizer_snapshot)
    items, manifest = prepare_full(plan, args.snapshot, tokenizer)
    lookup = {r['id']: r for r in manifest['selected_records']}
    old_test = [r for r in json.loads(old_path.read_text())['selected_records'] if r['split'] == 'test']
    for old in old_test:
        if any(lookup[old['id']][k] != old[k] for k in
               ('row_sha256', 'prompt_sha256', 'input_ids_sha256', 'input_tokens')):
            raise ValueError('Conteúdo/prompt/tokens divergem do piloto.')
    manifest['pilot_overlap'] = {'count': len(old_test), 'content_prompt_tokens_identical': True,
                                 'manifest_sha256': MANIFEST_SHA256}
    manifest['environment'] = {'python': platform.python_version(),
        'packages': {p: version(p) for p in ('pyarrow', 'transformers', 'tokenizers', 'huggingface-hub')},
        'disk_free_bytes': shutil.disk_usage('.').free}
    manifest['source_sha256'] = {str(p): file_hash(p) for p in
        (Path('diagnostics/mmlu_full_preflight.py'), Path('diagnostics/mmlu_dataset.py'),
         Path('src/benchmark/mmlu.py'), Path('src/benchmark/prompts.py'), plan_path)}
    if 'mlx.core' in sys.modules or 'mlx_lm' in sys.modules:
        raise RuntimeError('Preflight não deve importar MLX.')
    args.output.mkdir(parents=True)
    data = args.output / 'selected_questions.jsonl'
    with data.open('x', encoding='utf-8') as stream:
        for item in items:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False, allow_nan=False)+'\n')
    manifest['selected_questions_jsonl_sha256'] = file_hash(data)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                       allow_nan=False, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in
        ('status', 'selected_count', 'over_limit_ids', 'contextual_error_ids', 'selection_sha256')}))


if __name__ == '__main__':
    main()
