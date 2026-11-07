"""Preparação 3A: seis Parquets e tokenizer local; nunca carrega pesos ou Runner."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import hashlib
from pathlib import Path
import platform
import sys

from benchmark.mmlu import (DATASET, SELECTION_METHOD, MMLUSource, canonical_hash,
                            duplicate_report, file_hash, load_mmlu_parquet, select_indices)
from benchmark.prompts import CURRENT_TEMPLATE, build_prompt

SUBJECTS = ["high_school_psychology", "high_school_world_history", "high_school_biology"]


def validate_plan(plan):
    fields = {"version", "dataset", "revision", "subjects", "counts", "seed",
              "selection_method", "files", "prompt_template", "max_prompt_tokens", "tokenizer"}
    if not isinstance(plan, dict) or set(plan) != fields:
        raise ValueError("Campos do plano de seleção incompatíveis.")
    if (type(plan['version']) is not int or plan['version'] != 1
            or plan['dataset'] != DATASET or plan['subjects'] != SUBJECTS
            or plan['counts'] != {'dev': 5, 'test': 20} or type(plan['seed']) is not int
            or plan['selection_method'] != SELECTION_METHOD
            or plan['prompt_template'] != CURRENT_TEMPLATE
            or type(plan['max_prompt_tokens']) is not int or plan['max_prompt_tokens'] != 1024):
        raise ValueError("Plano diverge do escopo/protocolo de preparação 3A.")
    if not isinstance(plan['files'], list) or any(
            not isinstance(entry, dict) or set(entry) != {'subject','split','sha256','size_bytes'}
            for entry in plan['files']):
        raise ValueError("Manifesto de arquivos inválido.")
    sources = [MMLUSource(revision=plan['revision'], **entry) for entry in plan['files']]
    expected = {(subject, split) for subject in SUBJECTS for split in ('dev', 'test')}
    if len(sources) != 6 or {(s.subject, s.split) for s in sources} != expected:
        raise ValueError("O plano exige exatamente os seis arquivos autorizados.")
    tokenizer = plan['tokenizer']
    if (not isinstance(tokenizer, dict) or set(tokenizer) != {'model','revision','files_sha256'}
            or not isinstance(tokenizer['files_sha256'], dict)
            or set(tokenizer['files_sha256']) != {'added_tokens.json','config.json','merges.txt',
                'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}):
        raise ValueError("Manifesto do tokenizer incompleto.")
    return sources


def source_paths(plan, sources, cache, download=False):
    # Imports opcionais; esta função só pede filenames individuais.
    from huggingface_hub import hf_hub_download
    return {
        (source.subject, source.split): Path(hf_hub_download(
            repo_id=DATASET, repo_type='dataset', revision=plan['revision'],
            filename=source.filename, cache_dir=str(cache), local_files_only=not download))
        for source in sources
    }


def load_local_tokenizer(plan, snapshot):
    if not snapshot.is_dir() or snapshot.name != plan['tokenizer']['revision']:
        raise ValueError("Tokenizador deve vir do snapshot local fixado em 2B.")
    for name, expected in plan['tokenizer']['files_sha256'].items():
        if file_hash(snapshot / name) != expected:
            raise ValueError(f"Arquivo do tokenizer diverge de 2B: {name}.")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    expected = json.loads((snapshot / 'tokenizer_config.json').read_text())['chat_template']
    if tokenizer.chat_template != expected:
        raise ValueError("Chat template diverge do snapshot aprovado.")
    return tokenizer


def token_budget(question, tokenizer, maximum):
    user, _ = build_prompt(question, CURRENT_TEMPLATE)
    messages = [{'role':'user','content':user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    # O wrapper MLX-LM também fixa return_dict=False; aqui usamos só o tokenizer.
    if ids != tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_dict=False):
        raise ValueError("Tokenização do texto renderizado e do template diverge.")
    return {'input_tokens':len(ids), 'max_prompt_tokens':maximum,
            'over_limit':len(ids)>maximum, 'prompt_sha256':canonical_hash(prompt),
            'input_ids_sha256':canonical_hash(ids)}


def prepare(plan, paths, tokenizer):
    sources = validate_plan(plan)
    all_items, selected, groups = [], [], []
    for source in sources:
        items = load_mmlu_parquet(paths[(source.subject,source.split)], source)
        indices = select_indices(len(items), plan['counts'][source.split], plan['seed'], source)
        chosen = [items[i] for i in indices]
        all_items.extend(items); selected.extend(chosen)
        groups.append({**asdict(source), 'source_file':source.filename,
                       'population_count':len(items), 'selected_indices':list(indices),
                       'selected_ids':[i.question.id for i in chosen]})
    # Todo índice foi fixado antes de inspecionar o orçamento de tokens.
    records = []
    for item in selected:
        records.append({**item.provenance(), **token_budget(item.question, tokenizer, plan['max_prompt_tokens'])})
    manifest = {
        'status':'dataset_prepared_no_model_evaluation', 'plan':plan,
        'selection_method':SELECTION_METHOD, 'index_base':0, 'groups':groups,
        'selected_count':len(selected), 'selected_records':records,
        'selection_sha256':canonical_hash([i.provenance() for i in selected]),
        'plan_sha256':canonical_hash(plan),
        'duplicates_all_downloaded_rows':duplicate_report(all_items),
        'duplicates_selected_rows':duplicate_report(selected),
        'over_limit_ids':[r['id'] for r in records if r['over_limit']],
        'chat_template':tokenizer.chat_template,
        'chat_template_sha256_utf8':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'token_policy':'no_truncation_no_filter_no_replacement',
    }
    return selected, manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('configs/mmlu_selection.json'))
    parser.add_argument('--cache',type=Path,default=Path('.cache/mmlu/huggingface'))
    parser.add_argument('--download',action='store_true',help='Permite baixar só os seis arquivos fixados.')
    parser.add_argument('--tokenizer-snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use uma pasta de saída nova; não sobrescrever seleções anteriores.')
    plan=json.loads(args.plan.read_text())
    sources=validate_plan(plan)
    paths=source_paths(plan,sources,args.cache,args.download)
    tokenizer=load_local_tokenizer(plan,args.tokenizer_snapshot)
    selected,manifest=prepare(plan,paths,tokenizer)
    if 'mlx.core' in sys.modules or 'mlx_lm' in sys.modules:
        raise RuntimeError('O diagnóstico não deve importar o backend de inferência.')
    manifest['environment']={'python':platform.python_version(),
        'packages':{p:version(p) for p in ('pyarrow','huggingface-hub','transformers','tokenizers')}}
    manifest['source_sha256']={str(p):file_hash(p) for p in
        (Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
         Path('src/benchmark/mmlu.py'), Path('src/benchmark/prompts.py'))}
    args.output.mkdir(parents=True,exist_ok=False)
    data=args.output/'selected_questions.jsonl'
    with data.open('x',encoding='utf-8') as stream:
        for item in selected:
            stream.write(json.dumps(item.to_dict(),ensure_ascii=False,allow_nan=False)+'\n')
    manifest['selected_questions_jsonl_sha256']=file_hash(data)
    temporary=args.output/'manifest.json.tmp'
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write('\n')
    temporary.replace(args.output/'manifest.json')
    print(json.dumps({'selected_count':len(selected),'over_limit_count':len(manifest['over_limit_ids']),
                      'selection_sha256':manifest['selection_sha256'],'status':manifest['status']}))


if __name__=='__main__':
    main()
