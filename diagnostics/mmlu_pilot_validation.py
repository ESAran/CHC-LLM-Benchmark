"""Auditoria de execuções MMLU persistidas; não executa nem carrega uma LLM."""

import argparse
import json
from pathlib import Path

from benchmark.config import load_config
from benchmark.metrics import summarize
from benchmark.mmlu import canonical_hash, file_hash
from benchmark.mmlu_selection import DATASET, load_frozen_mmlu
from benchmark.schemas import RunMetadata
from benchmark.results import read_record


def audit_run(directory: Path) -> dict:
    required = {'config.yaml', 'results.jsonl', 'summary.json', 'metadata.json'}
    if {p.name for p in directory.iterdir()} != required:
        raise ValueError('Execução incompleta ou com arquivos inesperados.')
    config = load_config(directory / 'config.yaml')
    if config.dataset_name != DATASET:
        raise ValueError('Esta auditoria exige seleção MMLU congelada.')
    meta = json.loads((directory / 'metadata.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    records = [read_record(json.loads(line)) for line in
               (directory / 'results.jsonl').read_text().splitlines()]
    metadata = RunMetadata(**{**meta, 'question_ids': tuple(meta['question_ids'])})
    metadata.validate_records(records, summary)
    if (meta['run_id'] != directory.name or meta['status'] != 'completed'
            or meta['simulation'] or meta['close_error'] is not None
            or summary['failed_count'] != 0 or summary['valid_count'] != config.limit):
        raise ValueError('Validação técnica falhou; não avançar de dev para test.')
    resolved = config.to_dict()
    for section in ('dataset', 'model'):
        if any(meta[section].get(k) != v for k, v in resolved[section].items()):
            raise ValueError('Configuração persistida diverge dos metadados.')
    protocol = {'forward_mode': 'full_no_kv_cache', 'batch_size': 1, 'fewshot_count': 0,
                'add_generation_prompt': True, 'add_special_tokens': False,
                'normalization_dtype': 'float32_to_python_float64', 'max_prompt_tokens': 1024}
    if any(meta['model'].get(k) != v for k, v in protocol.items()):
        raise ValueError('Protocolo persistido diverge de 2B.')
    items, source = load_frozen_mmlu(Path(config.selection_manifest),
                                   Path(config.dataset_snapshot_path), config.subjects, config.dataset_split)
    for key in ('manifest_sha256', 'frozen_selection_sha256', 'groups',
                'question_provenance', 'preparation_records', 'anomaly_policy'):
        if source[key] != meta['dataset'].get(key):
            raise ValueError(f'Procedência persistida diverge do conjunto aprovado: {key}.')
    compact = []
    for record, item, expected in zip(records, items, source['preparation_records'], strict=True):
        if record.question != item.question or record.prompt_template != config.prompt_template:
            raise ValueError('Texto, alternativas, gabarito ou template divergem da origem.')
        scoring = record.scoring
        token = scoring.tokenization
        if (scoring.method != config.scoring_method or token is None
                or token.candidate_token_ids != (32, 33, 34, 35)
                or canonical_hash(record.prompt) != expected['prompt_sha256']
                or canonical_hash(token.input_ids) != expected['input_ids_sha256']
                or len(token.input_ids) != expected['input_tokens']):
            raise ValueError('Pontuação/prompt/tokenização divergentes da preparação aprovada.')
        origin = record.provenance
        compact.append({
            **{k: origin[k] for k in ('id', 'subject', 'split', 'original_index',
                'selection_index', 'execution_index', 'row_sha256', 'anomalies')},
            'prompt': record.prompt, 'prompt_sha256': canonical_hash(record.prompt),
            'correct_index': record.question.correct_index, 'predicted_index': scoring.predicted_index,
            'correct': record.correct, 'raw_scores': scoring.raw_scores,
            'probabilities': scoring.probabilities, 'normalized_entropy': record.normalized_entropy,
            'original_label_probability_mass': scoring.original_label_probability_mass,
            'original_label_probabilities': scoring.original_label_probabilities,
            'input_tokens': len(token.input_ids), 'input_ids_sha256': canonical_hash(token.input_ids),
            'candidate_token_ids': token.candidate_token_ids, 'logits_position': token.position,
        })
    # Somente validade técnica; acurácia e entropia nunca são condições de avanço.
    return {
        'phase': config.dataset_split, 'technical_validation_passed': True,
        'run_directory': str(directory), 'config': resolved, 'metadata': meta,
        'summary': summary,
        'by_subject': {subject: summarize([r for r in records if r.question.subject == subject])
                       for subject in config.subjects},
        'artifact_sha256': {name: file_hash(directory / name) for name in sorted(required)},
        'records': compact,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = {'status': 'pending_technical_review', 'dev_is_technical_only': True,
                'accuracy_threshold': None, 'runs': [audit_run(path) for path in args.run]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, allow_nan=False,
                                      separators=(',', ':')) + '\n', encoding='utf-8')
    for run in evidence['runs']:
        print(json.dumps({'phase': run['phase'], 'technical_validation_passed': True,
                          'summary': run['summary'], 'by_subject': run['by_subject']}))


if __name__ == '__main__':
    main()
