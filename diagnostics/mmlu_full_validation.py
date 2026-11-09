"""Auditoria e comparação do run integral; sem inferência ou retentativas."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from benchmark.analysis import (PILOT_EVIDENCE_SHA256, analyze_test, load_approved_test,
                                macro_accuracy)
from benchmark.mmlu import canonical_hash, file_hash
from benchmark.mmlu_selection import FULL_SELECTION_ID, load_frozen_mmlu
from benchmark.results import read_run

# Fixadas antes da inferência. BF16 promovido: logits devem ser idênticos;
# demais floats seguem a tolerância absoluta dos contratos já existentes.
TOLERANCES = {'raw_scores': 0., 'probabilities': 1e-12, 'normalized_entropy': 1e-12,
              'original_label_probabilities': 1e-12, 'original_label_probability_mass': 1e-12}


def compare_pilot(pilot, full):
    lookup = {r.question.id: r for r in full}
    if len(lookup) != len(full) or len({r.question.id for r in pilot}) != len(pilot):
        raise ValueError('Comparação exige IDs únicos.')
    differences, maxima, checks = [], dict.fromkeys(TOLERANCES, 0.), []
    for old in pilot:
        current = lookup.get(old.question.id)
        errors, deltas = [], {}
        if current is None or current.error is not None or old.error is not None:
            errors.append('missing_or_failed_record')
        else:
            if (old.question != current.question or old.prompt != current.prompt
                    or old.prompt_template != current.prompt_template
                    or old.scoring.tokenization != current.scoring.tokenization
                    or old.scoring.method != current.scoring.method):
                errors.append('content_prompt_tokenization_or_scoring_changed')
            if old.correct != current.correct or old.scoring.predicted_index != current.scoring.predicted_index:
                errors.append('prediction_or_correctness_changed')
            for field, tolerance in TOLERANCES.items():
                a, b = (getattr(old, field), getattr(current, field)) if field == 'normalized_entropy' else (
                    getattr(old.scoring, field), getattr(current.scoring, field))
                if not isinstance(a, tuple):
                    a, b = (a,), (b,)
                delta = max(abs(x-y) for x, y in zip(a, b, strict=True))
                maxima[field] = max(maxima[field], delta)
                deltas[field] = delta
                if delta > tolerance:
                    errors.append(field)
        entry = {'id': old.question.id, 'max_absolute_differences': deltas, 'errors': errors}
        checks.append(entry)
        if errors:
            differences.append(entry)
    return {'count': len(pilot), 'passed': not differences, 'absolute_tolerances': TOLERANCES,
            'relative_tolerance': 0., 'max_absolute_differences': maxima,
            'differences': differences, 'per_record': checks}


def audit_full(directory):
    hashes = {name: file_hash(directory/name) for name in
              ('config.yaml', 'metadata.json', 'results.jsonl', 'summary.json')}
    config, meta, records, summary = read_run(directory, hashes)
    if config.selection_id != FULL_SELECTION_ID or meta.simulation:
        raise ValueError('Exige-se a execução integral real autorizada.')
    items, source = load_frozen_mmlu(Path(config.selection_manifest), Path(config.dataset_snapshot_path),
                                   config.subjects, config.dataset_split, config.selection_id)
    if any(meta.dataset.get(k) != v for k, v in source.items()):
        raise ValueError('Metadados divergem da seleção integral congelada.')
    protocol = {'forward_mode': 'full_no_kv_cache', 'batch_size': 1, 'fewshot_count': 0,
                'add_generation_prompt': True, 'add_special_tokens': False,
                'normalization_dtype': 'float32_to_python_float64', 'max_prompt_tokens': 1024}
    if any(meta.model.get(k) != v for k, v in protocol.items()):
        raise ValueError('Protocolo diverge do aprovado.')
    for record, item, expected, origin in zip(records, items, source['preparation_records'],
                                             source['question_provenance'], strict=True):
        if record.question != item.question or record.provenance != origin:
            raise ValueError('Conteúdo, ordem, gabarito ou procedência divergente.')
        if record.error is None:
            token = record.scoring.tokenization
            if (token is None or token.candidate_token_ids != (32,33,34,35)
                    or canonical_hash(record.prompt) != expected['prompt_sha256']
                    or canonical_hash(token.input_ids) != expected['input_ids_sha256']
                    or len(token.input_ids) != expected['input_tokens']):
                raise ValueError('Prompt ou tokenização diverge do preflight.')
    pilot, pilot_source = load_approved_test(Path('docs/reports/MMLU_PILOT_EVIDENCE.json'),
                                           Path('docs/reports/MMLU_SELECTION_MANIFEST.json'))
    comparison = compare_pilot(pilot, records)
    result = {'status': 'pending_technical_review' if comparison['passed'] else 'blocked_overlap_divergence',
              'run_directory': str(directory), 'artifact_sha256': hashes,
              'config': config.to_dict(), 'metadata': asdict(meta), 'summary': summary,
              'overlap_comparison': comparison, 'pilot_evidence_sha256': PILOT_EVIDENCE_SHA256,
              'pilot_run_id': pilot_source['run_id'],
              'failures': [{'id': r.question.id, 'error': r.error} for r in records if r.error],
              'close_error': meta.close_error}
    # Divergências bloqueiam a consolidação; preservar a auditoria para investigação.
    if comparison['passed']:
        analysis = analyze_test(records, selection_id=FULL_SELECTION_ID)
        result.update(analysis=analysis, macro_accuracy=macro_accuracy(analysis),
                      pilot_analysis=analyze_test(pilot))
        result['records'] = [asdict(r) for r in records]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Não sobrescrever uma auditoria existente.')
    result = audit_full(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    print(json.dumps({'status': result['status'], 'summary': result['summary'],
                      'overlap': {k:v for k,v in result['overlap_comparison'].items() if k!='per_record'}}))


if __name__ == '__main__':
    main()
