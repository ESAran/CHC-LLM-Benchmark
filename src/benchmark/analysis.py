"""Análise descritiva de registros test; nunca executa modelo ou scoring."""

from bisect import bisect_right
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import fmean, median, pstdev, quantiles
from typing import Sequence

from .metrics import summarize
from .mmlu import canonical_hash, file_hash
from .mmlu_selection import DATASET, REVISION, SELECTION_ID, SUBJECTS, MANIFEST_SHA256, FULL_SELECTION_ID
from .results import read_run
from .schemas import EvaluationRecord

PILOT_EVIDENCE_SHA256 = '974f22e63c06c753182ac574b8199a1579115cce2fd47f59eb35bc6f329ba6ed'
HISTOGRAM_EDGES = tuple(i / 10 for i in range(11))
LABELS = 'ABCD'


def macro_accuracy(analysis: dict) -> dict:
    """Média simples das acurácias disciplinares; não omite grupos sem válidos."""
    groups = analysis['by_subject']
    missing = [s for s, g in groups.items() if g['accuracy'] is None]
    return {'value': fmean(g['accuracy'] for g in groups.values()) if groups and not missing else None,
            'subject_count': len(groups), 'undefined_subjects': missing,
            'weighting': 'equal_subject_weight'}


def describe(values: Sequence[float]) -> dict:
    """Quartis inclusivos (interpolação linear n−1); desvio populacional descritivo."""
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('Valores descritivos devem ser números finitos.')
    if not values:
        return {k: (0 if k == 'count' else None) for k in
                ('count', 'mean', 'std_population', 'min', 'q1', 'median', 'q3', 'max')}
    q1, _, q3 = quantiles(values, n=4, method='inclusive') if len(values) > 1 else (values[0],)*3
    return {'count': len(values), 'mean': fmean(values), 'std_population': pstdev(values),
            'min': min(values), 'q1': q1, 'median': median(values), 'q3': q3, 'max': max(values)}


def entropy_distribution(values: Sequence[float]) -> dict:
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Entropias devem estar no intervalo [0, 1].')
    counts = [0]*10
    for value in values:
        counts[min(bisect_right(HISTOGRAM_EDGES, value)-1, 9)] += 1
    return {'edges': list(HISTOGRAM_EDGES), 'counts': counts,
            'intervals': '[left,right), last [0.9,1]', 'denominator': len(values)}


def _aggregate(records: Sequence[EvaluationRecord], threshold: float) -> dict:
    valid = [r for r in records if r.error is None]
    correct = [r for r in valid if r.correct]
    incorrect = [r for r in valid if not r.correct]
    entropies = [r.normalized_entropy for r in valid]
    high = [r for r in incorrect if max(r.scoring.probabilities) >= threshold]
    counts = {label: sum(r.question.correct_index == i for r in records) for i, label in enumerate(LABELS)}
    predicted = {label: sum(r.scoring.predicted_index == i for r in valid) for i, label in enumerate(LABELS)}
    return {
        **summarize(records), 'correct_count': len(correct), 'incorrect_count': len(incorrect),
        'accuracy_denominator': len(valid),
        'entropy': describe(entropies), 'entropy_histogram': entropy_distribution(entropies),
        'entropy_by_outcome': {name: describe([r.normalized_entropy for r in group])
                               for name, group in (('correct', correct), ('incorrect', incorrect))},
        'original_label_mass': describe([r.scoring.original_label_probability_mass for r in valid]),
        'gold_frequencies': {'denominator': len(records), 'counts': counts},
        'prediction_frequencies': {'denominator': len(valid), 'counts': predicted},
        'high_concentration_errors': {'threshold_inclusive': threshold, 'count': len(high),
            'incorrect_denominator': len(incorrect),
            'fraction_of_incorrect': len(high)/len(incorrect) if incorrect else None,
            'ids': [r.question.id for r in sorted(high, key=lambda r: (-max(r.scoring.probabilities), r.question.id))]},
    }


def analyze_test(records: Sequence[EvaluationRecord], *, subjects: tuple[str, ...] = SUBJECTS,
                 high_concentration_threshold: float = 0.90,
                 selection_id: str = SELECTION_ID) -> dict:
    """Não filtra dev: rejeita entradas dev/mistas para impedir contaminação."""
    if (type(high_concentration_threshold) not in (int, float)
            or not math.isfinite(high_concentration_threshold) or not 0 <= high_concentration_threshold <= 1):
        raise ValueError('Limiar descritivo deve estar em [0, 1].')
    if not isinstance(subjects, tuple) or len(subjects) != len(set(subjects)):
        raise ValueError('Disciplinas devem ser uma tupla sem duplicações.')
    if selection_id not in (SELECTION_ID, FULL_SELECTION_ID):
        raise ValueError('Seleção de análise não autorizada.')
    ids = set()
    for record in records:
        if not isinstance(record, EvaluationRecord):
            raise ValueError('A análise exige EvaluationRecord.')
        replace(record, question=replace(record.question),
                scoring=replace(record.scoring) if record.scoring else None)
        origin = record.provenance
        if (origin is None or origin['split'] != 'test' or origin['dataset'] != DATASET
                or origin['revision'] != REVISION or origin['selection_id'] != selection_id
                or record.question.subject not in subjects):
            raise ValueError('A análise experimental aceita somente test do conjunto aprovado.')
        if record.question.id in ids:
            raise ValueError('IDs duplicados alterariam os denominadores.')
        if record.scoring is not None and record.scoring.method != 'label_next_token':
            raise ValueError('Registros válidos exigem probabilidades e massa do protocolo real.')
        ids.add(record.question.id)
    groups = {s: [r for r in records if r.question.subject == s] for s in subjects}
    examples = []
    for subject, group in groups.items():
        for outcome in (True, False):
            eligible = [r for r in group if r.error is None and r.correct == outcome]
            if not eligible:
                continue
            for role, key in (('min_entropy', lambda r: (r.normalized_entropy, r.question.id)),
                              ('max_entropy', lambda r: (-r.normalized_entropy, r.question.id))):
                chosen = min(eligible, key=key)
                existing = next((item for item in examples if item['id'] == chosen.question.id), None)
                if existing:
                    existing['roles'].append(role)
                else:
                    examples.append({'id': chosen.question.id, 'subject': subject,
                                     'outcome': 'correct' if outcome else 'incorrect', 'roles': [role]})
    rows = []
    for record in records:
        q, score, origin = record.question, record.scoring, record.provenance
        rows.append({
            'id': q.id, 'subject': q.subject, 'split': 'test',
            'original_index': origin['original_index'], 'selection_index': origin['selection_index'],
            'row_sha256': origin['row_sha256'], 'question': q.text, 'choices': list(q.choices),
            'gold': LABELS[q.correct_index], 'prediction': LABELS[score.predicted_index] if score else None,
            'correct': record.correct, 'error': record.error,
            'probabilities': list(score.probabilities) if score else None,
            'normalized_entropy': record.normalized_entropy,
            'max_conditional_probability': max(score.probabilities) if score else None,
            'original_label_probability_mass': score.original_label_probability_mass if score else None,
            'prompt_sha256': canonical_hash(record.prompt),
        })
    return {
        'split': 'test', 'selection_id': selection_id,
        'policy': {'high_concentration_threshold_inclusive': high_concentration_threshold,
                   'threshold_is_calibrated': False, 'quartiles': 'inclusive_linear_n_minus_1',
                   'standard_deviation': 'population_descriptive', 'invalid_records': 'explicit_exclusion_from_valid_denominator',
                   'examples': 'min_and_max_entropy_per_subject_and_outcome_tie_by_id_no_duplicates'},
        'overall': _aggregate(records, high_concentration_threshold),
        'by_subject': {s: _aggregate(group, high_concentration_threshold) for s, group in groups.items()},
        'examples': examples, 'records': rows,
    }


def load_approved_test(evidence_path: Path, manifest_path: Path, *, root: Path = Path('.')):
    """Lê só os quatro artefatos test; não necessita dos Parquets, pesos ou MLX."""
    if file_hash(evidence_path) != PILOT_EVIDENCE_SHA256 or file_hash(manifest_path) != MANIFEST_SHA256:
        raise ValueError('Evidência de 3B ou manifesto de 3A diverge da versão aprovada.')
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    phases = [r for r in evidence['runs'] if r['phase'] == 'test']
    if len(phases) != 1:
        raise ValueError('Exige-se exatamente uma execução test aprovada.')
    phase = phases[0]
    config, meta, records, summary = read_run(root / phase['run_directory'], phase['artifact_sha256'])
    if (config.dataset_name != DATASET or config.dataset_revision != REVISION
            or config.selection_id != SELECTION_ID or config.dataset_split != 'test'
            or config.subjects != SUBJECTS or config.limit != 60
            or meta.status != 'completed' or meta.simulation or summary['valid_count'] != 60
            or summary != phase['summary']):
        raise ValueError('O piloto experimental aprovado exige 60 test válidos, sem dev.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = [r for r in manifest['selected_records'] if r['split'] == 'test']
    if [r.question.id for r in records] != [r['id'] for r in expected]:
        raise ValueError('IDs ou ordem divergem da seleção congelada.')
    for record, frozen in zip(records, expected, strict=True):
        q = record.question
        raw = {'question': q.text, 'subject': q.subject, 'choices': q.choices, 'answer': q.correct_index}
        if (canonical_hash(raw) != frozen['row_sha256'] or record.provenance['split'] != 'test'
                or record.provenance['original_index'] != frozen['original_index']
                or canonical_hash(record.prompt) != frozen['prompt_sha256']
                or canonical_hash(record.scoring.tokenization.input_ids) != frozen['input_ids_sha256']):
            raise ValueError('Registro não corresponde ao conteúdo/prompt/tokens aprovados.')
    return records, {
        'pilot_evidence_sha256': PILOT_EVIDENCE_SHA256, 'selection_manifest_sha256': MANIFEST_SHA256,
        'run_id': meta.run_id, 'run_directory': phase['run_directory'],
        'artifact_sha256': phase['artifact_sha256'], 'config': config.to_dict(),
        'model': meta.model, 'dataset_revision': REVISION, 'execution_code': meta.code,
        'dev_excluded': [{'run_id': r['metadata']['run_id'], 'count': r['summary']['total_count'],
                          'reason': 'technical_validation_only_not_loaded_into_analysis'}
                         for r in evidence['runs'] if r['phase'] == 'dev'],
    }
