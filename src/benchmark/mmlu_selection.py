"""Consumo local das seleções MMLU congeladas: piloto e test integral autorizado."""

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from .mmlu import DATASET, MMLUSource, file_hash, load_mmlu_parquet

REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
SELECTION_ID = "mmlu-3a-v1"
MANIFEST_SHA256 = "2e030192cdba40f31ec35ab33b32da1cdfe0207501e09b0edbdd1107dfd7fae1"
SUBJECTS = ("high_school_psychology", "high_school_world_history", "high_school_biology")
FULL_SELECTION_ID = "mmlu-full-test-v1"
FULL_COUNTS = dict(zip(SUBJECTS, (545, 237, 310), strict=True))
FULL_MANIFEST_SHA256 = "b3bb4c92c4745e69f191900a75d2a977e57b15a6da8dcba4805edc51045004da"


def validate_scope(subjects: tuple[str, ...], split: str, selection_id: str = SELECTION_ID) -> int:
    if selection_id == FULL_SELECTION_ID:
        if subjects != SUBJECTS or split != 'test':
            raise ValueError('A avaliação integral exige as três disciplinas e somente test.')
        return sum(FULL_COUNTS.values())
    if selection_id != SELECTION_ID:
        raise ValueError('Seleção MMLU não autorizada.')
    if (not isinstance(subjects, tuple) or not subjects
            or subjects != tuple(s for s in SUBJECTS if s in subjects)
            or split not in ("dev", "test")):
        raise ValueError("MMLU exige disciplinas únicas na ordem aprovada e split dev/test.")
    return len(subjects) * (5 if split == "dev" else 20)


def load_frozen_mmlu(manifest_path: Path, snapshot_path: Path,
                     subjects: tuple[str, ...], split: str, selection_id: str = SELECTION_ID):
    """Verifica manifesto e Parquets antes de retornar qualquer questão ao Runner.

    Não consulta o Hub, sorteia índices, filtra conteúdo ou depende de MLX.
    A assinatura do manifesto fixa também índices, hashes de linhas e tokenização.
    """
    expected_count = validate_scope(subjects, split, selection_id)
    full = selection_id == FULL_SELECTION_ID
    expected_hash = FULL_MANIFEST_SHA256 if full else MANIFEST_SHA256
    if file_hash(manifest_path) != expected_hash:
        raise ValueError("Manifesto MMLU diverge da seleção aprovada em 3A.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if full and (manifest['selection_id'] != FULL_SELECTION_ID
                 or manifest['status'] != 'preflight_passed_no_model_evaluation'
                 or manifest['over_limit_ids'] or manifest['contextual_error_ids']):
        raise ValueError('Seleção integral exige preflight aprovado, sem exclusões.')
    if not snapshot_path.is_dir() or snapshot_path.name != REVISION:
        raise ValueError("MMLU exige snapshot local da revisão aprovada; sem download automático.")
    frozen = {r['id']: (i, r) for i, r in enumerate(manifest['selected_records'])}
    selected, provenance, preparation, groups = [], [], [], []
    for group in manifest['groups']:
        if group['subject'] not in subjects or group['split'] != split:
            continue
        source = MMLUSource(**{k: group[k] for k in
            ('revision', 'subject', 'split', 'sha256', 'size_bytes')})
        items = load_mmlu_parquet(snapshot_path / source.filename, source)
        if len(items) != group['population_count']:
            raise ValueError("População MMLU incompatível com o manifesto.")
        if full and (len(items) != FULL_COUNTS[source.subject]
                     or group['selected_indices'] != list(range(len(items)))):
            raise ValueError('Seleção integral deve preservar todos os índices originais.')
        chosen = [items[i] for i in group['selected_indices']]
        if [item.question.id for item in chosen] != group['selected_ids']:
            raise ValueError("IDs/índices MMLU incompatíveis com a seleção congelada.")
        groups.append(group)
        for item in chosen:
            selection_index, expected = frozen[item.question.id]
            origin = item.provenance()
            if any(expected.get(k) != v for k, v in origin.items()):
                raise ValueError("Conteúdo/procedência MMLU diverge da linha aprovada.")
            provenance.append({**origin, 'selection_id': selection_id,
                               'selection_index': selection_index,
                               'execution_index': len(selected),
                               'anomalies': (['repeated_choice_text']
                                   if len(set(item.question.choices)) != 4 else [])})
            selected.append(item)
            preparation.append(expected)
    if len(selected) != expected_count:
        raise ValueError("Quantidade MMLU incompatível com o conjunto congelado.")
    try:
        reader_version = version('pyarrow')
    except PackageNotFoundError:
        reader_version = None
    return tuple(selected), {
        'selection': 'all_original_indices_in_source_order_v1' if full else 'frozen_3a_indices_no_resampling',
        'source': DATASET, 'filters': [], 'index_base': 0,
        'manifest_sha256': expected_hash,
        'frozen_selection_sha256': manifest['selection_sha256'],
        'groups': groups, 'question_provenance': provenance,
        'preparation_records': preparation,
        'anomaly_policy': 'preserve_all_original_rows' if full else 'preserve_original_approved_3a',
        'reader': {'library': 'pyarrow', 'version': reader_version},
    }
