"""Leitura de resultados persistidos: contratos existentes, sem inferência ou rede."""

import json
from pathlib import Path

from .config import load_config
from .mmlu import file_hash
from .schemas import EvaluationRecord, Question, RunMetadata, ScoringResult, TokenizationEvidence


def read_record(data):
    question = Question(**{**data['question'], 'choices': tuple(data['question']['choices'])})
    scoring = data['scoring']
    if scoring is not None:
        scoring = dict(scoring)
        for key in ('labels', 'raw_scores', 'probabilities', 'original_label_probabilities'):
            if scoring[key] is not None:
                scoring[key] = tuple(scoring[key])
        token = scoring['tokenization']
        if token is not None:
            scoring['tokenization'] = TokenizationEvidence(**{
                **token, 'input_ids': tuple(token['input_ids']),
                'candidate_token_ids': tuple(token['candidate_token_ids'])})
        scoring = ScoringResult(**scoring)
    return EvaluationRecord(**{**data, 'question': question, 'scoring': scoring})



def read_run(directory: Path, expected_hashes: dict[str, str]):
    """Exige os quatro artefatos completos e os hashes da evidência aprovada."""
    names = {'config.yaml', 'results.jsonl', 'summary.json', 'metadata.json'}
    if set(expected_hashes) != names or {p.name for p in directory.iterdir()} != names:
        raise ValueError('Execução incompleta ou lista de artefatos incompatível.')
    for name in names:
        if file_hash(directory / name) != expected_hashes[name]:
            raise ValueError(f'Artefato persistido diverge da evidência: {name}.')
    config = load_config(directory / 'config.yaml')
    data = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    metadata = RunMetadata(**{**data, 'question_ids': tuple(data['question_ids'])})
    records = tuple(read_record(json.loads(line)) for line in
                    (directory / 'results.jsonl').read_text(encoding='utf-8').splitlines())
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    metadata.validate_records(records, summary)
    if metadata.run_id != directory.name or len(records) != config.limit:
        raise ValueError('Identidade ou contagem incompatível com a execução.')
    for section in ('dataset', 'model'):
        if any(getattr(metadata, section).get(k) != v for k, v in config.to_dict()[section].items()):
            raise ValueError(f'Configuração incompatível com metadados: {section}.')
    if any(r.prompt_template != config.prompt_template or
           (r.scoring is not None and r.scoring.method != config.scoring_method) for r in records):
        raise ValueError('Protocolo dos registros incompatível com a configuração.')
    return config, metadata, records, summary
