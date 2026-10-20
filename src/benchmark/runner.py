"""Orquestração sequencial da fundação simulada."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .backends.base import ModelBackend
from .backends.simulated import SimulatedBackend
from .config import RunConfig
from .datasets import load_synthetic
from .metrics import normalized_entropy, summarize
from .prompts import TEMPLATE, build_prompt
from .schemas import EvaluationRecord, RunMetadata
from .scoring import score_choices
from .storage import code_metadata, environment_metadata, save_run


def run(config: RunConfig, backend: ModelBackend | None = None) -> Path:
    """Executa a fixture; injeção de backend existe para testar o contrato simulado."""
    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%S%fZ}_{uuid4().hex}"
    resolved = config.to_dict()
    questions = load_synthetic(config.limit, config.seed)
    backend = backend if backend is not None else SimulatedBackend()
    if "simulated_scores" not in backend.capabilities:
        raise ValueError("A Entrega 1 exige a capacidade simulated_scores.")
    if (backend.name, backend.revision) != (
        resolved["model"]["name"], resolved["model"]["revision"]
    ):
        raise ValueError("Identidade do backend incompatível com a configuração.")
    code = code_metadata()
    environment = environment_metadata()
    records = []
    primary_error = None
    close_error = None
    try:
        backend.load()
        for question in questions:
            prompt = ""
            try:
                prompt, labels = build_prompt(question)
                scoring = score_choices(backend, prompt, labels)
                record = EvaluationRecord(
                    question, prompt, TEMPLATE, scoring,
                    scoring.predicted_index == question.correct_index,
                    normalized_entropy(scoring.probabilities),
                )
            except Exception as error:
                # Falhas por item são evidências explícitas, nunca descartes silenciosos.
                record = EvaluationRecord(
                    question, prompt, TEMPLATE, None, None, None,
                    error=f"{type(error).__name__}: {error}",
                )
            records.append(record)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            backend.close()
        except Exception as error:
            close_error = f"{type(error).__name__}: {error}"
            if primary_error is not None:
                primary_error.add_note(f"Falha adicional no encerramento do backend: {close_error}")
    summary = summarize(records)
    failed = summary["failed_count"]
    status = "completed" if not failed and close_error is None else "completed_with_errors"
    if not summary["valid_count"]:
        status = "failed"
    metadata = RunMetadata(
        run_id=run_id,
        started_at=started.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        config_version=resolved["version"],
        code=code,
        dataset={
            **resolved["dataset"],
            "selection": "seeded_shuffle_then_limit",
            "source": "bundled_original_synthetic_fixture",
            "filters": [],
        },
        model={
            **resolved["model"],
            "implementation": type(backend).__name__,
            "capabilities": list(backend.capabilities),
            "score_rule": "score[i] = float(i) in the default simulated backend",
            "chat_template": None,
        },
        environment=environment,
        question_ids=tuple(question.id for question in questions),
        valid_count=summary["valid_count"],
        failed_count=failed,
        status=status,
        close_error=close_error,
    )
    directory = Path(config.output_directory) / run_id
    try:
        save_run(directory, resolved, metadata, records, summary)
    except Exception as error:
        if close_error is not None:
            error.add_note(f"Falha anterior no encerramento do backend: {close_error}")
        for record in records:
            if record.error is not None:
                error.add_note(f"Falha anterior na questão {record.question.id}: {record.error}")
        raise
    return directory
