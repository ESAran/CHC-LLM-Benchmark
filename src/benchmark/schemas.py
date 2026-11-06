"""Contratos compartilhados, sem acesso a disco, datasets ou modelos."""

from dataclasses import dataclass, replace
from datetime import datetime
from math import fsum, isclose
import re
from typing import Sequence

from .metrics import summarize
from .numerics import (
    is_finite_number, normalized_entropy, predicted_index, softmax, validate_probabilities,
)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    choices: tuple[str, ...]
    correct_index: int
    subject: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("A questão precisa de um ID não vazio.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("O enunciado não pode ser vazio.")
        if not isinstance(self.choices, tuple) or len(self.choices) < 2:
            raise ValueError("choices deve ser uma tupla com pelo menos duas alternativas.")
        if any(not isinstance(c, str) or not c.strip() for c in self.choices):
            raise ValueError("Alternativas devem ser textos não vazios.")
        if type(self.correct_index) is not int or not 0 <= self.correct_index < len(self.choices):
            raise ValueError("correct_index deve ser um índice inteiro válido.")
        if self.subject is not None and not isinstance(self.subject, str):
            raise ValueError("subject deve ser texto ou None.")


@dataclass(frozen=True)
class TokenizationEvidence:
    input_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    position: int
    vocabulary_size: int
    inference_dtype: str
    forward_mode: str = "full_no_kv_cache"
    normalization_dtype: str = "float32_to_python_float64"

    def __post_init__(self) -> None:
        if type(self.vocabulary_size) is not int or self.vocabulary_size < 4:
            raise ValueError("Vocabulário inválido.")
        for ids in (self.input_ids, self.candidate_token_ids):
            if (not isinstance(ids, tuple) or not ids or
                    any(type(i) is not int or not 0 <= i < self.vocabulary_size for i in ids)):
                raise ValueError("IDs devem ser tuplas de índices válidos no vocabulário.")
        if len(self.candidate_token_ids) != 4 or len(set(self.candidate_token_ids)) != 4:
            raise ValueError("Exigem-se quatro tokens candidatos distintos.")
        if type(self.position) is not int or self.position != len(self.input_ids) - 1:
            raise ValueError("A posição deve prever o próximo token: T-1.")
        if (not _text(self.inference_dtype) or self.forward_mode != "full_no_kv_cache"
                or self.normalization_dtype != "float32_to_python_float64"):
            raise ValueError("Caminho/precisão de inferência incompatível com o protocolo.")


@dataclass(frozen=True)
class NextTokenOutput:
    vocabulary_logits: tuple[float, ...]
    tokenization: TokenizationEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.tokenization, TokenizationEvidence):
            raise ValueError("Evidência de tokenização obrigatória.")
        replace(self.tokenization)
        if (not isinstance(self.vocabulary_logits, tuple)
                or len(self.vocabulary_logits) != self.tokenization.vocabulary_size
                or any(not is_finite_number(x) for x in self.vocabulary_logits)):
            raise ValueError("Logits devem cobrir o vocabulário e ser finitos.")


@dataclass(frozen=True)
class ScoringResult:
    labels: tuple[str, ...]
    raw_scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    predicted_index: int
    method: str = "simulated_scores"
    score_unit: str = "synthetic_logits"
    normalization: str = "softmax_over_choices"
    tie_break: str = "lowest_index"
    tokenization_diagnostic: str = "not_applicable_simulation"
    original_label_probability_mass: float | None = None
    original_label_probabilities: tuple[float, ...] | None = None
    tokenization: TokenizationEvidence | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(value, tuple) for value in (
            self.labels, self.raw_scores, self.probabilities
        )):
            raise ValueError("Rótulos, escores e probabilidades devem ser tuplas.")
        validate_probabilities(self.probabilities)
        if len(self.labels) != len(self.probabilities) or len(self.raw_scores) != len(self.labels):
            raise ValueError("Rótulos, escores e probabilidades devem ter o mesmo tamanho.")
        if any(not _text(label) for label in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Rótulos devem ser únicos e não vazios.")
        expected_probabilities = softmax(self.raw_scores)
        if any(not isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
               for actual, expected in zip(self.probabilities, expected_probabilities, strict=True)):
            raise ValueError("Probabilidades incompatíveis com o softmax dos escores.")
        expected = predicted_index(self.probabilities)
        if type(self.predicted_index) is not int or self.predicted_index != expected:
            raise ValueError("Predição deve respeitar argmax e desempate pelo menor índice.")
        if self.normalization != "softmax_over_choices" or self.tie_break != "lowest_index":
            raise ValueError("Normalização/desempate incompatíveis.")
        if self.method == "simulated_scores":
            if (self.score_unit != "synthetic_logits"
                    or self.tokenization_diagnostic != "not_applicable_simulation"
                    or self.original_label_probability_mass is not None
                    or self.original_label_probabilities is not None or self.tokenization is not None):
                raise ValueError("A simulação não fornece evidência de tokens ou massa original.")
        elif self.method == "label_next_token":
            if (self.labels != tuple("ABCD") or self.score_unit != "logits"
                    or self.tokenization_diagnostic != "contextual_single_token_validated"
                    or not isinstance(self.tokenization, TokenizationEvidence)):
                raise ValueError("label_next_token exige A–D e evidência contextual válida.")
            replace(self.tokenization)
            mass, original = self.original_label_probability_mass, self.original_label_probabilities
            if (not is_finite_number(mass) or not 0 <= mass <= 1
                    or not isinstance(original, tuple) or len(original) != 4
                    or any(not is_finite_number(p) or not 0 <= p <= 1 for p in original)
                    or not isclose(fsum(original), mass, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Massa e probabilidades originais incompatíveis.")
            if mass > 0 and any(not isclose(p / mass, q, rel_tol=0, abs_tol=1e-12)
                                for p, q in zip(original, self.probabilities, strict=True)):
                raise ValueError("Distribuições original e condicionada incompatíveis.")
        else:
            raise ValueError("Método de pontuação não suportado.")


@dataclass(frozen=True)
class EvaluationRecord:
    question: Question
    prompt: str
    prompt_template: str
    scoring: ScoringResult | None
    correct: bool | None
    normalized_entropy: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question, Question):
            raise ValueError("O registro deve conter uma Question válida.")
        if not isinstance(self.prompt, str):
            raise ValueError("O prompt deve ser texto.")
        if not _text(self.prompt_template):
            raise ValueError("A versão do prompt deve ser registrada.")
        if self.error is not None:
            if not _text(self.error) or any(
                value is not None for value in (self.scoring, self.correct, self.normalized_entropy)
            ):
                raise ValueError("Uma falha deve ter erro explícito e métricas ausentes.")
            return
        if not isinstance(self.scoring, ScoringResult) or not _text(self.prompt):
            raise ValueError("Um resultado válido precisa de prompt e pontuação.")
        if len(self.scoring.labels) != len(self.question.choices):
            raise ValueError("A pontuação deve corresponder às alternativas da questão.")
        expected = self.scoring.predicted_index == self.question.correct_index
        if type(self.correct) is not bool or self.correct != expected:
            raise ValueError("O indicador de acerto deve corresponder ao gabarito.")
        if not is_finite_number(self.normalized_entropy):
            raise ValueError("A entropia deve ser finita.")
        if not 0 <= self.normalized_entropy <= 1:
            raise ValueError("A entropia deve pertencer a [0, 1].")
        if not isclose(self.normalized_entropy, normalized_entropy(self.scoring.probabilities),
                       rel_tol=0, abs_tol=1e-12):
            raise ValueError("Entropia incompatível com as probabilidades do registro.")


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    started_at: str
    finished_at: str
    config_version: int
    code: dict[str, object]
    dataset: dict[str, object]
    model: dict[str, object]
    environment: dict[str, object]
    question_ids: tuple[str, ...]
    valid_count: int
    failed_count: int
    status: str
    simulation: bool = True
    close_error: str | None = None

    def __post_init__(self) -> None:
        if not _text(self.run_id):
            raise ValueError("Metadados exigem identificação da execução.")
        try:
            started = datetime.fromisoformat(self.started_at)
            finished = datetime.fromisoformat(self.finished_at)
            if started.utcoffset() is None or finished.utcoffset() is None or finished < started:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("Horários devem ser ISO 8601, com fuso e fim não anterior ao início.") from None
        if type(self.config_version) is not int or self.config_version != 1 or type(self.simulation) is not bool:
            raise ValueError("Exigem-se configuração v1 e indicador explícito de simulação.")
        if not isinstance(self.model, dict):
            raise ValueError("Metadados do modelo devem ser um dicionário.")
        if not self.simulation and self.model.get("backend") != "mlx":
            raise ValueError("Execução real exige backend MLX identificado.")
        if self.simulation and self.model.get("backend", "simulated") != "simulated":
            raise ValueError("Execução simulada não pode declarar backend real.")
        if any(type(count) is not int or count < 0 for count in (self.valid_count, self.failed_count)):
            raise ValueError("Contagens devem ser inteiros não negativos.")
        if (not isinstance(self.question_ids, tuple) or not self.question_ids
                or any(not _text(item) for item in self.question_ids)
                or len(set(self.question_ids)) != len(self.question_ids)):
            raise ValueError("IDs da execução devem ser uma tupla não vazia de textos únicos.")
        if self.valid_count + self.failed_count != len(self.question_ids):
            raise ValueError("Contagens devem corresponder aos IDs avaliados.")
        if self.close_error is not None and not _text(self.close_error):
            raise ValueError("Erro de encerramento deve ser texto não vazio ou None.")
        expected_status = (
            "failed" if self.valid_count == 0 else
            "completed_with_errors" if self.failed_count or self.close_error else "completed"
        )
        if self.status != expected_status:
            raise ValueError("Status incompatível com contagens e erro de encerramento.")
        for identity in (self.dataset, self.model):
            if not isinstance(identity, dict) or any(
                not _text(identity.get(key)) for key in ("name", "revision")
            ):
                raise ValueError("Dataset e modelo exigem nome e revisão.")
        if not isinstance(self.code, dict) or not {"package_version", "commit", "dirty"} <= self.code.keys():
            raise ValueError("Metadados de código exigem versão, commit e estado local explícitos.")
        commit = self.code["commit"]
        dirty = self.code["dirty"]
        if (not _text(self.code["package_version"])
                or (commit is not None and (
                    not isinstance(commit, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
                ))
                or (dirty is not None and type(dirty) is not bool)
                or (commit is None) != (dirty is None)):
            raise ValueError("Versão/commit/estado local inválidos; ausências devem ser explícitas.")
        if not isinstance(self.environment, dict) or any(
            not _text(self.environment.get(key)) for key in ("python", "system", "release", "machine")
        ):
            raise ValueError("Ambiente exige versão Python, sistema, release e arquitetura.")
        dependencies = self.environment.get("dependencies")
        if (not isinstance(dependencies, dict)
                or not {"PyYAML", "llm-benchmark"} <= dependencies.keys()
                or any(not _text(name) or (value is not None and not _text(value))
                       for name, value in dependencies.items())):
            raise ValueError("Dependências exigem nomes e versões, ou None quando indisponíveis.")

    def validate_records(self, records: Sequence[EvaluationRecord], summary: dict) -> None:
        """Valida o conjunto antes da escrita usando os cálculos compartilhados."""
        replace(self)  # Revalida dicionários de metadados, que podem ter sido mutados.
        for record in records:
            if not isinstance(record, EvaluationRecord):
                raise ValueError("Resultados devem ser EvaluationRecord.")
            replace(record, question=replace(record.question),
                    scoring=replace(record.scoring) if record.scoring is not None else None)
        if tuple(record.question.id for record in records) != self.question_ids:
            raise ValueError("IDs/ordem dos registros incompatíveis com os metadados.")
        expected = summarize(records)
        if self.valid_count != expected["valid_count"] or self.failed_count != expected["failed_count"]:
            raise ValueError("Contagens incompatíveis com os registros.")
        if not isinstance(summary, dict) or summary.keys() != expected.keys():
            raise ValueError("Campos do resumo incompatíveis com os registros.")
        for key, value in expected.items():
            actual = summary[key]
            if type(actual) is not type(value) or (
                isinstance(value, float) and not isclose(actual, value, rel_tol=0, abs_tol=1e-12)
            ) or (not isinstance(value, float) and actual != value):
                raise ValueError(f"Resumo incompatível com os registros: {key}.")
