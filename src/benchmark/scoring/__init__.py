"""Transforma evidência dos backends em distribuições identificadas."""

from math import fsum
from ..backends.base import ModelBackend
from ..metrics import predicted_index
from ..schemas import NextTokenOutput, ScoringResult
from .normalization import softmax


def score_choices(backend: ModelBackend, prompt: str, labels: tuple[str, ...],
                  method: str = "simulated_scores") -> ScoringResult:
    if method not in backend.capabilities:
        raise ValueError("Backend não suporta o método solicitado.")
    if method == "simulated_scores":
        scores = tuple(backend.score(prompt, labels))
        if len(scores) != len(labels):
            raise ValueError("O backend devolveu quantidade de escores incompatível.")
        probabilities = softmax(scores)
        return ScoringResult(labels, scores, probabilities, predicted_index(probabilities))
    if method != "label_next_token" or labels != tuple("ABCD"):
        raise ValueError("Método ou convenção de rótulos não suportados.")
    output = backend.next_token_logits(prompt, labels)
    if not isinstance(output, NextTokenOutput):
        raise ValueError("O backend deve fornecer NextTokenOutput.")
    ids = output.tokenization.candidate_token_ids
    scores = tuple(output.vocabulary_logits[i] for i in ids)
    probabilities = softmax(scores)
    vocabulary_probabilities = softmax(output.vocabulary_logits)
    original = tuple(vocabulary_probabilities[i] for i in ids)
    return ScoringResult(
        labels, scores, probabilities, predicted_index(probabilities),
        method=method, score_unit="logits",
        tokenization_diagnostic="contextual_single_token_validated",
        original_label_probability_mass=fsum(original),
        original_label_probabilities=original, tokenization=output.tokenization,
    )
