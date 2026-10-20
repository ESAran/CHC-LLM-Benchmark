"""Pontuação controlada usada exclusivamente pela simulação."""

from ..backends.base import ModelBackend
from ..metrics import predicted_index
from ..schemas import ScoringResult
from .normalization import softmax


def score_choices(backend: ModelBackend, prompt: str, labels: tuple[str, ...]) -> ScoringResult:
    scores = tuple(backend.score(prompt, labels))
    if len(scores) != len(labels):
        raise ValueError("O backend devolveu quantidade de escores incompatível.")
    probabilities = softmax(scores)
    return ScoringResult(labels, scores, probabilities, predicted_index(probabilities))
