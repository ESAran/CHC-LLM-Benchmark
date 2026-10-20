"""Funções numéricas puras compartilhadas por métricas, scoring e contratos."""

from math import exp, fsum, isclose, isfinite, log2
from typing import Sequence


def is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def validate_probabilities(values: Sequence[float]) -> None:
    if len(values) < 2:
        raise ValueError("A distribuição precisa de pelo menos duas alternativas.")
    if any(not is_finite_number(p) or p < 0 or p > 1 for p in values):
        raise ValueError("Probabilidades devem ser números finitos em [0, 1].")
    if not isclose(fsum(values), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("A soma das probabilidades deve ser 1 (tolerância 1e-12).")


def softmax(scores: Sequence[float]) -> tuple[float, ...]:
    if len(scores) < 2 or any(not is_finite_number(score) for score in scores):
        raise ValueError("Softmax requer pelo menos dois escores numéricos finitos.")
    maximum = max(scores)
    weights = tuple(exp(score - maximum) for score in scores)
    total = fsum(weights)
    return tuple(weight / total for weight in weights)


def predicted_index(probabilities: Sequence[float]) -> int:
    """Argmax com desempate pelo menor índice na ordem original."""
    validate_probabilities(probabilities)
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def normalized_entropy(probabilities: Sequence[float]) -> float:
    validate_probabilities(probabilities)
    value = -fsum(p * log2(p) for p in probabilities if p > 0) / log2(len(probabilities))
    # Corrige apenas erro de arredondamento após validar a distribuição.
    return min(1.0, max(0.0, value))
