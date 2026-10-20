"""Métricas puras sobre distribuições condicionadas às alternativas."""

from __future__ import annotations

from math import fsum
from typing import TYPE_CHECKING, Sequence

from .numerics import normalized_entropy, predicted_index

if TYPE_CHECKING:
    from .schemas import EvaluationRecord


def accuracy(correct: Sequence[bool]) -> float:
    if not correct or any(type(value) is not bool for value in correct):
        raise ValueError("Acurácia requer uma sequência não vazia de booleanos.")
    return sum(correct) / len(correct)


def summarize(records: Sequence[EvaluationRecord]) -> dict[str, int | float | None]:
    valid = [record for record in records if record.error is None]
    return {
        "total_count": len(records),
        "valid_count": len(valid),
        "failed_count": len(records) - len(valid),
        "accuracy": accuracy([record.correct for record in valid]) if valid else None,
        "mean_normalized_entropy": (
            fsum(record.normalized_entropy for record in valid) / len(valid) if valid else None
        ),
    }
