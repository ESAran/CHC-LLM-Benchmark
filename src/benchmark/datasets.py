"""Pequena fixture original; não contém nem carrega exemplos do MMLU."""

import random
from collections.abc import Iterable, Mapping

from .schemas import Question


SYNTHETIC_ROWS = (
    {"id": "synthetic-001", "text": "Quanto é 1 + 1?", "choices": ("1", "2"), "correct_index": 1},
    {"id": "synthetic-002", "text": "Qual número é par?", "choices": ("2", "3", "5"), "correct_index": 0},
    {"id": "synthetic-003", "text": "Quanto é 2 + 2?", "choices": ("1", "2", "3", "4"), "correct_index": 3},
)


def adapt_rows(rows: Iterable[Mapping[str, object]]) -> tuple[Question, ...]:
    questions = []
    seen = set()
    for row in rows:
        if not isinstance(row.get("choices"), (tuple, list)):
            raise ValueError("choices deve ser uma lista ou tupla de alternativas.")
        question = Question(
            id=row["id"],
            text=row["text"],
            choices=tuple(row["choices"]),
            correct_index=row["correct_index"],
            subject=row.get("subject"),
        )
        if question.id in seen:
            raise ValueError(f"ID de questão duplicado: {question.id}")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise ValueError("O dataset não pode estar vazio.")
    return tuple(questions)


def load_synthetic(limit: int, seed: int) -> tuple[Question, ...]:
    questions = list(adapt_rows(SYNTHETIC_ROWS))
    if type(limit) is not int or not 1 <= limit <= len(questions):
        raise ValueError(f"limit deve estar entre 1 e {len(questions)}.")
    if type(seed) is not int:
        raise ValueError("seed deve ser inteiro.")
    random.Random(seed).shuffle(questions)
    return tuple(questions[:limit])


# Fixture de regressão de 2B: mesmos casos da comparação aprovada, sem MMLU.
CONTROLLED_QUESTIONS = (
    ("triangle", "How many sides does a triangle have?", ("3", "2", "4", "5"), "elementary_mathematics"),
    ("planet", "Which planet is known as the Red Planet?", ("Mars", "Venus", "Jupiter", "Saturn"), "astronomy"),
    ("organ", "Which organ pumps blood through the human body?", ("Heart", "Liver", "Lung", "Stomach"), "anatomy"),
    ("meter", "How many centimeters are in one meter?", ("100", "10", "1000", "1"), "elementary_mathematics"),
)


def load_controlled(limit: int, seed: int) -> tuple[Question, ...]:
    questions = [Question(f"controlled-{qid}-{rotation}", text,
                         choices[-rotation:] + choices[:-rotation] if rotation else choices,
                         rotation, subject)
                 for qid, text, choices, subject in CONTROLLED_QUESTIONS
                 for rotation in range(4)]
    if type(limit) is not int or not 1 <= limit <= len(questions) or type(seed) is not int:
        raise ValueError("Limite (1–16) ou seed inválido na fixture controlada.")
    random.Random(seed).shuffle(questions)
    return tuple(questions[:limit])
