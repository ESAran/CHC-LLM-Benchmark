"""Template versionado da simulação; não valida o protocolo de uma LLM."""

from .schemas import Question

TEMPLATE = "synthetic_multiple_choice_v1"


def build_prompt(question: Question) -> tuple[str, tuple[str, ...]]:
    # Identificadores numéricos suportam qualquer K >= 2 sem pressupor A–D.
    labels = tuple(str(index + 1) for index in range(len(question.choices)))
    alternatives = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, question.choices, strict=True)
    )
    prompt = (
        "Selecione uma alternativa e responda somente com seu identificador.\n\n"
        f"{question.text}\n{alternatives}\n\nResposta:"
    )
    return prompt, labels
