"""Templates versionados: simulação e texto zero-shot aprovado, sem inferência."""

from .schemas import Question

TEMPLATE = "synthetic_multiple_choice_v1"
CURRENT_TEMPLATE = "current-no-enumeration-v1"
CURRENT_INSTRUCTION = "Reply with exactly one letter. Do not add spaces or explanations."
TEMPLATES = (TEMPLATE, CURRENT_TEMPLATE)


def build_prompt(question: Question, template: str = TEMPLATE) -> tuple[str, tuple[str, ...]]:
    if template == CURRENT_TEMPLATE:
        if len(question.choices) != 4:
            raise ValueError("current-no-enumeration-v1 exige quatro alternativas.")
        labels = tuple("ABCD")
        alternatives = "\n".join(f"{k}. {v}" for k, v in zip(labels, question.choices, strict=True))
        return question.text + "\n" + alternatives + "\n" + CURRENT_INSTRUCTION, labels
    if template != TEMPLATE:
        raise ValueError("Template desconhecido.")
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
