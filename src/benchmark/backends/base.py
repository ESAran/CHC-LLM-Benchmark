"""Contratos de ciclo de vida e capacidades independentes da biblioteca de inferência."""

from typing import Protocol
from ..schemas import NextTokenOutput


class ModelBackend(Protocol):
    name: str
    revision: str
    capabilities: tuple[str, ...]

    def load(self) -> None: ...
    def close(self) -> None: ...
    def metadata(self) -> dict: ...


class SimulatedScoresBackend(ModelBackend, Protocol):
    def score(self, prompt: str, labels: tuple[str, ...]) -> tuple[float, ...]: ...


class NextTokenBackend(ModelBackend, Protocol):
    def prepare_prompt(self, user_prompt: str) -> str: ...
    def next_token_logits(self, prompt: str, labels: tuple[str, ...]) -> NextTokenOutput: ...
