"""Contrato mínimo da fundação, sem prometer acesso a tokens ou logits reais."""

from typing import Protocol


class ModelBackend(Protocol):
    name: str
    revision: str
    capabilities: tuple[str, ...]

    def load(self) -> None: ...

    def score(self, prompt: str, labels: tuple[str, ...]) -> tuple[float, ...]: ...

    def close(self) -> None: ...
