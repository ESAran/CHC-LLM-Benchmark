"""Escores controlados, independentes do gabarito e do texto da questão."""


class SimulatedBackend:
    name = "controlled_scores"
    revision = "v1"
    capabilities = ("simulated_scores",)

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def score(self, prompt: str, labels: tuple[str, ...]) -> tuple[float, ...]:
        if not self.loaded:
            raise RuntimeError("O backend simulado precisa ser carregado.")
        # A alternativa de índice i recebe o escore i, sem consultar gabaritos.
        return tuple(float(index) for index in range(len(labels)))

    def close(self) -> None:
        self.loaded = False
