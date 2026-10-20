"""Configuração estrita apenas para a Entrega 1."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import yaml

from .prompts import TEMPLATE


def _fields(value: object, expected: set[str], location: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{location}: campos obrigatórios e únicos: {', '.join(sorted(expected))}.")
    return value


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    limit: int
    seed: int
    output_directory: str

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment.name deve ser texto não vazio.")
        if type(self.limit) is not int or not 1 <= self.limit <= 3:
            raise ValueError("dataset.limit deve ser inteiro entre 1 e 3 na fixture v1.")
        if type(self.seed) is not int:
            raise ValueError("dataset.seed deve ser inteiro.")
        if not isinstance(self.output_directory, str) or not self.output_directory.strip():
            raise ValueError("output.directory deve ser texto não vazio.")

    def to_dict(self) -> dict:
        """Configuração completa; todos os métodos/revisões são fixos nesta entrega."""
        return {
            "version": 1,
            "experiment": {"name": self.experiment_name},
            "dataset": {
                "name": "synthetic", "revision": "v1", "split": "synthetic",
                "limit": self.limit, "seed": self.seed,
            },
            "model": {"backend": "simulated", "name": "controlled_scores", "revision": "v1"},
            "prompt": {"template": TEMPLATE},
            "scoring": {"method": "simulated_scores"},
            "metrics": ["accuracy", "normalized_entropy"],
            "output": {"directory": self.output_directory},
        }

    @classmethod
    def from_dict(cls, data: object) -> "RunConfig":
        schema = cls("example", 3, 42, "outputs").to_dict()
        data = _fields(data, set(schema), "config")
        for section, expected in schema.items():
            if isinstance(expected, dict):
                _fields(data[section], set(expected), section)
        config = cls(
            data["experiment"]["name"], data["dataset"]["limit"],
            data["dataset"]["seed"], data["output"]["directory"],
        )
        # Impede ativar silenciosamente datasets, backends ou métodos posteriores.
        resolved = config.to_dict()
        for section in resolved:
            if data[section] != resolved[section]:
                raise ValueError(f"{section}: valor incompatível com a fundação simulada.")
        if type(data["version"]) is not int:
            raise ValueError("version deve ser o inteiro 1.")
        return config


class _UniqueKeyLoader(yaml.SafeLoader):
    """Evita que uma chave YAML repetida substitua silenciosamente a anterior."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in mapping:
            raise ValueError("Chaves YAML devem ser textos únicos dentro de cada seção.")
        mapping[key] = loader.construct_object(value_node)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def load_config(path: Path) -> RunConfig:
    with path.open(encoding="utf-8") as stream:
        return RunConfig.from_dict(yaml.load(stream, Loader=_UniqueKeyLoader))
