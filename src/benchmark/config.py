"""Configuração estrita para simulação e protocolo MLX provisório aprovado."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import yaml

from .prompts import TEMPLATE, CURRENT_TEMPLATE, TEMPLATES
from .backends.mlx import MODEL, REVISION


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
    backend: str = "simulated"
    dataset_revision: str = "v1"
    prompt_template: str = TEMPLATE
    snapshot_path: str | None = None
    max_prompt_tokens: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment.name deve ser texto não vazio.")
        if self.backend not in ("simulated", "mlx") or self.dataset_revision not in ("v1", "controlled-v1"):
            raise ValueError("Backend ou revisão de fixture não suportado.")
        maximum = 3 if self.dataset_revision == "v1" else 16
        if type(self.limit) is not int or not 1 <= self.limit <= maximum:
            raise ValueError(f"dataset.limit deve ser inteiro entre 1 e {maximum}.")
        if type(self.seed) is not int:
            raise ValueError("dataset.seed deve ser inteiro.")
        if not isinstance(self.output_directory, str) or not self.output_directory.strip():
            raise ValueError("output.directory deve ser texto não vazio.")
        if self.prompt_template not in TEMPLATES:
            raise ValueError("Template desconhecido.")
        if type(self.max_prompt_tokens) is not int or not 1 <= self.max_prompt_tokens <= 1024:
            raise ValueError("max_prompt_tokens deve estar entre 1 e 1024 nesta integração inicial.")
        if self.backend == "mlx":
            if self.dataset_revision != "controlled-v1" or self.prompt_template != CURRENT_TEMPLATE:
                raise ValueError("MLX exige fixture de quatro opções e template aprovado nesta etapa.")
            if not isinstance(self.snapshot_path, str) or not self.snapshot_path.strip():
                raise ValueError("MLX exige snapshot_path local explícito.")
        elif self.snapshot_path is not None or self.max_prompt_tokens != 1024:
            raise ValueError("Opções de inferência MLX não se aplicam à simulação.")
        if self.prompt_template == CURRENT_TEMPLATE and self.dataset_revision != "controlled-v1":
            raise ValueError("O template aprovado exige fixture de quatro opções.")

    @property
    def scoring_method(self) -> str:
        return "label_next_token" if self.backend == "mlx" else "simulated_scores"

    def to_dict(self) -> dict:
        model = {"backend": "simulated", "name": "controlled_scores", "revision": "v1"}
        if self.backend == "mlx":
            model = {"backend": "mlx", "name": MODEL, "revision": REVISION,
                     "snapshot_path": self.snapshot_path, "max_prompt_tokens": self.max_prompt_tokens}
        return {
            "version": 1,
            "experiment": {"name": self.experiment_name},
            "dataset": {"name": "synthetic", "revision": self.dataset_revision, "split": "synthetic",
                        "limit": self.limit, "seed": self.seed},
            "model": model,
            "prompt": {"template": self.prompt_template},
            "scoring": {"method": self.scoring_method},
            "metrics": ["accuracy", "normalized_entropy"],
            "output": {"directory": self.output_directory},
        }

    @classmethod
    def from_dict(cls, data: object) -> "RunConfig":
        schema = cls("example", 3, 42, "outputs").to_dict()
        data = _fields(data, set(schema), "config")
        for section, expected in schema.items():
            if isinstance(expected, dict):
                fields = set(expected)
                if section == "model" and isinstance(data[section], Mapping) and data[section].get("backend") == "mlx":
                    fields |= {"snapshot_path", "max_prompt_tokens"}
                _fields(data[section], fields, section)
        config = cls(data["experiment"]["name"], data["dataset"]["limit"], data["dataset"]["seed"],
                     data["output"]["directory"], data["model"]["backend"], data["dataset"]["revision"],
                     data["prompt"]["template"], data["model"].get("snapshot_path"),
                     data["model"].get("max_prompt_tokens", 1024))
        for section, expected in config.to_dict().items():
            if data[section] != expected:
                raise ValueError(f"{section}: valor incompatível com o protocolo suportado.")
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
