"""Persistência exclusiva por execução e proveniência local."""

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Sequence

import yaml

from . import __version__
from .config import RunConfig
from .schemas import EvaluationRecord, RunMetadata


def code_metadata() -> dict[str, object]:
    # Não usa o cwd do comando, que pode pertencer a outro repositório.
    source_root = Path(__file__).resolve().parents[2]
    result: dict[str, object] = {
        "package_version": __version__, "commit": None, "dirty": None,
    }
    if not (source_root / ".git").exists():
        return result
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    result.update(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()))
    return result


def environment_metadata() -> dict[str, object]:
    dependencies = {}
    for package in ("PyYAML", "llm-benchmark"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "dependencies": dependencies,
    }


def save_run(
    directory: Path,
    config: dict,
    metadata: RunMetadata,
    records: Sequence[EvaluationRecord],
    summary: dict,
) -> None:
    if directory.exists():
        raise FileExistsError(f"A pasta da execução já existe: {directory}")
    metadata.validate_records(records, summary)
    resolved = RunConfig.from_dict(config).to_dict()
    if metadata.run_id != directory.name or metadata.config_version != resolved["version"]:
        raise ValueError("Identificação da pasta/configuração incompatível com os metadados.")
    for section in ("dataset", "model"):
        identity = getattr(metadata, section)
        if any(identity.get(key) != value for key, value in resolved[section].items()):
            raise ValueError(f"Configuração e metadados incompatíveis: {section}.")
    if len(records) != resolved["dataset"]["limit"] or any(
        record.prompt_template != resolved["prompt"]["template"]
        or (record.scoring is not None and record.scoring.method != resolved["scoring"]["method"])
        for record in records
    ):
        raise ValueError("Configuração incompatível com os registros.")
    # Serializa/valida antes de criar a pasta; storage não contém fórmulas.
    contents = {
        "config.yaml": yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        "results.jsonl": "".join(
            json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ),
        "summary.json": json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        "metadata.json.tmp": json.dumps(asdict(metadata), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    }
    # A criação exclusiva reserva a pasta; uma colisão nunca sobrescreve outra execução.
    directory.mkdir(parents=True, exist_ok=False)
    try:
        for name, content in contents.items():
            with (directory / name).open("x", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        # Único sinal de finalização: publicado atomicamente após fechar todos os arquivos.
        # Se algo falhar antes, a pasta parcial permanece sem metadata.json.
        (directory / "metadata.json.tmp").replace(directory / "metadata.json")
    except Exception as error:
        error.add_note(f"Execução incompleta em {directory}; sem metadata.json final.")
        raise
