"""Entrada de linha de comando; sem fórmulas ou regras de pontuação."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from .config import load_config
from .runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark — execução sintética offline")
    parser.add_argument("--config", type=Path, required=True, help="Arquivo YAML da execução")
    args = parser.parse_args(argv)
    try:
        directory = run(load_config(args.config))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        for note in getattr(error, "__notes__", ()):
            print(note, file=sys.stderr)
        return 2
    print(f"Simulação salva em: {directory}")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if metadata["close_error"] is not None:
        print(f"Falha no encerramento do backend: {metadata['close_error']}", file=sys.stderr)
    return 1 if metadata["status"] != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
