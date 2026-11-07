"""Adaptador local MMLU: sem rede, backend, inferência ou seleção por gabarito."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from .schemas import Question

DATASET = "cais/mmlu"
SELECTION_METHOD = "sha256_index_rank_v1"


def canonical_hash(value: object) -> str:
    """JSON UTF-8, chaves ordenadas, sem espaços, sem normalizar strings."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class MMLUSource:
    revision: str
    subject: str
    split: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("A revisão MMLU deve ser um SHA completo, não uma branch.")
        if not isinstance(self.subject, str) or not re.fullmatch(r"[a-z_]+", self.subject):
            raise ValueError("Disciplina inválida.")
        if self.split not in ("dev", "test"):
            raise ValueError("Esta preparação aceita somente dev/test autorizados.")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("SHA-256 do arquivo obrigatório.")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("Tamanho do arquivo deve ser inteiro positivo.")

    @property
    def filename(self) -> str:
        return f"{self.subject}/{self.split}-00000-of-00001.parquet"

    def question_id(self, index: int) -> str:
        if type(index) is not int or index < 0:
            raise ValueError("Índice original deve ser inteiro não negativo.")
        return f"{DATASET}@{self.revision}/{self.subject}/{self.split}/{index}"


@dataclass(frozen=True)
class MMLUItem:
    question: Question
    source: MMLUSource
    original_index: int
    row_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, MMLUSource) or not isinstance(self.question, Question):
            raise ValueError("Questão e procedência tipadas são obrigatórias.")
        q = self.question
        if (q.id != self.source.question_id(self.original_index)
                or q.subject != self.source.subject or len(q.choices) != 4):
            raise ValueError("Questão incompatível com identidade, posição ou disciplina.")
        expected = canonical_hash({"question": q.text, "subject": q.subject,
                                   "choices": q.choices, "answer": q.correct_index})
        if self.row_sha256 != expected:
            raise ValueError("Hash da linha incompatível com o conteúdo original.")

    def provenance(self) -> dict:
        return {"id": self.question.id, "dataset": DATASET,
                "revision": self.source.revision, "subject": self.source.subject,
                "split": self.source.split, "original_index": self.original_index,
                "source_file": self.source.filename, "source_sha256": self.source.sha256,
                "row_sha256": self.row_sha256}

    def to_dict(self) -> dict:
        return {"question": asdict(self.question), "provenance": self.provenance()}


def adapt_mmlu_rows(rows: Sequence[Mapping], source: MMLUSource) -> tuple[MMLUItem, ...]:
    if not rows:
        raise ValueError("Arquivo MMLU vazio.")
    items = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"question", "subject", "choices", "answer"}:
            raise ValueError(f"Linha {index}: schema MMLU inesperado.")
        if row["subject"] != source.subject:
            raise ValueError(f"Linha {index}: disciplina incompatível com a origem.")
        if not isinstance(row["choices"], (list, tuple)) or len(row["choices"]) != 4:
            raise ValueError(f"Linha {index}: exigem-se quatro alternativas.")
        question = Question(source.question_id(index), row["question"],
                            tuple(row["choices"]), row["answer"], source.subject)
        # Question valida textos não vazios e índice inteiro 0..3, sem coerção/strip.
        items.append(MMLUItem(question, source, index, canonical_hash(dict(row))))
    return tuple(items)


def read_parquet_rows(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Leitura Parquet requer diagnostics/mmlu-requirements.txt.") from error
    # Arquivo exato, sem descoberta de diretórios ou download de outros splits.
    return pq.ParquetFile(path).read(use_threads=False).to_pylist()


def load_mmlu_parquet(path: Path, source: MMLUSource) -> tuple[MMLUItem, ...]:
    if path.stat().st_size != source.size_bytes or file_hash(path) != source.sha256:
        raise ValueError(f"Arquivo diverge do tamanho/hash fixado: {source.filename}.")
    return adapt_mmlu_rows(read_parquet_rows(path), source)


def select_indices(population: int, count: int, seed: int, source: MMLUSource) -> tuple[int, ...]:
    if (type(population) is not int or type(count) is not int
            or not 1 <= count <= population or type(seed) is not int):
        raise ValueError("População, quantidade ou seed inválida.")
    # Só identidade, índice e seed: textos, respostas e resultados não são argumentos.
    def rank(index):
        return canonical_hash([SELECTION_METHOD, DATASET, source.revision,
                               source.subject, source.split, seed, index]), index
    chosen = sorted(range(population), key=rank)[:count]
    return tuple(sorted(chosen))  # Apresentação na ordem original do arquivo.


def duplicate_report(items: Sequence[MMLUItem]) -> dict:
    """Sinaliza repetições exatas sem remover/substituir qualquer linha."""
    ids = [item.question.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Identidades repetidas: mesma origem/posição foi carregada duas vezes.")
    text_groups, content_groups = defaultdict(list), defaultdict(list)
    repeated_choices = []
    for item in items:
        q = item.question
        text_groups[canonical_hash(q.text)].append(item)
        content_groups[canonical_hash([q.text, q.choices])].append(item)
        if len(set(q.choices)) != 4:
            repeated_choices.append(q.id)
    return {
        "same_question_text": [[i.question.id for i in group] for group in text_groups.values() if len(group) > 1],
        "same_question_and_choices": [
            {"ids": [i.question.id for i in group],
             "conflicting_answers": len({i.question.correct_index for i in group}) > 1}
            for group in content_groups.values() if len(group) > 1],
        "repeated_choices_within_question": repeated_choices,
        "policy": "report_only_no_filter_no_replacement",
    }
