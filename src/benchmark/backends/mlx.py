"""Inferência local MLX: template do snapshot e forward completo, sem KV cache.

Imports MLX são tardios: instalar/importar o benchmark simulado não requer Metal.
O backend não constrói questões, não conhece gabaritos e não normaliza escores.
"""

import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from ..schemas import NextTokenOutput, TokenizationEvidence

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
REVISION = "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
# Manifesto do snapshot aprovado em 2A; recusa deriva de pesos/tokenizer.
APPROVED_SHA256 = {'added_tokens.json': '58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b',
 'config.json': 'bc8d587c364e4905e8510be14d07c5e69c84347c17f7c7607d5ee4470e72cb50',
 'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
 'model.safetensors': '39da1e4174f12953ad7ace17239da672aa2b4b9ba723a09835963484c5a05b41',
 'model.safetensors.index.json': 'db341d98a68822279de81d9fbe29cb9bf0077ad032ce7bd10ed0a9f4c24f68fa',
 'special_tokens_map.json': '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd',
 'tokenizer.json': 'a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf',
 'tokenizer_config.json': 'f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7',
 'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}


def _load_runtime():
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as error:
        raise RuntimeError("MLX-LM não instalado; use o ambiente de diagnostics/mlx-requirements.txt.") from error
    return mx, load


class MLXBackend:
    capabilities = ("label_next_token",)

    def __init__(self, snapshot_path: str, *, max_prompt_tokens: int = 1024):
        self.name, self.revision = MODEL, REVISION
        self.snapshot_path = Path(snapshot_path)
        if type(max_prompt_tokens) is not int or max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens deve ser inteiro positivo.")
        self.max_prompt_tokens = max_prompt_tokens
        self._mx = self._model = self._tokenizer = None
        self._previous_cache_limit = None
        self._details = {}

    def _verify_snapshot(self) -> dict:
        if not self.snapshot_path.is_dir() or self.snapshot_path.name != self.revision:
            raise ValueError("Informe um snapshot local existente da revisão fixada; não há download automático.")
        allowed = set(APPROVED_SHA256) | {"README.md"}
        if any(p.name not in allowed for p in self.snapshot_path.iterdir()):
            raise ValueError("Snapshot contém arquivos não previstos no manifesto aprovado.")
        hashes = {}
        for name, expected in APPROVED_SHA256.items():
            with (self.snapshot_path / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            if hashes[name] != expected:
                raise ValueError(f"Snapshot diverge do manifesto aprovado: {name}.")
        return hashes

    def load(self) -> None:
        if self._model is not None:
            raise RuntimeError("Backend já carregado.")
        hashes = self._verify_snapshot()
        self._mx, load = _load_runtime()
        self._previous_cache_limit = self._mx.set_cache_limit(256 * 1024**2)
        self._mx.reset_peak_memory()
        self._model, self._tokenizer = load(
            str(self.snapshot_path), trust_remote_code=False,
            tokenizer_config={"local_files_only": True, "trust_remote_code": False},
        )
        self._model.eval()
        expected_template = json.loads((self.snapshot_path / "tokenizer_config.json").read_text())["chat_template"]
        if self._tokenizer.chat_template != expected_template:
            raise ValueError("O chat template carregado diverge do snapshot.")
        self._details = {
            "snapshot_sha256": hashes, "chat_template": self._tokenizer.chat_template,
            "chat_template_sha256": hashlib.sha256(self._tokenizer.chat_template.encode()).hexdigest(),
            "forward_mode": "full_no_kv_cache", "batch_size": 1, "fewshot_count": 0,
            "add_generation_prompt": True, "add_special_tokens": False,
            "normalization_dtype": "float32_to_python_float64", "cache_limit_bytes": 256 * 1024**2,
            "max_prompt_tokens": self.max_prompt_tokens,
            "dependencies": {p: version(p) for p in ("mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "huggingface-hub", "numpy", "safetensors")},
        }

    def _require_loaded(self):
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("O backend MLX precisa ser carregado.")

    def prepare_prompt(self, user_prompt: str) -> str:
        self._require_loaded()
        messages = [{"role": "user", "content": user_prompt}]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self._tokenizer.encode(prompt, add_special_tokens=False) != self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True):
            raise ValueError("Renderização e tokenização do chat template divergem.")
        return prompt

    def next_token_logits(self, prompt: str, labels: tuple[str, ...]) -> NextTokenOutput:
        self._require_loaded()
        if labels != tuple("ABCD"):
            raise ValueError("O protocolo inicial exige A, B, C, D sem espaços.")
        tokenizer, mx = self._tokenizer, self._mx
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.max_prompt_tokens:
            raise ValueError("Comprimento do prompt fora do limite; não há truncamento silencioso.")
        candidate_ids = []
        for label in labels:
            full = tokenizer.encode(prompt + label, add_special_tokens=False)
            if full[:len(ids)] != ids or len(full) != len(ids) + 1:
                raise ValueError(f"Rótulo {label}: fronteira contextual alterada ou continuação não unitária.")
            token_id = full[-1]
            if token_id in tokenizer.all_special_ids or tokenizer.decode([token_id]) != label:
                raise ValueError(f"Rótulo {label}: token especial ou decodificação incompatível.")
            candidate_ids.append(token_id)
        if len(set(candidate_ids)) != len(labels):
            raise ValueError("Tokens candidatos duplicados.")
        # Sem cache, sem generate(), sem processadores ou penalidades de logits.
        logits = self._model(mx.array([ids]))
        try:
            if len(logits.shape) != 3 or logits.shape[:2] != (1, len(ids)):
                raise ValueError("Shape dos logits incompatível com batch 1 e prompt completo.")
            evidence = TokenizationEvidence(tuple(ids), tuple(candidate_ids), len(ids) - 1,
                                            logits.shape[2], str(logits.dtype))
            last = logits[0, -1, :].astype(mx.float32)
            mx.eval(last)
            values = tuple(last.tolist())  # float Python: double, antes de qualquer softmax.
            return NextTokenOutput(values, evidence)
        finally:
            del logits
            mx.clear_cache()

    def metadata(self) -> dict:
        return dict(self._details)

    def close(self) -> None:
        self._model = self._tokenizer = None
        gc.collect()
        if self._mx is not None:
            try:
                self._details["mlx_peak_bytes"] = self._mx.get_peak_memory()
                self._mx.clear_cache()
                self._details["mlx_active_bytes_after_close"] = self._mx.get_active_memory()
            finally:
                if self._previous_cache_limit is not None:
                    self._mx.set_cache_limit(self._previous_cache_limit)
                    self._previous_cache_limit = None
