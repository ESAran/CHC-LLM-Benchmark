"""Diagnóstico 2A isolado; não é um backend nem um experimento MMLU.

Executar com PYTHONPATH=src no ambiente descrito em docs/reports/MLX_DIAGNOSTIC.md.
O snapshot deve ter sido baixado previamente na revisão fixada abaixo.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import generate, load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import numpy as np

from benchmark.numerics import softmax


MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
REVISION = "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
LABELS = "ABCD"
PREFIXES = ("", " ", "\n", "\n ", "\n\n")


def memory():
    return {
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "swap": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip(),
    }


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contextual_rows(tokenizer, prompt):
    base = tokenizer.encode(prompt, add_special_tokens=False)
    rows = []
    for prefix in PREFIXES:
        for label in LABELS:
            candidate = prefix + label
            full = tokenizer.encode(prompt + candidate, add_special_tokens=False)
            preserved = full[:len(base)] == base
            suffix = full[len(base):] if preserved else None
            common = 0
            for a, b in zip(base, full):
                if a != b:
                    break
                common += 1
            rows.append({
                "candidate": candidate,
                "standalone_ids": tokenizer.encode(candidate, add_special_tokens=False),
                "prefix_preserved": preserved,
                "common_prefix_length": common,
                "contextual_suffix_ids": suffix,
                "single_token": preserved and len(suffix) == 1,
                "full_tail_ids": full[-8:],
            })
    return rows


def inspect_prompt(model, tokenizer, prompt, name, output):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    rows = contextual_rows(tokenizer, prompt)
    candidates = [row for row in rows if row["candidate"] in LABELS]
    if not all(row["single_token"] for row in candidates):
        return {"name": name, "prompt": prompt, "input_ids": ids,
                "tokenization": rows, "scoring": None,
                "reason": "Bare labels are not single-token continuations of this prefix."}
    token_ids = [row["contextual_suffix_ids"][0] for row in candidates]
    assert len(set(token_ids)) == 4
    assert not set(token_ids).intersection(tokenizer.all_special_ids)
    started = time.perf_counter()
    output_logits = model(mx.array([ids]))
    last = output_logits[0, -1, :].astype(mx.float32)
    mx.eval(last)
    logits = np.array(last).astype(np.float64)
    assert np.isfinite(logits).all()
    np.save(output / f"{name}-logits.npy", logits)
    scores = logits[token_ids]
    conditional = np.array(softmax(scores.tolist()))
    # Referência independente sobre o vocabulário completo, em float64.
    weights = np.exp(logits - logits.max())
    vocab_probs = weights / weights.sum()
    original = vocab_probs[token_ids]
    mass = float(original.sum())
    assert 0 < mass <= 1
    assert np.allclose(conditional, original / mass, rtol=0, atol=1e-12)
    mx_probs = np.array(mx.softmax(mx.array(scores.tolist(), dtype=mx.float32)))
    assert np.allclose(conditional, mx_probs, rtol=0, atol=1e-6)
    full_mx_probs = mx.softmax(mx.array(logits.astype(np.float32)))
    full_mx_mass = float(mx.sum(full_mx_probs[mx.array(token_ids)]).item())
    assert math.isclose(full_mx_mass, mass, rel_tol=1e-5, abs_tol=1e-7)
    shape, dtype = list(output_logits.shape), str(output_logits.dtype)
    previous = output_logits[0, -2, :].astype(mx.float32)
    mx.eval(previous)
    previous_scores = np.array(previous)[token_ids].tolist()
    previous_argmax = int(mx.argmax(previous).item())
    del previous
    del output_logits, last
    mx.clear_cache()

    # Referência da biblioteca, primeiro token, sem penalidades/temperatura.
    captured = []
    def capture_logits(history, raw_logits):
        captured.append(raw_logits[0].astype(mx.float32))
        return raw_logits
    first = next(stream_generate(model, tokenizer, ids, max_tokens=1,
                                sampler=make_sampler(temp=0),
                                logits_processors=[capture_logits]))
    library_logits = np.array(captured[0]).astype(np.float64)
    np.save(output / f"{name}-generation-logits.npy", library_logits)
    library_raw_error = float(np.max(np.abs(library_logits - logits)))
    library_weights = np.exp(library_logits - library_logits.max())
    library_ref_logprobs = library_logits - (library_logits.max() + np.log(library_weights.sum()))
    library_logprobs = np.array(first.logprobs.astype(mx.float32)).astype(np.float64)
    reference_logprobs = logits - (logits.max() + np.log(weights.sum()))
    logprob_error = float(np.max(np.abs(reference_logprobs - library_logprobs)))
    # Cache: pré-preencher até T-2 e avaliar somente o último token de entrada.
    cache = make_prompt_cache(model)
    model(mx.array([ids[:-1]]), cache=cache)
    cached = model(mx.array([ids[-1:]]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(cached)
    cache_error = float(np.max(np.abs(np.array(cached) - logits)))
    top = np.argsort(logits)[-5:][::-1].tolist()
    result = {
        "name": name, "prompt": prompt, "input_ids": ids,
        "tokenization": rows, "output_shape": shape, "output_dtype": dtype,
        "position": len(ids) - 1, "candidate_token_ids": token_ids,
        "scores": scores.tolist(), "conditional_probabilities": conditional.tolist(),
        "original_probabilities": original.tolist(), "valid_label_mass": mass,
        "vocabulary_probability_sum": float(vocab_probs.sum()),
        "mlx_softmax_max_abs_error": float(np.max(np.abs(mx_probs - conditional))),
        "library_logprobs_max_abs_error": logprob_error,
        "library_raw_logits_max_abs_error": library_raw_error,
        "library_scores": library_logits[token_ids].tolist(),
        "library_conditional_probabilities_float64": list(softmax(library_logits[token_ids].tolist())),
        "library_normalization_only_max_abs_error": float(np.max(np.abs(library_ref_logprobs - library_logprobs))),
        "library_logprobs_dtype": str(first.logprobs.dtype),
        "library_exp_logprobs_sum": float(np.exp(library_logprobs).sum()),
        "penultimate_position_control": {"scores": previous_scores,
                                         "argmax": previous_argmax},
        "library_first_token": first.token,
        "direct_argmax": int(logits.argmax()),
        "cached_logits_max_abs_error": cache_error,
        "cached_vs_library_logits_max_abs_error": float(np.max(np.abs(np.array(cached) - library_logits))),
        "full_vocabulary_mlx_float32_mass": full_mx_mass,
        "cached_argmax": int(mx.argmax(cached).item()),
        "top_tokens": [{"id": i, "text": tokenizer.decode([i]),
                        "probability": float(vocab_probs[i])} for i in top],
        "elapsed_seconds": time.perf_counter() - started,
        "memory": memory(),
    }
    spaced = [row for row in rows if row["candidate"] in [" " + x for x in LABELS]]
    if all(row["single_token"] for row in spaced):
        spaced_ids = [row["contextual_suffix_ids"][0] for row in spaced]
        result["space_prefixed_alternatives"] = {
            "token_ids": spaced_ids, "scores": logits[spaced_ids].tolist(),
            "conditional_probabilities": list(softmax(logits[spaced_ids].tolist())),
            "original_probabilities": vocab_probs[spaced_ids].tolist(),
            "mass": float(vocab_probs[spaced_ids].sum()),
        }
    result["generation_matches_direct_argmax"] = first.token == int(logits.argmax())
    assert math.isclose(float(conditional.sum()), 1, abs_tol=1e-12)
    del cache, cached, first
    gc.collect()
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--behavior", action="store_true", help="Investiga rotações e prompts sem alterar o runner.")
    parser.add_argument("--prompt-comparison", action="store_true", help="Compara adaptações de formatos com pontuação fixa.")
    args = parser.parse_args()
    if args.behavior and args.prompt_comparison:
        parser.error("Escolha somente um modo de investigação.")
    if args.snapshot.name != REVISION:
        parser.error("Use o diretório de snapshot da revisão fixada, sem renomeá-lo.")
    args.output.mkdir(parents=True, exist_ok=False)
    mx.set_cache_limit(256 * 1024**2)
    mx.reset_peak_memory()
    report = {
        "model": MODEL, "revision": REVISION,
        "python": platform.python_version(), "macos": platform.mac_ver()[0],
        "machine": platform.machine(), "metal_device": mx.device_info(),
        "cache_limit_bytes": 256 * 1024**2, "before_load": memory(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers",
                      "huggingface-hub", "numpy", "safetensors")},
        "script_sha256": digest(Path(__file__)),
        "snapshot_sha256": {p.name: digest(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()},
    }
    started = time.perf_counter()
    model, tokenizer = load(str(args.snapshot), trust_remote_code=False)
    model.eval()
    report["load_seconds"] = time.perf_counter() - started
    report["after_load"] = memory()
    report["parameter_dtypes"] = sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())})
    report["parameter_bytes"] = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    report["chat_template"] = tokenizer.chat_template
    simple = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with the single word OK."}],
        tokenize=False, add_generation_prompt=True)
    report["simple_inference"] = {"prompt": simple, "text": generate(
        model, tokenizer, prompt=tokenizer.encode(simple, add_special_tokens=False),
        max_tokens=16, sampler=make_sampler(temp=0), verbose=False)}
    report["cases"] = []
    if args.behavior:
        from mlx_behavior import investigate
        report["behavior_script_sha256"] = digest(Path(__file__).with_name("mlx_behavior.py"))
        report.update(investigate(model, tokenizer, args.output))
    if args.prompt_comparison:
        from prompt_comparison import compare
        report["comparison_script_sha256"] = digest(Path(__file__).with_name("prompt_comparison.py"))
        report.update(compare(model, tokenizer, args.output))
    for rotation in ([] if args.behavior or args.prompt_comparison else range(4)):
        options = ["4", "3", "5", "6"]
        options = options[-rotation:] + options[:-rotation] if rotation else options
        user = "What is 2 + 2?\n" + "\n".join(
            f"{label}. {value}" for label, value in zip(LABELS, options))
        user += "\nReply with exactly one letter: A, B, C, or D. Do not add spaces or explanations."
        messages = [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) == tokenizer.encode(prompt, add_special_tokens=False)
        variants = [(f"chat-{rotation}", prompt)]
        if rotation == 0:
            variants += [("no-generation-header", tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)),
                ("chat-trailing-space", prompt + " "),
                ("chat-trailing-newline", prompt + "\n"), ("raw-user", user)]
        for name, text in variants:
            print(f"Inspecting {name}", flush=True)
            case = inspect_prompt(model, tokenizer, text, name, args.output)
            report["cases"].append(case)
    report["after_diagnostic"] = memory()
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    report["after_release"] = memory()
    report["status"] = "diagnostic_completed_not_backend_approved"
    with (args.output / "diagnostic.json").open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
