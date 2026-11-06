"""Matriz controlada complementar; invocada por mlx_labels.py --behavior."""
import json
import re

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler


QUESTIONS = [
    ("sum", "What is 2 + 2?", ["4", "3", "5", "6"]),
    ("capital", "What is the capital of France?", ["Paris", "Rome", "Berlin", "Madrid"]),
    ("week", "How many days are in one week?", ["7", "5", "6", "8"]),
    ("opposite", "What is the opposite of hot?", ["cold", "warm", "large", "fast"]),
]
INSTRUCTIONS = {
    "original": "Reply with exactly one letter: A, B, C, or D. Do not add spaces or explanations.",
    "reverse-enumeration": "Reply with exactly one letter: D, C, B, or A. Do not add spaces or explanations.",
    "no-enumeration": "Reply with exactly one letter. Do not add spaces or explanations.",
    "semantic": "Choose the correct answer to the question. Reply only with the letter of that option.",
}


def generate_response(model, tokenizer, ids):
    responses = list(stream_generate(model, tokenizer, ids, max_tokens=24,
                                    sampler=make_sampler(temp=0)))
    text = "".join(r.text for r in responses)
    # Sem resgatar letras dentro de prosa: saídas fora do formato são explícitas.
    strict = re.fullmatch(r"[ABCD]", text.strip())
    return {"text": text, "token_ids": [r.token for r in responses],
            "first_token": responses[0].token if responses else None,
            "finish_reason": responses[-1].finish_reason if responses else None,
            "strict_label": strict.group() if strict else None}


def investigate(model, tokenizer, output):
    # __main__ é o script existente; não carrega uma segunda cópia do modelo.
    from __main__ import inspect_prompt, LABELS
    cases = []
    controls = []
    for qid, question, canonical in QUESTIONS:
        for rotation in range(4):
            options = canonical[-rotation:] + canonical[:-rotation] if rotation else canonical[:]
            correct = LABELS[options.index(canonical[0])]
            for variant, instruction in INSTRUCTIONS.items():
                user = question + "\n" + "\n".join(f"{k}. {v}" for k, v in zip(LABELS, options)) + "\n" + instruction
                messages = [{"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                assert ids == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                name = f"{qid}-{rotation}-{variant}"
                print("Inspecting", name, flush=True)
                case = inspect_prompt(model, tokenizer, prompt, name, output)
                assert case.get("candidate_token_ids") == [32, 33, 34, 35]
                assert [tokenizer.decode([i]) for i in case['candidate_token_ids']] == list(LABELS)
                case.update(question_id=qid, question=question, options=options,
                            correct_label=correct, variant=variant, messages=messages)
                case['scored_label'] = LABELS[max(range(4), key=lambda i: case['scores'][i])]
                case['generated'] = generate_response(model, tokenizer, ids)
                case['generated_first_matches_probe'] = case['generated']['first_token'] == case['library_first_token']
                cases.append(case)
                # Registro incremental explícito; somente diagnostic.json sinaliza o fim.
                with (output / "behavior-progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(case, ensure_ascii=False) + "\n")
        control = question + " Reply with the answer only."
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": control}], tokenize=False, add_generation_prompt=True)
        controls.append({"kind": "open-answer", "question_id": qid, "prompt": prompt,
                         "generated": generate_response(model, tokenizer, tokenizer.encode(prompt, add_special_tokens=False))})
    for label in LABELS:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": f"Reply with exactly the letter {label}."}], tokenize=False, add_generation_prompt=True)
        controls.append({"kind": "copy-letter", "target": label, "prompt": prompt,
                         "generated": generate_response(model, tokenizer, tokenizer.encode(prompt, add_special_tokens=False))})
    return {"cases": cases, "controls": controls, "design": {
        "questions": QUESTIONS, "instructions": INSTRUCTIONS, "rotations": 4,
        "temperature": 0, "max_generated_tokens": 24,
        "order": "question, rotation, instruction insertion order", "seed": None}}
