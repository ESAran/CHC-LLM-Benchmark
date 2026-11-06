"""Comparação diagnóstica zero-shot de textos adaptados, sem carregar MMLU."""
from collections import Counter
from math import fsum

from benchmark.metrics import accuracy
from benchmark.numerics import normalized_entropy, predicted_index


QUESTIONS = [
    ("triangle", "elementary_mathematics", "How many sides does a triangle have?", ["3", "2", "4", "5"]),
    ("planet", "astronomy", "Which planet is known as the Red Planet?", ["Mars", "Venus", "Jupiter", "Saturn"]),
    ("organ", "anatomy", "Which organ pumps blood through the human body?", ["Heart", "Liver", "Lung", "Stomach"]),
    ("meter", "elementary_mathematics", "How many centimeters are in one meter?", ["100", "10", "1000", "1"]),
]
FORMATS = ("current-no-enumeration", "harness-adapted", "lighteval-adapted")


def render(format_name, question, options, subject):
    choices = "\n".join(f"{label}. {text}" for label, text in zip("ABCD", options))
    if format_name == FORMATS[0]:
        return question + "\n" + choices + "\nReply with exactly one letter. Do not add spaces or explanations."
    description = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    prefix = "Question: " if format_name == FORMATS[2] else ""
    return description + prefix + question.strip() + "\n" + choices + "\nAnswer:"


def compare(model, tokenizer, output):
    from __main__ import contextual_rows, inspect_prompt
    cases = []
    for qid, subject, question, canonical in QUESTIONS:
        for rotation in range(4):
            options = canonical[-rotation:] + canonical[:-rotation] if rotation else canonical[:]
            correct = "ABCD"[options.index(canonical[0])]
            for fmt in FORMATS:
                user = render(fmt, question, options, subject)
                messages = [{"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                assert tokenizer.encode(prompt, add_special_tokens=False) == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                name = f"{qid}-{rotation}-{fmt}"
                print("Inspecting", name, flush=True)
                case = inspect_prompt(model, tokenizer, prompt, name, output)
                assert case.get('candidate_token_ids') == [32,33,34,35]
                assert [tokenizer.decode([i]) for i in case['candidate_token_ids']] == list('ABCD')
                probs = case['conditional_probabilities']
                pred = "ABCD"[predicted_index(probs)]
                case.update(format=fmt, question_id=qid, subject=subject, options=options,
                            correct_label=correct, messages=messages, user_prompt=user,
                            selected_label=pred, correct=pred==correct,
                            normalized_entropy=normalized_entropy(probs),
                            scoring_method="label_next_token_bare_full_forward")
                # Só tokenização do texto sem chat; não entra na comparação de escores.
                case['unwrapped_tokenization_control'] = contextual_rows(tokenizer, user)
                spaced = case['space_prefixed_alternatives']
                assert spaced['token_ids'] == [362,425,356,422]
                spaced['selected_label'] = "ABCD"[predicted_index(spaced['conditional_probabilities'])]
                spaced['correct'] = spaced['selected_label']==correct
                spaced['normalized_entropy'] = normalized_entropy(spaced['conditional_probabilities'])
                cases.append(case)
    summary = {}
    for fmt in FORMATS:
        rows = [c for c in cases if c['format']==fmt]
        def aggregate(items):
            counts=Counter(r['selected_label'] for r in items)
            return {'count':len(items),'accuracy':accuracy([r['correct'] for r in items]),
                    'mean_normalized_entropy':fsum(r['normalized_entropy'] for r in items)/len(items),
                    'selected_counts':{k:counts[k] for k in 'ABCD'}}
        summary[fmt] = {'primary_bare':aggregate(rows),
                        'sensitivity_space_only':aggregate([r['space_prefixed_alternatives'] for r in rows])}
    return {'cases':cases,'summary':summary,'design':{'questions':QUESTIONS,'formats':FORMATS,
            'fewshot_count':0,'primary_candidates':['A','B','C','D'],
            'sensitivity_candidates':[' A',' B',' C',' D'],
            'sampling':'greedy API probe only','temperature':0,'max_probe_tokens':1,
            'seed':None,'order':'question, rotation, format',
            'adaptation':'entire reference text in user message; default Qwen chat template; no assistant prefill'}}
