"""Live activation steering during generation: add a "truthfulness" direction
(difference-of-means, from extract_and_probe.py --directions_out) to the
residual stream at a chosen layer, on every decoding step, and measure
whether it reduces entity-confusion errors on distractor-primed prompts.

Uses plain transformers generate() + a forward hook rather than nnsight --
nnsight's tracing API is built for the interactive single-pass experiments
in extract_and_probe.py / causal_patch.py; an always-on intervention across
an entire generation loop is simpler and more standard as a raw PyTorch
forward hook.
"""
import argparse
import re
import unicodedata

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import DOMAINS, build_distractor_dataset


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

SYSTEM_PROMPT = "Answer the question in as few words as possible. Give only the final answer, with no explanation."


class Steerer:
    """Registers a forward hook on model.model.layers[layer] that adds
    coeff * direction to the last-position residual stream on every forward
    call (prefill and each incremental decode step alike).

    If gate_clf is given, the intervention only fires when the live probe
    (fit on the same layer's completed-statement activations) predicts the
    *current* partial generation is leaning false -- rather than blasting
    every single token indiscriminately, most of which have no claim content
    yet to judge true/false about."""

    def __init__(self, model, layer, direction, coeff, gate_clf=None):
        self.direction = direction.to(dtype=model.dtype, device=model.device)
        self.coeff = coeff
        self.gate_clf = gate_clf
        self.n_fired = 0
        self.n_calls = 0
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.coeff == 0:
            return output
        self.n_calls += 1
        if self.gate_clf is not None:
            last = output[:, -1, :].float().cpu().numpy()
            p_true = self.gate_clf.predict_proba(last)[0, 1]
            if p_true >= 0.5:
                return output  # probe thinks it's already leaning truthful -- leave it alone
        self.n_fired += 1
        output = output.clone()
        output[:, -1:, :] = output[:, -1:, :] + self.coeff * self.direction
        return output

    def remove(self):
        self.handle.remove()


def generate_answer(model, tokenizer, question_text, max_new_tokens=12):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def classify_answer(generated, correct_answer, wrong_answer):
    g = strip_accents(generated.lower())
    correct = bool(re.search(r"\b" + re.escape(strip_accents(correct_answer.lower())) + r"\b", g))
    confused = bool(re.search(r"\b" + re.escape(strip_accents(wrong_answer.lower())) + r"\b", g))
    return correct, confused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", choices=["none", "nf4", "prequantized"], default="none")
    parser.add_argument("--directions", required=True, help="Path to directions .npz from extract_and_probe.py")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--coeffs", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument("--gate_activations", default=None,
                         help="If set, path to raw activations .npz (from extract_and_probe.py --out) "
                              "used to fit a live gating probe at --layer -- steering only fires when "
                              "the probe predicts the current step is leaning false.")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=["capitals"],
                         help="Which fact domain(s) to evaluate on (default: capitals, matching the "
                              "README's documented numbers). Use a domain different from the one the "
                              "direction was trained on to test transfer.")
    args = parser.parse_args()

    if args.quant == "none":
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    elif args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        load_kwargs = dict(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    else:
        load_kwargs = dict(device_map="auto")

    print(f"Loading {args.model} (quant={args.quant})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    direction_npz = np.load(args.directions)
    direction = torch.from_numpy(direction_npz[f"layer_{args.layer}"]).float()
    direction_unit_norm = direction / direction.norm()
    print(f"Steering at layer {args.layer}, direction norm {direction.norm():.3f}")

    gate_clf = None
    if args.gate_activations:
        from sklearn.linear_model import LogisticRegression
        gate_data = np.load(args.gate_activations)
        X = gate_data[f"layer_{args.layer}"]
        y = gate_data["labels"]
        gate_clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
        print(f"Fit live gating probe on layer {args.layer} ({len(y)} examples, "
              f"train accuracy {gate_clf.score(X, y):.3f})")

    examples = build_distractor_dataset(domains=args.domains)
    print(f"Evaluating on domains: {args.domains} ({len(examples)} examples)")

    for coeff in args.coeffs:
        steerer = Steerer(model, args.layer, direction, coeff, gate_clf=gate_clf)
        n_correct_plain = n_correct_distractor = n_confused_distractor = 0
        rows = []
        for ex in examples:
            ans_plain = generate_answer(model, tokenizer, ex["prompt_plain"])
            ans_distr = generate_answer(model, tokenizer, ex["prompt_distractor"])
            c_plain, _ = classify_answer(ans_plain, ex["correct_answer"], ex["wrong_answer"])
            c_distr, conf_distr = classify_answer(ans_distr, ex["correct_answer"], ex["wrong_answer"])
            n_correct_plain += c_plain
            n_correct_distractor += c_distr
            n_confused_distractor += conf_distr
            rows.append((ex["item"], ans_plain, c_plain, ans_distr, c_distr, conf_distr))
        steerer.remove()

        n = len(examples)
        print(f"\n=== coeff={coeff} ===")
        if gate_clf is not None:
            print(f"Gate fired on {steerer.n_fired}/{steerer.n_calls} forward calls "
                  f"({100 * steerer.n_fired / max(1, steerer.n_calls):.1f}%)")
        print(f"Plain prompt accuracy:      {n_correct_plain}/{n}")
        print(f"Distractor prompt accuracy: {n_correct_distractor}/{n}  (confused->distractor: {n_confused_distractor}/{n})")
        for item, ans_plain, c_plain, ans_distr, c_distr, conf_distr in rows:
            flag = "OK" if c_distr else ("CONFUSED" if conf_distr else "WRONG")
            print(f"  {item:14s} plain={ans_plain!r:20s} distractor={ans_distr!r:20s} [{flag}]")


if __name__ == "__main__":
    main()
