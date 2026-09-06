"""Proof-of-concept demo for live activation steering (README section 4):
shows direct before/after generations on the same prompt, rather than the
statistical sweep steer.py's __main__ produces.

Procedure: generate a baseline (unsteered) answer for every distractor-primed
prompt, keep only the ones the model got wrong without help, then regenerate
those same prompts with steering switched on and report how many self-correct
-- plus a quick sanity check that plain (non-distractor) prompts are
unaffected by the intervention.

Defaults assume you've already run:
    python collect_rollouts.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --out rollout_activations.npz --directions_out rollout_directions.npz
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import DOMAINS, build_distractor_dataset
from steer import Steerer, classify_answer, generate_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", choices=["none", "nf4", "prequantized"], default="none")
    parser.add_argument("--directions", default="rollout_directions.npz")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--coeff", type=float, default=1.0)
    parser.add_argument("--gate_activations", default=None,
                         help="Optional: gate steering on a live probe fit from this activations "
                              ".npz (e.g. rollout_activations.npz) instead of firing every token.")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap the number of distractor examples checked (default: all).")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=["capitals"],
                         help="Which fact domain(s) to demo on (default: capitals, matching the "
                              "README's documented numbers). Pass e.g. --domains animals to test "
                              "a direction trained on a different domain than it's evaluated on.")
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

    direction = torch.from_numpy(np.load(args.directions)[f"layer_{args.layer}"]).float()

    gate_clf = None
    if args.gate_activations:
        from sklearn.linear_model import LogisticRegression
        gate_data = np.load(args.gate_activations)
        X, y = gate_data[f"layer_{args.layer}"], gate_data["labels"]
        gate_clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)

    examples = build_distractor_dataset(domains=args.domains)
    if args.max_examples:
        examples = examples[: args.max_examples]

    print(f"\nGenerating baseline (no steering) answers for {len(examples)} distractor-primed prompts "
          f"(domains={args.domains})...")
    baseline = []
    for ex in examples:
        ans = generate_answer(model, tokenizer, ex["prompt_distractor"])
        correct, confused = classify_answer(ans, ex["correct_answer"], ex["wrong_answer"])
        baseline.append((ex, ans, correct, confused))

    wrong_baseline = [(ex, ans, confused) for ex, ans, correct, confused in baseline if not correct]

    mode = "gated" if gate_clf is not None else "unconditional"
    print(f"\n=== Live steering demo (layer {args.layer}, coeff {args.coeff}, mode {mode}) ===")
    print(f"{len(wrong_baseline)}/{len(examples)} prompts were WRONG without steering. "
          f"Re-running those same prompts with steering ON:\n")

    steerer = Steerer(model, args.layer, direction, args.coeff, gate_clf=gate_clf)
    n_fixed = 0
    for ex, base_ans, base_confused in wrong_baseline:
        steered_ans = generate_answer(model, tokenizer, ex["prompt_distractor"])
        correct, confused = classify_answer(steered_ans, ex["correct_answer"], ex["wrong_answer"])
        n_fixed += correct
        tag = "FIXED" if correct else ("STILL CONFUSED" if confused else "STILL WRONG")
        print(f"[{ex['item']}] (false premise: {ex['wrong_answer']})")
        print(f"  Without steering: {base_ans!r:30s} {'[CONFUSED]' if base_confused else '[WRONG]'}")
        print(f"  With steering:    {steered_ans!r:30s} [{tag}]")
        print()
    steerer.remove()

    print(f"Summary: {n_fixed}/{len(wrong_baseline)} baseline errors corrected by live steering.")

    print("\nSanity check -- plain prompts (no distractor), steering still ON:")
    steerer = Steerer(model, args.layer, direction, args.coeff, gate_clf=gate_clf)
    sample = examples[:8]
    n_ok = 0
    for ex in sample:
        ans = generate_answer(model, tokenizer, ex["prompt_plain"])
        correct, _ = classify_answer(ans, ex["correct_answer"], ex["wrong_answer"])
        n_ok += correct
        print(f"  {ex['item']:14s} -> {ans!r:20s} {'[OK]' if correct else '[WRONG]'}")
    steerer.remove()
    print(f"{n_ok}/{len(sample)} plain prompts still correct with steering on "
          f"(should match the no-steering baseline -- steering shouldn't break what already works).")


if __name__ == "__main__":
    main()
