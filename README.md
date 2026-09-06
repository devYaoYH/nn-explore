# nn-explore

A minimal, working base for mechanistic-interpretability experiments (linear
probing + causal activation patching) on a single GPU, using
[`nnsight`](https://nnsight.net/) against real Hugging Face checkpoints —
including 4-bit-quantized 32B-parameter models. This repo is meant to be
**cloned/forked as a starting point** on a fresh Vast.ai (or any single-GPU)
instance for follow-on experiments, not as a finished research artifact.

It currently demonstrates the full pipeline end-to-end on a toy task (true vs.
false factual statements) across two model scales:

- `Qwen/Qwen2.5-1.5B-Instruct` (bf16, full precision)
- `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (pre-quantized NF4, 4-bit)

## Results

### 1. Linear probes: does the model "know" true vs. false?

We extract the residual-stream activation at the **last token** of each
statement, at several layers spread across the network depth, and train a
held-out logistic-regression probe per layer to classify true vs. false.

![Probe AUROC by layer](results/figures/probe_auroc_by_layer.png)

| Model | Peak AUROC | Layer (depth %) |
|---|---|---|
| Qwen2.5-1.5B (bf16) | 0.96 | ~43-54% |
| Qwen2.5-32B (NF4) | **1.00** | ~50%+ |

Both models linearly encode truth value well before the middle of the
network, and the 32B model reaches **perfect** held-out separability —
consistent with the broader finding (e.g. Azaria & Mitchell 2023, Marks &
Tegmark 2023, Anthropic's deception-probing work) that truth value is one of
the most linearly-decodable properties in a transformer's residual stream,
and that this gets *more* linear with scale, not less.

### 2. Causal patching: is that representation load-bearing, or just a correlate?

A linear probe only shows *correlation*. To test *causation*, we patch the
residual stream at the **subject-token position** (e.g. "France" / "Germany")
from a corrupted prompt into a clean prompt's forward pass, one layer at a
time, and watch whether the output flips:

- Clean:   `"The capital of France is"` → `" Paris"`
- Corrupt: `"The capital of Germany is"` → `" Berlin"`

![Causal patch by layer](results/figures/causal_patch_by_layer.png)

Both models show the same qualitative signature: patching early/mid layers
does nothing (the fact is "read off" from elsewhere later); there's a **sharp
crossover window** where the patch flips the answer; then the effect fades
in the very last layers because there's no computation left to use it.

| Model | Crossover window (depth %) |
|---|---|
| Qwen2.5-1.5B (bf16) | ~75-82% |
| Qwen2.5-32B (NF4) | ~72-80% |

**Notable finding:** the causal crossover happens *later* in the network than
where the linear probe already saturates (~45-50% depth). In other words, the
fact becomes linearly *readable* well before it becomes *causally load-bearing*
for the actual output token — the representation is present but not yet
"consumed" by the computation that produces the answer. This is a real
mechanistic-interpretability result, not just a pipeline sanity check.

### Why patch the subject token, not the last token?

The original naive approach — patch only the *last* token's residual stream —
does **not** reliably flip the output, even at the "correct" layer. Downstream
attention layers can still see the *unperturbed* subject token elsewhere in
the sequence and reconstruct the original answer from it. Patching the subject
token itself (standard "causal tracing", as in Meng et al.'s ROME paper) is
what actually isolates the causal effect. See `causal_patch.py` for the
token-alignment logic.

## Repo layout

```
dataset.py            synthetic true/false factual-statement dataset (104 examples)
inspect_model.py       quick architecture sanity check for a new model (layer count, module names)
extract_and_probe.py   extract last-token residual activations, train per-layer probes
causal_patch.py        subject-token causal patching sweep across layers
plot_results.py        regenerate results/figures/*.png from results/*.csv
results/               probe_auroc.csv, causal_patch.csv, and the plots above
setup.sh               one-shot environment bootstrap for a new GPU box
requirements.txt       pinned dependency versions
```

## Setup

```bash
git clone git@github.com:devYaoYH/nn-explore.git
cd nn-explore
./setup.sh
source .venv/bin/activate
```

`setup.sh` detects the GPU's CUDA compute capability and picks a matching
torch build automatically (see **Environment notes** below for why this
matters). If you'd rather do it by hand:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # or cu124 on older GPUs
uv pip install -r requirements.txt
```

## Reproducing / extending

```bash
# 1. Sanity-check a model's architecture before writing hooks against it
python inspect_model.py

# 2. Extract activations + train probes (small model, full precision)
python extract_and_probe.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --out /tmp/activations.npz --results_csv results/probe_auroc.csv

# 3. Same, but for a 4-bit-quantized 32B model
python extract_and_probe.py \
    --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --quant prequantized \
    --out /tmp/activations.npz --results_csv results/probe_auroc.csv

# 4. Causal patching sweep
python causal_patch.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --results_csv results/causal_patch.csv
python causal_patch.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --quant prequantized \
    --results_csv results/causal_patch.csv

# 5. Regenerate the figures above
python plot_results.py
```

`--quant` has three modes:
- `none` — load at full precision (bf16). Use for models that fit comfortably.
- `nf4` — quantize on load via `bitsandbytes` `BitsAndBytesConfig`. This
  downloads the **full-precision** checkpoint first, then quantizes locally —
  use only when no pre-quantized checkpoint exists.
- `prequantized` — load a checkpoint that's already stored in 4-bit (e.g. an
  `unsloth/*-bnb-4bit` repo). Much less bandwidth (~19GB vs. ~65GB for a 32B
  model) and just as fast to run — **prefer this when available**.

Swapping to a different model or a different dataset (e.g. an
honest-vs-deceptive-instruction dataset instead of true/false facts) should
only require editing `dataset.py` and re-running steps 2-5 — the extraction,
probing, and patching code is model- and dataset-agnostic as long as the
model exposes `.model.layers[i]` and `.lm_head` (true of most Llama-family /
Qwen-family causal LMs).

## Dataset

`dataset.py` generates 104 templated true/false statement pairs across four
factual categories (country capitals, chemical symbols, small multiplication
facts, animal taxonomic classes), with the false variant swapping in a
plausible-but-wrong value from the same category so the *only* thing that
differs between a true/false pair is truth value, not surface form or
sentence structure. This is a toy stand-in for a real contrastive dataset —
swap in something closer to your actual research question (e.g. the
honest-vs-deceptive instruction-following setup from Anthropic's
["Detecting Strategic Deception Using Linear Probes"](https://alignment.anthropic.com/2024/deception-probes/))
when you're ready to move past pipeline validation.

## Environment notes (read before bootstrapping a new instance)

These are real gotchas hit while building this, worth checking again on any
new machine before assuming things will just work:

- **GPU architecture determines the required CUDA wheel, not the driver
  version.** Blackwell GPUs (RTX 50-series, B200; compute capability ≥ 10.0)
  need **CUDA ≥ 12.8** torch wheels. An older `cu124` build *installs
  cleanly* but fails at the first GPU op with `no kernel image is available
  for execution on the device`. `setup.sh` checks `nvidia-smi
  --query-gpu=compute_cap` and picks the right `--index-url` automatically —
  don't skip this check on a new box.
- **`transformers` 5.x changed decoder-layer output shape.** In older
  `transformers` versions, `model.model.layers[i].output` was a `(tensor,)`
  tuple, so you'd index `.output[0]`. In `transformers>=5.x` it's the
  `hidden_states` tensor directly — `.output[0]` throws `IndexError: too many
  indices`. Check `inspect_model.py`'s smoke test output before assuming a
  hook pattern from an older tutorial will work unmodified.
- **`nnsight` 0.7's `.save()` returns the realized tensor directly** outside
  the `trace()` context — no `.value` attribute access needed (older nnsight
  tutorials use `proxy.value`).
- **Prefer pre-quantized checkpoints over quantize-on-load** when available
  (e.g. `unsloth/*-bnb-4bit` repos) — `BitsAndBytesConfig(load_in_4bit=True)`
  against a non-quantized repo still downloads the full-precision weights
  first, which for a 32B model is ~65GB vs. ~19GB pre-quantized.
- **Workspace storage on Vast.ai is not guaranteed persistent** — check
  `vast-capabilities | jq '.instance.workspace_is_volume'` before assuming
  downloaded model weights or datasets survive a recycle/destroy. This repo's
  code is designed to be re-cloned and re-run from scratch cheaply for
  exactly this reason (activation `.npz` dumps are gitignored; only the small
  summary CSVs and figures are checked in).

## License

No license file yet — treat as all-rights-reserved until one is added.
