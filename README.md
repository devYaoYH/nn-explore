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

### 3. Live steering: can we push generation toward truthfulness in real time?

The natural follow-up question: since a single forward pass already causally
determines its own output (§2), can we intervene *live*, mid-generation, to
correct a model that's about to say something wrong — not just in an offline
oracle-patching demo, but as an actual online steering mechanism? Short
answer: we tried the obvious approach and it doesn't work, for an instructive
reason. Full writeup, including a corrected finding from an earlier
mis-designed experiment, below.

**3a. Forced-continuation per-token probing (and a correction).** `dataset.py`'s
`build_steering_dataset()` extends the true/false template into full
teacher-forced continuations (e.g. `"Question: What is the capital of
France?\nAnswer:"` + `" The capital of France is Paris."` vs. `"...Berlin."`),
and `extract_continuation_probe.py` probes *every token position* of the
continuation, not just the last one. The pooled AUROC comes out much lower
than the static-statement result (~0.72 vs. ~0.96) — but breaking it down by
position relative to the end of the continuation shows why:

![Position breakdown](results/figures/position_breakdown.png)

Every position before the answer word sits at exactly chance. **This is a
construction artifact, not a finding about the model.** The true and false
continuations share an identical token prefix up to the branch point (`"...is
Paris."` vs `"...is Berlin."` — everything before "Paris"/"Berlin" is the same
tokens either way), so there is *by construction* zero information available
at those positions for a probe to find, regardless of what the model
internally "knows." An earlier pass at this analysis concluded from this that
"the model doesn't know the answer before committing to it" — that conclusion
was wrong; the null result is guaranteed by the matched-prefix design and says
nothing about the model's actual internal state. Real pre-commitment signal
(if it exists) has to come from genuine variation across *different*
questions, not an artificially matched prefix — see Future Work.

**3b. Live steering attempt.** Despite (3a), we already know from §2 that
*something* causal exists to intervene on — so we tried it anyway, live,
using `steer.py`: a forward hook on a mid-to-late layer that adds a generic
"truthfulness" direction (difference-of-means between true/false statement
activations, computed by `extract_and_probe.py --directions_out`) to the
residual stream on every decoding step. To get a genuine error case to correct
(the model gets essentially all of our original 20 well-known capitals right
even under provocation), `dataset.py`'s `build_distractor_dataset()` primes
each question with a confidently-asserted false premise about a *different*
country's capital (e.g. *"Context: I just learned that the capital of France
is Berlin.\nQuestion: What is the capital of France?"*) — a known, reproducible
way to induce entity-confusion errors, over a broadened set of 37 capitals
(20 common + 17 deliberately less-known ones, to get baseline uncertainty
above zero).

![Steering sweep](results/figures/steering_sweep.png)

| Approach | Result |
|---|---|
| Unconditional (every token) | Monotonic collapse — confusion rate rises slightly (11%→19%) before **total incoherence** by coeff=8 (output degenerates to `"The Capital To Its It The The The..."`). No coefficient ever improves accuracy. |
| Gated (live probe decides per-token) | Gate fires on 43-53% of tokens — far too promiscuous to be "targeted" — and shows no accuracy improvement at any coefficient tested. |

**Diagnosis:** the direction/classifier was trained on exactly one
distribution — the last token of a *complete* statement, where "is this claim
true or false" is a well-posed question. We applied it to a completely
different distribution — the last token of an *in-progress, incomplete*
phrase (including tokens like `"The"`, `"capital"`, `"of"` that have no claim
content yet to judge). Its behavior there is uncalibrated noise, not a
meaningful signal — confirmed by the direction's norm (29.1) already being
~29% of the layer's typical activation norm (~99), meaning even the smallest
tested coefficient was a large, indiscriminate perturbation on every single
token regardless of relevance. More coefficient/threshold tuning won't fix a
train/deploy distribution mismatch.

**What would actually be needed:** train the direction (and any live gate) on
activations from the *actual* target distribution — genuine intermediate
generation states, not post-hoc-labeled complete statements. Concretely: let
the model *actually generate* (sampled, not forced) answers to a broad
mixed-difficulty question set, auto-verify correctness against a known answer
key, and extract activations from those real generation trajectories. This is
a materially bigger lift (needs a much larger, harder, auto-verifiable
question set to get enough natural errors, plus careful separation between
"is a mistake about to happen" detection and the resulting steering vector)
and is the natural next experiment for whoever picks this repo up next — see
Future Work.

### 4. Fixing it: training on genuine free-generation rollouts

Directly informed by §3b's diagnosis, we retrained the direction (and gate)
on the actual deploy-time distribution instead of a proxy for it.
`collect_rollouts.py` samples **real generations** (`do_sample=True`,
temperature 0.8, top_p 0.95 — genuinely sampled, not teacher-forced) from
`build_distractor_dataset()`'s plain + distractor prompts — up to 8 rollouts
per question — auto-labels each
rollout correct/incorrect with the same regex ground-truth check `steer.py`
uses, and records the residual stream at each layer via a forward hook that
skips the (multi-token) prefill call and keeps only genuinely-generated
token positions — i.e. actual in-progress generation states, not
completed-statement snapshots or teacher-forced continuations.

Two representations come out of this per rollout:
- a **mean-pooled vector** (averaged over every generated position) — the
  same pooling the ["Detecting Strategic Deception"](https://alignment.anthropic.com/2024/deception-probes/)
  paper uses over generation-token activations — used to train a
  rollout-level probe (held out **by question**, same grouped-split
  discipline as §3a) and to compute the diff-of-means direction.
- **per-token rows** (every generated position individually, each inheriting
  the rollout's sequence-level label) — used to fit a live gate that's
  actually evaluated on the same kind of activation it'll see at
  steering-time, unlike §3b's gate.

Since multiple rollouts are sampled per question, a question where the
model's answer genuinely varies across samples yields a matched,
**on-policy contrastive pair for that exact question** — so besides the
usual pooled diff-of-means (all correct rollouts vs. all incorrect ones,
across *different* questions), we also compute a paired diff-of-means
restricted to just those matched-pair questions and average the per-question
differences — the same matched-pair logic CAA (Rimsky et al. 2023) uses,
now made on-policy.

On Qwen2.5-1.5B (592 rollouts from 74 prompts), the mean-pooled rollout-level
probe reaches **AUROC 0.86–0.92 across layers 6–21** on a held-out,
by-question split — a real signal, not the construction artifact §3a
diagnosed:

![Rollout probe AUROC by layer](results/figures/rollout_probe_auroc_by_layer.png)

Layer 18 (AUROC 0.924) is the peak, which is why it's the layer used below.
Rerunning `steer.py`'s exact distractor evaluation with this direction
(layer 18, paired diff-of-means) in place of the old completed-statement
direction:

![Rollout steering sweep](results/figures/rollout_steering_sweep.png)

| Approach | Result |
|---|---|
| Unconditional, coeff=1.0 | Distractor accuracy **31/37 → 35/37**, confusion **4/37 → 1/37**, plain-prompt accuracy unchanged (36/37). Degrades past coeff≈1.5, collapses by coeff=3 — same fluency-collapse shape as §3b, just at a different, now-*useful* coefficient range. |
| Gated (fires on **7–8%** of forward calls, vs. 43–53% for §3b's out-of-distribution gate) | Monotonic improvement across the whole tested range (coeff 0→4), reaching confusion **0/37** at coeff=4 with plain-prompt accuracy pinned at 36/37 throughout — never collapses in the tested range, trading a slightly lower ceiling for robustness. |

`demo_steering.py` makes this concrete rather than statistical: it generates
a baseline (unsteered) answer for every distractor prompt, keeps only the
ones the model got wrong, and regenerates *those exact same prompts* with
steering on — direct before/after pairs on the same input, e.g.:

```
[Portugal] (false premise: capital is Mbabane)
  Without steering: 'Mbabane'                      [CONFUSED]
  With steering:    'Lisbon.'                      [FIXED]

[Myanmar] (false premise: capital is Ottawa)
  Without steering: 'Ottawa.'                      [CONFUSED]
  With steering:    'The capital of Myanmar is Naypyidaw.' [FIXED]
```

4 of 6 baseline errors self-correct this way at coeff=1.0 (layer 18), and a
sanity pass over plain (non-distractor) prompts confirms steering doesn't
regress answers that were already correct.

This confirms §3b's diagnosis directly: the causal mechanism (§2) always
worked; what was missing was training the direction on the deploy-time
distribution instead of a proxy for it. The gate's fire-rate drop alone
(43–53% → 7–8%) is a strong independent signal that it's now well-calibrated
rather than just luckily thresholded. Still a toy-scale result (37 examples,
one model, one layer) — see Future Work for what's next.

### 5. Is this generalized truthfulness, or a capitals-specific fix?

A fair objection to §4: everything so far — training data and eval alike —
is country capitals. Fixing "resist a false premise about a country's
capital" isn't the same as finding a domain-general "is this claim true"
direction; it could just as easily be "ignore context that contradicts a
memorized country→capital mapping," a much narrower skill. `dataset.py`'s
`build_distractor_dataset()` (and every consumer downstream of it) was
generalized to a `{domain, item, correct_answer, wrong_answer,
prompt_plain, prompt_distractor}` shape spanning all four fact domains in
this repo — capitals, chemical symbols, multiplication, animal taxonomic
class — specifically to test this (`--domains` on `collect_rollouts.py`,
`steer.py`, and `demo_steering.py` selects which domain(s) to use; default
is `capitals` alone, to keep §4's numbers reproducible unmodified).

Three conditions, all measured with the same refactored `steer.py` so
they're directly comparable:

![Cross-domain generalization](results/figures/cross_domain_generalization.png)

| Condition | Baseline → steered | Net effect |
|---|---|---|
| Capitals direction → capitals eval (§4, same numbers as before the refactor — confirmed byte-identical) | 31/37 → 35/37, confused 4→1 | Clean win, zero regressions |
| Capitals direction → **animals** eval (zero-shot transfer) | 5/10 → 5/10, confused 5→4 | **Net zero** — one example (`frog`) gets fixed, but a *previously-correct* example (`snake`) gets newly broken. Confusion count alone made it look like a small win; per-example inspection shows it's a wash. |
| Animals-only direction (same recipe as §4, applied to its own domain) → animals eval | 5/10 → 6/10, confused 5→4 | Small clean win (one fix, zero regressions), but far weaker than capitals' effect size |

The **animals-in-domain result is the important one**: the on-policy
rollout recipe does still work when pointed at a new domain (a genuine,
regression-free improvement), so §4's fix isn't capitals-specific *by
construction*. But the effect is much smaller than capitals' — plausibly
because the animals training set is far smaller (10 questions / 4 matched
pairs vs. capitals' 37 questions / 8 pairs) and its distractor premise is a
blunter direct contradiction ("I just learned that a dog is a reptile")
than a geography fact. The **capitals→animals transfer result is the
negative one**: the specific vector learned from capitals data does not
carry over to a domain it never saw, even though it's the same "kind" of
task (resist a false premise, answer correctly).

Cosine similarity between the two directions, layer by layer, gives a
mechanistic reading on why:

![Direction cosine similarity](results/figures/direction_cosine_similarity.png)

In a ~1536-dim residual stream, two unrelated random vectors would sit near
cosine 0 (std ≈ 0.026), so a similarity of 0.05–0.19 is a real signal, not
noise — but it's small, and it **grows with depth, peaking at layer 18**
(0.192), the same layer where capitals steering works best. Reading: there
*is* a modest shared component — consistent with some genuine
domain-general "truthfulness-ish" direction existing and strengthening
through the network — but it's dominated by domain-specific structure (which
fact, which surface form), which is exactly why zero-shot transfer is weak
rather than either working cleanly or being pure noise.

**Bottom line:** §4 is a real fix for the *train/deploy distribution*
mismatch, not a demonstration of generalized truthfulness. Getting the
latter, if it exists as a clean linear feature at all, needs multi-domain
joint training (do many domains share enough of that small common
component to average out the domain-specific noise?) — see Future Work.

## Repo layout

```
dataset.py                     true/false statements, forced-continuation, and distractor datasets
inspect_model.py               quick architecture sanity check for a new model (layer count, module names)
extract_and_probe.py           extract last-token residual activations, train per-layer probes + steering directions
causal_patch.py                subject-token causal patching sweep across layers
extract_continuation_probe.py  per-token probing on forced (teacher-forced) continuations
analyze_position_breakdown.py  per-position AUROC breakdown (why (3a)'s pooled AUROC is misleading)
steer.py                       live activation steering during generation + distractor evaluation harness
collect_rollouts.py            sample real generations, auto-label, extract on-policy activations + directions (§4)
demo_steering.py                proof-of-concept: direct before/after generations showing live steering fix errors (§4)
plot_results.py                regenerate results/figures/*.png from results/*.csv
results/                       *.csv result tables and the plots above
setup.sh                       one-shot environment bootstrap for a new GPU box
requirements.txt               pinned dependency versions
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

# 6. Forced-continuation per-token probing + position breakdown (§3a)
python extract_continuation_probe.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --out /tmp/cont_activations.npz --results_csv results/continuation_probe.csv
python analyze_position_breakdown.py --activations /tmp/cont_activations.npz --layer 15 \
    --model_label "Qwen2.5-1.5B" --results_csv results/position_breakdown.csv

# 7. Save a generic steering direction, then try live steering (§3b)
python extract_and_probe.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --out /tmp/activations.npz --directions_out /tmp/directions.npz
python steer.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --directions /tmp/directions.npz --layer 21 --coeffs 0.0 0.5 1.0 2.0 4.0
# ...or gated (fires only when a live probe flags risk -- still doesn't help, see §3b):
python steer.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --directions /tmp/directions.npz --layer 21 --coeffs 0.0 0.5 1.0 2.0 4.0 \
    --gate_activations /tmp/activations.npz

# 8. Collect on-policy free-generation rollouts and steer with the fixed direction (§4)
python collect_rollouts.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --samples_per_prompt 8 --max_new_tokens 16 --layers 18 \
    --out /tmp/rollout_activations.npz --directions_out /tmp/rollout_directions.npz \
    --results_csv results/rollout_probe.csv
python steer.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --directions /tmp/rollout_directions.npz --layer 18 --coeffs 0.0 0.5 1.0 1.5 2.0 3.0
python steer.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --directions /tmp/rollout_directions.npz --layer 18 --coeffs 0.0 0.5 1.0 2.0 3.0 4.0 \
    --gate_activations /tmp/rollout_activations.npz

# 9. Proof-of-concept: direct before/after generations on the same prompts
python demo_steering.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --directions /tmp/rollout_directions.npz --layer 18 --coeff 1.0

# 10. Cross-domain generalization check (§5) -- collect a domain-specific
# direction and check both in-domain and zero-shot transfer.
python collect_rollouts.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none \
    --domains animals --samples_per_prompt 8 --max_new_tokens 16 --layers 9 \
    --out /tmp/rollout_activations_animals.npz --directions_out /tmp/rollout_directions_animals.npz
python steer.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none --domains animals \
    --directions /tmp/rollout_directions_animals.npz --layer 9 --coeffs 0.0 1.0        # in-domain
python steer.py --model Qwen/Qwen2.5-1.5B-Instruct --quant none --domains animals \
    --directions /tmp/rollout_directions.npz --layer 18 --coeffs 0.0 1.0               # zero-shot transfer
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

## Dataset builders

`dataset.py` generates true/false statements across four factual categories
(country capitals, chemical symbols, small multiplication facts, animal
taxonomic classes), with the false variant swapping in a plausible-but-wrong
value from the same category so the *only* thing that differs is truth value,
not surface form or sentence structure. It's grown four generators as the
experiments progressed:

- `build_dataset()` — 104 static true/false statement pairs (§1, §2 inputs).
- `build_steering_dataset()` — the same facts recast as `(prompt,
  true_continuation, false_continuation)` triples for teacher-forcing (§3a).
- `build_distractor_dataset(domains=...)` — factual questions across all
  four domains (`DOMAINS = ("capitals", "elements", "math", "animals")`,
  default), each paired with a false-premise distractor context, for the
  live steering evaluation (§3b) and the cross-domain generalization check
  (§5). Every example has the same generic shape (`domain`, `item`,
  `correct_answer`, `wrong_answer`, `prompt_plain`, `prompt_distractor`)
  regardless of domain, so `collect_rollouts.py`/`steer.py`/
  `demo_steering.py` need no domain-specific logic — pass `--domains` to
  restrict which one(s) to use (default: `capitals` alone, matching §4's
  documented numbers).
- `CAPITALS` / `HARD_CAPITALS` / `ELEMENTS` / `MATH` / `ANIMALS` — the
  underlying fact lists all generators draw from.

This is a toy stand-in for a real contrastive dataset — swap in something
closer to your actual research question (e.g. the honest-vs-deceptive
instruction-following setup from Anthropic's
["Detecting Strategic Deception Using Linear Probes"](https://alignment.anthropic.com/2024/deception-probes/))
when you're ready to move past pipeline validation.

## Future work

§4 closed the loop on §3's negative result — on-policy rollout training does
fix live steering — and §5 showed that fix is domain-specific rather than a
generalized truthfulness direction, though a small (~0.05-0.19 cosine)
shared component does exist and grows with depth. Natural next steps:

1. **Multi-domain joint training.** Collect rollouts across all four domains
   combined and train one direction on the pooled data. If domains share
   enough of that small common component, joint training should average out
   the domain-specific noise and produce something closer to general;
   if it doesn't help, that's further evidence there's no single linear
   "truthfulness" feature at this scale worth chasing.
2. **Replicate §4-§5 at 32B scale.** All results so far are on Qwen2.5-1.5B
   only, for iteration speed. Check whether the domain-specificity finding
   holds, or whether a larger model shares more cross-domain structure.
3. **More matched pairs.** Only 8 of the ~52 capitals training-split prompts
   (and just 4 of ~15 for animals) yielded both a correct and incorrect
   rollout at `samples_per_prompt=8`; either raising the sample count or
   widening the fact set would give the paired diff-of-means a
   lower-variance estimate, and animals in particular would benefit --
   its effect size was much noisier than capitals' with the same recipe.
4. **Beyond induced distractors.** The whole eval set relies on a
   confidently-asserted false premise to induce errors. Check whether the
   same recipe (real rollouts, auto-verified, mean-pooled + paired
   diff-of-means) also produces a working steering direction for
   *naturally occurring* hallucinations (no induced distractor context) —
   this needs a harder/larger question set to get a usable natural error
   rate, but is a more realistic target than entity-confusion.
5. Swap the toy capitals/elements/math dataset for something closer to a real
   research question — e.g. the honest-vs-deceptive instruction-following
   setup from
   ["Detecting Strategic Deception Using Linear Probes"](https://alignment.anthropic.com/2024/deception-probes/) —
   now that the full pipeline (rollout collection → on-policy direction →
   live steering → cross-domain check) is validated end-to-end on a toy task.

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
