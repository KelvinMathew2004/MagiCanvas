# Model training candidates and evidence

The research is **not limited to the two existing Qwen3 bases**. Six model packages were prepared. The fresh 0.6B control finished in Colab, was evaluated on 89+59 tasks and was not promoted. The two remaining jobs now use a separate uniform-data bundle: VL-2B first, then 0.8B. The existing v2 adapter remains the production incumbent. No GPU training was performed on this Mac.

| Package | Why it earns a training trial | Current evidence / limitation |
|---|---|---|
| Qwen3-0.6B | Fresh same-base control for audited labels/current prompts | Existing v2 adapter is the reference; a fresh adapter need not win |
| Qwen3-1.7B | Capacity control on exactly the same curated rows | Earlier adapter improved dense inventory but did not consistently improve final layouts |
| Qwen2.5-Coder-1.5B-Instruct | Contrasting code-pretrained text control | Older release, not advertised as a newer model; untuned structured output did not establish a layout win |
| Qwen3.5-0.8B | Newer small hybrid-attention architecture at a practical size | Newly downloaded and loaded through its text backbone with no missing weights; native MPS tiny-JSON smoke succeeded. This is compatibility evidence, not a layout-quality result |
| Qwen3.5-2B | Larger version of the newer architecture | Prior native feasibility pilot exists; portable fallback kernels were slow. CUDA training speed does not establish Linux/amd64 judging speed |
| Qwen3-VL-2B-Instruct | Source-image awareness for subject/background relationships | Own adapter, JSON plus source PNG, frozen vision encoder; never gets a resized target image. Untuned image-aware pilots were not task-matched training |

Pins and input modalities are in `config/training_candidates.json`. All six
use the same 1,043 curated training / 309 validation pairs and normalized-v2
output contract. Each uses its native tokenizer/chat template; relation
compaction follows the actual provider's token count. All train/validation
responses pass exact prefix masking and fit within 6,144 tokens. The largest
observed training sequence is 5,003 tokens for the image-conditioned model.

Each selected model gets its **own** rank-16 LoRA. Language projection discovery includes
Qwen3.5's gated-delta projections, not just Q/K/V/O. LM heads and vision
projections are not adapted. This is more than renaming a model in the old
notebook. Image inputs make the VLM a capability comparison, not an isolated
parameter-count comparison.

## Prescribed first-round queue

1. **Fresh Qwen3-0.6B**: isolate whether the audited data/current prompts help the known architecture.
2. **Qwen3.5-0.8B**: strongest small newer-architecture hypothesis with a successful native loader/JSON smoke and much less memory than larger alternatives.
3. **Qwen3-VL-2B**: test a distinct missing capability—source-image conditioning—rather than paying for another text-only size variant.

The existing trained 1.7B is already a useful capacity reference. A fresh 1.7B,
Coder-1.5B and Qwen3.5-2B remain prepared reserves; the driver does not train
them automatically. The three-job queue is an experiment priority judgment,
not evidence that any challenger already beats the champion.

## Evidence boundaries

The new 0.8B smoke used the official pinned checkpoint, native MPS FP16 and
non-thinking generation. It produced the requested empty JSON in about 1.05s
after a 3.6s load on one M1 Pro run. It generated only a few tokens, so that
measurement is not a full resize benchmark or a speed forecast for M5/Docker.
The source JSON/geometry benchmarks and all older model pilots retain their
own recorded snapshots. The current production viewer uses recorded 0.6B
predictions; it does not depict untrained new adapters.

The earlier generic-model losses do **not** show that no better trained model
exists. Nor do rejected layout heuristics prove an engine Pareto frontier.
The next useful evidence is matched training followed by fresh model proposals,
raw inventory/geometry/font checks, unchanged-selector repair, and visual review
of changed winners across 89 reference pairs plus 59 development tasks. Runtime,
full stored model size, input modality and raw/final defects must be reported
separately. Model names and generic benchmark scores cannot settle the choice.

The model-size/approved-model ceiling is not numeric in the supplied challenge
brief. These are public small-model research candidates, not a certification of
eligibility. The selected final package still needs the organizer's confirmed
limits, public adapter publication and offline Docker verification.

## New dense-layout probe: evidence against premature training claims

Four fresh native Qwen3.5-0.8B grammar probes covered 17, 23, 22 and 35
source elements and actual extreme/reverse resizes. Three completed exact
inventories; the fourth exhausted its output budget with repeated whitespace.
The unchanged current selector chose deterministic layouts on all four. Raw
stretching, alignment, contacts and clipping remain; the final four outputs have
no automated flags because the existing engine retained safer alternatives.
This is **not** an immediate quality upgrade and is not proof that its trained
adapter will be better. The 0.8B job remains a bounded architecture experiment,
not a promised win. See local `data/training_next/new_model_layout_summary.json`
and the [two-panel feasibility viewer](http://127.0.0.1:8766/review/training_candidates/index.html).

## Implementation and next action

Follow [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md). The current run selects VL-2B only and defers 0.8B, using unit weights and single-pass sampling. The launcher
resumes only matching uniform checkpoints on Drive and downloads separate adapters. The available Colab
T4 may attempt the unchanged sequential recipe using FP16; an L4 or larger also
works. Each model first checks actual forward/backward/AdamW memory and finite
gradients on long-input/long-answer rows. GPU training and 4-bit memory peaks
remain unverified on this Mac; a model that fails preflight needs a larger GPU.
The current default remains unchanged until a candidate earns promotion. Native
new-provider evaluation is isolated in `training/inference.py`; the production
loader does not silently switch to a multimodal interface.

Larger/gated/less-compatible alternatives remain in the
[research ledger](archive/UPGRADE_EXPLORATION_AND_EVALUATION.md), with their
license, total-weight, backend or evidence barriers. This is a bounded search
for the strongest tested project package, not a claim to have exhausted every
model or found an absolute global optimum.

Primary sources: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B),
[Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
[Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
[Transformers Qwen3.5 support](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

## October 29 frontier recheck

Public metadata was rechecked and pinned in local `frontier_screen.json`:

- **Qwen3.8**: no accessible official 0.8B/2B checkpoint was verified at the queried IDs (Hub returned 401). This is not proof that none exists privately. The official repository advertises much larger released packages; newer branding alone does not make them viable substitutes for a small offline engine. [Official release repository](https://github.com/QwenLM/Qwen3.8).
- **Gemma 4 E2B**: the public checkpoint contains **5.123B stored parameters**, despite the effective-size name. Its FP16 weights alone are roughly 9.54 GiB before activations, vision buffers and the selector. It is not rejected as inaccurate; the additional deployment/training cost lacks project-specific evidence and exact judge RAM is unconfirmed. [Official model card](https://huggingface.co/google/gemma-4-E2B-it).
- **LFM3-1.7B**: an interesting newer efficient candidate, but the installed Transformers has no native `lfm3` mapping, and its official loading example requires remote model code. Its custom license also needs explicit review. A pinned code/backend/adapter audit is still necessary before putting it in a user training queue. [Official model card](https://huggingface.co/LiquidAI/LFM3-1.7B).
- **LFM2.5-1.2B-Instruct**: a native causal loader is available and the checkpoint is public, so it remains a credible reserve—not dismissed. It has custom terms and no local task-matched quality/training pilot. Adding it would be an additional experiment, not an evidence-backed replacement for the current queue. [Publisher model library](https://docs.liquid.ai/lfm/models/complete-library).

All of these remain possible future contenders. None of the general benchmark,
model-card or token-throughput claims establishes graphic-layout quality. The
three first-round jobs maximize distinct evidence under the current runtime and
implementation constraints; they are not a guarantee that every training run
will improve quality.
