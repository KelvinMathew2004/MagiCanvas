> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Upgrade exploration and evaluation plan

Reviewed and revised October 28, 2026. Covers **every suggestion** in the initial Gemini attachment, the subsequent Google AI pipeline table, the follow-up model/license/zero-dependency audit, and a fresh primary-source model search. Proposal IDs match `config/upgrade_experiments.json`. This is a research and experiment checklist, not a claim that proposed features have shipped or that any new model already beats the trained baseline.

**Implementation follow-up:** the user subsequently authorized downloads and
experiments. See [executed pilot results](UPGRADE_PILOT_RESULTS.md) and the
[pilot viewer](http://127.0.0.1:8766/review/upgrade_pilots/index.html).
Metadata-only statements below describe the initial research phase. Four
pinned checkpoints have since been downloaded; grammar, solver and critic
pilots now exist outside production. No new adapter has been trained or published.
The subsequent source-owned rotated-label repair passed all 391 tests and
34 subtests; all 59 final regression renders remain identical. A four-task
three-decode pilot added no final improvement, so production sampling remains
unchanged. See the results document for rejected-proposal evidence and limits.

**Expanded no-training evaluation:** [134-task results](TRAINING_FREE_SCALE_RESULTS.md)
now cover all 89 held-out reference pairs, six independent layout algorithm
arms, full-engine tiled selection, the existing structural head and all 178
human preferences with fixed critics. Optional tiling is available but remains
off by default. Several utilization-improving alternatives visibly broke source
lockups and were not promoted. Native assisted decoding was slower on three
cases and matched exact target tokens on two. Remaining backend ports, semantic
subject grounding and new training contracts are not declared complete.

**Surgical follow-up:** the [decision ledger](../SURGICAL_ALGORITHM_DECISIONS.md)
tests all six later Gemini suggestions against the same 134 tasks, the 178
recorded preferences, and 1,231 distinct train/validation targets. A narrower
raw-font recovery remains experimental; blanket type-scale bands and the
suggested topology/occupancy and selective-critic policies are not promoted.
Target line-count sidecars and an isolated/leave-one-out glyph auditor are
available offline. Native target auditing covers a 32-target sample, not all
1,231. Production behavior and the existing v2 training files remain unchanged.

## Recommendation

Keep the current production model and selector while testing separate changes. First demonstrate **Qwen3-0.6B + its adapter ↔ Qwen3-1.7B + its different adapter** through the same engine interface; both packages already exist locally. For a different text-model family, **Qwen2.5-Coder-1.5B-Instruct** is the simplest Apache-licensed causal-LM training pilot, with SmolLM3-3B as another candidate. Neither accepts the existing Qwen3 LoRA.

The most promising newer capability experiment remains **Qwen3.5-2B**, compared with **Qwen3-VL-2B-Instruct** for image awareness. It must pass native CPU/MPS and offline-loader smoke tests before training; class registration alone does not establish kernel performance. **Qwen3.5-0.8B** is the smaller control. These are hypotheses, not measured layout winners. Each different base needs its own compatible layout training.

**LFM3-1.7B / LFM3-VL-2B move to conditional research**, rather than the preferred judge-facing replacement. The publisher announced them on October 27, 2026, but they require custom-code compatibility work and carry commercial-use conditions that matter for transfer to an enterprise. Keep the idea; prioritize Apache alternatives until intended evaluation/distribution rights are clear. This is not a finding that research evaluation is automatically prohibited. [Publisher announcement](https://huggingface.co/blog/LiquidAI/lfm3), [model](https://huggingface.co/LiquidAI/LFM3-1.7B), [license conditions](https://www.liquid.ai/lfm-license).

The highest-value engine experiments are **ordered-ID constrained decoding**, **bounded macro-layout search using measured text variants**, and **an aspect-aware visual critic comparison**. A critic remains subordinate to geometry and raster safety. Do not train a new large judge or replace the engine with diffusion before measuring where valid learned proposals still fail.

Suggested order:

0. Verify and document paired model profiles; keep the proposal/repair/selector interface independent of the chosen base. Record missing release pins honestly.
1. Inventory grammar feasibility on the current trained model; test exact IDs and complete output, then final layout quality.
2. New-model loader smoke tests, followed by matched training pilots for Qwen3.5-2B and one contrasting model, not ten simultaneous training runs.
3. Compare current SigLIP at native-aspect tiles against SigLIP2 NaFlex and a small VLM critic on newly labeled difficult pairs.
4. Try CP-SAT as one additional, bounded macro candidate family; preserve source ownership and run the same raster verifier.
5. Quantized GGUF CPU inference after choosing a promising model; separately benchmark native Mac and Linux/amd64 Docker.
6. Macro-contract training and dense-data augmentation if the leaf-coordinate pilot remains weak.

## Judge preference: what swappable models means (B01)

The engine should accept **interchangeable generator packages**, not require one LoRA to work on every model. A LoRA is a small learned update to specific matrices in its training base. Its checkpoint records that base and target modules; it does not contain an independent layout model. Different dimensions, module names, weights or tokenizer/contract can invalidate the combination. Matching tensor sizes alone is insufficient evidence of semantic compatibility. [PEFT checkpoint documentation](https://huggingface.co/docs/peft/developer_guides/checkpoint), [LoRA paper](https://arxiv.org/abs/2106.09685).

### Verified local options

| Package | Base / hidden width | Matching adapter | Current state |
| --- | --- | --- | --- |
| Default | Qwen3-0.6B / 1024 | `qwen3-layout-lora-v2` | Local metadata and q-projection shape checks pass; existing trained default |
| Optional | Qwen3-1.7B / 2048 | `qwen3-layout-lora-v2-expanded-1.7b` | Same checks pass; existing A/B alternative, not promoted |
| Crossed pair | Qwen3-1.7B / 2048 with the 0.6B adapter | Incompatible input width 1024 | Explicitly tested and rejected |
| Different-family pilot | Qwen2.5-Coder-1.5B-Instruct / 1536 | **Needs a new matching adapter** | Compatible broad loader category; actual Coder weights are not local |
| Visual pilot | Qwen3-VL-2B-Instruct | **Needs a new VLM adapter** | New image processor/input path required |

The local `models/qwen2.5-1.5b-instruct` directory contains **general Qwen2.5-Instruct**, not Coder. It cannot serve as evidence that Coder is already available. The existing 0.6B adapter records an Unsloth quantized training-base alias; the 1.7B adapter records its Qwen base. Changing or merging an adapter does not convert it to a different architecture or size.

No retraining is needed to demonstrate swapping the two trained packages already local. For a new base, train a separate adapter or evaluate its untuned baseline explicitly; do not silently attach the old adapter. For the **same exact base** on another runtime or precision, merging/converting the existing adapter may be possible without retraining, but conversion, quantization and final outputs still need verification. That is backend portability, not cross-model LoRA transfer.

`config/layout_model_profiles.json` describes the two existing packages; `scripts/check_model_profiles.py` verifies them without loading a generator or changing the default:

```bash
.venv/bin/python scripts/check_model_profiles.py
.venv/bin/python scripts/check_model_profiles.py --profile qwen3_1p7b
```

This profile file is currently an **audit manifest**, not a newly wired runtime selector. The checker reports the environment settings already supported by the existing loader. Both local tokenizer hashes match, but the base hidden widths differ; one shared tokenizer still does not make adapters interchangeable. The training metadata has no pinned base revision, so this is a documented provenance gap, not an invented immutable release.

To demonstrate the optional package in the existing native preview, stop the current preview server first, then start a separate process from the project root:

```bash
MAGICANVAS_QWEN_ENABLED=1 \
MAGICANVAS_REQUIRE_QWEN_ADAPTER=1 \
MAGICANVAS_QWEN_DIR=models/qwen3-1.7b \
MAGICANVAS_QWEN_ADAPTER_DIR=models/qwen3-layout-lora-v2-expanded-1.7b \
MAGICANVAS_QWEN_PROMPT_VERSION=2 \
.venv/bin/python viewer/server.py --port 5050
```

The paths and loaded model are cached at process startup; this is a restart-time swap, not instant hot-swapping. A user changing only the base path can leave the default 0.6B adapter selected. A future runtime profile selector should choose the whole pair atomically and fail closed on incompatibility.

### Model-independent interface to build next

Keep `source JSON + target canvas → normalized element proposal + diagnostics` stable. Implement providers for causal text, image-conditioned VLM and optional local quantized inference behind that boundary. Each provider owns its tokenizer/processor, chat template, prompt version, context/output budget, decoder, device policy and base/adapter pair. Shared repair and the rendered selector consume the same normalized proposal; they do not contain model-name-specific layout rules.

The next profile contract must pin base, adapter and tokenizer revisions; specify architecture and target-module shapes; distinguish separate adapters from already merged weights; declare optional image support; reject unsupported configurations; and support a generator-only smoke test. The current verifier is Qwen-oriented: it checks family and `q_proj` input width, not every matrix or exact base identity. A model with fused `qkv_proj` needs another capability check rather than removal of validation. Full validation and runtime profile integration are planned, not already shipped here.

**Judge-facing claim we can support:** “The layout engine separates its learned generator from repair and verification. We can switch between two locally trained model/adapter packages using configuration, with the same layout contract.” A third, different-family trained model would strengthen the demonstration but requires a new adapter and evaluation; a universal reusable LoRA is not required for this architecture.

## Follow-up Gemini audit: accepted points and corrections (B02–B03)

| Follow-up claim / suggestion | Research disposition |
| --- | --- |
| Keep 0.6B default; 1.7B optional | Agree for the current evidence; no automatic promotion. Its historical quality/runtime tradeoff remains mixed. Neither “strictly superior” speed nor perfect legality follows from one timing or a license tag. |
| Qwen3.5 has hybrid attention and incompatible old LoRA | Agree on architecture/adapter incompatibility. Installed Transformers 5.20.1 maps `qwen3_5` to both causal and multimodal classes, so a blanket missing-loader claim is incorrect. PyTorch chunk/recurrent fallbacks exist; “always too slow on MPS/CPU” remains a runtime hypothesis. Check actual checkpoint loading, unsupported operators and peak memory before training. |
| Liquid license makes it unusable for Adobe | Verified commercial revenue conditions, but contest/research evaluation is not automatically the same as enterprise production use. Deprioritize for a low-friction distributable default; clarify intended usage before adopting. |
| Qwen2.5 3B research terms; gated Gemma3 | Verified research/non-commercial terms for both cited 3B checkpoints. Gated Gemma3 currently blocks anonymous config access. These are deployment/permission obstacles, not a claim that every possible authorized evaluation is disqualified. Prefer clearer public alternatives. |
| 512px SigLIP preserves native aspect and costs the same | Correct the distinction: fixed 512 preprocessing is square; NaFlex is the varying-aspect variant. Similar weight count does not mean equal compute or patch memory. Keep the critic experiment. |
| Prompt lookup had zero n-gram hits and Metal shape thrashing | Plausible mechanisms, **not measured by the current probe**. It did not log draft acceptance, n-gram hits or allocator events. Observed slowdown, one prefix mismatch and a Metal warning do not establish the warning's cause or prove all prompt lookup is inferior. Keep disabled by default; instrument before causal claims. |
| Complete inventories with 1.7B prove the selector is the only bottleneck | Disagree with the inference. Complete raw IDs plus poor final choices can reflect bad model geometry, insufficient candidates, grouping/repair, selection, or several at once. Retain the raw→repair→selected ablation rather than attributing everything to one stage. |
| A 40-line forced-prefix decoder guarantees every ID and is 35% faster | Good experiment direction, inaccurate guarantee. A `LogitsProcessor` can constrain a real token automaton; forcing headers does not validate numbers or ensure completion within context. Appending token IDs does not populate the KV cache: forced chunks still need a model forward pass, and cache positions/masks must match. Test batched prefill versus token masking separately. No speed claim yet. |
| SigLIP patch L2 norms or patch/text cosine are free semantic masks | Investigate as soft features. Norms are not calibrated object masks; patch features may need projection into text space, and added source-image passes/feature retention cost compute and memory. Human-marked subject masks are needed to test localization. |
| Exempt all overflowing tiny source text; weaken micro-banner floors | Retain F01/F02 safeguards. Thin visible elements are not inert simply because width≤1; visible source overflow is not permission to hide content. Do not lower a safety floor only to remove the last flag. No guaranteed zero-regression claim from a special size threshold. |
| Absolute CTA caps, fewer columns, sparse proportional penalty | Retain source-relative hierarchy, measured short-copy ownership and safe-candidate ranking experiments. Avoid unconditional percentages, column limits and winner-name triggers; they can punish valid CTA-led/asymmetric layouts. |
| Add a pure-Python Hungarian solver | Already implemented as `engine.retrieve.assign`, with no SciPy dependency. Do not replace it with factorial search or duplicate matching code. Library expansion/ownership-safe merging remains A07. |
| Center glyph labels after returning the winner | Glyph-bounds centering is already present. Any further correction must occur before final safety verification/selection; a ±3px post-selection shift can create clipping, new contact or rotated-label misalignment. |

Primary checks: [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [Liquid license](https://www.liquid.ai/lfm-license), [Coder-3B license](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE), [VL-3B license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE), [fixed-512 processor](https://huggingface.co/google/siglip2-base-patch16-512/blob/main/preprocessor_config.json). License implications for the specific judging arrangement need clarification; the plan does not make a legal eligibility determination.

## What was actually checked in this exploration

- Read the complete Gemini attachment and subsequent Google AI suggestions. The original is preserved locally under `data/upgrade_exploration/`; it is not vendored as a competition document in Git.
- Checked current engine, model loader, prompt budgeting, ordered peer wrapping, rendered text centering, font caches, retrieval library, ranker reports, and historical 0.6B/1.7B comparisons.
- Fetched public metadata/configuration for 30 model checkpoints, pinned each observed Hub revision, recorded licenses, architecture and stored tensor counts. No weights downloaded or remote model code executed. Gated Gemma3 metadata was readable, but its config returned 401; that is not a successful loader check.
- Ran a small offline same-model/same-adapter prompt-lookup feasibility probe. See the runtime section for its negative result and limitations.
- Rechecked the challenge PDF: page 7 leaves the exact approved-model list and size limit pending; page 11 requires offline Docker execution and public adapted weights for final submission. No numerical sub-4B approval can be inferred from this version of the brief. Publication remains deferred as requested.
- No production engine, selector, renderer, model default, or training dataset was changed by this research task. The current 59-case viewer retains its actual engine snapshot.

The read-only probe is reproducible:

```bash
.venv/bin/python scripts/probe_upgrade_readiness.py
# Optional: refresh only small public metadata, never weights or model code.
.venv/bin/python scripts/probe_upgrade_readiness.py --refresh-hub
```

Local evidence: `data/upgrade_exploration/readiness.json`, `prompt_lookup.json`, and the source archive. Reports are excluded from Git by the existing `/data` rule and included selectively in the Gemini audit packet.

## Ground rules for comparing upgrades

**Three separate questions:** Does the raw model produce useful layouts? Does repair improve those layouts? Does selection choose the best safe candidate? Record all three; a deterministic winner with Qwen enabled is not a model win. Grammar guarantees do not establish visual quality. Zero flags do not establish good design.

**Contract and permissions:** Preserve Stage-1 IDs, content, font families and supplied asset identity; image cover cropping may be allowed while aspect distortion is not. Do not redraw artwork, automatically delete legal copy, or turn visible source content transparent to pass checks. The creative Stage-2 mode needs explicit user-approved content edits and its own export contract.

**Training interpretation:** the current language model learns token likelihood, not a coordinate-MSE regression objective. “It averages layouts” is a hypothesis about observed behavior, not an established explanation. Fixed tensor outputs solve inventory representation only when their serialization is correct; they still require a compatible numerical runtime and learned geometry.

**Hardware:** Native Darwin/arm64 can use MPS; Linux/amd64 Docker cannot use Apple's Metal. This workstation is an Apple M1 Pro, not the M5 Pro reference. CPU instruction availability under Docker emulation must be detected; AVX2/VNNI cannot simply be assumed. Weight storage, KV cache, image tokens, processor memory, rendering and all concurrently resident models contribute to memory.

**Runtime reproducibility:** `requirements.txt` currently has broad lower bounds (`transformers>=4.51`, `peft>=0.14`), while this probe used Transformers 5.20.1, Torch 2.14.1 and PEFT 0.21.2. A newer architecture appearing in the local class mapping does not prove that an older installation or final container can load it. Pin and test a backend-specific dependency set for each promoted candidate, including all processor/custom-code files needed offline.

**Evidence maturity:** Existing 59 development cases, the 89 test rows and the 309 validation rows have been inspected or used during development to varying degrees. They are useful regression cohorts, not a newly sealed unseen test. Reserve new source families and fresh human preference pairs before tuning the next upgrade. Do not tune to previously consumed preference-test outcomes.

## Modern model shortlist and verified caveats (N01–N02)

The approximate counts below are **Hub-reported stored tensor totals**, including available vision components; they are not a definition of unique trainable/active parameters or peak RAM. Some publishers count tied embeddings differently. `readiness.json` records exact totals and revisions. “Fits the challenge” remains conditional on the organizers' final list and counting rule.

| Candidate | Approximate stored tensors | Why investigate / main barrier | Initial priority |
| --- | ---: | --- | --- |
| Qwen3.5-0.8B | 0.87B | Small newer multimodal control; new architecture, processor and adapter | High, cheap smoke test |
| Qwen3.5-2B | 2.27B | Best first quality/size hypothesis; compare text-only versus image-conditioned input | Highest new generator pilot |
| Qwen3.5-4B | 4.66B | Larger quality ceiling, greater memory; model name does not prove eligibility | Conditional |
| Qwen3-VL-2B-Instruct | 2.13B | Image grounding comparator without reasoning-output overhead by default | High |
| Qwen3-VL-4B-Instruct | 4.44B | Larger visual comparator; eligibility and CPU cost | Conditional |
| LFM3-1.7B | 1.73B | Latest CPU-oriented text alternative; custom code and LFM license review | Conditional research |
| LFM3-VL-2B | 1.92B | Small new vision alternative; specialized processor/runtime and license | Conditional research |
| LFM2.5-1.2B-Instruct / 2.6B | 1.17B / 2.70B | More established Liquid comparison if LFM3 tooling is immature | Secondary |
| LFM2.5-VL-1.6B | 1.60B | Low-size vision alternative | Secondary |
| SmolLM3-3B | 3.08B | Apache-licensed text alternative; no demonstrated layout advantage | Secondary |
| SmolVLM2-2.2B-Instruct | 2.25B | Apache-licensed small VLM control | Secondary |
| Gemma 4 E2B / E4B | 5.12B / 8.00B | Effective-size naming hides larger stored weights; Apache metadata, eligibility first | Conditional, not default |
| Phi-4-mini-flash-reasoning | 3.85B | Hybrid runtime experiment; custom-code example, reasoning cost and CPU compatibility | Secondary |

Primary model cards: [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [2B](https://huggingface.co/Qwen/Qwen3.5-2B), [4B](https://huggingface.co/Qwen/Qwen3.5-4B); [Qwen3-VL 2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct); [Gemma E2B](https://huggingface.co/google/gemma-4-E2B-it), [E4B](https://huggingface.co/google/gemma-4-E4B-it); [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct); [LFM2.5 text](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct), [2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B), [vision](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B); [Phi flash](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning). Config metadata for LFM3-VL was fetched successfully even though one browser card fetch failed.

Qwen3.5's card warns that small models can loop in thinking mode. Use bounded non-thinking inference first and include timeout/truncation rates in the comparison. Multimodal checkpoints need an explicit processor/input adapter and may need a different AutoModel class. A shared engine proposal interface does not make the existing text-only loader universal.

No verified small Qwen3.8 layout checkpoint emerged from the official model search. Larger MoE releases with a small **active** count are not small-memory substitutes. Image generators, ASR models and safety classifiers are not geometry generators merely because their parameter counts fit.

## Full Gemini proposal inventory

Status describes the present code or proposed experiment, not an approval to ship.

### I. Models and perception

| ID | Suggestion | Decision and smallest useful experiment |
| --- | --- | --- |
| M01 | Qwen2.5-Coder 1.5B / 3B | Worth a controlled text-model baseline. Same broad causal-LM interface, different trained weights/chat behavior; train a matching adapter. No primary evidence found for the asserted 10-point advantage on this task. 1.5B Apache is the simpler comparison; 3B uses Qwen research terms rather than the same license. |
| M02 | Phi-4-mini for dense constraints | Retain as an alternative, not a proven best constraint solver. Run a bounded dense pilot with matched data. A language model's reasoning score does not guarantee non-overlap; deterministic constraints remain necessary. |
| M03 | Qwen2.5-VL / Gemma3 image-conditioned generation | Useful capability hypothesis, now compare newer small VLMs too. Run source-PNG + JSON versus JSON-only using the same model. Test crop-specific subject protection separately. Gemma3 is gated and uses Gemma terms; no gate acceptance performed. |
| M04 | Fixed-tensor transformer / DiT / Group-CVAE | Longer-track independent learned candidate. Fixed input slots can preserve inventory if the serializer maps slots exactly, but geometry can still collapse or overlap. Train and compare a simple deterministic tensor regressor before diffusion. Model size, speed and training-data adequacy are unmeasured. |
| M05 | SigLIP2 384/512 or NaFlex | Strong critic experiment. Fixed 384/512 is higher resolution, not native aspect. Compare native-aspect tiles of current weights, base-512 and base-NaFlex on new difficult labels. SO400 is a much larger full dual-encoder checkpoint (~1.14B stored tensors), so include CPU/memory cost. |
| M06 | DINOv2-small / AIMv2 / SigLIP feature saliency | Try localization as a soft feature only. DINO small is ~22M; AIMv2-large is ~309M and has Apple-specific terms. Attention/feature norms are not guaranteed semantic masks for every face or bottle. Some vision towers lack the assumed CLS attention output. Measure actual subject hits and false protection. |

Sources: [Coder 1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), [Coder 3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct), [Phi-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Gemma3](https://huggingface.co/google/gemma-3-4b-it), [DINO](https://huggingface.co/facebook/dinov2-small), [AIM](https://huggingface.co/apple/aimv2-large-patch14-224), [SigLIP NaFlex](https://huggingface.co/google/siglip2-base-patch16-naflex), [base-512](https://huggingface.co/google/siglip2-base-patch16-512).

Reusing an existing vision tower avoids another checkpoint but is not automatically free computation: dense feature retention, source-background inference and mask processing need profiling. Saliency is evaluated as an additional feature, not a guarantee that text can never touch a subject.

**Specific corrections:** 3B floating weights require roughly 6 GB at two bytes per parameter, not 3.1 GB; runtime overhead is additional. Qwen2.5-VL's report describes grounding with actual image-dimension coordinates; do not assume its contract is the same normalized 0..1000 convention as a different generation. Verify coordinate transforms against image preprocessing. [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923).

### II. Decoding and training

| ID | Suggestion | Decision, risk and test |
| --- | --- | --- |
| D01 | XGrammar / Outlines / custom exact-ID grammar | Highest-priority feasibility prototype. Force ordered aliases once each, exact list length, field types and numeric domains. A schema with only an enum of IDs does **not** prevent duplicates or omissions. Grammar completion can guarantee inventory, but EOS/token-budget failure still exists. Measure completed-output rate, grammar dead ends, geometry and latency; forced tokens do not inherently skip neural forward passes. |
| D02 | Remove font-size prediction; predict box/lines and fit with renderer | Promising ablation, not immediate contract replacement. Compare unchanged raw boxes with predicted font versus exact font fitting before retraining. Then test a new line-count contract with matched training. A bad tiny box still gives tiny text; fitting alone cannot restore hierarchy. Preserve tracking ratios, source padding and content. |
| D03 | Macro-tree / guillotine / CSS Grid adapter | Strong next training contract if grammar-clean leaf proposals remain weak. Use source-derived 5–12 blocks and a bounded layout DSL; preserve atomic ownership. A slicing tree cannot express every overlapping or asymmetric design, so include a keep-arrangement escape and existing v2 route. Measure grouping errors separately from planner errors. |
| D04 | GRPO / DPO using rendered rewards | Explore after reward audit. Prefer offline rejection sampling + small DPO pilot before online GRPO. Do not use the unvalidated ranker as a large truth reward or optimize only the engine's own blind spots. Split by family; human-audit reward winners and exploit cases. 500×6 is an experiment budget, not an established optimum. |
| D05 | Best-of-K batched diverse proposals | Good isolated experiment: greedy plus 2–3 seeded samples, deduplicate, same repair/selector. A greedy and sampled arm are not a single standard generation setting without explicit batching logic. Compare quality at equal wall-clock and equal render budgets. KV memory grows; the asserted 25% latency overhead is unverified. |
| D06 | Synthetic dense, transitive and partial training pairs | Worth a data pilot. Include 22–40 elements, cards, menus, edge attachments and all ratio directions. Enforce unique target cap and family splits before augmentation. Solver-produced targets must pass raster checks and human spot review; otherwise training reproduces the fallback's mistakes. Compare 0/10/25% synthetic mixtures, not automatic 1,500-pair acceptance. |

[XGrammar's Transformers integration](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html) provides a mechanism for output masks, not visual guarantees. [DPO](https://huggingface.co/docs/trl/dpo_trainer) uses preference pairs; [GRPO](https://huggingface.co/docs/trl/grpo_trainer) has different sampling and reward requirements. Neither is equivalent to running one more supervised epoch.

**Grammar prototype acceptance:** build grammar from current ordered aliases, forbid early closure, reject unknown fields/IDs and nonfinite values, test zero/one/40 elements, escaped original IDs through aliases, max-token exhaustion and EOS. Keep an explicit unsuccessful-generation diagnostic rather than claiming a recovered proposal was raw-valid. Do not alter prompt v2 or retrain until a same-adapter grammar ablation shows benefit.

**Data requirements:** fit normalization and filtering on training only. Keep original source/family provenance for all generated targets. Do not turn current regression samples, preference test pairs, or held-out targets into training data. New adapters use response-only masking, equal target inventory and matched hyperparameter budgets. A base-model smoke test assesses compatibility, not a fair comparison with a task-trained adapter.

### III. Remaining defects and outliers

Current evidence still reports one flagged 320×50 conference conversion. This case is a useful capacity failure, not a reason to hide content or weaken a metric globally.

| ID | Suggestion | Decision and safe version |
| --- | --- | --- |
| F01 | Quarantine degenerate/whitespace and >300% overflowing source text | Truly inert handling already exists in part. Audit what is visible in the source. A thin rule, narrow label or authored overflowing visible word is not automatically inert. Never make visible text transparent or exempt it wholesale. Preserve IDs and report malformed source geometry separately. |
| F02 | 22/58/20 micro-strip with 3.5px type | Test generic capacity-aware allocation with measured cards, not those fixed zones or speaker-specific engine logic. Preserve every required element and visible style. Do not lower font floors merely to obtain zero flags. Report an honest readable-capacity compromise if no feasible layout exists. |
| F03 | Absolute CTA width/hierarchy caps | Existing source-relative hierarchy/measurement is preferable to unconditional 45% canvas / 60% headline caps. Those constants break CTA-dominant designs. Compare bounded caps only when source hierarchy supports them and no readability is lost. |
| F04 | Fold tiny annotations into the main copy column | Useful generic role/ownership experiment. Use measured height, source proximity and reading order; text-length percentage alone cannot classify legal text or a caption. Reflow the owning block together and verify neighboring media. |
| F05 | Enable structural ranker on sparse zero-flag outputs | Worth an isolated safe-candidate selection ablation, not automatic deployment. Define the sparse trigger before evaluation. Never override hard raster defects. Collect new difficult human pairs; controlled shrink examples inflate apparent success. |

The ranker report has 14/17 validation and 31/37 reconstructed test matches, but actual campaign validation tied the comparator at 5/8; deployment was explicitly not approved. Five pixel-reconstruction failures were excluded. That test has been consumed. Source: `data/chameleon_upgrade_audit/structural_ranker/report.json`. The older 13-case model comparison improved exact raw inventory 7/13→9/13, but ten final outputs stayed identical and generation time rose; it does not justify a default switch. Its selector snapshot is historical.

### IV. Solvers and mathematical layout improvements

| ID | Suggestion | Decision and safe experiment |
| --- | --- | --- |
| A01 | OR-Tools CP-SAT `NoOverlap2D` + discrete arrangement modes | Strong optional candidate prototype. Premeasure text/card variants, use integer positions with documented quantization, constrain ownership/order/edge anchors, allow only source-supported overlaps, and limit solver time. Return FEASIBLE/OPTIMAL/UNKNOWN/INFEASIBLE honestly. A timeout falls back to existing candidates. Run raster verification after realization. |
| A02 | Ordered variable-width peer-row dynamic programming | Already implemented in `container_flow.ordered_peer_rows`; retain tests and improve objective only if measured cases fail. It is ordered line breaking inspired by typesetting, not a complete implementation of Knuth–Plass. No second parallel implementation. |
| A03 | L-BFGS-B micro-polish | Conditional experiment on macro translations only. Bound drift, keep children rigid, start feasible and reject new contacts after raster verification. L-BFGS-B provides coordinate bounds, not arbitrary pairwise hard constraints. A smooth penalty cannot promise non-overlap or equal margins. Profile 30 iterations; 4 ms is not established. |
| A04 | 64×64 distance transform / large whitespace holes | Compare against existing maximal empty rectangles. Normalize grid distances to actual target pixel aspect. A circle peak does not guarantee that an entire rectangular stamp fits. Whitespace can be intentional; condition any penalty on source hierarchy and actual crowding. |
| A05 | Optical glyph centering | Actual glyph-bounds centering already exists in measured controls and `_center_control_labels`. Test rotated/composite labels and padding ownership before adding more shifts. Ink bounding-box center is different from alpha-weighted centroid; neither is a universal visual center. |
| A06 | Nearby calm-zone search / existing scrim | Source-aware contrast repair already exists. Try bounded nearest-zone movement of whole owned lockups, verify subject/reading-order/containment and rescore actual glyph-mask contrast. Reuse a source scrim only if it belongs to that copy block; do not repurpose unrelated decorations or invent a new overlay/style in Stage 1. |
| A07 | Larger human library + hierarchical matching | Current experimental library has 134 layouts / 58 training families; extreme buckets have only 2 and 1. Audit rejection causes and label retained source-supported overlap explicitly. Do not relax constraints just to reach 500 or merge unrelated blocks because zones are fewer. Keep training-only provenance and the 0.70 prominence guard as a tested hypothesis. |

Primary mechanisms: [OR-Tools implementation](https://github.com/google/or-tools/blob/stable/ortools/sat/python/cp_model.py), [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html), [Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html). None establishes the attachment's universal runtime estimates. Generic packing is combinatorial; adding valid aspect/line-count modes improves expressiveness but costs search time.

**CP-SAT pilot design:** freeze current raw proposals; select development inputs by block count and ratio, not by known failures alone. For each block, supply measured keep/stack/row/grid variants. Penalize drift from the model's allocation, broken source axes and lost hierarchy; preserve allowable overlap rather than imposing no-overlap on every painted rectangle. Compare with current macro candidates at 25/100/300 ms budgets and deterministic seeds/single solver worker. If only family-specific examples improve, do not promote it as universal.

**Optical-balance caution:** Gemini's `(0.5, 0.52)` and “slightly above at 0.48” conflict. There is no evidence for imposing either on all posters. Use source/candidate balance as a soft feature; a full-bleed subject and an intentionally asymmetric editorial composition should not be globally centered.

### V. Runtime

| ID | Suggestion | Decision and experiment |
| --- | --- | --- |
| R01 | GGUF Q4_K_M/Q8 or ONNX INT8 CPU backend | High-value CPU experiment after model selection. Merge a matching adapter once; compare FP reference, Q8 and Q4 on exact inventory and rendered quality, then latency/RSS on Linux/amd64. Preserve native MPS as a separate backend. Quantization can alter coordinates; 4–6× improvement is not established on the judge's device. |
| R02 | 0.6B draft with 1.7B/3B target | Worth a bounded compatibility probe. Verify tokenizer compatibility, target/draft protocol, grammar interaction and supported kernels. Two models use more memory. Correct speculative algorithms preserve the target distribution under their assumptions; byte-identical output across devices/precisions is not guaranteed. |
| R03 | `prompt_lookup_num_tokens=10` | **Probed; not recommended as default from current evidence.** Supported by this Transformers installation, but slower on the local prefixes tested. Full-layout and reference-CPU tests remain separate. |
| R04 | Precompute 100px glyph advances/kerning | Existing exact Skia/Pillow measurement caches already avoid much repeated work. A unit-size approximation can change hinting, tracking, ligatures and wrap decisions. Profile cache misses first; require wrap/glyph equivalence before replacing exact measurements. |
| R05 | Geometry/glyph prefilter, render only four candidates | Useful quota-preserving funnel experiment. Geometry cannot see background contrast or every ink collision. Reserve learned/raw, repaired learned, macro, proportional and other required families; avoid optimistic irreversible pruning. Compare winner recall and changed-winner human preferences against full rendering, not only speed. |

[llama.cpp grammar support](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) can combine quantized inference and structured output. [Optimum ONNX quantization](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization) has architecture-specific configurations; verify the exact CPU target rather than assuming a native Mac instruction profile applies inside Docker. [Transformers generation strategies](https://huggingface.co/docs/transformers/generation_strategies) documents assisted/prompt-lookup decoding; compatibility was also checked in installed 5.20.1 code.

**Local R03 probe, not release evidence:** current merged 0.6B adapter, MPS/FP16, 96 output tokens, three alternating-order repeats after warming both paths. For an 11-element prompt: median greedy 2.94 s, lookup 5.73 s. For 35 elements: 4.93 s versus 9.30 s. First prompt's token prefixes matched in 3/3 repeats; dense prompt matched in 2/3. A Metal command-buffer recovery warning occurred during dense execution, so these are diagnostic observations, not trusted portable latency or quality claims. Outputs were deliberately capped prefixes, not valid final layout completions. No inference default changed.

```bash
.venv/bin/python scripts/benchmark_decode_lookup.py \
  --audits data/gemini_review_qwen_projection_fixes_2026-10-25/review/regression_suite/00_audit.json \
           data/gemini_review_qwen_projection_fixes_2026-10-25/review/regression_suite/48_audit.json
```

Other speed claims in the attachment—3–5 s M5 generation, 60–90 tokens/s Docker, 1.5–2× lookup, 2.2–2.8× speculative, 25 ms packing—remain unverified for MagiCanvas. Native generation, repair, rendering, cold loading and repeated-request caching must be timed separately. Docker execution remains unverified while its daemon is unavailable.

### VI. Stage-2 product improvements

| ID | Suggestion | Decision, constraints and smallest demo |
| --- | --- | --- |
| U01 | Six-format campaign matrix + brand score | Good first product addition. One source, independently measured outputs for Story 1080×1920, Portrait 1080×1350, Square 1080×1080, Widescreen 1600×400, Leaderboard 728×90, Skyscraper 160×600. Report concrete palette/font/asset/hierarchy checks, not an unsupported “brand preserved” score. |
| U02 | Pin/lock and reflow | Worth adding after a typed lock contract. Hard-pin actual composites, test infeasible pins, return a conflict instead of silently moving them. Initial synchronous measured reflow can be cached; <200 ms is not established. |
| U03 | Top-three diverse layout carousel | High benefit using existing candidates. Deduplicate by macro arrangement, keep full audits and safety checks, disclose deterministic versus learned candidates. Cache the shortlist for a fast choice action. Do not show three cosmetically identical variants as diversity. |
| U04 | Density slider, copy shortening, graceful omissions | Stage 2 only, opt-in with an explicit content diff and undo. Never automatically drop legal copy. User identifies protected/optional elements. Stage-1 output must retain the fixed original content/IDs. A local copy-fitter requires independent fact-preservation checks. |
| U05 | X-ray guides, contrast heatmap, morph | Good explainability feature using real measured axes/gaps/ink bounds. Do not label existing approximate contrast statistics as WCAG/APCA certification or a saliency heuristic as subject detection certainty. Animation must interpolate groups with correct source/target transforms and respect reduced motion. |
| U06 | Natural-language steering | Translate into validated typed operations (scale group, align axis, reorder peers), show intended edits, enforce invariants and support undo. No arbitrary generated Python/HTML execution. Contradictory requests produce a clear conflict. |
| U07 | Animated HTML5 export | Separate Stage-2 export project, with bundled permitted assets/fonts, safe escaping, offline playback and reduced-motion/static fallback. Preserve the required JSON/PNG scoring path. Research supports editable code representations, not automatic compliance or animation quality. |

## Google AI follow-up pipeline: resolve the ambiguous names (N03)

1. **“Qwen3-Coder sub-3B + CFG”**: no official sub-3B total-weight checkpoint was verified. Qwen3-Coder-Next advertises 3B **activated** out of **80B total**. That is unsuitable as a presumed lightweight drop-in. Keep Qwen2.5-Coder-1.5B as the code-model control and test small Qwen3.5 instead. Grammar still does not imply lower font/box MAE. [Official Coder-Next card](https://huggingface.co/Qwen/Qwen3-Coder-Next).
2. **“Phi-4 Reasoning variant / MoE-Lite”**: the concrete small candidate is Phi-4-mini-reasoning (~3.8B class), plus the flash variant. Full Phi-4-reasoning is much larger; “MoE-Lite” is not a reproducible checkpoint identifier. Reasoning token cost, total resident weights and final eligibility must be explicit. [Mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning), [publisher family](https://huggingface.co/collections/microsoft/phi-4).
3. **“Qwen3-VL small / 4B–8B”**: official 2B and 4B variants exist and deserve image-conditioned experiments. Begin at 2B; the larger tier is not automatically compatible with the final size/resource limits. Use a new processor and matching training, not the old adapter.
4. **“DiT flow matching ~10MB/~10ms”**: legitimate research direction, not a specified ready checkpoint. Flow matching may need multiple integration steps; one-pass output is an architectural/distillation choice. Test an N-slot regression baseline, then CVAE/flow variants under identical family-held-out data. Exact serialization can keep inventory, while positions, content ownership and readability still need verification. No 10ms claim without the built model and benchmark.

## Research grounding and what transfers

[LayoutDM](https://arxiv.org/abs/2303.08137) establishes discrete diffusion layout generation; [RALF](https://arxiv.org/abs/2311.13602) studies retrieval augmentation for content-aware layouts. These support experiments with alternate representations and retrieval, not the quoted 5–20 MB / 8 ms estimates for our constraints. A fixed N×4 geometry head is distinct from either complete published method.

[DesignAsCode](https://arxiv.org/abs/2602.17690) and its [author project](https://liuziyuan1109.github.io/design-as-code/) support hierarchical code representations and render feedback. Asset generation and open-ended rewriting in such systems cannot be imported into a fixed-material Stage-1 engine wholesale. Transfer the measured planning/verification ideas first. [LayoutRectifier](https://arxiv.org/abs/2508.11177) is directly relevant to an optimization experiment; its reported benchmark results are not MagiCanvas results. The attachment's exact PLAYGROUND citation is not yet pinned to a checked primary release; it is not being used as performance evidence.

## Staged experiments and decision gates

| Experiment | Change one factor | Freeze / inspect | Ship only when |
| --- | --- | --- | --- |
| E00 swappable packages | Select a validated whole base/adapter/tokenizer profile | Same proposal API and downstream engine; restart process; record prompt and artifact hashes | Each profile loads offline, correct pairs pass, crossed pairs fail, no silent fallback; pin provenance before release |
| E01 inventory grammar | Existing adapter, unconstrained vs exact ordered inventory | Same prompts, token limits, dtype, repair and selector | Completion improves without worse raw/final hard-defect rates or blind visual loss |
| E02 model frontier | Current 0.6/1.7, Qwen3.5 0.8/2, one VLM or LFM | Same train families, unique targets, budget; model-specific processor/adapter recorded | Better quality-cost frontier on new families, not a generic benchmark assertion |
| E03 font contract | Predicted font vs rendered fit; then lines contract | First use identical raw boxes; train contract arms equally if needed | Font/readability/hierarchy improve without tiny-box compensation regressions |
| E04 critic resolution | Current global, tiled current, 512, NaFlex, small VLM | Same safe rendered candidate pairs and new labels | Better real hard-pair agreement, including extreme ratios, at acceptable cost |
| E05 solver candidate | Existing pool vs +bounded CP-SAT measured variants | Exact saved raw proposals; same final raster checks | Real changed-winner wins, no new hard defects, bounded timeout/memory |
| E06 safe ranker | Off vs predeclared sparse zero-flag trigger | Same shortlist and new blinded human votes | Real campaign improvement; no synthetic-shrink-only justification |
| E07 runtime backend | Torch reference vs merged Q8/Q4/ONNX | Same prompts, weights revision, tokenizer, decoding contract | Native and required Docker measurements separately; quality loss quantified |
| E08 rendering funnel | Full shortlist vs quota-preserving prefilter | Same candidates and source assets | High best-safe-candidate recall, no hard-regression increase, real request speed gain |
| E09 dense training | Original vs synthetic fractions / macro contract | Train-family-only labels, unseen dense and reverse-ratio families | Better dense raw proposals and final outputs; no leakage or old-design regression |

For **each raw / repaired / selected** output record counts and severity of stretch, shared-axis drift, unintended overlap, bounds, clipping/text fit, parent containment, lost attachments, edge detachment, source lockup error, contrast loss, minimum type/hierarchy, hero prominence and occupied-space balance. Include candidate provenance, selector reason, actual generated-token completion and any inventory recovery. Geometry MAE is only defined where a real human target exists and is not the sole quality metric: multiple valid reflows exist.

Report per aspect and density: ≤15 / 16–22 / 23–40 elements; similar-ratio, portrait→leaderboard, portrait→skyscraper, banner→square/portrait, and 320×50. A→B→A is a stability diagnostic, not an exact-invertibility requirement after cropping or rewrap. Do not average away extreme cases. Source overlap may be intentional; only unintended target contact is defective.

Suggested pilot: select 12–20 families by density/role/aspect before examining outputs, include clean controls and tough cases. Give the baseline identical inference/render budgets. For a model change, generate **fresh** proposals; for a repair/selector change replay exact stored proposals. Save source+raw+repair+selection records, compare blind randomized image pairs, report ties/neither-good and family-level uncertainty. Promote only a clear Pareto improvement; if evidence is mixed, retain an opt-in experiment.

For a release candidate run the appropriate automated tests, the frozen 59-case engine regression, the wider 89/309 comparison and independent new-family review. All accepted engine changes must refresh the canonical Original/Resized viewer and Gemini packet. Existing reviewed cohorts cannot establish zero regressions on every future design.

## Action checklist

- [x] Catalogue all Gemini suggestions and the newer Google AI pipeline.
- [x] Review the follow-up audit; verify two local model/adapter pairs and a rejected crossed pair; document the difference between model swapping and adapter transfer.
- [x] Research official modern alternatives and pin metadata without downloading weights.
- [x] Check official brief's offline/model/public-adapter conditions; defer publication.
- [x] Verify current implemented primitives before proposing duplicates.
- [x] Run an isolated prompt-lookup feasibility/timing probe; retain its adverse result and warning.
- [x] Define per-experiment baselines, datasets, defect reports and promotion gates.
- [x] E01: implement exact-ID grammar behind an experimental interface and run complete-output comparisons; distinguish emitted IDs from usable numeric fields.
- [ ] E00: wire validated atomic profiles into runtime selection and expand the verifier beyond Qwen-specific projection checks; the current manifest/checker is audit tooling only.
- [ ] E02: compatibility smoke tests for Qwen3.5-2B, Qwen3-VL-2B and LFM3; choose at most two matching training pilots.
- [ ] E04: prepare new difficult preference labels and compare tiled/NaFlex/VLM critic features.
- [x] E05: prototype bounded discrete macro packing and measure solver statuses; paired six-case result does not justify promotion.
- [ ] E06–E09: run independently once prerequisites above pass; preserve failed experiments as evidence.
- [ ] Add Stage-2 matrix and diverse-candidate choice before more expensive steering/export features.

Training is likely needed for a **new base or output contract**, but not for the grammar, solver, quantization, font-fitting or critic ablations. The initial research phase did not download weights. Its authorized implementation follow-up now has pinned downloads and executed pilots; see UPGRADE_PILOT_RESULTS.md for measured findings and limitations. No new training run, public adapter upload or production promotion is claimed.
