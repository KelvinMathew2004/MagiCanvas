# Qwen layout LoRA: first Colab run

This workflow trains the v2 exact-pair JSONL prepared by
`scripts/prepare_qwen_training.py`. It does not upload the Crello files to
Hugging Face. Training may still be subject to Crello/VistaCreate terms; check
`CRELLO_DATA_AND_TRAINING.md` and the event's model rules before publishing an
adapter. Qwen3 fine-tuning with Unsloth is supported in Colab in the [Qwen
training guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/unsloth.md).

The current engine changes are inference-time container projection, shared-axis
snapping, candidate selection, and geometry repairs. They do not change the
Qwen prompt, target JSON, coordinates, or text labels, so they do not require
regenerating the prepared JSONL or changing the response-only masking code
below. Rebuild the dataset only after changing `scripts/prepare_qwen_training.py`
or its source pair manifest.

## 1. Mine held-out defects and prepare training data

Run the resumable batch miner over the frozen 89-case test and 309-case
validation files. It stores a checkpoint/result for every case and creates a
viewer containing only the ten highest-ranked failing cases.

```bash
python3 scripts/mine_qwen_batch_defects.py \
  --test data/qwen_lora_expanded_v2/test.jsonl \
  --validation data/qwen_lora_expanded_v2/validation.jsonl \
  --output data/qwen_lora/batch_defect_mining_v1 \
  --workers 2
```

The rank uses rendered layout defect cost, low-contrast glyph count, invisible
text, Skia-rendered glyph overflow, source lockup error, source-derived axis
alignment, and the final Qwen-on geometry MAE change against Qwen-off. The
engine's existing SigLIP score remains a coarse tie-breaker after its rendered
and geometry checks.

The current training export now has **1,122 pairs from 2,706**. Each
`target_template_id` occurs once; the selected rows still cover 295 connected
families and all measured aspect transitions. The frozen 309 validation and 89
test rows are preserved. `train_pre_target_id_cap.jsonl` is an audit backup;
upload only `train.jsonl` and `validation.jsonl`.

To repeat only the training-set cap on the frozen export:

```bash
python3 scripts/filter_qwen_training_targets.py \
  --input data/qwen_lora_expanded_v2/train_pre_target_id_cap.jsonl \
  --output data/qwen_lora_expanded_v2/train.jsonl \
  --max-per-target 1
```

New pair exports can enforce the same rule with
`scripts/prepare_qwen_training.py --train-target-cap 1`.

Upload only `train.jsonl` and `validation.jsonl` in Colab for training. Keep
`test.jsonl` for the final local evaluation. These files contain design text and
geometry, so only use a Colab account/storage location permitted for the source
data.

## 2. Start a GPU Colab notebook

In Colab choose **Runtime → Change runtime type → T4 GPU**. Run these cells in
order.

```python
!pip -q install --upgrade unsloth unsloth_zoo datasets trl
```

Upload `train.jsonl` and `validation.jsonl` from
`data/qwen_lora_expanded_v2/`. Keep `test.jsonl` local for final evaluation.

```python
from google.colab import files
uploaded = files.upload()
assert "train.jsonl" in uploaded and "validation.jsonl" in uploaded
```

Run the training cells once per model. Choose `MODEL_KEY`, execute the notebook
from this model cell through the save cell, download the adapter ZIP, then
restart the Colab runtime and repeat for the other key. This produces two
independent adapters trained on the same unique-target training rows.

Load either model and format the messages with its own chat template:

```python
import torch
from datasets import load_dataset
from math import ceil
from unsloth import FastLanguageModel
from huggingface_hub import model_info

MODEL_CONFIGS = {
    "qwen3_1_7b": {
        "model_name": "Qwen/Qwen3-1.7B",
        "adapter_name": "qwen3-layout-lora-v2-expanded-1.7b",
    },
    "qwen25_1_5b": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_name": "qwen25-layout-lora-v2-expanded-1.5b",
    },
}
MODEL_KEY = "qwen3_1_7b"  # Set to qwen25_1_5b for the second run.
MODEL_NAME = MODEL_CONFIGS[MODEL_KEY]["model_name"]
ADAPTER_NAME = MODEL_CONFIGS[MODEL_KEY]["adapter_name"]

# Prevent automatic remapping to a different pre-quantized repository so the
# recorded revision identifies the model actually used by both A/B runs.
BASE_REVISION = model_info(MODEL_NAME).sha
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    revision=BASE_REVISION,
    use_exact_model_name=True,
    max_seq_length=6144,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

raw = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "validation.jsonl",
})

PROMPT_VERSION = int(raw["train"][0].get("prompt_version", 2))
assert PROMPT_VERSION == 2
assert all(int(row.get("prompt_version", 2)) == PROMPT_VERSION
           for split in raw.values() for row in split)
def make_prompt(messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:  # Some model/tokenizer templates do not expose this option.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

def format_chat(row):
    prompt = make_prompt(row["messages"][:-1])
    answer = row["messages"][-1]["content"]
    return {"text": prompt + answer + tokenizer.eos_token}

data = raw.map(format_chat)
max_length = 6144
before = {split: len(data[split]) for split in data}
lengths = {split: [len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
                  for row in data[split]] for split in data}
print({split: {"rows": len(values), "max_tokens": max(values)}
       for split, values in lengths.items()})
assert all(max(values) <= max_length for values in lengths.values()), \
    "Increase context length before training; do not silently drop complex layouts"
assert len({row["target_template_id"] for row in raw["train"]}) == len(raw["train"])
for left, right in (("train", "validation"),):
    assert not ({r["family_id"] for r in raw[left]} & {r["family_id"] for r in raw[right]})
```

Use three epochs as the first run, then select based on validation loss and
validation layout quality rather than assuming that more training is better. Runtime varies
with the Colab GPU, package versions, and sequence lengths; do not rely on a
fixed training-time estimate.

```python
from trl import SFTConfig, SFTTrainer

EVAL_STEPS = max(1, min(100, ceil(len(data["train"]) / 16) // 2))
args = SFTConfig(
    output_dir="qwen-layout-checkpoints",
    dataset_text_field="text",
    max_length=6144,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    report_to="none",
    seed=3407,
    data_seed=3407,
    packing=False,
)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    args=args,
)
```

Mask system and user prompt tokens so loss is computed on the assistant response
containing the target layout JSON:

```python
from unsloth.chat_templates import train_on_responses_only

response_prefix = make_prompt(data["train"][0]["messages"][:-1])
assistant_start = response_prefix.rfind("<|im_start|>assistant")
assert assistant_start >= 0
response_part = response_prefix[assistant_start:]
assert response_part in data["train"][0]["text"]
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part=response_part,
)
```

```python
# Audit every train and validation row, not only a short first example.
for split, prepared in (("train", trainer.train_dataset), ("validation", trainer.eval_dataset)):
    for index, sample in enumerate(prepared):
        ids = list(sample["input_ids"])
        labels = list(sample["labels"])
        prefix = make_prompt(data[split][index]["messages"][:-1])
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        assert ids[:len(prefix_ids)] == prefix_ids, (split, index, "inference prefix mismatch")
        assert all(label == -100 for label in labels[:len(prefix_ids)]), (split, index, "prompt loss")
        assert labels[len(prefix_ids):] == ids[len(prefix_ids):], (split, index, "answer masked")
        assert any(label != -100 for label in labels), (split, index, "all labels masked")
print("All train/validation rows have exact response-only labels.")
```

Training constructs the same generation prefix used at inference, then appends
the assistant JSON and EOS token. The response marker is derived from the
selected tokenizer's generation prefix, so a template change cannot silently
mask every answer. See the
[Unsloth fine-tuning notebooks](https://github.com/unslothai/notebooks) for
maintained response-only masking examples; this helper is applied after trainer
construction in this workflow.

Then start training:

```python
trainer.train()
```

Save the adapter files, then download them as one archive:

```python
model.save_pretrained(ADAPTER_NAME)
tokenizer.save_pretrained(ADAPTER_NAME)
from pathlib import Path
import shutil
Path(ADAPTER_NAME, "magicanvas_prompt_version.txt").write_text(f"{PROMPT_VERSION}\n")
# Preserve enough metadata to reproduce this A/B run.
import hashlib, importlib.metadata, json, subprocess
metadata = {
    "base_model": MODEL_NAME, "base_revision": BASE_REVISION,
    "model_key": MODEL_KEY, "prompt_version": PROMPT_VERSION,
    "train_sha256": hashlib.sha256(Path("train.jsonl").read_bytes()).hexdigest(),
    "validation_sha256": hashlib.sha256(Path("validation.jsonl").read_bytes()).hexdigest(),
    "training_arguments": args.to_dict(),
    "versions": {name: importlib.metadata.version(name) for name in
                 ("torch", "transformers", "trl", "unsloth", "peft", "datasets")},
}
Path(ADAPTER_NAME, "training_run.json").write_text(json.dumps(metadata, indent=2, default=str))
Path(ADAPTER_NAME, "requirements.freeze.txt").write_text(
    subprocess.check_output(["pip", "freeze"], text=True))
archive = shutil.make_archive(ADAPTER_NAME, "zip", ".", ADAPTER_NAME)
from google.colab import files
files.download(archive)
```

## 3. Evaluate before wiring/publishing

Do not choose a checkpoint only because its training loss fell. Evaluate held-out
test prompts for valid JSON, exactly the expected IDs, numeric/in-canvas
coordinates, preserved image proportions, and no content edits. The A/B command
below then compares raw proposals and engine-selected layouts against the
currently installed v2 adapter. Inspect the per-case results and renders before
switching adapters. A malformed or incomplete response is rejected by the
runtime wrapper; the deterministic candidate path remains available.

Both base models are already downloaded in this checkout at
`models/qwen3-1.7b/` and `models/qwen2.5-1.5b-instruct/`. The commands below are
only needed when setting up another machine. Drop each Colab adapter in its own
`models/` folder, keeping the existing 0.6B adapter untouched:

```bash
python3 scripts/fetch_qwen.py --model Qwen/Qwen3-1.7B
python3 scripts/fetch_qwen.py --model Qwen/Qwen2.5-1.5B-Instruct
```

The base revision recorded in the downloaded adapter's `training_run.json`
should match the weights used for local evaluation. If it differs from the local
download, fetch that model again with `--revision <base_revision>` before A/B.

Unzip the Colab downloads to
`models/qwen3-layout-lora-v2-expanded-1.7b/` and
`models/qwen25-layout-lora-v2-expanded-1.5b/`. Evaluate either adapter on the
same 89-row held-out test split against the installed Qwen3-0.6B baseline:

```bash
MAGICANVAS_QWEN_ENABLED=1 python3 scripts/evaluate_qwen_adapter.py \
  --input data/qwen_lora_expanded_v2/test.jsonl \
  --adapter-dir models/qwen3-layout-lora-v2-expanded-1.7b \
  --model-dir models/qwen3-1.7b \
  --compare-adapter-dir models/qwen3-layout-lora-v2 \
  --compare-model-dir models/qwen3-0.6b \
  --engine \
  --output data/qwen_lora_expanded_v2/adapter_comparison_qwen3_1_7b.json
```

For the Qwen2.5 adapter, use:

```bash
MAGICANVAS_QWEN_ENABLED=1 python3 scripts/evaluate_qwen_adapter.py \
  --input data/qwen_lora_expanded_v2/test.jsonl \
  --adapter-dir models/qwen25-layout-lora-v2-expanded-1.5b \
  --model-dir models/qwen2.5-1.5b-instruct \
  --compare-adapter-dir models/qwen3-layout-lora-v2 \
  --compare-model-dir models/qwen3-0.6b \
  --engine \
  --output data/qwen_lora_expanded_v2/adapter_comparison_qwen25_1_5b.json
```

Raw JSON completion is a non-regression gate (a 100% baseline cannot be strictly improved).
Geometry MAE and Qwen win rate must improve; final hard checks must all pass.

The report includes raw JSON completion and exact-ID validity, raw geometry MAE,
post-constraint geometry MAE, Qwen-family win rate, and hard-check pass rate for
both model/adapter pairs. The promotion flag requires no loss of raw JSON completion,
lower post-constraint geometry MAE, a higher Qwen-family win rate, no regression
in valid raw proposals, complete final layouts, and a 100% hard-check pass rate.
No adapter is installed automatically. This full-engine comparison renders the
held-out cases for both candidates, so it takes longer than raw JSON evaluation.

The existing baseline remains at `models/qwen3-layout-lora-v2/`. Run with
`MAGICANVAS_QWEN_ENABLED=1` to enable model candidates. The Dockerfile supports
build-time public adapter fetching through `QWEN_ADAPTER_REPO` and
`QWEN_ADAPTER_REVISION`; the training cells do not upload anything publicly.

Runtime now compares the decoder-checked proposal with source-lockup/column,
measured-font, bounded local-search, and contrast-repair alternatives. Font
calibration is a runtime candidate, not a change to human target labels. Raw
model completeness is reported separately from recovered runtime layouts, so
fallbacks no longer inflate the model's JSON/ID validity score. Current limits
are 4096 input tokens, up to 4096 generated tokens, and an 8192-token combined
ceiling (or the model's smaller context limit). Output budget is estimated from
the tokenized answer schema, with omitted IDs and token exhaustion logged.

## Training preflight evidence

The local tokenizer preflight checked all 1,122 training and 309 validation rows
for both base models. Qwen3 has a maximum sequence of 4,749 tokens; Qwen2.5 has
4,745. A 4,096-token filter removed 14 of the more complex training examples,
so this recipe now uses 6,144 tokens and fails explicitly if that is insufficient.
There are zero shared families between any train/validation/test split.

Qwen3-1.7B is a reasonable first lightweight candidate; there is no evidence yet
that it is optimal for this layout task. Use the same rows, seed, LoRA rank,
optimizer settings, and evaluation engine for both runs. Three epochs with
r=16/alpha=32 and 1e-4 learning rate is a starting configuration, not a measured
optimum. Select the checkpoint on validation; use the frozen test set to report
results after decisions are made, not to choose repeated training settings.

Run the CPU training preflight before uploading files:

```bash
python3 scripts/verify_qwen_training.py
```

Run the installed adapter stress suite independently of the later model A/B:

```bash
python3 scripts/evaluate_qwen_adapter.py \
  --input data/qwen_lora_expanded_v2/test.jsonl --engine --engine-only \
  --official-samples input --workers 2 --resume \
  --model-dir models/qwen3-0.6b --adapter-dir models/qwen3-layout-lora-v2 \
  --output data/qwen_lora/day4_5_stress_v4/report.json
```

There are three official source JSONs in `input/`; six formats produce 18 official
combinations, in addition to 89 held-out Crello cases. These are different test
sources. The generated outlier viewer contains only the ten worst flagged cases.
Checkpoints include an engine/data/adapter fingerprint; use a new output directory
when code or inputs change.

The exact-model loading option follows the [Unsloth model loader](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py):
without it a requested model may resolve to a pre-quantized mirror, where the
original model revision does not identify the resolved weights.
