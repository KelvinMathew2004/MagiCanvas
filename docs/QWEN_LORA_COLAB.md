# Qwen layout LoRA: first Colab run

This workflow trains the v2 exact-pair JSONL prepared by
`scripts/prepare_qwen_training.py`. It does not upload the Crello files to
Hugging Face. Training may still be subject to Crello/VistaCreate terms; check
`CRELLO_DATA_AND_TRAINING.md` and the event's model rules before publishing an
adapter. Qwen3 fine-tuning with Unsloth is supported in Colab in the [Qwen
training guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/unsloth.md).

The current v11 engine changes are inference-time candidate selection and
geometry repairs. They do not change the Qwen prompt, target JSON, coordinates,
or text labels, so they do not require regenerating the prepared JSONL or
changing the response-only masking code below. Rebuild the dataset only after
changing `scripts/prepare_qwen_training.py` or its source pair manifest.

## 1. Prepare the training data

From the repository root, with the full pair scan and converted splits already
present, create the selected exact-pair training set. It preserves target font
sizes and isolates connected template families across train, validation, and
test.

```bash
python3 scripts/prepare_qwen_training.py \
  --output data/qwen_lora_expanded_v2 \
  --train-source-cap 4 \
  --train-family-cap 40 \
  --stratify-aspect-transitions
```

The current manifest provides **2,706** expanded training directions from
4,242 eligible exact directions. The 309 validation and 89 test rows remain
byte-for-byte identical to the current capped-v2 evaluation split. The
stratified cap prioritizes portrait, square, landscape, and wide transitions,
then balances directions across sources and families.

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

Load the model and format the messages with Qwen's own chat template:

```python
import torch
from datasets import load_dataset
from math import ceil
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=4096,
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
ADAPTER_NAME = "qwen3-layout-lora-v2-expanded"

def format_chat(row):
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    answer = row["messages"][-1]["content"]
    return {"text": prompt + answer + tokenizer.eos_token}

data = raw.map(format_chat)
max_length = 4096
before = {split: len(data[split]) for split in data}
data = data.filter(lambda row: len(tokenizer(
    row["text"], add_special_tokens=False)["input_ids"]
) <= max_length)
print("Rows removed because full prompt+answer exceeds context:",
      {split: before[split] - len(data[split]) for split in data})
```

Use three epochs as the first run, then select based on validation and held-out
layout quality rather than assuming that more training is better. Runtime varies
with the Colab GPU, package versions, and sequence lengths; do not rely on a
fixed training-time estimate.

```python
from trl import SFTConfig, SFTTrainer

EVAL_STEPS = max(1, min(100, ceil(len(data["train"]) / 16) // 2))
args = SFTConfig(
    output_dir="qwen-layout-checkpoints",
    dataset_text_field="text",
    max_length=4096,
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

response_prefix = tokenizer.apply_chat_template(
    data["train"][0]["messages"][:-1], tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
)
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
sample_index = 0
sample = trainer.train_dataset[sample_index]
input_ids = sample["input_ids"].tolist() if hasattr(sample["input_ids"], "tolist") else sample["input_ids"]
labels = sample["labels"].tolist() if hasattr(sample["labels"], "tolist") else sample["labels"]

# The exact generation prefix must match the start of the training sequence,
# and every system/user/prefix token must have label -100 (masked from loss).
prefix_ids = tokenizer(response_prefix, add_special_tokens=False)["input_ids"]
assert input_ids[:len(prefix_ids)] == prefix_ids, "Training prefix differs from inference prefix"
assert all(label == -100 for label in labels[:len(prefix_ids)]), \
    "System/user/generation prefix is not fully masked"

# Decode only unmasked labels. This avoids printing the tokenizer's pad marker
# (for example, repeated <|vision_pad|> tokens) in place of every masked token.
unmasked_label_ids = [int(token) for token, label in zip(input_ids, labels)
                      if int(label) != -100]
decoded_labels = tokenizer.decode(unmasked_label_ids, skip_special_tokens=False)
assistant_json = data["train"][sample_index]["messages"][-1]["content"]
system_prompt = data["train"][sample_index]["messages"][0]["content"]
user_prompt = data["train"][sample_index]["messages"][1]["content"]
assert assistant_json in decoded_labels, "Assistant JSON/coordinates are missing from unmasked labels"
assert system_prompt not in decoded_labels and user_prompt not in decoded_labels, \
    "System or user prompt text is present in unmasked labels"
print(decoded_labels)
```

Training constructs the same generation prefix used at inference, then appends
the assistant JSON and EOS token. The response marker is derived from the
installed tokenizer's generation prefix, including Qwen3 thinking-control
tokens, so a template change cannot silently mask every answer. See the
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

Keep the new adapter in `models/qwen3-layout-lora-v2-expanded/` so the current
v2 adapter remains available. Evaluate it on the expanded export's held-out
89-row test split and compare against the current adapter:

```bash
MAGICANVAS_QWEN_ENABLED=1 python3 scripts/evaluate_qwen_adapter.py \
  --input data/qwen_lora_expanded_v2/test.jsonl \
  --adapter-dir models/qwen3-layout-lora-v2-expanded \
  --compare-adapter-dir models/qwen3-layout-lora-v2 \
  --engine \
  --output data/qwen_lora_expanded_v2/adapter_comparison.json
```

The report includes raw model JSON completion and valid-proposal rates, raw
geometry and relative font MAE, post-constraint geometry and relative font MAE,
Qwen-family winner counts, and hard-check pass rates for both adapters. The
`candidate_beats_both_comparison_metrics` flag is true only when the new adapter
has lower post-constraint geometry and font MAE, more Qwen-family wins, complete
valid layouts, and a 100% hard-check pass rate. Review per-example results and
renders before changing the installed adapter. This full-engine comparison
renders the held-out cases for both adapters, so it takes longer than raw JSON
evaluation.

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
