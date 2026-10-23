# Qwen layout LoRA: first Colab run

This workflow trains the v2 exact-pair JSONL prepared by
`scripts/prepare_qwen_training.py`. It does not upload the Crello files to
Hugging Face. Training may still be subject to Crello/VistaCreate terms; check
`CRELLO_DATA_AND_TRAINING.md` and the event's model rules before publishing an
adapter. Qwen3 fine-tuning with Unsloth is supported in Colab in the [Qwen
training guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/unsloth.md).

## 1. Create the exact-only files locally

From the repository root, with the full pair scan and converted splits already
present:

```bash
python3 scripts/prepare_qwen_training.py
```

This creates exact-only chat JSONL with source/reference font sizes preserved.
Text labels are never shrunk through the engine's font fitter. Connected exact
and core template families stay together in a stable 80/10/10 split; core pairs
remain review-only. Each row records its `family_id`, `source_template_id`, and
`prompt_version`.

The new defaults cap each *source template* at **2 outgoing directions** in each
split. Connected components still define train/validation/test isolation, but a
large component no longer loses nearly all its examples to one family-wide cap.
This retains more resize variety while limiting repeated-source domination.
`--train-source-cap 0 --eval-source-cap 0` disables caps. Evaluate the cap with
the held-out split; it is a sampling choice, not evidence of higher model quality.

To prepare the next experiment without replacing the installed adapter's data:

```bash
python3 scripts/prepare_qwen_training.py --output data/qwen_lora_source_capped_v2
```

This dataset uses the existing v2 contract (0..1000 boxes and font size relative
to the short side). An optional v3 experiment changes only font normalization:

```bash
python3 scripts/prepare_qwen_training.py --prompt-version 3 --output data/qwen_lora_v3_source_capped
```

V3 uses `sqrt(width * height)` as the font reference. Both normalizations are
reversible, and the old short-side encoding does **not** inherently amplify pixel
quantization error on banners. V3 is an experiment requiring a new adapter,
not a demonstrated correction to v2. Never use a v2 adapter to decode v3 tokens.
Keep training/validation/test from one prepared directory together. Existing
`data/qwen_lora/` and its 90-case review remain the historical baseline; do not
claim that repeatedly inspected cases are a fresh blind test.

Upload only `train.jsonl` and `validation.jsonl` in Colab for training. Keep
`test.jsonl` for the final local evaluation. These files contain design text and
geometry, so only use a Colab account/storage location permitted for the source
data.

## 2. Start a GPU Colab notebook

In Colab choose **Runtime → Change runtime type → T4 GPU** (or another available
CUDA GPU). Run these cells in order. If the install reports a version conflict,
restart the runtime and follow the current install cell in the linked Qwen guide.

```python
!pip -q install unsloth datasets trl
```

If the response-mask helper reports a marker or masking error, upgrade the
training packages and restart the Colab runtime before rerunning the notebook:

```python
!pip -q install --upgrade unsloth unsloth_zoo trl
```

Upload `train.jsonl` and `validation.jsonl` from
`data/qwen_lora_source_capped_v2/` when prompted:

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
assert PROMPT_VERSION in (2, 3)
assert all(int(row.get("prompt_version", 2)) == PROMPT_VERSION
           for split in raw.values() for row in split)
ADAPTER_NAME = f"qwen3-layout-lora-v{PROMPT_VERSION}"

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
coordinates, preserved image proportions, and no content edits. Then feed the
adapter's proposals through the engine and compare rendered PNGs with the
non-Qwen baseline. Inspect several examples, including dense and unusual target
ratios. A malformed/incomplete response is rejected by the runtime wrapper; the
deterministic candidate path remains available.

The installed v2 adapter continues to load from `models/qwen3-layout-lora-v2/`.
For a v3 experiment, install the new adapter into `models/qwen3-layout-lora-v3/`
and set `MAGICANVAS_QWEN_ADAPTER_DIR` to that directory. The version marker must
match the evaluation JSONL. Run with `MAGICANVAS_QWEN_ENABLED=1` to enable model
candidates. The Dockerfile supports build-time public adapter fetching through
`QWEN_ADAPTER_REPO` and `QWEN_ADAPTER_REVISION`; the training cells do not upload
anything publicly.

Runtime now compares the decoder-checked proposal with source-lockup/column,
measured-font, bounded local-search, and contrast-repair alternatives. Font
calibration is a runtime candidate, not a change to human target labels. Raw
model completeness is reported separately from recovered runtime layouts, so
fallbacks no longer inflate the model's JSON/ID validity score. Current limits
are 4096 input tokens, up to 4096 generated tokens, and an 8192-token combined
ceiling (or the model's smaller context limit). Output budget is estimated from
the tokenized answer schema, with omitted IDs and token exhaustion logged.
