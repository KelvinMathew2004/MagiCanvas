# Qwen layout LoRA: first Colab run

This is a starter workflow for the exact-pair JSONL prepared by
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

This creates `data/qwen_lora/train.jsonl`, `validation.jsonl`, `test.jsonl`,
`core_review.jsonl`, and `summary.json`. The exporter filters exact-content
mismatches, projects target coordinates onto the engine's canvas and image
aspect constraints, and fits target text sizes to the source text. It assigns
whole connected template families (connected by exact and core matches) to one stable 80/10/10 split, preventing
variants from leaking across training and evaluation even when upstream Crello
splits disagree. Training families are capped at six directed examples; the
validation and test cap is twelve per family. Core pairs stay review-only
because unmatched decorations make target mapping noisy. Check
`data/qwen_lora/summary.json` for current counts and filtering totals. Keep the
test file untouched until model selection is finished.

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

Upload `train.jsonl` and `validation.jsonl` from the local `data/qwen_lora/`
folder when prompted:

```python
from google.colab import files
uploaded = files.upload()
assert "train.jsonl" in uploaded and "validation.jsonl" in uploaded
```

Load the model and format the messages with Qwen's own chat template:

```python
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=2048,
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

def format_chat(row):
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    answer = row["messages"][-1]["content"]
    return {"text": prompt + answer + tokenizer.eos_token}

data = raw.map(format_chat)
max_length = 2048
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

args = SFTConfig(
    output_dir="qwen-layout-checkpoints",
    dataset_text_field="text",
    max_length=2048,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
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
    raw["train"][0]["messages"][:-1], tokenize=False,
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

Before training, inspect a masked example and confirm the decoded labels contain
the assistant's JSON answer and coordinates, while the system/user prompt is
masked. Training constructs the exact same generation prefix used at inference,
then appends the assistant JSON and EOS token. The response marker is derived
from the installed tokenizer's prefix, including any Qwen3 thinking-control
tokens, so a Transformers template change cannot silently mask every answer.
Install current `unsloth`, `unsloth_zoo`, and TRL versions if the helper reports
a marker or masking error.
See the [Unsloth fine-tuning notebooks](https://github.com/unslothai/notebooks)
for the maintained response-only masking examples; the helper is applied after
trainer construction in this workflow.

```python
sample = trainer.train_dataset[0]
visible_labels = [
    token if label != -100 else tokenizer.pad_token_id
    for token, label in zip(sample["input_ids"], sample["labels"])
]
print(tokenizer.decode(visible_labels, skip_special_tokens=False))
assert raw["train"][0]["messages"][-1]["content"] in tokenizer.decode(
    visible_labels, skip_special_tokens=False
), "Assistant answer is missing from the unmasked labels"
```

Then start training:

```python
trainer.train()
```

Save the adapter files, then download them as one archive:

```python
model.save_pretrained("qwen3-layout-lora")
tokenizer.save_pretrained("qwen3-layout-lora")
!zip -qr qwen3-layout-lora.zip qwen3-layout-lora
from google.colab import files
files.download("qwen3-layout-lora.zip")
```

## 3. Evaluate before wiring/publishing

Do not choose a checkpoint only because its training loss fell. Evaluate held-out
test prompts for valid JSON, exactly the expected IDs, numeric/in-canvas
coordinates, preserved image proportions, and no content edits. Then feed the
adapter's proposals through the engine and compare rendered PNGs with the
non-Qwen baseline. Inspect several examples, including dense and unusual target
ratios. A malformed/incomplete response is rejected by the runtime wrapper; the
deterministic candidate path remains available.

The engine can load a local adapter from `models/qwen3-layout-lora/` (or from
`MAGICANVAS_QWEN_ADAPTER_DIR`) after it is copied there. Run the project with
`MAGICANVAS_QWEN_ENABLED=1` to enable the Qwen candidate. The Dockerfile currently
downloads only the public base model; it will need a build-time adapter fetch
step after a reviewed, approved adapter has a public repository. The model
weights are not uploaded by these Colab cells.
