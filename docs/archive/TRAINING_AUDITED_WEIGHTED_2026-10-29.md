# Train and compare the next layout adapters

Run the numbered blocks in order. Use VS Code's terminal on your Mac for steps 1 and 5; use one Colab **GPU** notebook for steps 2–4. Select your available **T4 GPU** (an L4 or larger also works). CUDA is required for training, not preparation. The same full-length recipe is attempted one model at a time; each model must pass an actual memory/numerical preflight before its epochs. T4 fit and speed have not been measured locally. Allow several hours for training/evaluation; no 45-minute promise.

This controlled experiment prepares six model packages but trains **three first-round adapters**: Qwen3-0.6B as the same-base data control, newer Qwen3.5-0.8B as the small architecture challenger, and Qwen3-VL-2B with source-image conditioning. The existing trained 1.7B remains a reference; fresh 1.7B, Coder-1.5B and Qwen3.5-2B are reserves, not automatic training jobs. This avoids paying for six overlapping experiments before evidence justifies them. It does not overwrite the current champion, upload weights, or change engine defaults. A larger adapter is not automatically better.

## 1. Prepare locally

From the MagiCanvas repository with its existing `.venv`, local Crello exports/fonts and the six pinned local model packages listed in `config/training_candidates.json`:

```bash
.venv/bin/python scripts/prepare_training_run.py --workers 2
.venv/bin/python scripts/expand_training_candidates.py --workers 2
```

The resumable audit writes `data/training_next/layout_training_bundle.zip`. Upload **that ZIP only** to Colab. It contains model-specific train/validation messages, hashes, training code and 971 rendered **source** PNGs for the VLM (about 281 MiB zipped). It contains no human target images, test rows, font files or model weights. Keep this dataset-derived bundle private; it is excluded from Gemini uploads. The source corpus remains unchanged. Confirmed newly invisible labels are excluded from training; severe new overflow is downweighted to 0.5. Uncertain shared-mask warnings alone never delete data. Tracking is scaled with font size before auditing, matching inference; older audit counts made without that scaling are superseded. Validation stays unchanged for comparison.

Training keeps one unique target per row and family-disjoint splits. Every row is visited once per epoch; existing ≥18-element rows receive three visits. This addresses scarcity modestly, not missing diversity: the original training export has only 35 dense rows. Do not train on official samples, reviewed alternate designs, the 59-case suite or test pairs.


The prepared run contains **1,043 training / 309 validation rows**. All 1,431
original train/validation rows were screened natively: 79 training rows have a
confirmed newly invisible label and are excluded; 78 retained rows receive
weight 0.5 for severe new overflow. There are 34 retained dense rows and 1,111
sampler visits per epoch. Maximum sequence lengths are 4,749 train / 3,537
validation tokens for the two reference tokenizers. All six providers pass response-mask checks; the VLM maximum is 5,003 train / 3,751 validation tokens, including source-image tokens. These checks improve label
screening; they do not certify aesthetic quality. The experiment combines
curation, current prompts and dense sampling, so a gain cannot be attributed to
one change without a later ablation.

A ready-to-run Colab notebook is also provided as `training/colab.ipynb`; it
contains the same ordered GPU cells below.

## 2. Install in Colab

Create a notebook, select **Runtime → Change runtime type → GPU**, then run:

```python
import torch
assert torch.cuda.is_available(), "Select a GPU runtime before continuing."
gpu = torch.cuda.get_device_properties(0)
assert gpu.total_memory >= 14 * 2**30, "Use a T4-class or larger CUDA GPU with at least 14 GiB visible VRAM."
print(gpu.name, round(gpu.total_memory / 2**30, 2), "GiB", torch.__version__)
print("Precision:", "BF16" if torch.cuda.is_bf16_supported(including_emulation=False) else "FP16", "— actual fit is checked per model before training.")
from pathlib import Path
Path("/content/torch-constraint.txt").write_text("torch==" + torch.__version__.split("+")[0] + "\n")

from google.colab import files
uploaded = files.upload()  # Select layout_training_bundle.zip
assert set(uploaded) == {"layout_training_bundle.zip"}

from pathlib import Path
from zipfile import ZipFile
bundle = Path("/content/layout_training")
bundle.mkdir(exist_ok=True)
with ZipFile("layout_training_bundle.zip") as z:
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(bundle)
```

```python
%pip install torchvision --index-url https://download.pytorch.org/whl/cpu -c /content/torch-constraint.txt
%pip install -r /content/layout_training/requirements.txt -c /content/torch-constraint.txt
```

Use Colab's requested **Restart session** after installation, then continue with step 3. The constraint preserves Colab's CUDA Torch. Only Torchvision preprocessing uses a matching CPU wheel; the model still trains on CUDA. Do not replace Torch with a CPU-only wheel. The recipe pins Transformers/PEFT/Accelerate/bitsandbytes and records the actual CUDA/Torch environment with each adapter.

To restart, choose **Runtime → Restart session** in Colab's top menu and confirm. This restarts Python and keeps the uploaded ZIP, extracted `/content/layout_training` files and installed packages. Then run step 3; do not rerun the upload/install cells. Do not choose **Disconnect and delete runtime**, which removes that temporary environment. Older bundles installed Hugging Face `datasets 5.1.0`, whose PyArrow dependency conflicts with Colab's unrelated preinstalled cuDF. Installation can still succeed; this trainer does not use cuDF. Do not downgrade PyArrow alone to satisfy that warning. Current bundles omit the unnecessary `datasets` dependency because training reads JSON directly into a PyTorch dataset.

## 3. Train the three selected models sequentially

**If Qwen3-0.6B already reached checkpoint 210**, use
[the continuation launcher](TRAINING_CONTINUATION.md) instead of this cell. It
skips that completed model even if its final export was interrupted, and adds
10-step recovery saves without changing the hashed training bundle.

```python
from google.colab import drive
from pathlib import Path
import subprocess, sys, psutil

drive.mount("/content/drive")
output = Path("/content/drive/MyDrive/MagiCanvas/audited_v2_candidates")
output.mkdir(parents=True, exist_ok=True)
script = "/content/layout_training/train.py"
# On a deliberate rerun, stop only an orphan from this exact training bundle.
# Saved checkpoints survive; progress since the last checkpoint may be lost.
old = []
for p in psutil.process_iter(["cmdline"]):
    command = p.info["cmdline"] or []
    if script in command and "--bundle" in command and command[command.index("--bundle") + 1] == "/content/layout_training":
        try:
            p.terminate()
            old.append(p)
        except psutil.NoSuchProcess:
            pass
_, alive = psutil.wait_procs(old, timeout=10)
for p in alive:
    p.kill()
psutil.wait_procs(alive, timeout=10)

def stream_training(command, log_path):
    with log_path.open("a", buffering=1) as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

print("Starting training; live output below. Console log:", output / "training-console.log", flush=True)
stream_training([
    sys.executable, "-u", script,
    "--bundle", "/content/layout_training",
    "--output", str(output),
], output / "training-console.log")
```

The launcher explicitly streams both stdout and stderr into Colab and saves the same output to Drive's `training-console.log`. Loading/tokenization and the GPU preflight happen before the training bar; the initial bar is not evidence that epochs already started. The earlier `subprocess.run` launcher could leave child output outside Colab's visible cell capture. If that cell is already running, changing notebook text cannot attach a live bar to its existing process. Switching launchers requires stopping the cell and rerunning this step; it resumes the last saved checkpoint, but unsaved progress can restart. Do not restart or delete the whole runtime merely to change the launcher. Checkpoints currently save at epoch boundaries.

One CUDA GPU, NF4 double quantization, response-only loss including EOS, no packing or truncation, 6,144-token maximum, rank 16 / alpha 32, all language linear projections (including the hybrid attention projections in Qwen3.5); vision encoder and LM head stay frozen, learning rate 1e-4, three epochs, microbatch 1 / accumulation 16, seed 3407. T4 uses FP16; GPUs with native BF16 use BF16. AdamW avoids tensor-list temporaries (`foreach=False`), and gradient checkpointing stays enabled. Before training, each package must pass CUDA forward/backward and an AdamW step at zero learning rate on its long-input and long-answer training rows, including existing gradient storage. Loss, finite gradients, time and peak allocated/reserved memory are checked; the probe does not alter the adapter. This catches common memory failures early but does not prove every row fits. Validation loss then selects the checkpoint. The full CUDA recipe remains unexecuted on this Mac. Checkpoints persist on Drive; rerunning the same command resumes an interrupted run. Completed matching runs are skipped. Use a new output folder if the bundle changes.

If a model fails its GPU memory preflight, the run stops with its name and keeps any earlier completed adapters. Restart the runtime to release GPU memory. Do not bypass that failure or shorten data just to force a run: that model needs a larger GPU for this unchanged recipe. The previous blanket 22 GiB check was overly restrictive for sequential small-model QLoRA; passing the new 14 GiB hardware floor only permits this real preflight.

The three training runs use the same audited rows and output schema. Each uses its native chat template and context-aware relation compaction. The VLM sees the source PNG plus JSON; text challengers see JSON only. This is a capability comparison, not a pure parameter-count ablation. The script computes vocabulary logits only for response positions to reduce memory. Target line counts are saved locally as audit sidecars, **not supplied as input** or added to the v2 output. Conditioning on an unknown target line count would leak the answer; changing the schema is a separate experiment.

## 4. Download adapters

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files

root = Path("/content/drive/MyDrive/MagiCanvas/audited_v2_candidates")
archive = Path("/content/audited_layout_adapters.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    import json
    specs = json.loads(Path("/content/layout_training/manifest.json").read_text())["models"]
    for spec in specs:
        if spec["key"] not in json.loads(Path("/content/layout_training/manifest.json").read_text())["training_queue"]:
            continue
        model = spec["key"]
        final = root / model / "final"
        assert (final / "adapter_model.safetensors").is_file(), model
        assert (final / "training_manifest.json").is_file(), model
        for p in sorted(final.iterdir()):
            if p.is_file():
                z.write(p, p.relative_to(root))
files.download(str(archive))
```

The three downloaded LoRAs are small relative to their bases. They require their **own matching base**: a 0.6B adapter cannot be attached to 1.7B. Publication to Hugging Face remains pending until a model is selected.

## 5. Evaluate locally before promotion

Save the download to `~/Downloads/audited_layout_adapters.zip`, then run from the repository:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
archive = Path.home() / 'Downloads/audited_layout_adapters.zip'
with ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall('models/audited_v2_candidates')
PY
.venv/bin/python scripts/refresh_regression_viewer.py --workers 2
.venv/bin/python scripts/evaluate_trained_layouts.py --trained models/audited_v2_candidates
```

The last command runs **fresh inference** for the existing champion and the three selected new adapters, using the same v2 layout contract and selector (native provider chat/image formatting) on 89 reference pairs plus all 59 review tasks. It resumes only matching engine/data/weight snapshots. It can take hours on a Mac. Qwen stays enabled in every arm. Audit JSON retains raw and final geometry; the comparison page shows only Original/Resized, with a resized-model selector:

http://127.0.0.1:8766/review/trained_adapters/index.html

Promotion requires raw inventory improvement on dense tasks (or continued 100% completion), no reference-ID regression, lower raw geometry/font error on common-valid pairs, no new flag types, ≤1 flagged review case, valid final IDs, lower reference geometry/font error and more learned-family wins. The fresh geometry baseline and historical .044773 reference must both be beaten. Passing makes a model eligible for visual review; **nothing auto-installs it**. These repeatedly used development cohorts cannot establish unseen generalization.

## What is already verified

CPU tests check exact response masking, EOS supervision, loss/gradient equivalence to full logits, accumulation normalization, dense-sampler coverage, source-preserving label decoding, T4 precision selection, optimizer configuration and the zero-rate probe leaving parameters unchanged. Local preparation tokenizes every retained training/validation row with every model's native tokenizer/processor and refuses overlength rows. The 4-case untuned 0.8B probe produced 3/4 complete inventories and no learned winners; no improvement is promised. CUDA training and new-adapter performance have not been executed on this Mac; those results must come from your run.

Reference implementation guidance: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [Qwen3 non-thinking mode](https://huggingface.co/Qwen/Qwen3-1.7B), [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).
