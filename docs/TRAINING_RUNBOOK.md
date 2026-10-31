# Train the VL-2B challenger

Run these steps in order for **Qwen3-VL-2B only**. Defer Qwen3.5-0.8B.
Use one CUDA GPU per job. Keep the completed 0.6B and old checkpoints.

## 1. Copy the prepared files

1. Upload `data/training_next/layout_training_bundle_uniform.zip` from this
   checkout to your private Drive folder `MyDrive/MagiCanvas/`.
2. Open `training/colab.ipynb` in Colab, or paste these cells into a notebook.
3. Select **Runtime → Change runtime type → T4 GPU**, or a larger CUDA GPU.
4. Run only one training process per model/output folder. Do not use the old
   all-model cell or `docs/archive/QWEN_LORA_COLAB.md`.
5. When moving accounts, copy the private ZIP and notebook to your destination
   Drive and mount that Drive. Do not use multiple accounts to bypass resource
   restrictions: [Colab policy](https://research.google.com/colaboratory/faq.html).

## 2. Restore and verify

```python
import torch, json, shutil
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from google.colab import drive

assert torch.cuda.is_available(), "Select a CUDA GPU runtime."
gpu = torch.cuda.get_device_properties(0)
assert gpu.total_memory >= 14 * 2**30, "Use a T4-class or larger GPU."
print(gpu.name, round(gpu.total_memory / 2**30, 2), "GiB", torch.__version__)
drive.mount("/content/drive")
source = Path("/content/drive/MyDrive/MagiCanvas/layout_training_bundle_uniform.zip")
assert source.is_file(), "Upload the new uniform ZIP to this Drive folder."
archive = Path("/content/layout_training_bundle_uniform.zip")
shutil.copy2(source, archive)
assert sha256(archive.read_bytes()).hexdigest() == "fbe34f10f418cf3d738af84927beca348eeb08c1ac15d44aa363e887298b513c", "Wrong or changed ZIP."
bundle = Path("/content/layout_training_uniform")
bundle.mkdir(exist_ok=True)
with ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(bundle)
manifest = json.loads((bundle / "manifest.json").read_text())
for name, digest in manifest["bundle_hashes"].items():
    assert sha256((bundle / name).read_bytes()).hexdigest() == digest, name
assert manifest["data_policy"]["name"] == "audited-uniform-v1"
assert manifest["splits"]["train"]["rows"] == 1043
assert manifest["splits"]["validation"]["rows"] == 309
Path("/content/torch-constraint.txt").write_text("torch==" + torch.__version__.split("+")[0] + "\n")
print("Verified: 1,043 train / 309 validation; unit weights, one visit per row.")
```

## 3. Install, then restart Python

```python
%pip install torchvision --index-url https://download.pytorch.org/whl/cpu -c /content/torch-constraint.txt
%pip install -r /content/layout_training_uniform/requirements.txt -c /content/torch-constraint.txt
```

Choose **Runtime → Restart session**. Continue with step 4; do not delete the
runtime or repeat installation. Keep CUDA Torch installed.

## 4. Train VL-2B

```python
from google.colab import drive
from pathlib import Path
import subprocess, sys, torch

assert torch.cuda.is_available(), "Reconnect to a CUDA GPU."
drive.mount("/content/drive")
MODEL = "qwen3-vl-2b-instruct"
bundle = Path("/content/layout_training_uniform")
assert (bundle / "continue_remaining.py").is_file(), "Restore the uniform bundle first."
output = Path("/content/drive/MyDrive/MagiCanvas/uniform_v2_candidates")
output.mkdir(parents=True, exist_ok=True)

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

stream_training([
    sys.executable, "-u", str(bundle / "continue_remaining.py"),
    "--bundle", str(bundle), "--output", str(output),
    "--models", MODEL, "--save-every", "10",
], output / (MODEL + "-console.log"))
```

Let the forward/backward/optimizer preflight finish before the training bar.
If it reports nonfinite gradients or insufficient memory, stop and use a larger
compatible GPU. On interruption, stop the prior process and rerun step 4 with
the **same uniform bundle and output folder**. Never copy weighted checkpoints
into `uniform_v2_candidates`.

## 5. Download the completed adapter

```python
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files
import json

root = Path("/content/drive/MyDrive/MagiCanvas/uniform_v2_candidates")
bundle = Path("/content/layout_training_uniform")
final = root / MODEL / "final"
assert (final / "adapter_model.safetensors").is_file(), "Training/export is incomplete."
saved = json.loads((final / "training_manifest.json").read_text())
assert saved["bundle_sha256"] == sha256((bundle / "manifest.json").read_bytes()).hexdigest()
assert saved["recipe"]["data_policy"] == "audited-uniform-v1"
archive = Path("/content") / (MODEL + "-uniform-adapter.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for p in sorted(final.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(root))
    z.write(root / "uniform_run_identity.json", "uniform_run_identity.json")
    z.write(bundle / "manifest.json", "uniform_bundle_manifest.json")
files.download(str(archive))
```

Keep Drive checkpoints until the download has been verified locally. Send the
adapter ZIP here for evaluation; keep the existing production adapter.

Stop after the VL-2B download. Do not start 0.8B or repeat 0.6B for this run.
