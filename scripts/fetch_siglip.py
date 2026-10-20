"""Download SigLIP 2 weights into ``models/siglip`` for offline use.

Runs at Docker build time (see the Dockerfile) so the weights are baked into
the image and the engine can load them under ``--network none`` at runtime.
Re-runnable locally for development. Requires network access ONLY while
fetching; the engine then loads these files offline.

Only the safetensors weights, config, and SentencePiece tokenizer are kept;
the redundant .bin/.h5/.msgpack/.onnx copies are skipped to keep the image
lean.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "google/siglip2-base-patch16-224"
TARGET = Path(__file__).resolve().parents[1] / "models" / "siglip"


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(TARGET),
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.tflite"],
    )
    print(f"SigLIP weights ready in {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
