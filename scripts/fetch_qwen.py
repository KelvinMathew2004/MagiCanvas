"""Download Qwen3-0.6B weights for offline local/Docker inference."""
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B"
TARGET = Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(TARGET),
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.tflite"],
    )
    print(f"Qwen weights ready in {TARGET}")


if __name__ == "__main__":
    main()
