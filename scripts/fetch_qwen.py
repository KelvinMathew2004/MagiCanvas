#!/usr/bin/env python3
"""Download a supported Qwen base model for offline, model-swappable inference."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    "Qwen/Qwen3-0.6B":"qwen3-0.6b",
    "Qwen/Qwen3-1.7B":"qwen3-1.7b",
    "Qwen/Qwen2.5-1.5B-Instruct":"qwen2.5-1.5b-instruct",
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=sorted(MODELS),default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--revision",default=None)
    parser.add_argument("--max-workers",type=int,default=2,
                        help="Parallel Hub downloads (small values are steadier on slow links)")
    args=parser.parse_args()
    if args.max_workers<1:parser.error("--max-workers must be positive")
    target=(args.output or ROOT/"models"/MODELS[args.model]).expanduser().resolve()
    target.mkdir(parents=True,exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.model,revision=args.revision,
                      local_dir=str(target),
                      ignore_patterns=["*.bin","*.h5","*.msgpack","*.onnx","*.tflite"],
                      max_workers=args.max_workers)
    print(f"{args.model} ready in {target}")


if __name__=="__main__":main()
