#!/usr/bin/env python3
"""Fetch a reviewed public Qwen adapter into the offline runtime image."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True,help="Public Hugging Face adapter repository id")
    parser.add_argument("--revision",default=None)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"models"/"qwen3-layout-lora")
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    args.output.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo_id=args.repo,revision=args.revision,
                      local_dir=str(args.output),
                      ignore_patterns=["*.bin","*.h5","*.msgpack","*.onnx","*.tflite"])
    print(f"Qwen adapter ready in {args.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
