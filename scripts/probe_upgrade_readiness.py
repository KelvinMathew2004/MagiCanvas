#!/usr/bin/env python3
"""Inventory upgrade prerequisites without loading weights or changing the engine.

Offline by default. --refresh-hub fetches small, public model metadata/configs
only, pins their observed revision, and never downloads or executes model code.
Metadata parameter counts describe stored tensors, not peak runtime memory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "MagiCanvas-research/1.0"})
    with urlopen(request, timeout=30) as response:
        data = response.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError("Metadata response exceeds 4 MB")
    return json.loads(data)


def hub_record(model):
    model_id = model["model_id"]
    result = {"model_id": model_id, "card_url": f"https://huggingface.co/{model_id}"}
    try:
        info = fetch_json(f"https://huggingface.co/api/models/{model_id}")
        revision = info["sha"]
        card = info.get("cardData") or {}
        tensors = info.get("safetensors") or {}
        result.update(revision=revision, license=card.get("license"),
                      license_name=card.get("license_name"), license_link=card.get("license_link"),
                      gated=info.get("gated"), created_at=info.get("createdAt"),
                      stored_parameters=tensors.get("total"),
                      parameter_dtypes=tensors.get("parameters"),
                      pipeline=info.get("pipeline_tag"),
                      library=info.get("library_name"))
        try:
            config = fetch_json(f"https://huggingface.co/{model_id}/resolve/{revision}/config.json")
            result.update(model_type=config.get("model_type"),
                          architectures=config.get("architectures"),
                          auto_map=config.get("auto_map"),
                          config_sha256=sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                          config=config)
            try:
                from transformers.models.auto.modeling_auto import (
                    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)
                kind = config.get("model_type")
                result["installed_loader_mapping"] = {
                    "causal_lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(kind),
                    "image_text_to_text": MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get(kind),
                    "note": "Class mapping only, not a model load or kernel/processor compatibility test."}
            except ImportError as exc:
                result["loader_probe_error"] = str(exc)
        except Exception as exc:
            result["config_error"] = str(exc)
        # Lower-bound dense floating-weight storage, excluding runtime buffers.
        if tensors.get("total"):
            result["fp16_weight_GiB_lower_bound"] = round(tensors["total"] * 2 / 2**30, 3)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def local_tensors(folder):
    total = 0
    files = []
    for path in sorted(folder.glob("*.safetensors")):
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated tensor header: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if length > 32_000_000:
                raise ValueError(f"Excessive tensor header: {path}")
            header = json.loads(stream.read(length))
        total += sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
        files.append({"name": path.name, "bytes": path.stat().st_size})
    return {"stored_parameters": total, "files": files,
            "note": "Stored tensor count; tied/omitted weights and runtime buffers require separate accounting."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "config/upgrade_experiments.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/upgrade_exploration/readiness.json")
    parser.add_argument("--refresh-hub", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    packages = {}
    for name in ("torch", "transformers", "peft", "scipy", "ortools", "xgrammar",
                 "outlines", "llama-cpp-python", "onnxruntime", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "registry_sha256": sha256(args.registry.read_bytes()).hexdigest(),
              "platform": {"system": platform.system(), "machine": platform.machine(),
                           "python": sys.version.split()[0]},
              "packages": packages, "local_models": {},
              "production_changed": False, "weight_downloads": False}
    for folder in sorted((ROOT / "models").iterdir()):
        if not folder.is_dir() or not (folder / "config.json").is_file():
            continue
        config = json.loads((folder / "config.json").read_text())
        report["local_models"][folder.name] = dict(model_type=config.get("model_type"),
                                                  **local_tensors(folder))
    try:
        import torch
        report["platform"].update(torch_mps=torch.backends.mps.is_available(),
                                   torch_cuda=torch.cuda.is_available())
        from transformers.generation.utils import GENERATION_MODES_MAPPING
        report["generation_modes"] = {str(k): v for k, v in GENERATION_MODES_MAPPING.items()}
    except (ImportError, AttributeError) as exc:
        report["runtime_probe_error"] = str(exc)
    if args.refresh_hub:
        with ThreadPoolExecutor(max_workers=4) as pool:
            report["hub_models"] = list(pool.map(hub_record, registry["models"]))
        report["hub_metadata_checked_at"] = report["created_at"]
    elif args.output.is_file():
        # Preserve pinned research evidence without calling it freshly fetched.
        previous = json.loads(args.output.read_text())
        if previous.get("registry_sha256") == report["registry_sha256"] and previous.get("hub_models"):
            report["hub_models"] = previous["hub_models"]
            report["hub_metadata_checked_at"] = previous.get("hub_metadata_checked_at", previous["created_at"])
            report["hub_metadata_reused"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "models": len(report.get("hub_models", [])),
                      "hub_errors": sum("error" in r for r in report.get("hub_models", [])),
                      "production_changed": False}))


if __name__ == "__main__":
    main()
