#!/usr/bin/env python3
"""Index the original Crello preview images for a vision-model experiment.

This reads the converted split manifests only. It does not render designs,
copy/re-encode preview files, or treat locally rendered images as positives.
The output contains paths to the original preview images and preserves the
upstream train/validation/test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train": ROOT / "data" / "crello_train",
    "validation": ROOT / "data" / "crello_validation",
    "test": ROOT / "data" / "crello_test",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "siglip_previews")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in SPLITS}
    missing = []
    for split, folder in SPLITS.items():
        manifest = folder / "manifest.jsonl"
        if not manifest.exists():
            raise SystemExit(f"Missing {manifest}; convert the Crello split first.")
        with (args.output / f"{split}.jsonl").open("w") as out:
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                if row.get("status") != "exported" or not row.get("preview"):
                    continue
                preview = (folder / row["preview"]).resolve()
                if not preview.is_file():
                    missing.append({"split": split, "id": row.get("template_id"),
                                    "path": str(preview)})
                    continue
                out.write(json.dumps({
                    "image": str(preview),
                    "split": split,
                    "label": "dataset_preview_positive_candidate",
                    "template_id": row.get("template_id"),
                    "origin": "cyberagent/crello preview column",
                }, ensure_ascii=False) + "\n")
                counts[split] += 1

    summary = {"counts": counts, "total_indexed": sum(counts.values()),
               "missing_previews": len(missing),
               "missing": missing,
               "note": "Original dataset previews are indexed in place; no local renders are used as positives."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
