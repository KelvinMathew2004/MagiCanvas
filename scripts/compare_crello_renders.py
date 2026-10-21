#!/usr/bin/env python3
"""Render converted Crello designs with the unchanged starter cr-renderer.

Each render is compared with the original dataset preview PNG stored during
conversion. The report is a triage aid: pixel similarity is affected by font
availability and rasterization differences, so a low score is not by itself
proof that the JSON conversion is incorrect.

Example:
    .venv/bin/python scripts/compare_crello_renders.py --splits train validation test
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback
from typing import Any

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dataset_renderer import render_design_to_png

_WORKER_FONTS: str | None = None
_WORKER_MAX_SIZE = 2048
_WORKER_TEMP: tempfile.TemporaryDirectory[str] | None = None


def _init_worker(fonts_file: str, max_size: int) -> None:
    global _WORKER_FONTS, _WORKER_MAX_SIZE, _WORKER_TEMP
    _WORKER_FONTS = fonts_file
    _WORKER_MAX_SIZE = max_size
    _WORKER_TEMP = tempfile.TemporaryDirectory(prefix="crello-render-worker-")


def _render_one(task: tuple[str, dict[str, Any], str, str | None]) -> dict[str, Any]:
    """Process-pool worker: render a single manifest row and compare pixels."""
    split, row, split_dir_raw, save_dir_raw = task
    split_dir = Path(split_dir_raw)
    design_path = split_dir / row["design"]
    preview_path = split_dir / row["preview"] if row.get("preview") else None
    result: dict[str, Any] = {"status": "error", "split": split,
                              "row_index": row.get("row_index"),
                              "template_id": row.get("template_id"),
                              "design": row.get("design"), "preview": row.get("preview")}
    try:
        if _WORKER_TEMP is None or _WORKER_FONTS is None:
            raise RuntimeError("Render worker was not initialized")
        temp_png = Path(_WORKER_TEMP.name) / "render.png"
        design = json.loads(design_path.read_text(encoding="utf-8"))
        warnings = render_design_to_png(
            design, split_dir, temp_png, _WORKER_FONTS, max_size=_WORKER_MAX_SIZE
        )
        result["warnings"] = warnings
        with Image.open(temp_png) as rendered:
            rendered_copy = rendered.copy()
        if preview_path is None or not preview_path.is_file():
            result["status"] = "rendered_no_preview"
        else:
            with Image.open(preview_path) as reference:
                result["metrics"] = compare_images(rendered_copy, reference.copy())
            result["status"] = "compared"
        if save_dir_raw:
            out_png = Path(save_dir_raw) / split / f"{row.get('template_id', row['row_index'])}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            rendered_copy.save(out_png)
            result["rendered_png"] = str(out_png)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def _rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)


def compare_images(rendered: Image.Image, reference: Image.Image) -> dict[str, float]:
    """Compare at reference dimensions; scores are diagnostic, not a gate."""
    rw, rh = rendered.size
    pw, ph = reference.size
    a, b = _rgb(rendered, (pw, ph)), _rgb(reference, (pw, ph))
    delta = a - b
    mae = float(np.abs(delta).mean() / 255.0)
    mse = float(np.square(delta).mean())
    # Global, channel-wise SSIM. This provides a stable coarse ranking without
    # requiring an extra imaging dependency such as scikit-image.
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(axis=(0, 1)), b.mean(axis=(0, 1))
    centered_a, centered_b = a - mu_a, b - mu_b
    var_a = np.square(centered_a).mean(axis=(0, 1))
    var_b = np.square(centered_b).mean(axis=(0, 1))
    covariance = (centered_a * centered_b).mean(axis=(0, 1))
    ssim = ((2 * mu_a * mu_b + c1) * (2 * covariance + c2) /
            ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))
    return {
        "mae": mae,
        "psnr_db": float(10 * math.log10(255**2 / max(mse, 1e-12))),
        "global_ssim": float(np.mean(ssim)),
        "render_width": rw,
        "render_height": rh,
        "preview_width": pw,
        "preview_height": ph,
        "aspect_ratio_delta": abs(rw / rh - pw / ph),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--fonts-file", type=Path, default=ROOT / "data" / "crello_fonts.pickle")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "output" / "crello-render-comparison")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum exported designs per split (0 means all).")
    parser.add_argument("--save-renders", action="store_true",
                        help="Keep rendered PNGs under report-dir/renders (can use substantial disk space).")
    parser.add_argument("--max-size", type=int, default=2048,
                        help="Maximum render dimension passed to cr-renderer (default: 2048).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel renderer processes (default: 4). Use 1 to run serially.")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("--limit must be non-negative and --workers must be at least 1")
    if not args.fonts_file.is_file():
        parser.error(f"Font bundle not found: {args.fonts_file}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    render_dir = args.report_dir / "renders"
    if args.save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)

    overall = {"splits": {}, "report_dir": str(args.report_dir), "saved_renders": args.save_renders}
    for split in args.splits:
        split_dir = args.data_root / f"crello_{split}"
        manifest_path = split_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            print(f"ERROR missing manifest: {manifest_path}", file=sys.stderr)
            overall["splits"][split] = {"fatal_error": f"Missing {manifest_path}"}
            continue
        report_path = args.report_dir / f"{split}.jsonl"
        summary = {"manifest_rows": 0, "exported": 0, "rendered": 0,
                   "compared": 0, "missing_preview": 0, "render_errors": 0,
                   "warnings": 0, "scores": []}
        started = time.monotonic()
        tasks: list[tuple[str, dict[str, Any], str, str | None]] = []
        skipped_rows: list[dict[str, Any]] = []
        with manifest_path.open(encoding="utf-8") as manifest:
            for line in manifest:
                if not line.strip():
                    continue
                row = json.loads(line)
                summary["manifest_rows"] += 1
                if row.get("status") != "exported":
                    skipped_rows.append({"status": "skipped", "row_index": row.get("row_index"),
                                         "template_id": row.get("template_id"),
                                         "reason": row.get("reason")})
                    continue
                if args.limit and summary["exported"] >= args.limit:
                    break
                summary["exported"] += 1
                tasks.append((split, row, str(split_dir), str(render_dir) if args.save_renders else None))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(str(args.fonts_file), args.max_size)) as pool, \
                report_path.open("w", encoding="utf-8") as report:
            for skipped in skipped_rows:
                report.write(json.dumps(skipped, ensure_ascii=False) + "\n")
            for result in pool.map(_render_one, tasks, chunksize=4):
                summary["rendered"] += result["status"] in {"compared", "rendered_no_preview"}
                summary["warnings"] += len(result.get("warnings", []))
                summary["missing_preview"] += result["status"] == "rendered_no_preview"
                summary["compared"] += result["status"] == "compared"
                summary["render_errors"] += result["status"] == "error"
                if result.get("metrics"):
                    summary["scores"].append(result["metrics"]["global_ssim"])
                report.write(json.dumps(result, ensure_ascii=False) + "\n")
                if summary["rendered"] and summary["rendered"] % 100 == 0:
                    print(f"{split}: rendered {summary['rendered']}/{summary['exported']}; "
                          f"errors {summary['render_errors']}", flush=True)
        scores = summary.pop("scores")
        summary["global_ssim_mean"] = statistics.fmean(scores) if scores else None
        summary["global_ssim_median"] = statistics.median(scores) if scores else None
        summary["global_ssim_min"] = min(scores) if scores else None
        summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
        summary["record_report"] = str(report_path)
        overall["splits"][split] = summary
        print(f"{split}: {summary['rendered']}/{summary['exported']} rendered, "
              f"{summary['compared']} compared, {summary['render_errors']} errors, "
              f"median SSIM {summary['global_ssim_median']}; report {report_path}", flush=True)
    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 1 if any(item.get("render_errors", 0) or item.get("fatal_error")
                    for item in overall["splits"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
