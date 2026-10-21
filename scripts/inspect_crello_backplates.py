#!/usr/bin/env python3
"""Measure converted Crello image layers that sit behind text boxes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crello_sample"
sys.path.insert(0, str(ROOT))
from renderer.converter.assets import visible_content_bounds
from engine.resize import _text_visible_bounds


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def box(e):
    return tuple(float(e[k]) for k in ("x", "y")) + (float(e["x"])+float(e["width"]), float(e["y"])+float(e["height"]))


def stats(path, asset_box, text_box, visible_area_ratio):
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3] >= 8
    visible = int(alpha.sum())
    if not visible:
        return None
    pixels = rgba[:, :, :3][alpha].astype(np.float32)
    std = float(pixels.std(axis=0).mean())
    # Alpha-mask compactness and bilateral symmetry are useful shape clues,
    # but only supporting signals; neither is a shape classifier by itself.
    ys, xs = np.where(alpha)
    crop = alpha[ys.min():ys.max()+1, xs.min():xs.max()+1]
    fill = float(crop.mean())
    sym_h = float(np.mean(crop == crop[:, ::-1]))
    sym_v = float(np.mean(crop == crop[::-1, :]))
    rgb = rgba[:, :, :3].astype(np.int16)
    gx = np.abs(np.diff(rgb, axis=1)).mean(axis=2)
    gy = np.abs(np.diff(rgb, axis=0)).mean(axis=2)
    opaque_x = alpha[:, 1:] & alpha[:, :-1]
    opaque_y = alpha[1:, :] & alpha[:-1, :]
    edge = float(((gx > 20) & opaque_x).sum() + ((gy > 20) & opaque_y).sum()) / max(1, opaque_x.sum()+opaque_y.sum())
    # Map the intersecting text rectangle into asset-pixel coordinates and
    # measure its opaque pixels and color variance there as well.
    ix0, iy0, ix1, iy1 = asset_box
    tx0, ty0, tx1, ty1 = text_box
    w, h = ix1-ix0, iy1-iy0
    if w <= 0 or h <= 0:
        return None
    x0 = max(0, min(rgba.shape[1]-1, int((max(ix0, tx0)-ix0)/w*rgba.shape[1])))
    y0 = max(0, min(rgba.shape[0]-1, int((max(iy0, ty0)-iy0)/h*rgba.shape[0])))
    x1 = max(x0+1, min(rgba.shape[1], int((min(ix1, tx1)-ix0)/w*rgba.shape[1])))
    y1 = max(y0+1, min(rgba.shape[0], int((min(iy1, ty1)-iy0)/h*rgba.shape[0])))
    patch = rgba[y0:y1, x0:x1]
    pmask = patch[:, :, 3] >= 8
    patch_std = float(patch[:, :, :3][pmask].astype(np.float32).std(axis=0).mean()) if pmask.any() else None
    patch_opaque = float(pmask.mean())
    return {"rgb_std": round(std, 2), "text_region_rgb_std": None if patch_std is None else round(patch_std, 2),
            "text_region_opaque_ratio": round(patch_opaque, 4),
            "edge_density": round(edge, 4), "alpha_fill_ratio": round(fill, 4),
            "horizontal_symmetry": round(sym_h, 4), "vertical_symmetry": round(sym_v, 4),
            "alpha_area_ratio": round(visible/(rgba.shape[0]*rgba.shape[1]), 4),
            "visible_bounds_area_ratio": round(visible_area_ratio, 4)}


def main():
    manifest = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text().splitlines() if line]
    hits = []
    for row in manifest:
        if row.get("status") != "exported":
            continue
        design = json.loads((DATA / row["design"]).read_text())
        canvas = design["canvas"]
        texts = [e for e in design["elements"] if e.get("type") == "text"]
        for img in (e for e in design["elements"] if e.get("type") == "image"):
            ibox = visible_content_bounds(img, str(DATA))
            behind = []
            for txt in texts:
                if abs(float(img.get("rotation", 0))) > .1 or abs(float(txt.get("rotation", 0))) > .1:
                    continue
                tbox = _text_visible_bounds(txt)
                ratio = overlap(ibox, tbox) / max(1e-9, (tbox[2]-tbox[0])*(tbox[3]-tbox[1]))
                if img.get("z_index", 0) < txt.get("z_index", 0) and ratio >= .85:
                    behind.append((txt, ratio, tbox))
            if not behind:
                continue
            path = DATA / img["asset_id"]
            measured = stats(path, box(img), behind[0][2],
                             (ibox[2]-ibox[0])*(ibox[3]-ibox[1])/(float(img["width"])*float(img["height"])))
            if measured is None:
                continue
            area = float(img["width"])*float(img["height"])/(canvas["width"]*canvas["height"])
            covers = (float(img["x"]) <= canvas["width"]*.05 and
                      float(img["y"]) <= canvas["height"]*.05 and
                      float(img["x"])+float(img["width"]) >= canvas["width"]*.95 and
                      float(img["y"])+float(img["height"]) >= canvas["height"]*.95)
            flat = (measured["rgb_std"] <= 12 and measured["edge_density"] <= .08
                    and measured["alpha_fill_ratio"] >= .82
                    and measured["horizontal_symmetry"] >= .90 and measured["vertical_symmetry"] >= .90
                    and measured["text_region_rgb_std"] is not None
                    and measured["text_region_rgb_std"] <= 12
                    and measured["text_region_opaque_ratio"] >= .85
                    and ibox[2]-ibox[0] >= (behind[0][2][2]-behind[0][2][0])*1.02
                    and ibox[3]-ibox[1] >= (behind[0][2][3]-behind[0][2][1])*1.06)
            kind = "full_bleed" if covers else "small_backplate" if area < .16 else "panel_candidate" if .06 <= area <= .80 else "other"
            hits.append({"template_id": row["template_id"], "title": row.get("title"), "image_id": img["id"],
                         "asset": img["asset_id"], "area_share": round(area, 4), "candidate_kind": kind,
                         "flat_low_variance": flat,
                         "behind_text": [{"id": t["id"], "content": t.get("content", "")[:80], "overlap": round(o, 4)} for t,o,_ in behind],
                         **measured})
    out = DATA / "backplate_diagnostics.json"
    out.write_text(json.dumps(hits, ensure_ascii=False, indent=2)+"\n")
    print(f"Scanned {sum(r.get('status')=='exported' for r in manifest)} designs; {len(hits)} image/text containment cases across {len({h['template_id'] for h in hits})} designs.")
    print(f"Strict pixel-confirmed flat backplates/cards: {sum(h['flat_low_variance'] for h in hits)} across {len({h['template_id'] for h in hits if h['flat_low_variance']})} designs.")
    print(f"Containment candidates: full bleed {sum(h['candidate_kind']=='full_bleed' for h in hits)}, small backplate {sum(h['candidate_kind']=='small_backplate' for h in hits)}, panel-sized {sum(h['candidate_kind']=='panel_candidate' for h in hits)} (these counts include photos/decor and are not confirmed backplates).")
    for hit in sorted(hits, key=lambda h: (not h['flat_low_variance'], h['rgb_std']))[:12]:
        print(json.dumps(hit, ensure_ascii=False))
    print(f"Detailed metrics: {out}")


if __name__ == "__main__":
    main()
