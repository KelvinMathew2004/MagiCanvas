#!/usr/bin/env python3
"""Render weak preference pairs for a later SigLIP ranking-head experiment.

The supplied target preview is a human-authored alternate-format reference;
the second image is today's engine output for that same source/target canvas.
This is a weak preference signal, not an assertion that every engine result is
bad or that every source preview is universally good.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep the engine's local candidate renders on the same Crello fonts/assets as
# the dataset target preview; this affects this preparation process only.
os.environ.setdefault("MAGICANVAS_DATASET_RENDERING", "1")
from engine.resize import resize, _text, _text_visible_bounds, _rgba, _asset_root
from engine.resize import _overlap, _intentional_stack_pair
from engine.asset_geometry import visible_content_bounds
from scripts.dataset_renderer import render_design_to_png
from scripts.compare_crello_renders import compare_images

PAIR_FILE = ROOT / "data" / "crello_verified_pairs.json"
SPLIT_DIR = ROOT / "data" / "crello_train"
OUT_DIR = ROOT / "data" / "siglip_pairs"
INSPECTION_DIR = ROOT / "data" / "siglip_negative_pairs"


def _composite_at_preview_size(source_png, destination, preview_size):
    with Image.open(source_png) as image:
        image.convert("RGB").resize(preview_size, Image.Resampling.LANCZOS).save(destination)


def _linear_luminance(rgb):
    values=[]
    for value in rgb:
        v=value/255.0
        values.append(v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4)
    return .2126*values[0]+.7152*values[1]+.0722*values[2]


def _text_contrast(image_path, element, canvas):
    try:
        import numpy as np
        color=_rgba(element.get("color", "#000000"))
        if color is None:return None
        box=_text_visible_bounds(element)
        with Image.open(image_path) as image:
            rgb=image.convert("RGB")
            sx,sy=rgb.width/canvas["width"],rgb.height/canvas["height"]
            x0=max(0,min(rgb.width,int(box[0]*sx)));y0=max(0,min(rgb.height,int(box[1]*sy)))
            x1=max(x0+1,min(rgb.width,int(box[2]*sx+1)));y1=max(y0+1,min(rgb.height,int(box[3]*sy+1)))
            pixels=np.asarray(rgb.crop((x0,y0,x1,y1)),dtype=np.uint8)
        backgrounds=np.median(pixels.reshape(-1,3),axis=0)
        bg_lum=_linear_luminance(backgrounds)
        alpha=float(color[3])
        fg_lum=_linear_luminance(color[:3])*alpha+bg_lum*(1-alpha)
        return (max(fg_lum,bg_lum)+.05)/(min(fg_lum,bg_lum)+.05)
    except (ImportError,ValueError,TypeError,OSError,KeyError):
        return None


def _candidate_defects(source, target_design, output, pair, source_meta, target_meta,
                       target_render_path, candidate_path,
                       target_background_path=None,candidate_background_path=None):
    """Return evidence-backed reasons an engine render is a weak negative."""
    source_is_a=source_meta["id"]==pair["a"]["id"]
    out_by_id={e["id"]:e for e in output["elements"]}
    source_to_target={}
    for match in pair.get("matches",[]):
        si=match["a_index"] if source_is_a else match["b_index"]
        ti=match["b_index"] if source_is_a else match["a_index"]
        source_to_target[si]=ti
    canvas=target_design["canvas"]
    center_deltas=[]
    text_deltas=[]
    defects=[]
    shrink_ratios=[]
    matched_visuals=[]
    target_elements=target_design["elements"]
    for source_index,target_index in source_to_target.items():
        original=source["elements"][source_index]
        target=target_elements[target_index]
        adapted=out_by_id.get(original["id"])
        if adapted is None:
            continue
        dx=((adapted["x"]+adapted["width"]/2)-(target["x"]+target["width"]/2))/canvas["width"]
        dy=((adapted["y"]+adapted["height"]/2)-(target["y"]+target["height"]/2))/canvas["height"]
        displacement=(dx*dx+dy*dy)**.5
        center_deltas.append(displacement)
        if target["type"]=="text":
            text_deltas.append(displacement)
            ratio=adapted.get("font_size",0)/max(1e-6,target.get("font_size",0))
            shrink_ratios.append(ratio)
            measured_h,measured_w,lines=_text(adapted)
            if measured_h>adapted["height"]*1.03 or measured_w>adapted["width"]*1.03:
                defects.append(f"text_overflow:{original['id']}")
            if adapted.get("max_lines") and lines>adapted["max_lines"]:
                defects.append(f"text_line_limit:{original['id']}")
            if ratio<.70:
                defects.append(f"text_shrunk:{original['id']}:{ratio:.2f}")
            target_contrast=_text_contrast(target_background_path or target_render_path,target,canvas)
            candidate_contrast=_text_contrast(candidate_background_path or candidate_path,adapted,canvas)
            if (target_contrast is not None and candidate_contrast is not None
                    and target_contrast>=2.0
                    and candidate_contrast<max(1.35,target_contrast*.60)):
                defects.append(f"contrast_drop:{original['id']}:{target_contrast:.2f}->{candidate_contrast:.2f}")
        elif original["type"] in ("image", "shape") and target["type"] in ("image", "shape"):
            matched_visuals.append((original, target, adapted))

    def canvas_background(element, design):
        cw,ch=design["canvas"]["width"],design["canvas"]["height"]
        full=(element["x"]<=cw*.02 and element["y"]<=ch*.02
              and element["x"]+element["width"]>=cw*.98
              and element["y"]+element["height"]>=ch*.98)
        behind=all(element.get("z_index",0)<=text.get("z_index",0)
                   for text in design["elements"] if text["type"]=="text")
        return full and behind and not element.get("group_id")

    def visual_area(element):
        if element["type"]=="image":
            box=visible_content_bounds(element,_asset_root(element))
        else:
            box=(element["x"],element["y"],element["x"]+element["width"],
                 element["y"]+element["height"])
        return max(1.,(box[2]-box[0])*(box[3]-box[1]))

    measured_visuals=[]
    for original,target,adapted in matched_visuals:
        if (canvas_background(original,source) or canvas_background(target,target_design)
                or adapted is None):
            continue
        target_area=visual_area(target)
        adapted_area=visual_area(adapted)
        area_ratio=adapted_area/target_area
        measured_visuals.append((original["id"],target_area,adapted_area))
        if area_ratio < .30 or area_ratio > 2.2:
            defects.append(f"visual_scale_distortion:{original['id']}:{area_ratio:.2f}")
    for i,(id_a,target_a,adapted_a) in enumerate(measured_visuals):
        for id_b,target_b,adapted_b in measured_visuals[i+1:]:
            if target_a >= 1.5*target_b and adapted_a < adapted_b:
                defects.append(f"visual_hierarchy_inversion:{id_a}:{id_b}")
            elif target_b >= 1.5*target_a and adapted_b < adapted_a:
                defects.append(f"visual_hierarchy_inversion:{id_b}:{id_a}")

    mean_displacement=sum(center_deltas)/max(1,len(center_deltas))
    max_text_displacement=max(text_deltas,default=0.)

    # Detect text that escaped a visible lower-z backing layer in the source.
    for text in (e for e in source["elements"] if e["type"]=="text"):
        text_box=_text_visible_bounds(text)
        text_area=max(1e-9,(text_box[2]-text_box[0])*(text_box[3]-text_box[1]))
        parents=[]
        for back in source["elements"]:
            if back["type"] not in ("image","shape") or back.get("z_index",0)>=text.get("z_index",0):
                continue
            share=(back["width"]*back["height"])/(source["canvas"]["width"]*source["canvas"]["height"])
            back_box=(visible_content_bounds(back,_asset_root(back)) if back["type"]=="image"
                      else (back["x"],back["y"],back["x"]+back["width"],back["y"]+back["height"]))
            containment=_overlap_box(back_box,text_box)/text_area
            threshold=.85 if share<.30 else .80 if share<=.85 else 1.1
            if containment>=threshold:
                parents.append(back)
        if not parents:
            continue
        adapted_text=out_by_id.get(text["id"])
        if adapted_text is None:continue
        adapted_box=_text_visible_bounds(adapted_text)
        adapted_area=max(1e-9,(adapted_box[2]-adapted_box[0])*(adapted_box[3]-adapted_box[1]))
        for back in parents:
            adapted_back=out_by_id.get(back["id"])
            if adapted_back is None:continue
            adapted_back_box=(visible_content_bounds(adapted_back,_asset_root(adapted_back))
                              if adapted_back["type"]=="image" else
                              (adapted_back["x"],adapted_back["y"],
                               adapted_back["x"]+adapted_back["width"],
                               adapted_back["y"]+adapted_back["height"]))
            if _overlap_box(adapted_back_box,adapted_box)/adapted_area<.80:
                defects.append(f"source_containment_lost:{back['id']}->{text['id']}")
                break
    # A newly created overlap can be a concrete layout defect; ordinary
    # source-evidenced collage/backplate overlaps are not negative evidence.
    source_by_id={e["id"]:e for e in source["elements"]}
    output_items=list(output["elements"])
    def visible_box(element):
        if element["type"]=="text":return _text_visible_bounds(element)
        if element["type"]=="image":return visible_content_bounds(element,_asset_root(element))
        return (element["x"],element["y"],element["x"]+element["width"],element["y"]+element["height"])
    for index,a in enumerate(output_items):
        for b in output_items[index+1:]:
            sa,sb=source_by_id.get(a["id"]),source_by_id.get(b["id"])
            if sa is None or sb is None:continue
            ab,bb=visible_box(a),visible_box(b)
            aa=max(1.,(ab[2]-ab[0])*(ab[3]-ab[1]));ba=max(1.,(bb[2]-bb[0])*(bb[3]-bb[1]))
            fraction=_overlap(ab,bb)/min(aa,ba)
            if fraction<.22:continue
            sab,sbb=visible_box(sa),visible_box(sb)
            saa=max(1.,(sab[2]-sab[0])*(sab[3]-sab[1]));sba=max(1.,(sbb[2]-sbb[0])*(sbb[3]-sbb[1]))
            source_fraction=_overlap(sab,sbb)/min(saa,sba)
            if source_fraction<.08 and not _intentional_stack_pair(sa,sb,sab,sbb):
                defects.append(f"new_unlayered_overlap:{a['id']}:{b['id']}:{fraction:.2f}")
    return sorted(set(defects)), {
        "mean_center_displacement": mean_displacement,
        "max_text_center_displacement": max_text_displacement,
        "min_text_font_ratio": min(shrink_ratios,default=None),
    }


def _overlap_box(element_box, box):
    return max(0.,min(element_box[2],box[2])-max(element_box[0],box[0])) * max(
        0.,min(element_box[3],box[3])-max(element_box[1],box[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=PAIR_FILE)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum verified train pairs (0 means all eligible pairs)")
    parser.add_argument("--both-directions", action="store_true",
                        help="Also resize B to A's canvas; doubles renderer work")
    parser.add_argument("--min-target-ssim", type=float, default=0.90,
                        help="Reject a target label if converted JSON differs from its supplied preview below this SSIM")
    parser.add_argument("--min-text-ssim", type=float, default=0.65,
                        help="Minimum mean local SSIM over converted target text regions")
    parser.add_argument("--min-text-p10-ssim", type=float, default=0.35,
                        help="Minimum 10th-percentile text-region SSIM; rejects a badly mismatched text block")
    parser.add_argument("--inspect-all", action="store_true",
                        help="Render every exact train pair for human review without target-fidelity filtering")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not all(0.0<=value<=1.0 for value in
               (args.min_target_ssim,args.min_text_ssim,args.min_text_p10_ssim)):
        parser.error("SSIM thresholds must be in [0, 1]")
    rows = {}
    for line in (SPLIT_DIR / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "exported":
            rows[row["template_id"]] = row
    pair_doc = json.loads(args.pairs.read_text())
    eligible = [p for p in pair_doc["pairs"]
                if p.get("classification") == "exact"
                and p["a"]["split"] == p["b"]["split"] == "train"
                and not p.get("cross_split")]
    if args.limit:
        eligible = eligible[:args.limit]
    if args.inspect_all and args.output == OUT_DIR:
        args.output = INSPECTION_DIR
    args.output.mkdir(parents=True, exist_ok=True)
    render_dir = args.output / ("negative_pairs" if args.inspect_all else "engine_renders")
    render_dir.mkdir(parents=True, exist_ok=True)
    target_render_dir=args.output/"target_renders"
    if not args.inspect_all:
        target_render_dir.mkdir(parents=True,exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    rendered = failures = rejected_fidelity = rejected_no_defect = accepted_negatives = 0
    with manifest_path.open("w") as manifest:
        for pair_index, pair in enumerate(eligible):
            directions = [(pair["a"], pair["b"])]
            if args.both_directions:
                directions.append((pair["b"], pair["a"]))
            for source_meta, target_meta in directions:
                source_row, target_row = rows.get(source_meta["id"]), rows.get(target_meta["id"])
                if source_row is None or target_row is None:
                    failures += 1
                    continue
                source_path = SPLIT_DIR / source_row["design"]
                source = json.loads(source_path.read_text())
                target_path = SPLIT_DIR / target_row["design"]
                target_design = json.loads(target_path.read_text())
                target_preview = SPLIT_DIR / target_row["preview"]
                target_metrics = None
                target_warnings = []
                target_canvas = {"width": target_meta["width"], "height": target_meta["height"]}
                stem = re.sub(r"[^A-Za-z0-9_-]", "_", f"{source_meta['id']}__{target_meta['id']}")
                target_render = target_render_dir / f"{stem}.png"
                if not args.inspect_all:
                    # Both local renders share a renderer domain. Keep the
                    # upstream preview as a second positive view only after
                    # checking converted-target fidelity against it.
                    native_target=target_render.with_name(target_render.stem+"-native.png")
                    target_warnings = render_design_to_png(
                        target_design, SPLIT_DIR, native_target,
                        ROOT / "data" / "crello_fonts.pickle")
                    with Image.open(native_target) as converted, Image.open(target_preview) as preview:
                        sx,sy=preview.width/target_design["canvas"]["width"],preview.height/target_design["canvas"]["height"]
                        text_regions=[]
                        for element in target_design["elements"]:
                            if element["type"]=="text":
                                box=_text_visible_bounds(element)
                                text_regions.append((int(box[0]*sx),int(box[1]*sy),
                                                     int(box[2]*sx),int(box[3]*sy)))
                        target_metrics = compare_images(converted.copy(), preview.copy(),text_regions)
                        preview_size=preview.size
                    _composite_at_preview_size(native_target,target_render,preview_size)
                    native_target.unlink(missing_ok=True)
                if (not args.inspect_all and
                        (target_metrics["global_ssim"] < args.min_target_ssim or target_warnings or
                         (target_metrics.get("text_region_ssim_mean") is not None and
                          target_metrics["text_region_ssim_mean"]<args.min_text_ssim) or
                         (target_metrics.get("text_region_ssim_p10") is not None and
                          target_metrics["text_region_ssim_p10"]<args.min_text_p10_ssim))):
                    rejected_fidelity += 1
                    manifest.write(json.dumps({
                        "pair_id": f"{source_meta['id']}__to__{target_meta['id']}",
                        "status": "rejected_target_render_fidelity",
                        "target_global_ssim": target_metrics["global_ssim"],
                        "target_text_region_ssim_mean":target_metrics.get("text_region_ssim_mean"),
                        "target_text_region_ssim_p10":target_metrics.get("text_region_ssim_p10"),
                        "target_render_warnings": target_warnings,
                        "minimum_ssim": args.min_target_ssim,
                        "target_preview": str(target_preview.resolve()),
                    }, ensure_ascii=False) + "\n")
                    continue
                output = resize(source, target_canvas)
                negative = render_dir / f"{stem}.png"
                try:
                    render_design_to_png(output, SPLIT_DIR, negative,
                                         ROOT / "data" / "crello_fonts.pickle")
                    # Keep the inspection candidate pixel dimensions identical
                    # to its paired original preview, regardless of the renderer
                    # output's native canvas raster size.
                    with Image.open(negative) as candidate, Image.open(target_preview) as preview:
                        target_size = preview.size
                        if candidate.size != target_size:
                            candidate.resize(target_size, Image.Resampling.LANCZOS).save(negative)
                except Exception as exc:
                    failures += 1
                    print(f"render failed {stem}: {exc}", file=sys.stderr, flush=True)
                    continue
                defects=[]
                quality_metrics={}
                if not args.inspect_all:
                    target_background=target_render.with_name(target_render.stem+"-background.png")
                    candidate_background=negative.with_name(negative.stem+"-background.png")
                    target_without_text=deepcopy(target_design)
                    target_without_text["elements"]=[e for e in target_without_text["elements"] if e["type"]!="text"]
                    output_without_text=deepcopy(output)
                    output_without_text["elements"]=[e for e in output_without_text["elements"] if e["type"]!="text"]
                    render_design_to_png(target_without_text,SPLIT_DIR,target_background,
                                         ROOT/"data"/"crello_fonts.pickle")
                    render_design_to_png(output_without_text,SPLIT_DIR,candidate_background,
                                         ROOT/"data"/"crello_fonts.pickle")
                    defects,quality_metrics=_candidate_defects(
                        source,target_design,output,pair,source_meta,target_meta,
                        target_render,negative,target_background,candidate_background)
                    target_background.unlink(missing_ok=True)
                    candidate_background.unlink(missing_ok=True)
                    if defects:
                        accepted_negatives+=1
                    else:
                        rejected_no_defect+=1
                status=("review_candidate" if args.inspect_all else
                        "accepted_negative" if defects else "candidate_without_measured_defect")
                record = {
                    "pair_id": stem,
                    "status": status,
                    "human_review": "pending" if not args.inspect_all else "unreviewed",
                    "preference": "unreviewed" if args.inspect_all else
                                 "target_render_over_engine_candidate" if defects else None,
                    "positive_images": [] if args.inspect_all else [str(target_render.resolve())],
                    "preview_fidelity_evidence": str(target_preview.resolve()),
                    "target_local_render": None if args.inspect_all else str(target_render.resolve()),
                    "engine_image": str(negative.resolve()),
                    "source_id": source_meta["id"],
                    "target_id": target_meta["id"],
                    "target_canvas": target_canvas,
                    "preview_pixel_size": list(target_size),
                    "match_ratio": pair.get("match_ratio"),
                    "target_render_global_ssim": (target_metrics["global_ssim"]
                                                   if target_metrics is not None else None),
                    "target_render_text_region_ssim_mean": (target_metrics.get("text_region_ssim_mean")
                                                            if target_metrics is not None else None),
                    "target_render_grid_ssim_p10": (target_metrics.get("grid_ssim_p10")
                                                     if target_metrics is not None else None),
                    "target_render_warnings": target_warnings,
                    "layout_defects": defects,
                    "layout_metrics": quality_metrics,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                rendered += 1
            if (pair_index + 1) % 25 == 0:
                print(f"Processed {pair_index + 1}/{len(eligible)} pairs; rendered {rendered}, "
                      f"negative labels {accepted_negatives}, failures {failures}",
                      flush=True)
    summary = {
        "source_pairs": len(eligible), "rendered_candidates": rendered,
        "accepted_negative_labels": accepted_negatives,
        "candidates_without_measured_defect": rejected_no_defect,
        "failures": failures, "rejected_target_fidelity": rejected_fidelity,
        "min_target_ssim": None if args.inspect_all else args.min_target_ssim,
        "min_text_ssim": None if args.inspect_all else args.min_text_ssim,
        "min_text_p10_ssim": None if args.inspect_all else args.min_text_p10_ssim,
        "inspect_all_without_fidelity_filter": args.inspect_all,
        "both_directions": args.both_directions,
        "warning": ("Human-review renders only; target conversion fidelity was not filtered."
                    if args.inspect_all else
                    "Weak preference data; audit samples before training. Do not label every dataset preview as universally good."),
        "manifest": str(manifest_path.resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
