#!/usr/bin/env python3
"""Rerender the first review candidates after an engine layout change."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SPLIT_DIR=ROOT/"data"/"crello_train"
os.environ.setdefault("MAGICANVAS_DATASET_RENDERING","1")
sys.path.insert(0,str(ROOT))
from engine.resize import resize
from scripts.dataset_renderer import render_design_to_png


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=ROOT/"data"/"siglip_negative_pairs")
    parser.add_argument("--limit",type=int,default=25)
    args=parser.parse_args()
    if args.limit<0:parser.error("--limit must be nonnegative")
    manifest_path=args.folder/"manifest.jsonl"
    records=[json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    indexes={}
    for line in (SPLIT_DIR/"manifest.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row.get("status")=="exported":indexes[row["template_id"]]=row
    selected=[row for row in records if row.get("engine_image")][:args.limit]
    updated=failures=0
    for row in selected:
        source_meta=indexes.get(row["source_id"])
        if source_meta is None:
            row["rerender_error"]="missing source design in train manifest"
            failures+=1
            continue
        try:
            source=json.loads((SPLIT_DIR/source_meta["design"]).read_text())
            output=resize(source,row["target_canvas"])
            png=Path(row["engine_image"])
            png.parent.mkdir(parents=True,exist_ok=True)
            warnings=render_design_to_png(output,SPLIT_DIR,png,
                                          ROOT/"data"/"crello_fonts.pickle")
            preview_path=(row.get("preview_fidelity_evidence") or row.get("positive_image")
                          or row.get("positive_images",[None])[0])
            if preview_path is None:
                raise ValueError("review row has no reference image")
            with Image.open(preview_path) as preview:
                size=preview.size
            with Image.open(png) as candidate:
                if candidate.size!=size:
                    candidate.resize(size,Image.Resampling.LANCZOS).save(png)
            row["preview_pixel_size"]=list(size)
            row["engine_revision"]="source_stack_containment_small_visual_scale_v3"
            row["engine_render_warnings"]=warnings
            row.pop("rerender_error",None)
            updated+=1
        except Exception as exc:
            row["rerender_error"]=f"{type(exc).__name__}: {exc}"
            failures+=1
        if (updated+failures)%5==0:
            print(f"Rerendered {updated}/{len(selected)}; failures {failures}",flush=True)

    with manifest_path.open("w") as manifest:
        for row in records:
            manifest.write(json.dumps(row,ensure_ascii=False)+"\n")
    summary={"requested":args.limit,"selected":len(selected),"updated":updated,
             "failures":failures,"engine_revision":"source_stack_containment_small_visual_scale_v3",
             "manifest":str(manifest_path.resolve())}
    (args.folder/"rerender_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
