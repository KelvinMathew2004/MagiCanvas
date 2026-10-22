#!/usr/bin/env python3
"""Compare the local Qwen base model and LoRA on held-out Qwen examples."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

SPLIT_DIRS={
    "train":ROOT/"data"/"crello_train",
    "validation":ROOT/"data"/"crello_validation",
    "test":ROOT/"data"/"crello_test",
}


def read_sources():
    indexes={}
    for split,folder in SPLIT_DIRS.items():
        index={}
        for line in (folder/"manifest.jsonl").read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get("status")=="exported":index[row["template_id"]]=row
        indexes[split]=index
    cache={}

    def get(split,design_id):
        key=(split,design_id)
        if key not in cache:
            row=indexes[split][design_id]
            cache[key]=json.loads((SPLIT_DIRS[split]/row["design"]).read_text())
        return cache[key]
    return get


def measure(prediction,gold,canvas):
    if prediction is None:
        return {"valid":False,"geometry_mae":None,"font_mae":None,
                "out_of_canvas":None,"exact_ids":False}
    expected={row["id"]:row for row in gold.get("elements",[])}
    actual={row["id"]:row for row in prediction}
    exact_ids=set(expected)==set(actual)
    if not exact_ids or not expected:
        return {"valid":False,"geometry_mae":None,"font_mae":None,
                "out_of_canvas":None,"exact_ids":exact_ids}
    cw,ch=float(canvas["width"]),float(canvas["height"])
    geometry=[];fonts=[];out_of_canvas=0
    for element_id,target in expected.items():
        result=actual[element_id]
        geometry.extend((abs(result["x"]-target["x"])/cw,
                         abs(result["y"]-target["y"])/ch,
                         abs(result["width"]-target["width"])/cw,
                         abs(result["height"]-target["height"])/ch))
        if "font_size" in target:
            fonts.append(abs(float(result.get("font_size",0))-float(target["font_size"]))
                         /max(1.,float(target["font_size"])))
        if (result["x"]<0 or result["y"]<0 or result["width"]<=0 or result["height"]<=0
                or result["x"]+result["width"]>cw+1e-4
                or result["y"]+result["height"]>ch+1e-4):
            out_of_canvas+=1
    return {"valid":True,"geometry_mae":sum(geometry)/len(geometry),
            "font_mae":sum(fonts)/len(fonts) if fonts else None,
            "out_of_canvas":out_of_canvas,"exact_ids":True}


def run_mode(records,source_for,adapter_dir):
    import torch
    import engine.qwen_seed as qwen_seed
    from engine.qwen_prompt import decode_v2_completion
    qwen_seed._load.cache_clear()
    qwen_seed.ADAPTER_DIR=adapter_dir
    results=[]
    for index,row in enumerate(records,1):
        started=perf_counter()
        source_id=row["pair_id"].split("__to__",1)[0]
        source=source_for(row["source_split"],source_id)
        proposal,_=qwen_seed.generate_seed(source,row["target_canvas"],source["elements"])
        gold=json.loads(row["messages"][-1]["content"])
        if gold.get("elements") and "box" in gold["elements"][0]:
            gold=decode_v2_completion(gold,source,row["target_canvas"])
        results.append(measure(proposal,gold,row["target_canvas"]))
        print(f"  {index}/{len(records)}: {'valid' if results[-1]['valid'] else 'invalid'} "
              f"({perf_counter()-started:.1f}s)",flush=True)
    qwen_seed._load.cache_clear()
    gc.collect()
    if hasattr(torch,"mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


def summarize(results):
    valid=[row for row in results if row["valid"]]
    def average(key):
        values=[row[key] for row in valid if row[key] is not None]
        return sum(values)/len(values) if values else None
    return {
        "examples":len(results),"valid_json_and_complete_ids":len(valid),
        "valid_rate":len(valid)/max(1,len(results)),
        "exact_ids_rate":sum(row["exact_ids"] for row in results)/max(1,len(results)),
        "mean_normalized_geometry_mae":average("geometry_mae"),
        "mean_relative_font_size_mae":average("font_mae"),
        "mean_out_of_canvas_elements":average("out_of_canvas"),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"data"/"qwen_lora"/"test.jsonl")
    parser.add_argument("--output",type=Path,default=ROOT/"data"/"qwen_lora"/"evaluation.json")
    parser.add_argument("--limit",type=int,default=0,
                        help="Examples to evaluate (0 means the full held-out test set)")
    args=parser.parse_args()
    if args.limit<0:parser.error("--limit must be nonnegative")
    records=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:records=records[:args.limit]
    if not records:raise SystemExit("No held-out examples found.")
    prompt_version=2 if "box" in json.loads(records[0]["messages"][-1]["content"])["elements"][0] else 1
    source_for=read_sources()
    adapter_dir=ROOT/"models"/"qwen3-layout-lora-v2"
    if not adapter_dir.is_dir():
        adapter_dir=ROOT/"models"/"qwen3-layout-lora"
    if not adapter_dir.is_dir():raise SystemExit(f"No adapter directory at {adapter_dir}")
    marker=adapter_dir/"magicanvas_prompt_version.txt"
    adapter_prompt_version=int(marker.read_text().strip()) if marker.is_file() else 1
    if adapter_prompt_version!=prompt_version:
        raise SystemExit(f"Dataset uses prompt v{prompt_version}, but {adapter_dir} uses prompt v{adapter_prompt_version}. Train/install the matching adapter first.")
    import os
    os.environ["MAGICANVAS_QWEN_PROMPT_VERSION"]=str(prompt_version)

    print("Evaluating base Qwen on held-out examples",flush=True)
    base_results=run_mode(records,source_for,ROOT/"models"/"no-qwen-adapter")
    print("Evaluating Qwen with the downloaded LoRA",flush=True)
    adapter_results=run_mode(records,source_for,adapter_dir)
    base_summary=summarize(base_results)
    adapter_summary=summarize(adapter_results)
    base_mae=base_summary["mean_normalized_geometry_mae"]
    adapter_mae=adapter_summary["mean_normalized_geometry_mae"]
    report={"test_file":str(args.input.resolve()),"adapter_dir":str(adapter_dir.resolve()),
            "examples":len(records),"base":base_summary,"adapter":adapter_summary,
            "adapter_beats_base_geometry":(adapter_mae<base_mae
                                             if adapter_mae is not None and base_mae is not None
                                             else None)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print(f"Saved: {args.output}")


if __name__=="__main__":main()
