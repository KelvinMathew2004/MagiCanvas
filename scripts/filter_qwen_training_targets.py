#!/usr/bin/env python3
"""Deduplicate Qwen SFT rows by target template without touching eval splits."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_INPUT=ROOT/"data"/"qwen_lora_expanded_v2"/"train_pre_target_id_cap.jsonl"
DEFAULT_OUTPUT=ROOT/"data"/"qwen_lora_expanded_v2"/"train.jsonl"


def _target_id(row):
    target=row.get("target_template_id")
    if target:return str(target)
    parts=str(row.get("pair_id","")).split("__to__",1)
    if len(parts)!=2 or not parts[1]:raise ValueError(f"Cannot parse target ID from {row.get('pair_id')!r}")
    row["target_template_id"]=parts[1]
    return parts[1]


def _aspect_bucket(ratio):
    if ratio<=.85:return "portrait"
    if ratio<=1.15:return "square"
    if ratio<=1.8:return "landscape"
    return "wide"


def _transition(row):
    transition=row.get("_aspect_transition")
    if transition:return transition
    text=row.get("messages",[{},{"content":""}])[1].get("content","")
    match=re.search(r"Source aspect ratio:\s*([0-9.]+)",text)
    if not match:return "unknown"
    source=float(match.group(1))
    canvas=row.get("target_canvas") or {}
    target=float(canvas.get("width",1))/max(1.,float(canvas.get("height",1)))
    return f"{_aspect_bucket(source)}->{_aspect_bucket(target)}"


def select(records,cap):
    if cap<1:raise ValueError("target cap must be positive")
    for row in records:_target_id(row)
    remaining=sorted(records,key=lambda row:row["pair_id"])
    selected=[];targets=Counter();families=Counter();sources=Counter();transitions=Counter()
    while remaining:
        eligible=[row for row in remaining if targets[row["target_template_id"]]<cap]
        if not eligible:break
        chosen=min(eligible,key=lambda row:(
            transitions[_transition(row)],
            families[row.get("family_id","")],
            sources[row.get("source_template_id","")],
            row["pair_id"],
        ))
        selected.append(chosen);remaining.remove(chosen)
        targets[chosen["target_template_id"]]+=1
        families[chosen.get("family_id","")]+=1
        sources[chosen.get("source_template_id","")]+=1
        transitions[_transition(chosen)]+=1
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-target",type=int,default=1)
    parser.add_argument("--report",type=Path,default=None)
    args=parser.parse_args()
    if args.max_per_target<1:parser.error("--max-per-target must be positive")
    records=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records:raise SystemExit(f"No training rows in {args.input}")
    chosen=select(records,args.max_per_target)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+".tmp")
    with temporary.open("w") as stream:
        for row in chosen:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    temporary.replace(args.output)
    report_path=args.report or args.output.with_name("target_cap_report.json")
    report={"input":str(args.input.resolve()),"output":str(args.output.resolve()),
            "input_pairs":len(records),"output_pairs":len(chosen),
            "removed_pairs":len(records)-len(chosen),
            "target_cap":args.max_per_target,
            "unique_target_ids":len({row["target_template_id"] for row in chosen}),
            "max_rows_for_any_target":max(Counter(row["target_template_id"] for row in chosen).values()),
            "families":len({row.get("family_id") for row in chosen}),
            "aspect_transition_counts":dict(sorted(Counter(_transition(row)
                                                           for row in chosen).items()))}
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
