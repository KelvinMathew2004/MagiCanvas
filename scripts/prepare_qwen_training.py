#!/usr/bin/env python3
"""Build Qwen chat-format SFT examples from verified Crello resize pairs.

Exact pairs are assigned to train/validation/test by connected template family.
Core pairs are exported to a separate review-only JSONL and never enter the
training/validation/test files. Optional direction caps apply per source
template, while connected components remain the split-isolation unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.qwen_prompt import messages_for
PAIR_FILE = ROOT / "data" / "crello_verified_pairs.json"
SPLIT_DIRS = {
    "train": ROOT / "data" / "crello_train",
    "validation": ROOT / "data" / "crello_validation",
    "test": ROOT / "data" / "crello_test",
}
DEFAULT_TRAIN_SOURCE_CAP = 2
DEFAULT_EVAL_SOURCE_CAP = 2


def manifest_index(root):
    by_id = {}
    for line in (root / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "exported":
            by_id[row["template_id"]] = row
    return by_id


def _is_background(element, elements, canvas):
    sw,sh=canvas["width"],canvas["height"]
    full=(element["x"]<=sw*.02 and element["y"]<=sh*.02
          and element["x"]+element["width"]>=sw*.98
          and element["y"]+element["height"]>=sh*.98)
    behind=all(element.get("z_index",0)<=other.get("z_index",0)
               for other in elements if other["type"]=="text")
    return full and behind and not element.get("group_id") and element["type"]!="text"


def _project_geometry(source_element, target_element, source, target):
    """Project a human target onto the engine's bounds/content invariants."""
    cw,ch=target["canvas"]["width"],target["canvas"]["height"]
    if _is_background(source_element,source["elements"],source["canvas"]):
        return {"x":0.,"y":0.,"width":float(cw),"height":float(ch)}
    width=min(float(cw),max(1.,float(target_element["width"])))
    height=min(float(ch),max(1.,float(target_element["height"])))
    if source_element["type"]=="image" and source_element.get("fit")!="cover":
        aspect=float(source_element["width"])/max(1e-6,float(source_element["height"]))
        scale=min(width/max(1e-6,float(target_element["width"])),
                  height/max(1e-6,float(target_element["height"])))
        width=min(float(cw),float(target_element["width"])*scale)
        height=width/aspect
        if height>ch:
            height=float(ch);width=height*aspect
    x=min(cw-width,max(0.,float(target_element["x"])))
    y=min(ch-height,max(0.,float(target_element["y"])))
    result={"x":x,"y":y,"width":width,"height":height}
    if source_element["type"]=="text":
        desired=max(1.,float(target_element.get("font_size") or source_element.get("font_size") or 1))
        # This is supervised geometry from a verified human-designed target.
        # Do not pass its font size through engine fit heuristics: local font
        # metrics and safety padding can shrink exact target labels even when
        # the source design intentionally uses a tightly fitted text box.
        result["font_size"]=desired
    return result


def build_example(pair, a_design, b_design, reverse=False, prompt_version=2):
    source, target = (b_design, a_design) if reverse else (a_design, b_design)
    source_info, target_info = (pair["b"], pair["a"]) if reverse else (pair["a"], pair["b"])
    mapping = {}
    for match in pair["matches"]:
        si = match["b_index"] if reverse else match["a_index"]
        ti = match["a_index"] if reverse else match["b_index"]
        mapping[si] = ti
    if len(mapping) != len(source["elements"]) or len(set(mapping.values())) != len(target["elements"]):
        return None
    target_by_index = {i: e for i, e in enumerate(target["elements"])}
    answers = []
    for source_index, source_element in enumerate(source["elements"]):
        target_element = target_by_index.get(mapping.get(source_index))
        if target_element is None:
            return None
        if (source_element["type"]=="text" and
                source_element.get("content","")!=target_element.get("content","")):
            return None
        geom=_project_geometry(source_element,target_element,source,target)
        answers.append({"id":source_element["id"],**{k:round(v,2) for k,v in geom.items()}})
    completion = json.dumps({"elements": answers}, ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": messages_for(source,target["canvas"],completion,version=prompt_version),
        "pair_id": f"{source_info['id']}__to__{target_info['id']}",
        "source_template_id":source_info["id"],
        "prompt_version":prompt_version,
        "source_split": source_info["split"], "target_split": target_info["split"],
        "target_canvas": target["canvas"],
    }


def pair_filter_reason(pair,a_design,b_design):
    for match in pair["matches"]:
        a=a_design["elements"][match["a_index"]]
        b=b_design["elements"][match["b_index"]]
        if a["type"]=="text" and a.get("content","")!=b.get("content",""):
            return "text_content_changed"
        if a["type"]=="image" and b["type"]=="image":
            is_bg_a=_is_background(a,a_design["elements"],a_design["canvas"])
            is_bg_b=_is_background(b,b_design["elements"],b_design["canvas"])
            if is_bg_a!=is_bg_b:
                return "background_role_changed"
        else:
            is_bg_a=is_bg_b=False
        if a["type"]=="image" and b["type"]=="image" and a.get("fit","contain")=="contain" and not is_bg_a:
            aspect_a=float(a["width"])/max(1e-6,float(a["height"]))
            aspect_b=float(b["width"])/max(1e-6,float(b["height"]))
            if abs(aspect_a-aspect_b)/max(1e-6,aspect_a)>.05:
                return "image_aspect_changed"
    return None


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=PAIR_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-source-cap",type=int,default=DEFAULT_TRAIN_SOURCE_CAP,
                        help="Maximum directed training examples per source template; 0 keeps all eligible directions")
    parser.add_argument("--eval-source-cap",type=int,default=DEFAULT_EVAL_SOURCE_CAP,
                        help="Maximum validation/test directions per source template; 0 keeps all")
    parser.add_argument("--prompt-version",type=int,choices=(2,3),default=2,
                        help="v2 short-side font encoding; experimental v3 geometric mean")
    args = parser.parse_args()
    if args.output is None:
        args.output=ROOT/"data"/("qwen_lora_source_capped_v2" if args.prompt_version==2
                                  else "qwen_lora_v3_source_capped")
    if args.eval_source_cap<0:parser.error("--eval-source-cap must be nonnegative")
    if args.train_source_cap < 0:
        parser.error("--train-source-cap must be zero (unlimited) or positive")
    pair_doc = json.loads(args.pairs.read_text())
    indexes = {split: manifest_index(root) for split, root in SPLIT_DIRS.items()}
    designs = {}
    def load_design(split, template_id):
        key = (split, template_id)
        if key not in designs:
            row = indexes[split].get(template_id)
            if row is None:
                raise KeyError(f"No converted design for {split}/{template_id}")
            designs[key] = json.loads((SPLIT_DIRS[split] / row["design"]).read_text())
        return designs[key]

    # Build exact-pair connected components before assigning examples. A family
    # that crosses upstream splits is assigned to exactly one model split.
    # Core links are excluded as supervised examples, but still connect likely
    # template families for leakage-safe grouping.
    family_edges=[p for p in pair_doc["pairs"]
                  if p.get("classification") in ("exact","core")]
    parent={}
    def find(value):
        parent.setdefault(value,value)
        if parent[value]!=value:parent[value]=find(parent[value])
        return parent[value]
    def union(a,b):parent[find(a)]=find(b)
    for pair in family_edges:union(pair["a"]["id"],pair["b"]["id"])
    members_by_root=defaultdict(list)
    for member in parent:
        members_by_root[find(member)].append(member)
    canonical_by_member={member:min(members) for members in members_by_root.values() for member in members}
    family_splits=defaultdict(set)
    for pair in family_edges:
        family=canonical_by_member[pair["a"]["id"]]
        family_splits[family].update((pair["a"]["split"],pair["b"]["split"]))
    # Split connected template families as units, so no variant from a family
    # can leak into both training and evaluation. A stable hash avoids moving
    # most mixed-source families into test merely because one member came from
    # the upstream test partition.
    family_assignment={}
    for family in family_splits:
        bucket=int.from_bytes(hashlib.sha256(family.encode()).digest()[:4],"big")%100
        family_assignment[family]="train" if bucket<80 else "validation" if bucket<90 else "test"
    exact_by_split = {"train": [], "validation": [], "test": []}
    core_review = []
    skipped = defaultdict(int)
    records_by_family=defaultdict(lambda:defaultdict(list))
    reassigned_cross_split_families={key for key,splits in family_splits.items()
                                     if len(splits)>1 or family_assignment[key] not in splits}
    cross_split_pairs_included=0
    for pair in pair_doc["pairs"]:
        category = pair["classification"]
        a, b = pair["a"], pair["b"]
        family=canonical_by_member.get(a["id"],a["id"])
        if category not in ("exact", "core"):
            continue
        ad, bd = load_design(a["split"], a["id"]), load_design(b["split"], b["id"])
        if category == "core":
            ma = indexes[a["split"]][a["id"]]
            mb = indexes[b["split"]][b["id"]]
            core_review.append({
                "classification": "core_review_only",
                "a": {"id": a["id"], "split": a["split"], "preview": ma["preview"]},
                "b": {"id": b["id"], "split": b["split"], "preview": mb["preview"]},
                "match_ratio": pair.get("match_ratio"),
                "matched_count": pair.get("matched_count"),
                "element_count_a": pair.get("element_count_a"),
                "element_count_b": pair.get("element_count_b"),
                "unmatched_count": pair.get("unmatched_count"),
                "matches": pair.get("matches", []),
            })
            continue
        assigned_split=family_assignment[family]
        if pair.get("cross_split") or a["split"] != b["split"]:
            cross_split_pairs_included+=1
        filter_reason=pair_filter_reason(pair,ad,bd)
        if filter_reason:
            skipped[filter_reason]+=2
            continue
        for reverse in (False, True):
            record = build_example(pair, ad, bd, reverse,prompt_version=args.prompt_version)
            if record is None:
                skipped["unmappable"] += 1
                continue
            record["model_split"]=assigned_split
            record["family_id"]=family
            records_by_family[family][assigned_split].append(record)

    capped_train=0
    capped_eval=0
    for family,split_records in records_by_family.items():
        for split,records in split_records.items():
            cap=args.train_source_cap if split=="train" else args.eval_source_cap
            by_source=defaultdict(list)
            for record in records:by_source[record["source_template_id"]].append(record)
            retained=[]
            for source_id,directions in by_source.items():
                directions.sort(key=lambda r:(r["target_canvas"]["width"]/r["target_canvas"]["height"],r["pair_id"]))
                if cap and len(directions)>cap:
                    n=cap
                    picks=sorted({round(i*(len(directions)-1)/max(1,n-1)) for i in range(n)})
                    if split=="train":capped_train+=len(directions)-len(picks)
                    else:capped_eval+=len(directions)-len(picks)
                    directions=[directions[i] for i in picks]
                retained.extend(directions)
            exact_by_split[split].extend(retained)

    write_jsonl(args.output / "train.jsonl", exact_by_split["train"])
    write_jsonl(args.output / "validation.jsonl", exact_by_split["validation"])
    write_jsonl(args.output / "test.jsonl", exact_by_split["test"])
    write_jsonl(args.output / "core_review.jsonl", core_review)
    summary = {"dataset": pair_doc.get("dataset"), "revision": pair_doc.get("revision"),
               "training_prompt_version": args.prompt_version,
               "coordinate_encoding": "normalized 0..1000 canvas fractions; font scale relative to "+("short side" if args.prompt_version==2 else "sqrt(width*height)"),
               "source_pair_manifest": str(args.pairs),
               "exact_examples": {key: len(value) for key, value in exact_by_split.items()},
               "core_review_examples_not_for_training": len(core_review),
               "skipped": dict(skipped),
               "cross_split_pairs_used_with_family_isolation": cross_split_pairs_included,
               "cross_split_families_reassigned": len(reassigned_cross_split_families),
               "family_split_policy":"stable hash of exact+core connected component (80/10/10)",
               "family_graph_edges":"exact and core matches; core remains review-only",
               "train_source_direction_cap":args.train_source_cap or None,
               "train_directions_removed_by_cap": capped_train,
               "eval_source_direction_cap":args.eval_source_cap or None,
               "families_per_split":{split:len({r["family_id"] for r in rows}) for split,rows in exact_by_split.items()},
               "validation_test_directions_removed_by_cap": capped_eval}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
