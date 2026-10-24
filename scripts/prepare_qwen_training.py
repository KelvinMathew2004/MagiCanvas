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
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUSIONS = ROOT / "data" / "qwen_lora" / "excluded_pairs.json"
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


def aspect_bucket(canvas):
    """Coarse layout orientation class used only to diversify training picks."""
    ratio = float(canvas["width"]) / max(1.0, float(canvas["height"]))
    if ratio <= 0.85:
        return "portrait"
    if ratio <= 1.15:
        return "square"
    if ratio <= 1.80:
        return "landscape"
    return "wide"


def select_directions(records, cap, stratified=False):
    """Select a deterministic set of outgoing directions from one source."""
    ordered = sorted(records, key=lambda r: (
        r["target_canvas"]["width"] / r["target_canvas"]["height"], r["pair_id"]))
    if not cap or len(ordered) <= cap:
        return ordered
    if not stratified:
        picks = sorted({round(i * (len(ordered) - 1) / max(1, cap - 1))
                        for i in range(cap)})
        return [ordered[i] for i in picks]

    # Greedy coverage first gives each source as many distinct orientation
    # transitions as the cap allows; repeated transitions are then spread
    # across target aspect ratios.
    selected = []
    transition_counts = Counter()
    selected_ratios = []
    remaining = list(ordered)
    while remaining and len(selected) < cap:
        def score(record):
            transition = record["_aspect_transition"]
            ratio = record["_target_aspect_ratio"]
            novelty = min((abs(ratio - old) for old in selected_ratios), default=0.0)
            return (transition_counts[transition], -novelty, record["pair_id"])
        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        transition_counts[chosen["_aspect_transition"]] += 1
        selected_ratios.append(chosen["_target_aspect_ratio"])
    return selected


def select_family_directions(records, cap, stratified=False):
    """Cap one family while covering transitions and source templates evenly."""
    if not cap or len(records) <= cap:
        return list(records)
    if not stratified:
        return sorted(records, key=lambda r: r["pair_id"])[:cap]
    selected = []
    transition_counts = Counter()
    source_counts = Counter()
    remaining = list(records)
    while remaining and len(selected) < cap:
        def score(record):
            return (transition_counts[record["_aspect_transition"]],
                    source_counts[record["source_template_id"]],
                    record["pair_id"])
        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        transition_counts[chosen["_aspect_transition"]] += 1
        source_counts[chosen["source_template_id"]] += 1
    return selected


def choose_split_records(records_by_family, split, source_cap,
                         family_cap=0, stratified=False):
    selected = []
    # Keep insertion order for legacy exports so the existing capped-v2
    # validation/test JSONL remains byte-for-byte reproducible.
    for family, split_records in records_by_family.items():
        records = split_records.get(split, [])
        if not records:
            continue
        by_source = defaultdict(list)
        for record in records:
            by_source[record["source_template_id"]].append(record)
        family_records = []
        for source_id, directions in by_source.items():
            family_records.extend(select_directions(directions,
                                                   source_cap,
                                                   stratified=stratified))
        family_records = select_family_directions(family_records, family_cap,
                                                  stratified=stratified)
        selected.extend(family_records)
    return selected


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0}
    return {"count": len(values), "min": values[0],
            "median": values[(len(values) - 1) // 2],
            "p90": values[int((len(values) - 1) * 0.90)], "max": values[-1]}


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
        "_aspect_transition": f"{aspect_bucket(source['canvas'])}->{aspect_bucket(target['canvas'])}",
        "_target_aspect_ratio": float(target["canvas"]["width"])/max(1.0,float(target["canvas"]["height"])),
        "prompt_version":prompt_version,
        "source_split": source_info["split"], "target_split": target_info["split"],
        "target_canvas": target["canvas"],
    }


def pair_filter_reason(pair,a_design,b_design):
    for match in pair["matches"]:
        a=a_design["elements"][match["a_index"]]
        b=b_design["elements"][match["b_index"]]
        # Current Qwen output formats predict box/font geometry only, and the
        # inference path preserves source rotation. Do not teach an impossible
        # target when the paired human design changes an element's angle.
        angle_delta=(float(b.get("rotation",0.))-float(a.get("rotation",0.))+180.)%360.-180.
        if abs(angle_delta)>1.:
            return "element_rotation_changed"
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
    parser.add_argument("--train-family-cap",type=int,default=0,
                        help="Optional maximum training directions per connected template family; 0 keeps all")
    parser.add_argument("--stratify-aspect-transitions",action="store_true",
                        help="Prioritize distinct portrait/square/landscape/wide source-to-target transitions under caps")
    parser.add_argument("--eval-source-cap",type=int,default=DEFAULT_EVAL_SOURCE_CAP,
                        help="Maximum validation/test directions per source template; 0 keeps all")
    parser.add_argument("--prompt-version",type=int,choices=(2,3),default=2,
                        help="v2 short-side font encoding; experimental v3 geometric mean")
    parser.add_argument("--exclusions",type=Path,default=DEFAULT_EXCLUSIONS,
                        help="JSON ledger of directed pair IDs excluded from supervised/evaluation exports")
    args = parser.parse_args()
    if args.output is None:
        args.output=ROOT/"data"/("qwen_lora_source_capped_v2" if args.prompt_version==2
                                  else "qwen_lora_v3_source_capped")
    if args.eval_source_cap<0:parser.error("--eval-source-cap must be nonnegative")
    if args.train_source_cap < 0:
        parser.error("--train-source-cap must be zero (unlimited) or positive")
    if args.train_family_cap < 0:
        parser.error("--train-family-cap must be zero (unlimited) or positive")
    pair_doc = json.loads(args.pairs.read_text())
    excluded_pairs = set()
    excluded_templates = set()
    if args.exclusions.is_file():
        exclusion_doc = json.loads(args.exclusions.read_text())
        excluded_pairs = {row["pair_id"] for row in exclusion_doc.get("excluded_pairs", [])}
        excluded_templates = {
            template_id
            for row in exclusion_doc.get("excluded_pairs", [])
            for template_id in row.get("template_ids", [])
        }
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
        directed_pair_ids = {f'{a["id"]}__to__{b["id"]}', f'{b["id"]}__to__{a["id"]}'}
        if (directed_pair_ids & excluded_pairs
                or a["id"] in excluded_templates or b["id"] in excluded_templates):
            skipped["curated_pair_exclusion"] += 1
            continue
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

    # Preserve legacy evenly-spaced cap behavior unless stratification is
    # explicitly requested. Eval files always use the same deterministic
    # direction cap so experiments can share identical held-out rows.
    exact_by_split["train"] = choose_split_records(
        records_by_family, "train", args.train_source_cap,
        family_cap=args.train_family_cap,
        stratified=args.stratify_aspect_transitions)
    for split in ("validation", "test"):
        exact_by_split[split] = choose_split_records(
            records_by_family, split, args.eval_source_cap, stratified=False)
    uncapped_by_split = {
        split: sum(len(split_records.get(split, []))
                   for split_records in records_by_family.values())
        for split in ("train", "validation", "test")}
    capped_train = uncapped_by_split["train"] - len(exact_by_split["train"])
    capped_eval = sum(uncapped_by_split[split] - len(exact_by_split[split])
                      for split in ("validation", "test"))

    # Include actual counts for comparable expansion policies and the uncapped
    # ceiling, all over the exact same family assignment and eligible examples.
    scenario_counts = {
        "current_source_cap_2_legacy": len(choose_split_records(
            records_by_family, "train", 2, stratified=False)),
        "source_cap_4_stratified": len(choose_split_records(
            records_by_family, "train", 4, stratified=True)),
        "source_cap_4_family_cap_16_stratified": len(choose_split_records(
            records_by_family, "train", 4, family_cap=16, stratified=True)),
        "source_cap_4_family_cap_20_stratified": len(choose_split_records(
            records_by_family, "train", 4, family_cap=20, stratified=True)),
        "source_cap_4_family_cap_32_stratified": len(choose_split_records(
            records_by_family, "train", 4, family_cap=32, stratified=True)),
        "source_cap_4_family_cap_40_stratified": len(choose_split_records(
            records_by_family, "train", 4, family_cap=40, stratified=True)),
        "uncapped_exact_directions": uncapped_by_split["train"],
    }
    after_source_cap = choose_split_records(records_by_family, "train", 4,
                                            stratified=True)
    by_capped_family = Counter(record["family_id"] for record in after_source_cap)
    family_candidate_counts = [len(rows.get("train", []))
                               for rows in records_by_family.values() if rows.get("train")]
    # Count raw directions by source template across the isolated train split.
    raw_by_source = defaultdict(int)
    for rows in records_by_family.values():
        for record in rows.get("train", []):
            raw_by_source[record["source_template_id"]] += 1
    source_candidate_counts = list(raw_by_source.values())
    transition_counts = Counter(record["_aspect_transition"]
                                for rows in records_by_family.values()
                                for record in rows.get("train", []))

    for split_records in exact_by_split.values():
        for record in split_records:
            record.pop("_aspect_transition", None)
            record.pop("_target_aspect_ratio", None)

    write_jsonl(args.output / "train.jsonl", exact_by_split["train"])
    write_jsonl(args.output / "validation.jsonl", exact_by_split["validation"])
    write_jsonl(args.output / "test.jsonl", exact_by_split["test"])
    write_jsonl(args.output / "core_review.jsonl", core_review)
    summary = {"dataset": pair_doc.get("dataset"), "revision": pair_doc.get("revision"),
               "training_prompt_version": args.prompt_version,
               "coordinate_encoding": "normalized 0..1000 canvas fractions; font scale relative to "+("short side" if args.prompt_version==2 else "sqrt(width*height)"),
               "source_pair_manifest": str(args.pairs),
               "curated_exclusions_file": str(args.exclusions),
               "curated_excluded_pair_count": len(excluded_pairs),
               "curated_excluded_template_count": len(excluded_templates),
               "exact_examples": {key: len(value) for key, value in exact_by_split.items()},
               "core_review_examples_not_for_training": len(core_review),
               "skipped": dict(skipped),
               "cross_split_pairs_used_with_family_isolation": cross_split_pairs_included,
               "cross_split_families_reassigned": len(reassigned_cross_split_families),
               "family_split_policy":"stable hash of exact+core connected component (80/10/10)",
               "family_graph_edges":"exact and core matches; core remains review-only",
               "train_source_direction_cap":args.train_source_cap or None,
               "train_family_direction_cap":args.train_family_cap or None,
               "aspect_transition_stratification":args.stratify_aspect_transitions,
               "train_directions_removed_by_cap": capped_train,
               "eval_source_direction_cap":args.eval_source_cap or None,
               "families_per_split":{split:len({r["family_id"] for r in rows}) for split,rows in exact_by_split.items()},
               "validation_test_directions_removed_by_cap": capped_eval,
               "uncapped_eligible_exact_directions": uncapped_by_split,
               "train_family_direction_distribution": distribution(family_candidate_counts),
               "train_family_distribution_after_source_cap_4": distribution(by_capped_family.values()),
               "train_source_direction_distribution": distribution(source_candidate_counts),
               "train_aspect_transition_distribution": dict(sorted(transition_counts.items())),
               "train_cap_scenario_counts": scenario_counts}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
