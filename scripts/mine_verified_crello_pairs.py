#!/usr/bin/env python3
"""Mine and structurally validate multi-size candidates in Crello 5.1.0."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from bisect import bisect_left, bisect_right
from itertools import islice
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from convert_crello_dataset import DATASET_ID, DATASET_REVISION, _decode_categorical, _element_image, _scalar, _text, _values


def norm(value):
    return " ".join(re.sub(r"[^\w]+", " ", (value or "").casefold()).split())


def image_vector(image):
    if image is None:
        return None
    rgba = image.convert("RGBA").resize((16, 16), Image.Resampling.LANCZOS)
    return np.asarray(rgba, dtype=np.float32).reshape(-1)


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float(np.array_equal(a, b))


def visual_role(e, design):
    area = e["width"] * e["height"] / max(1, design["width"] * design["height"])
    full_bleed = area >= .75
    vector = e.get("vector")
    flat = False
    if vector is not None:
        rgb = vector.reshape(16, 16, 4)[:, :, :3]
        flat = float(rgb.std(axis=(0, 1)).mean()) <= 12
    return "background" if full_bleed else "panel" if flat and area >= .03 else "object"


def boxes_overlap(x, y):
    return (max(x["x"], y["x"]) < min(x["x"] + x["width"], y["x"] + y["width"])
            and max(x["y"], y["y"]) < min(x["y"] + x["height"], y["y"] + y["height"]))


def candidate_pairs_for_group(indexes, rows):
    """Apply cheap structure/text/aspect filters before materializing pairs."""
    by_count = defaultdict(list)
    postings = defaultdict(lambda: defaultdict(set))
    textless = defaultdict(set)
    for i in indexes:
        row = rows[i]
        count = len(row["elements"])
        by_count[count].append(i)
        if row["text_set"]:
            for token in row["text_set"]:
                postings[count][token].add(i)
        else:
            textless[count].add(i)
    counts = sorted(by_count)
    result = set()
    for ai in indexes:
        a = rows[ai]
        n = len(a["elements"])
        lo_count, hi_count = bisect_left(counts, n - 2), bisect_right(counts, n + 2)
        possible_counts = counts[lo_count:hi_count]
        possible = set()
        for count in possible_counts:
            possible.update(textless[count])
            if not a["text_set"]:
                possible.update(by_count[count])
            else:
                for token in a["text_set"]:
                    possible.update(postings[count].get(token, ()))
        ar_a = a["width"] / max(1, a["height"])
        # Keep the candidate list small before applying the aspect threshold;
        # avoid constructing a near-full complement set for every row.
        possible = {bi for bi in possible
                    if (rows[bi]["width"] / max(1, rows[bi]["height"]) <= ar_a * .85
                        or rows[bi]["width"] / max(1, rows[bi]["height"]) >= ar_a / .85)}
        for bi in possible:
            if ai >= bi:
                continue
            b = rows[bi]
            if a["text_set"] and b["text_set"] and not (set(a["text_set"]) & set(b["text_set"])):
                continue
            if (a["width"], a["height"]) == (b["width"], b["height"]):
                continue
            result.add((ai, bi))
    return result


def compact_row(row, features, split, index):
    types = _values(_decode_categorical(features, "type", row.get("type")))
    texts = _values(row.get("text"))
    fonts = _values(_decode_categorical(features, "font", row.get("font")))
    widths, heights = _values(row.get("width")), _values(row.get("height"))
    positions = _values(row.get("left")), _values(row.get("top"))
    width, height = int(_scalar(row.get("canvas_width"), 0)), int(_scalar(row.get("canvas_height"), 0))
    length = int(_scalar(row.get("length"), len(types)))
    elements = []
    for i in range(length):
        kind = str(_scalar(types[i], ""))
        e = {"index": i, "type": kind,
             "x": float(_scalar(positions[0][i], 0)), "y": float(_scalar(positions[1][i], 0)),
             "width": float(_scalar(widths[i], 0)), "height": float(_scalar(heights[i], 0))}
        if "text" in kind.casefold():
            content = str(_scalar(texts[i], ""))
            font = str(_scalar(fonts[i], "")) if i < len(fonts) else ""
            e.update(text=content, text_norm=norm(content), font=norm(font),
                     vector=None)
        else:
            e.update(vector=None)
        elements.append(e)
    text_set = sorted({e["text_norm"] for e in elements if e.get("text_norm")})
    title = norm(_text(row.get("title")))
    return {"id": _text(row.get("id")), "split": split, "row_index": index,
            "title": _text(row.get("title")), "title_norm": title,
            "width": width, "height": height,
            "elements": elements, "text_set": text_set}


def matched_text(a, b):
    left = defaultdict(list)
    right = defaultdict(list)
    for e in a:
        if "text_norm" in e:
            left[(e["text_norm"], e["font"])].append(e)
    for e in b:
        if "text_norm" in e:
            right[(e["text_norm"], e["font"])].append(e)
    result = []
    for key in left.keys() & right.keys():
        for x, y in zip(left[key], right[key]):
            result.append((x, y, 1.0))
    return result


def validate(a, b):
    ea, eb = a["elements"], b["elements"]
    texts = matched_text(ea, eb)
    used_a = {x["index"] for x,_,_ in texts}
    used_b = {y["index"] for _,y,_ in texts}
    remaining_a = [e for e in ea if e["index"] not in used_a and e.get("vector") is not None]
    remaining_b = [e for e in eb if e["index"] not in used_b and e.get("vector") is not None]
    candidates = []
    for x in remaining_a:
        for y in remaining_b:
            sim = cosine(x["vector"], y["vector"])
            if sim >= .95:
                arx = x["width"] / max(1e-9, x["height"])
                ary = y["width"] / max(1e-9, y["height"])
                ar_similarity = min(arx/ary, ary/arx)
                role_x, role_y = visual_role(x, a), visual_role(y, b)
                relative_area_x = x["width"] * x["height"] / max(1, a["width"] * a["height"])
                relative_area_y = y["width"] * y["height"] / max(1, b["width"] * b["height"])
                role_match = role_x == role_y
                aspect_ok = ar_similarity >= .95 or (role_match and role_x in {"background", "panel"}
                                                     and min(relative_area_x/ max(1e-9,relative_area_y),
                                                             relative_area_y/max(1e-9,relative_area_x)) >= .5)
                if aspect_ok and role_match:
                    candidates.append((sim * ar_similarity, sim, ar_similarity, x, y))
    candidates.sort(key=lambda c: c[0], reverse=True)
    visuals = []
    for _, sim, ar_sim, x, y in candidates:
        if x["index"] in used_a or y["index"] in used_b:
            continue
        used_a.add(x["index"]); used_b.add(y["index"])
        visuals.append((x, y, sim, ar_sim))
    matches = len(texts) + len(visuals)
    text_count_a = sum("text_norm" in e for e in ea)
    text_count_b = sum("text_norm" in e for e in eb)
    text_complete = len(texts) == text_count_a == text_count_b
    total = max(len(ea), len(eb), 1)
    ratio = matches / total
    unmatched_a = [e for e in ea if e["index"] not in used_a]
    unmatched_b = [e for e in eb if e["index"] not in used_b]
    unmatched = unmatched_a + unmatched_b
    primary_a = [e for e in ea if "text_norm" not in e and e["width"]*e["height"] >= a["width"]*a["height"]*.03]
    primary_b = [e for e in eb if "text_norm" not in e and e["width"]*e["height"] >= b["width"]*b["height"]*.03]
    matched_primary = sum(x["index"] in used_a for x in primary_a) == len(primary_a) and sum(y["index"] in used_b for y in primary_b) == len(primary_b)
    order_ok = True
    ordered = [(x, y) for x,y,_ in texts] + [(x,y) for x,y,_,_ in visuals]
    for (x1,y1),(x2,y2) in combinations(ordered, 2):
        overlaps = (boxes_overlap(x1, x2) or boxes_overlap(y1, y2))
        if overlaps and (x1["index"]-x2["index"])*(y1["index"]-y2["index"]) < 0:
            order_ok = False
            break
    if len(ea) == len(eb) and matches == len(ea) and order_ok:
        category = "exact"
    elif (text_complete and matched_primary and ratio >= .80 and len(unmatched) <= 2
          and all(e.get("text_norm") is None and e["width"]*e["height"] <
                  a["width"]*a["height"]*.01 for e in unmatched_a)
          and all(e.get("text_norm") is None and e["width"]*e["height"] <
                  b["width"]*b["height"]*.01 for e in unmatched_b)):
        category = "core"
    else:
        category = "rejected"
    return {"classification": category, "element_count_a": len(ea), "element_count_b": len(eb),
            "matched_count": matches, "match_ratio": round(ratio, 4), "text_matches": len(texts),
            "text_count_a": text_count_a, "text_count_b": text_count_b,
            "visual_matches": len(visuals), "unmatched_count": len(unmatched),
            "order_preserved": order_ok, "primary_visuals_matched": matched_primary,
            "matches": [{"a_index": x["index"], "b_index": y["index"], "kind": "text", "similarity": 1.0}
                        for x,y,_ in texts] +
                       [{"a_index": x["index"], "b_index": y["index"], "kind": "visual",
                         "rgba_cosine": round(sim, 4), "aspect_similarity": round(ar, 4)}
                        for x,y,sim,ar in visuals]}


def main():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install optional dependency 'datasets' to scan Crello") from exc
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-split", type=int, default=0,
                        help="Rows per split for a bounded smoke scan; 0 scans full splits")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "crello_verified_pairs.json")
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("--limit-per-split must be >= 0")
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, streaming=True)
    metadata_columns=("id","title","canvas_width","canvas_height","length","type","text","font","width","height","left","top")
    all_rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        rows = ds[split].select_columns(list(metadata_columns))
        selected_rows = rows if args.limit_per_split == 0 else islice(rows, args.limit_per_split)
        for i, row in enumerate(selected_rows):
            all_rows.append(compact_row(row, rows.features, split, i))
            if (i+1) % 5000 == 0:
                print(f"Read {split}: {i+1} rows; total {len(all_rows)}", file=sys.stderr, flush=True)
    groups = defaultdict(set)
    for i, row in enumerate(all_rows):
        if row["title_norm"] and row["title_norm"] != "untitled":
            groups[("title", row["title_norm"])].add(i)
        if row["text_set"]:
            groups[("text_set", tuple(row["text_set"]))].add(i)
    candidate_pairs = set()
    for group_index, (key, indexes) in enumerate(groups.items(), 1):
        if len(indexes) < 2:
            continue
        candidate_pairs.update(candidate_pairs_for_group(indexes, all_rows))
        if group_index % 1000 == 0:
            print(f"Candidate groups: {group_index}; pairs so far: {len(candidate_pairs)}", file=sys.stderr, flush=True)
    # Download/decode layer raster columns only for rows participating in a
    # candidate. The metadata-only pass is column-pruned by Parquet.
    candidate_row_indexes={i for pair in candidate_pairs for i in pair}
    by_split=defaultdict(list)
    for i in candidate_row_indexes:
        by_split[all_rows[i]["split"]].append(i)
    ids_by_split={split:{all_rows[i]["id"]:i for i in indexes} for split,indexes in by_split.items()}
    for split, wanted in ids_by_split.items():
        rows=ds[split].select_columns(["id","image"])
        selected_rows = rows if args.limit_per_split == 0 else islice(rows, args.limit_per_split)
        for row in selected_rows:
            row_id=_text(row.get("id"))
            if row_id not in wanted:
                continue
            index=wanted[row_id]
            image_values=_values(row.get("image"))
            for element in all_rows[index]["elements"]:
                if "text_norm" not in element:
                    i=element["index"]
                    image=_element_image(image_values[i]) if i<len(image_values) else None
                    element["vector"]=image_vector(image)
    counts = defaultdict(int)
    records = []
    for ai, bi in sorted(candidate_pairs):
        a, b = all_rows[ai], all_rows[bi]
        result = validate(a, b)
        counts[result["classification"]] += 1
        records.append({"a": {k:a[k] for k in ("id","split","row_index","title","width","height")},
                        "b": {k:b[k] for k in ("id","split","row_index","title","width","height")},
                        "cross_split": a["split"] != b["split"], **result})
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset":DATASET_ID,"revision":DATASET_REVISION,
                                  "rows_scanned":len(all_rows),"candidate_pairs":len(candidate_pairs),
                                  "counts":dict(counts),"pairs":records}, ensure_ascii=False, indent=2)+"\n")
    print(f"Rows scanned: {len(all_rows)}; candidate pairs: {len(candidate_pairs)}")
    print(f"Exact 1-to-1: {counts['exact']}; core-matched: {counts['core']}; rejected: {counts['rejected']}")
    print(f"Verified index: {output}")
    for r in records:
        if r["classification"] != "rejected":
            print(json.dumps({k:r[k] for k in ("classification","a","b","cross_split","match_ratio","text_matches","visual_matches","unmatched_count")}))


if __name__ == "__main__":
    main()
