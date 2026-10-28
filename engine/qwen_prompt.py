"""Shared prompt contracts for Qwen layout prediction.

Version 1 is the original pixel-coordinate format. Version 2 uses normalized
canvas coordinates and compact aliases so the model learns layout ratios
instead of memorizing UUIDs and resolutions.
"""
from __future__ import annotations

import json
from math import sqrt
SYSTEM = (
    "You predict geometry for resizing an existing graphic design. Return "
    "only the requested JSON. Preserve every element, its content, type, and "
    "source-evidenced relationships."
)


def font_reference(canvas, version=2):
    width,height=float(canvas['width']),float(canvas['height'])
    return sqrt(width*height) if version==3 else min(width,height)


def record_prompt_version(row):
    if 'prompt_version' in row:return int(row['prompt_version'])
    answer=json.loads(row['messages'][-1]['content'])
    return 2 if answer.get('elements') and 'box' in answer['elements'][0] else 1


def _v1_prompt(source, target_canvas):
    items = []
    for element in source["elements"]:
        row = {
            "id": element["id"], "type": element["type"],
            "x": round(float(element["x"]), 1),
            "y": round(float(element["y"]), 1),
            "width": round(float(element["width"]), 1),
            "height": round(float(element["height"]), 1),
            "z": element.get("z_index", 0),
            "rotation": round(float(element.get("rotation", 0)), 1),
        }
        if element.get("group_id") is not None:
            row["group"] = element["group_id"]
        if element["type"] == "text":
            row.update(text=str(element.get("content", ""))[:100],
                       font_size=element.get("font_size"),
                       font=element.get("font_family", ""),
                       align=element.get("text_align", "left"))
        items.append(row)
    return (
        f"Source canvas: {source['canvas']['width']}x{source['canvas']['height']}\n"
        f"Target canvas: {target_canvas['width']}x{target_canvas['height']}\n"
        "Source elements (pixel geometry):\n"
        + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        + "\nCoordinates are target-canvas pixels. Keep every element within the target canvas. "
        "Preserve aspect ratios for non-background images; full-canvas backgrounds may fill the target canvas. "
        "Preserve element IDs. Do not rewrite text or change types. "
        "Text font_size may change to fit its box. Return every input ID exactly once. "
        "Return JSON only: {\"elements\":[{\"id\":string,\"x\":number,"
        "\"y\":number,\"width\":number,\"height\":number,"
        "\"font_size\":number for text elements}]}. Include all IDs."
    )


def _overlap(a, b):
    iw = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"])
             - max(a["x"], b["x"]))
    ih = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"])
             - max(a["y"], b["y"]))
    return iw * ih


def _compact_relations(source, aliases, relations, count_tokens, budget):
    """Keep the trained pair-tuple vocabulary; prune redundant dense links.

    Local ownership precedes a minimum spanning forest of existing alignment
    edges. Full-canvas paint does not consume the relationship budget. Collect
    across every element before pruning so late IDs have equal consideration.
    """
    by_alias={aliases[e['id']]:e for e in source['elements']}
    area=source['canvas']['width']*source['canvas']['height']
    active={k:e for k,e in by_alias.items() if e['width']>0 and e['height']>0
            and e['width']*e['height']<.90*area
            and (e['type']!='text' or str(e.get('content','')).strip())}
    rows=[r for r in relations if r[1] in active and r[2] in active]
    owners={}
    for r in rows:
        if r[0]=='backing':
            p=active[r[1]];cost=(p['width']*p['height'],r[1])
            if r[2] not in owners or cost<owners[r[2]][0]:owners[r[2]]=(cost,r)
    selected=[v[1] for _,v in sorted(owners.items())]
    for kind in ('same-row','same-column'):
        parent={k:k for k in active}
        def root(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        def distance(r):
            a,b=active[r[1]],active[r[2]]
            return ((a['x']+a['width']/2-b['x']-b['width']/2)/source['canvas']['width'])**2+((a['y']+a['height']/2-b['y']-b['height']/2)/source['canvas']['height'])**2
        for r in sorted((r for r in rows if r[0]==kind),key=lambda r:(distance(r),r[1:])):
            a,b=root(r[1]),root(r[2])
            if a!=b:parent[b]=a;selected.append(r)
    backed={frozenset(r[1:3]) for r in selected if r[0]=='backing'}
    selected.extend(sorted((r for r in rows if r[0]=='overlap' and frozenset(r[1:3]) not in backed),
                           key=lambda r:(min(active[r[1]]['width']*active[r[1]]['height'],active[r[2]]['width']*active[r[2]]['height']),r[1:])))
    result=[]
    for r in selected:
        if count_tokens(json.dumps(result+[r],ensure_ascii=False,separators=(',',':')))<=budget:result.append(r)
    return result


def _v2_prompt(source, target_canvas, version=2, relation_token_counter=None, relation_token_budget=None):
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    tw, th = float(target_canvas["width"]), float(target_canvas["height"])
    aliases = {e["id"]: f"e{i:03d}" for i, e in enumerate(source["elements"])}
    items = []
    for element in source["elements"]:
        row = {
            "id": aliases[element["id"]], "type": element["type"],
            "box": [round(float(element[k]) / d * 1000, 1)
                    for k, d in (("x", sw), ("y", sh), ("width", sw), ("height", sh))],
            "z": int(element.get("z_index", 0)),
            "rotation": round(float(element.get("rotation", 0)), 1),
        }
        if element.get("group_id") is not None:
            row["group"] = str(element["group_id"])
        if element["type"] == "text":
            row.update(text=str(element.get("content", ""))[:120],
                       font=element.get("font_family", ""),
                       font_scale=round(float(element.get("font_size", 1)) / font_reference(source["canvas"],version) * 1000, 2),
                       align=element.get("text_align", "left"),
                       weight=element.get("font_weight", "regular"),
                       color=element.get("color", ""))
        else:
            # Crello asset paths are ordinal element filenames, not semantic
            # labels. Keep useful shape/fit clues without pretending the name
            # tells the model what the pictured object is.
            row["aspect"] = round(float(element["width"]) / max(1e-6, float(element["height"])), 3)
            row["fit"] = element.get("fit", "contain")
        items.append(row)

    # Explicitly state source relationships that are otherwise hard to infer
    # from a long flat element list. Alignment groups are based on centers;
    # overlap links retain source-supported layered artwork and backplates.
    relations = []
    elements = source["elements"]
    for i, a in enumerate(elements):
        ax, ay = float(a["x"]), float(a["y"])
        aw, ah = float(a["width"]), float(a["height"])
        for b in elements[i + 1:]:
            bx, by = float(b["x"]), float(b["y"])
            bw, bh = float(b["width"]), float(b["height"])
            shared = _overlap(a, b) / max(1.0, min(aw * ah, bw * bh))
            if shared >= .18:
                relations.append(["overlap", aliases[a["id"]], aliases[b["id"]],
                                  "a-below-b" if a.get("z_index", 0) < b.get("z_index", 0) else "b-below-a"])
            acx, bcx = ax + aw / 2, bx + bw / 2
            acy, bcy = ay + ah / 2, by + bh / 2
            if abs(acy - bcy) <= .025 * sh and abs(acx - bcx) >= .04 * sw:
                relations.append(["same-row", aliases[a["id"]], aliases[b["id"]]])
            if abs(acx - bcx) <= .025 * sw and abs(acy - bcy) >= .04 * sh:
                relations.append(["same-column", aliases[a["id"]], aliases[b["id"]]])
            # A smaller foreground element contained in a larger lower layer
            # is a likely panel/backplate relationship.
            contained = (ax <= bx and ay <= by and ax + aw >= bx + bw and ay + ah >= by + bh)
            reverse_contained = (bx <= ax and by <= ay and bx + bw >= ax + aw and by + bh >= ay + ah)
            if contained and a.get("z_index", 0) < b.get("z_index", 0):
                relations.append(["backing", aliases[a["id"]], aliases[b["id"]]])
            elif reverse_contained and b.get("z_index", 0) < a.get("z_index", 0):
                relations.append(["backing", aliases[b["id"]], aliases[a["id"]]])
            if relation_token_budget is None and len(relations) >= 120:
                break
        if relation_token_budget is None and len(relations) >= 120:
            break

    if relation_token_budget is not None:
        relations=_compact_relations(source,aliases,relations,relation_token_counter,relation_token_budget)

    return (
        f"Layout prediction format v{version}. Every box is normalized independently to its canvas: "
        "[x,y,width,height] where 0..1000 spans the full canvas width or height. "
        + ("font_scale is font size divided by " + ("the canvas short side" if version==2 else "sqrt(canvas width * canvas height)") + ", then multiplied by 1000.\n")
        + f"Source aspect ratio: {sw / sh:.5f}; target aspect ratio: {tw / th:.5f}; "
        f"target/source {'short-side' if version==2 else 'font-reference'} ratio: {font_reference(target_canvas,version) / font_reference(source['canvas'],version):.5f}.\n"
        "Source elements:\n" + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        + "\nSource relationships [kind, element IDs, optional layer order]:\n"
        + json.dumps(relations, ensure_ascii=False, separators=(",", ":"))
        + "\nReflow the whole composition for the target aspect ratio. Use the source relationships to keep "
        "paired, aligned, overlapping, or backed elements together. Preserve the text hierarchy, content, "
        "and relative prominence. Avoid simply scaling all source positions from the top-left. "
        "Keep all boxes inside [0,1000]. Preserve non-background image aspect ratios. "
        "Return each compact element ID exactly once and no other IDs. Text outputs also need font_scale. "
        "Return only JSON: {\"elements\":[{\"id\":\"e000\",\"box\":[x,y,width,height],"
        "\"font_scale\":number}]}. Omit font_scale for non-text elements."
    )


def encode_v2_completion(completion, source, target_canvas, version=2):
    payload = json.loads(completion) if isinstance(completion, str) else completion
    elements = source["elements"]
    aliases = {e["id"]: f"e{i:03d}" for i, e in enumerate(elements)}
    cw, ch = float(target_canvas["width"]), float(target_canvas["height"])
    short = font_reference(target_canvas,version)
    encoded = []
    for row in payload["elements"]:
        eid = row["id"]
        box = [round(float(row[k]) / d * 1000, 2)
               for k, d in (("x", cw), ("y", ch), ("width", cw), ("height", ch))]
        answer = {"id": aliases[eid], "box": box}
        if "font_size" in row:
            answer["font_scale"] = round(float(row["font_size"]) / short * 1000, 3)
        encoded.append(answer)
    return json.dumps({"elements": encoded}, ensure_ascii=False, separators=(",", ":"))


def decode_v2_completion(payload, source, target_canvas, version=2):
    """Decode compact normalized supervision into ordinary engine geometry."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    elements = source["elements"]
    cw, ch = float(target_canvas["width"]), float(target_canvas["height"])
    short = font_reference(target_canvas,version)
    result = []
    for row in payload["elements"]:
        try:
            alias=row["id"]
            if not isinstance(alias,str) or not alias.startswith("e") or not alias[1:].isdigit():continue
            index = int(alias[1:])
            if index<0 or index>=len(elements):continue
            source_element = elements[index]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        x, y, width, height = (float(value) for value in row["box"])
        decoded = {"id": source_element["id"],
                   "x": x * cw / 1000, "y": y * ch / 1000,
                   "width": width * cw / 1000, "height": height * ch / 1000}
        if "font_scale" in row:
            decoded["font_size"] = float(row["font_scale"]) * short / 1000
        result.append(decoded)
    return {"elements": result}


def messages_for(source, target_canvas, completion=None, version=1, *, relation_token_counter=None, relation_token_budget=None):
    if version not in (1,2,3):raise ValueError(f"Unsupported Qwen prompt version: {version}")
    prompt = _v2_prompt(source, target_canvas,version,relation_token_counter,relation_token_budget) if version in (2,3) else _v1_prompt(source, target_canvas)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if completion is not None:
        if version in (2,3):
            completion = encode_v2_completion(completion, source, target_canvas,version)
        messages.append({"role": "assistant", "content": completion})
    return messages
