"""Shared prompt and chat-message contract for Qwen data and inference."""
from __future__ import annotations

import json

SYSTEM = (
    "You predict pixel geometry for resizing an existing graphic design. "
    "Return only the requested JSON. Preserve every element ID and do not "
    "rewrite content or change element types."
)


def prompt_for(source, target_canvas):
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
        "Preserve source image aspect ratios and element IDs. Do not rewrite text or change types. "
        "Text font_size may change to fit its box. Return every input ID exactly once. "
        "Return JSON only: {\"elements\":[{\"id\":string,\"x\":number,"
        "\"y\":number,\"width\":number,\"height\":number,"
        "\"font_size\":number for text elements}]}. Include all IDs."
    )


def messages_for(source, target_canvas, completion=None):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(source, target_canvas)}]
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
    return messages
