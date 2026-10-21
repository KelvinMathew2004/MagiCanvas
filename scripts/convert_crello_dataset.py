#!/usr/bin/env python3
"""Export Crello rows as CHAMELEON v0.9 designs for this repo's renderer.

The project renderer accepts CHAMELEON JSON, then converts that document to
cr-renderer's internal Crello representation. This script therefore exports
one ``{canvas, elements}`` design per supported dataset row, plus the
pre-rendered element image assets those designs reference.

Requires the optional Hugging Face ``datasets`` package. It is deliberately
not a runtime dependency of the scored engine.
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
import re
import sys
from pathlib import Path
from typing import Any


DATASET_ID = "cyberagent/crello"
DATASET_REVISION = "5.1.0"


def _scalar(value: Any, default: Any = None) -> Any:
    """Unwrap common NumPy / Arrow scalar values without importing NumPy."""
    if value is None:
        return default
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (ValueError, TypeError):
            pass
    return default if value is None else value


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _decode_categorical(features: Any, key: str, value: Any) -> Any:
    """Decode ClassLabel values (including ClassLabel sequences) to strings."""
    feature = features.get(key) if features is not None else None
    if hasattr(feature, "feature"):
        feature = feature.feature
    int2str = getattr(feature, "int2str", None)
    if not callable(int2str) or value is None:
        return value

    def decode(item: Any) -> Any:
        item = _scalar(item)
        if isinstance(item, int):
            try:
                return int2str(item)
            except (ValueError, IndexError):
                return item
        return item

    values = _values(value)
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return [decode(item) for item in values]
    return decode(value)


def _text(value: Any, default: str = "") -> str:
    value = _scalar(value, default)
    return value if isinstance(value, str) else str(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug[:120] or "example"


def _bool_at(values: list[Any], index: int, default: bool = False) -> bool:
    if not values:
        return default
    return bool(_scalar(values[min(index, len(values) - 1)], default))


def _color(value: Any, fallback: str = "rgba(0,0,0,1)") -> str:
    """Convert common Crello color encodings to the renderer's CSS rgba form."""
    value = _scalar(value)
    if value is None:
        return fallback
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", s):
            raw = s[1:]
            r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
            a = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1
            return f"rgba({r},{g},{b},{a:.4g})"
        if s.lower().startswith(("rgb(", "rgba(")):
            return s
        try:
            value = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return fallback
    channels = _values(value)
    if len(channels) < 3:
        return fallback
    try:
        nums = [float(_scalar(c)) for c in channels[:4]]
    except (TypeError, ValueError):
        return fallback
    rgb = [int(round(max(0, min(255, n)))) for n in nums[:3]]
    alpha = nums[3] if len(nums) > 3 else 1.0
    if alpha > 1:
        alpha /= 255
    alpha = max(0.0, min(1.0, alpha))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.4g})"


def _style_runs(text: str, colors: list[Any], bold: list[Any], italic: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Encode character-level Crello styles as CHAMELEON's range overrides."""
    if not text:
        return "rgba(0,0,0,1)", []

    def style_at(i: int) -> tuple[str, bool, bool]:
        color = _color(colors[min(i, len(colors) - 1)]) if colors else "rgba(0,0,0,1)"
        return color, _bool_at(bold, i), _bool_at(italic, i)

    base = style_at(0)
    runs: list[dict[str, Any]] = []
    start = 0
    current = style_at(0)
    for i in range(1, len(text) + 1):
        nxt = style_at(i) if i < len(text) else None
        if nxt == current:
            continue
        if current != base:
            run: dict[str, Any] = {"start": start, "end": i}
            if current[0] != base[0]:
                run["color"] = current[0]
            if current[1] != base[1]:
                run["font_weight"] = "bold" if current[1] else "regular"
            if current[2] != base[2]:
                run["font_style"] = "italic" if current[2] else "normal"
            runs.append(run)
        start = i
        current = nxt  # type: ignore[assignment]

    # Include an override at character zero if the first defined style differs
    # from a neutral default. That first style is used as the element default.
    return base[0], runs


def _format_text(content: str, line_numbers: list[Any], colors: list[Any],
                 bold: list[Any], italic: list[Any], capitalize: bool):
    """Transfer Crello's per-character line and style maps into literal text.

    CHAMELEON represents line breaks in ``content`` and rich styling as
    end-exclusive style runs. Crello stores line breaks separately in
    ``text_line`` and capitalization as a renderer flag. Rebuild those as
    explicit newlines and displayed casing, while keeping style offsets
    aligned with the resulting string.
    """
    glyphs = [char for char in content if char != "\n"]
    line_values = [_scalar(value) for value in line_numbers]
    has_line_map = len(line_values) == len(glyphs) and bool(glyphs)

    rendered: list[str] = []
    styled_colors: list[Any] = []
    styled_bold: list[Any] = []
    styled_italic: list[Any] = []
    glyph_index = 0
    prior_line = None

    def append_newline(adjacent_index: int) -> None:
        """Add a line break styled like its nearest adjacent source glyph."""
        rendered.append("\n")
        if styled_colors:
            styled_colors.append(styled_colors[-1])
            styled_bold.append(styled_bold[-1])
            styled_italic.append(styled_italic[-1])
        elif adjacent_index < len(colors):
            styled_colors.append(colors[adjacent_index])
            styled_bold.append(bold[adjacent_index] if adjacent_index < len(bold) else False)
            styled_italic.append(italic[adjacent_index] if adjacent_index < len(italic) else False)
        else:
            styled_colors.append(None)
            styled_bold.append(False)
            styled_italic.append(False)

    for char in content:
        if char == "\n":
            # A valid Crello line map is authoritative; its transitions below
            # recreate breaks without relying on ignored source newline chars.
            if not has_line_map and (not rendered or rendered[-1] != "\n"):
                append_newline(glyph_index)
            continue

        line = line_values[glyph_index] if has_line_map else None
        if has_line_map and glyph_index and line != prior_line:
            append_newline(glyph_index - 1)

        output_chars = char.upper() if capitalize else char
        for output_char in output_chars:
            rendered.append(output_char)
            styled_colors.append(colors[glyph_index] if glyph_index < len(colors) else None)
            styled_bold.append(bold[glyph_index] if glyph_index < len(bold) else False)
            styled_italic.append(italic[glyph_index] if glyph_index < len(italic) else False)
        prior_line = line
        glyph_index += 1

    return "".join(rendered), styled_colors, styled_bold, styled_italic


def _element_image(value: Any):
    """Return a decoded PIL image when the dataset row contains one."""
    if value is None:
        return None
    if hasattr(value, "save") and hasattr(value, "size"):
        return value
    if isinstance(value, dict):
        image = value.get("image")
        if image is not None and hasattr(image, "save"):
            return image
        raw = value.get("bytes")
        if raw:
            from io import BytesIO
            from PIL import Image
            return Image.open(BytesIO(raw))
    return None


def convert_row(row: dict[str, Any], assets_root: Path, features: Any = None) -> dict[str, Any]:
    """Convert one decoded Hugging Face Crello row to a CHAMELEON design."""
    width = int(_scalar(row.get("canvas_width"), 0))
    height = int(_scalar(row.get("canvas_height"), 0))
    if width <= 0 or height <= 0:
        raise ValueError("invalid canvas dimensions")

    length = int(_scalar(row.get("length"), 0))
    if length <= 0:
        raise ValueError("design has no elements")

    template_id = _text(row.get("id"), "unknown")
    folder = _slug(template_id)
    fields = ("type", "left", "top", "width", "height", "angle", "opacity",
              "text", "font", "font_size", "text_align", "font_bold",
              "font_italic", "text_color", "text_line", "line_height",
              "letter_spacing", "capitalize", "image")
    arrays = {
        key: _values(_decode_categorical(features, key, row.get(key)))
        for key in fields
    }
    # Crello releases have appeared with normalized coordinates and with
    # pixel-valued coordinates. Detect the encoding from the actual row. If
    # all four geometry columns fit in [−1, 1], they are fractions; otherwise
    # these values are already canvas pixels.
    geometry = [
        float(_scalar(value, 0.0))
        for key in ("left", "top", "width", "height")
        for value in arrays[key]
    ]
    coordinate_scale = max(width, height) if geometry and max(abs(v) for v in geometry) <= 1.0 else 1.0
    elements: list[dict[str, Any]] = []
    asset_dir = assets_root / folder

    def at(key: str, index: int, default: Any = None) -> Any:
        values = arrays[key]
        return values[index] if index < len(values) else default

    for i in range(length):
        kind = _text(at("type", i, "")).casefold()
        x = float(_scalar(at("left", i, 0.0), 0.0)) * coordinate_scale
        y = float(_scalar(at("top", i, 0.0), 0.0)) * coordinate_scale
        w = float(_scalar(at("width", i, 0.0), 0.0)) * coordinate_scale
        h = float(_scalar(at("height", i, 0.0), 0.0)) * coordinate_scale
        if w <= 0 or h <= 0:
            raise ValueError(f"element {i} has non-positive dimensions")

        common: dict[str, Any] = {
            "id": f"{folder}_{i:03d}", "x": x, "y": y,
            "width": w, "height": h,
            "z_index": i,
            "rotation": float(_scalar(at("angle", i, 0.0), 0.0)),
            "opacity": max(0.0, min(1.0, float(_scalar(at("opacity", i, 1.0), 1.0)))),
        }

        if "text" in kind:
            content = _text(at("text", i, ""))
            colors = _values(at("text_color", i, []))
            bold = _values(at("font_bold", i, []))
            italic = _values(at("font_italic", i, []))
            line_numbers = _values(at("text_line", i, []))
            content, colors, bold, italic = _format_text(
                content, line_numbers, colors, bold, italic,
                _bool_at(arrays["capitalize"], i),
            )
            base_color, runs = _style_runs(content, colors, bold, italic)
            first_bold = _bool_at(bold, 0)
            first_italic = _bool_at(italic, 0)
            align = _text(at("text_align", i, "left"), "left").casefold()
            if align not in {"left", "center", "right"}:
                align = "left"
            element: dict[str, Any] = {
                **common, "type": "text", "content": content,
                "font_family": _text(at("font", i, ""), ""),
                "font_size": max(1.0, float(_scalar(at("font_size", i, 1.0), 1.0))),
                "font_weight": "bold" if first_bold else "regular",
                "font_style": "italic" if first_italic else "normal",
                "color": base_color, "text_align": align,
                "line_height": float(_scalar(at("line_height", i, 1.2), 1.2)),
                "letter_spacing": float(_scalar(at("letter_spacing", i, 0.0), 0.0)),
            }
            if runs:
                element["style_runs"] = runs
            elements.append(element)
        else:
            image = _element_image(at("image", i))
            if image is None:
                raise ValueError(f"non-text element {i} has no decoded raster preview")
            asset_dir.mkdir(parents=True, exist_ok=True)
            filename = f"element_{i:03d}.png"
            image_path = asset_dir / filename
            image.convert("RGBA").save(image_path, format="PNG")
            elements.append({
                **common, "type": "image",
                "asset_id": f"assets/{folder}/{filename}", "fit": "contain",
            })

    return {"canvas": {"width": width, "height": height}, "elements": elements}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for designs/, assets/, and manifest.jsonl")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum rows to inspect; use 0 for the full split")
    parser.add_argument("--revision", default=DATASET_REVISION,
                        help=f"Pinned Hugging Face dataset revision (default: {DATASET_REVISION})")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing optional dependency 'datasets'. Install it with: pip install datasets", file=sys.stderr)
        return 2

    out = args.output_dir.resolve()
    designs_dir = out / "designs"
    assets_dir = out / "assets"
    previews_dir = out / "previews"
    designs_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, revision=args.revision, streaming=True)
    if args.split not in dataset:
        parser.error(f"dataset revision {args.revision!r} has no split {args.split!r}")
    rows = dataset[args.split]
    selected_rows = rows if args.limit == 0 else islice(rows, args.limit)
    manifest_path = out / "manifest.jsonl"
    exported = skipped = inspected = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(selected_rows):
            inspected += 1
            source_id = _text(row.get("id"), f"{args.split}-{index}")
            stem = f"{index:06d}_{_slug(source_id)}"
            try:
                design = convert_row(row, assets_dir, rows.features)
                path = designs_dir / f"{stem}.json"
                path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                preview = _element_image(row.get("preview"))
                preview_path = None
                if preview is not None:
                    preview_path = previews_dir / f"{stem}.png"
                    preview.convert("RGBA").save(preview_path, format="PNG")
                manifest.write(json.dumps({
                    "status": "exported", "split": args.split, "row_index": index,
                    "template_id": source_id, "design": f"designs/{path.name}",
                    "preview": f"previews/{preview_path.name}" if preview_path else None,
                    "element_count": len(design["elements"]),
                    "canvas": design["canvas"],
                    "group": _text(_decode_categorical(rows.features, "group", row.get("group"))),
                    "format": _text(_decode_categorical(rows.features, "format", row.get("format"))),
                    "category": _text(_decode_categorical(rows.features, "category", row.get("category"))),
                    "title": _text(row.get("title")),
                    "keywords": _values(row.get("keywords")),
                    "industries": _values(_decode_categorical(rows.features, "industries", row.get("industries"))),
                    "suitability": _values(_decode_categorical(rows.features, "suitability", row.get("suitability"))),
                    "cluster_index": _scalar(row.get("cluster_index")),
                }, ensure_ascii=False) + "\n")
                exported += 1
            except Exception as exc:
                manifest.write(json.dumps({
                    "status": "skipped", "split": args.split, "row_index": index,
                    "template_id": source_id, "reason": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False) + "\n")
                skipped += 1
            if inspected % 25 == 0:
                manifest.flush()
                print(f"Processed {inspected} rows ({exported} exported, {skipped} skipped)...", file=sys.stderr)
    print(f"Exported {exported}/{inspected}; skipped {skipped}. Output: {out}")
    print("Render a design with renderer.converter.convert.render_design_to_png(), passing the output directory as assets_dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
