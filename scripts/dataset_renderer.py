"""Dataset-only rendering through the starter's unchanged cr-renderer.

This adapter is for inspecting Crello samples locally. Submission code must
continue using ``renderer.converter.convert`` and the supplied brand fonts.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=4)
def _font_manager(fonts_file: str, mtime_ns: int):
    """Load the large Crello font pickle once for a batch of renders."""
    from cr_renderer.fonts import FontManager, normalize_family

    manager = FontManager(fonts_file)
    # The resource's keys differ from the dataset's labels in these two cases.
    for alias, stored in (("Brusher", "Brusher Free Font"), ("Exo 2", "Exo\\ 2")):
        if stored in manager._fonts:
            manager._fonts[normalize_family(alias)] = manager._fonts[stored]
    return manager


def render_design_to_png(
    design: dict[str, Any],
    assets_dir: str | Path,
    png_path: str | Path,
    fonts_file: str | Path,
    max_size: int | None = None,
    adjust_tight_text_widths: bool = True,
) -> list[str]:
    """Render with Crello's companion font bundle without changing starter files.

    ``adjust_tight_text_widths=False`` skips the dataset-only in-memory width
    tolerance and exposes the starter converter's unadjusted wrapping behavior.
    """
    from cr_renderer import CrelloV5Renderer
    from renderer.converter.convert import (
        _BOLD_WEIGHTS,
        _is_italic,
        _normalize_weight,
        _weight_style_to_font_style,
        convert_chameleon_to_crello,
    )
    from renderer.converter.features_stub import make_stub_features
    from renderer.converter.wrap import char_widths

    fonts_file = Path(fonts_file).resolve()
    font_manager = _font_manager(str(fonts_file), fonts_file.stat().st_mtime_ns)

    adapted = deepcopy(design)
    # CHAMELEON style_runs use end-exclusive offsets; the starter converter's
    # internal map is inclusive, so adjust only this private rendering copy.
    metric_warnings: list[str] = []
    for element in adapted.get("elements", []):
        for run in element.get("style_runs") or []:
            run["end"] = int(run.get("end", 0)) - 1
        if (not adjust_tight_text_widths or element.get("type") != "text"
                or element.get("style_runs")):
            continue

        # Crello's preview preserves its authored text_line map. The JSON
        # exporter represents those line breaks in `content`, but cr-renderer
        # also greedily wraps against local font metrics. For a tight source
        # line this can add a false newline and create an obvious collision.
        # Widen only the renderer input copy by a small metric tolerance; keep
        # the alignment anchor fixed and never mutate the saved design.
        content = str(element.get("content", ""))
        old_width = float(element.get("width", 1.0))
        family = element.get("font_family", "Montserrat")
        size = float(element.get("font_size", 24))
        weight = _normalize_weight(element.get("font_weight"))
        italic = _is_italic(element.get("font_style") or element.get("italic"))
        style = _weight_style_to_font_style(weight in _BOLD_WEIGHTS, italic)
        max_line_width = max((
            sum(char_widths(font_manager, family, weight, style, size, line))
            for line in content.split("\n")
        ), default=0.0)
        if old_width < max_line_width <= old_width * 1.10:
            new_width = max_line_width + 1.0
            align = str(element.get("text_align", "left")).lower()
            old_x=float(element.get("x",0.0))
            new_x=old_x-(new_width-old_width)/2 if align=="center" else (
                old_x-(new_width-old_width) if align=="right" else old_x)
            canvas=adapted.get("canvas",{})
            canvas_w=float(canvas.get("width",max(old_x+old_width,new_width)))
            can_widen=new_x>=0 and new_x+new_width<=canvas_w
            if can_widen:
                for other in adapted.get("elements",[]):
                    if other is element or other.get("type")!="text":continue
                    ox,oy=float(other.get("x",0)),float(other.get("y",0))
                    ow,oh=float(other.get("width",0)),float(other.get("height",0))
                    ey,eh=float(element.get("y",0)),float(element.get("height",0))
                    if (max(0.,min(new_x+new_width,ox+ow)-max(new_x,ox))>1 and
                            max(0.,min(ey+eh,oy+oh)-max(ey,oy))>1):
                        can_widen=False;break
            if can_widen:
                element["x"],element["width"]=new_x,new_width
            else:
                factor=max(.5,min(1.,old_width/max_line_width*.985))
                element["font_size"]=size*factor
                element["letter_spacing"]=float(element.get("letter_spacing",0))*factor
        elif max_line_width > old_width:
            # The preview uses Crello's authored line map while the local
            # renderer remeasures with its installed font. First give a tight
            # frame up to 10% more room when that does not create a text/text
            # collision. If the canvas or a neighbouring text box blocks that,
            # scale only this diagnostic render copy's type metrics enough to
            # fit the authored line; the exported JSON remains untouched.
            needed=max_line_width+1.0
            align = str(element.get("text_align", "left")).lower()
            old_x=float(element.get("x",0.0))
            proposed_x=old_x-(needed-old_width)/2 if align=="center" else (
                old_x-(needed-old_width) if align=="right" else old_x)
            canvas=adapted.get("canvas",{})
            canvas_w=float(canvas.get("width",max(old_x+old_width,needed)))
            can_widen=needed<=old_width*1.10 and proposed_x>=0 and proposed_x+needed<=canvas_w
            if can_widen:
                for other in adapted.get("elements",[]):
                    if other is element or other.get("type")!="text":continue
                    ox,oy=float(other.get("x",0)),float(other.get("y",0))
                    ow,oh=float(other.get("width",0)),float(other.get("height",0))
                    ey,eh=float(element.get("y",0)),float(element.get("height",0))
                    overlap_x=max(0.,min(proposed_x+needed,ox+ow)-max(proposed_x,ox))
                    overlap_y=max(0.,min(ey+eh,oy+oh)-max(ey,oy))
                    if overlap_x>1 and overlap_y>1:
                        can_widen=False;break
            if can_widen:
                element["x"],element["width"]=proposed_x,needed
            else:
                factor=max(.5,min(1.,old_width/max_line_width*.985))
                element["font_size"]=size*factor
                element["letter_spacing"]=float(element.get("letter_spacing",0))*factor

    renderer = CrelloV5Renderer.__new__(CrelloV5Renderer)
    renderer.features = make_stub_features()
    renderer.font_manager = font_manager
    example, warnings = convert_chameleon_to_crello(
        adapted, font_manager, str(assets_dir)
    )
    rendered = renderer.render(example, max_size=max_size or 2048, format="png")
    Path(png_path).write_bytes(rendered)
    return warnings + metric_warnings
