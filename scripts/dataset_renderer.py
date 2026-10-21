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
) -> list[str]:
    """Render with Crello's companion font bundle without changing starter files."""
    from cr_renderer import CrelloV5Renderer
    from renderer.converter.convert import convert_chameleon_to_crello
    from renderer.converter.features_stub import make_stub_features

    fonts_file = Path(fonts_file).resolve()
    font_manager = _font_manager(str(fonts_file), fonts_file.stat().st_mtime_ns)

    adapted = deepcopy(design)
    # CHAMELEON style_runs use end-exclusive offsets; the starter converter's
    # internal map is inclusive, so adjust only this private rendering copy.
    for element in adapted.get("elements", []):
        for run in element.get("style_runs") or []:
            run["end"] = int(run.get("end", 0)) - 1

    renderer = CrelloV5Renderer.__new__(CrelloV5Renderer)
    renderer.features = make_stub_features()
    renderer.font_manager = font_manager
    example, warnings = convert_chameleon_to_crello(
        adapted, font_manager, str(assets_dir)
    )
    rendered = renderer.render(example, max_size=max_size or 2048, format="png")
    Path(png_path).write_bytes(rendered)
    return warnings
