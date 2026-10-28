#!/usr/bin/env python3
"""Render source/output pairs from recorded reflow audits; never run inference.

Each input JSON contains source, target, design, diagnostics, defects and an
optional frozen_model_prediction marker. Intermediate model coordinates remain
in the copied audit JSON and are never displayed as extra panels.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import sys
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STYLE = '''body{font:17px/1.5 system-ui;background:#101820;color:#edf2f7;margin:0}header,main{max-width:1500px;margin:auto;padding:24px}a{color:#9ed4ff}article{padding:20px;margin:0 0 24px;background:#182532;border:1px solid #334552;border-radius:12px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}figure{margin:0}figcaption{display:flex;justify-content:space-between;gap:10px;margin-bottom:10px}.size,.muted{color:#b9c8d3}img{width:100%;height:520px;object-fit:contain;background:#101820}h1,h2{margin:8px 0 16px}p{overflow-wrap:anywhere}details{margin-top:16px}@media(max-width:800px){.panels{grid-template-columns:1fr}img{height:420px}}'''


def _preview_nonempty(render, doc, path):
    """Zero-area layers cannot paint pixels and fail cr-renderer validation.

    Omit only in this preview copy; retain exact source/output JSON and defects.
    """
    omitted=[e['id'] for e in doc['elements'] if e.get('width',0)<=0 or e.get('height',0)<=0]
    if not omitted:return render(doc,path)
    preview=deepcopy(doc)
    preview['elements']=[e for e in preview['elements'] if e.get('width',0)>0 and e.get('height',0)>0]
    warnings=render(preview,path)
    return [f'Preview omitted zero-area layers (audit JSON unchanged): {omitted}',*(warnings or [])]


def _renderer(source: dict, source_split: str | None = None):
    """Use the same local font family resource as each source's test cohort."""
    fonts = {e.get('font_family', '') for e in source.get('elements', []) if e.get('type') == 'text'}
    if fonts and not fonts <= {'Solstice Serif', 'Solstice Sans', ''} and (ROOT / 'data/crello_fonts.pickle').is_file():
        from scripts.dataset_renderer import render_design_to_png
        split=source_split or 'train'
        if split not in ('train','validation','test'):
            raise ValueError(f'Unknown dataset source split: {split}')
        return lambda doc, path: render_design_to_png(doc, ROOT / ('data/crello_'+split), path,
                  ROOT / 'data/crello_fonts.pickle', max_size=1400, adjust_tight_text_widths=False)
    from renderer.converter.convert import render_design_to_png
    return lambda doc, path: render_design_to_png(doc, ROOT / 'brand_kit/assets', path, max_size=1400)


def build(run: Path, output: Path, title: str = 'Current reflow review') -> list[dict]:
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for path in sorted(run.glob('*.json')):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or not {'source', 'design', 'target', 'diagnostics'} <= record.keys():
            continue
        source, design = record['source'], record['design']
        render = _renderer(source,record.get('source_split'))
        case = str(record.get('name', path.stem))
        prefix = path.stem
        panels = []
        warnings = {}
        for kind, label, doc in (('source', 'Original', source), ('output', 'Resized', design)):
            png = output / f'{prefix}_{kind}.png'
            warnings[kind] = _preview_nonempty(render, doc, png)
            jpg = png.with_suffix('.jpg')
            with Image.open(png) as im:
                im.convert('RGB').save(jpg, quality=93, optimize=True)
            png.unlink()
            canvas = doc['canvas']
            panels.append(f'<figure><figcaption>{label}<span class="size">{canvas["width"]} × {canvas["height"]}</span></figcaption>'
                          f'<a href="{html.escape(jpg.name, quote=True)}"><img loading="lazy" src="{html.escape(jpg.name, quote=True)}" alt="{label} · {html.escape(case, quote=True)}"></a></figure>')
        audit_name = f'{prefix}_audit.json'
        (output / audit_name).write_text(json.dumps(record, indent=2, default=str) + '\n')
        output_name = f'{prefix}_output.json'
        (output / output_name).write_text(json.dumps(design, indent=2) + '\n')
        diagnostics = record['diagnostics']
        flags = record.get('defects', {}).get('defect_flags', [])
        winner = str(diagnostics.get('winner', 'unknown'))
        provenance = 'Recorded Qwen coordinates replayed through the current engine.' if record.get('frozen_model_prediction') else 'Fresh Qwen-enabled engine run.'
        row = dict(case=case, source=f'{prefix}_source.jpg', image=f'{prefix}_output.jpg', json=output_name,
                   audit=audit_name, winner=winner, defect_flags=flags, frozen_model_prediction=bool(record.get('frozen_model_prediction')),
                   input_sha256=sha256(path.read_bytes()).hexdigest(), render_warnings=warnings)
        rows.append(row)
        cards.append(f'<article><h2>{html.escape(case.replace("_", " "))}</h2><div class="panels">{"".join(panels)}</div>'
                     f'<details><summary>Audit details</summary><p>{html.escape(provenance)}</p><p>Selected: {html.escape(winner)}. '
                     f'Automated flags: {html.escape(str(flags))}.</p><p><a href="{output_name}">Output JSON</a> · <a href="{audit_name}">Full diagnostics</a></p></details></article>')
    if not rows:
        raise ValueError(f'No source/output reflow audit records found in {run}')
    (output / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
       f'<title>{html.escape(title)}</title><style>{STYLE}</style><header><p><a href="../index.html">Review overview</a></p>'
       f'<h1>{html.escape(title)}</h1><p>Original and resized designs. Qwen is enabled in these runs; the engine may select a deterministic candidate. '
       'Intermediate proposals and rejection reasons remain in the audit files.</p></header><main>' + ''.join(cards) + '</main></html>')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='Current reflow review')
    args = parser.parse_args()
    rows = build(args.run, args.output, args.title)
    print(json.dumps(dict(cases=len(rows), output=str(args.output / 'index.html'))))


if __name__ == '__main__':
    main()
