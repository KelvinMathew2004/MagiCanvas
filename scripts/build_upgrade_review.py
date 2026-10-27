#!/usr/bin/env python3
"""Publish fresh A/B evidence and verify the frozen regression's actual pixels.

No model invocation, selector change, human approval or weight promotion occurs.
Public pages display only Original and Resized; raw geometry stays in JSON.
"""
from pathlib import Path
from collections import defaultdict
import argparse, html, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(args):
    from PIL import Image, ImageChops
    from scripts.build_focus_reflow_viewer import _renderer, STYLE
    from engine.resize import _analyze, _bounds
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / 'data/qwen_lora/engine_upgrade'
    raster = evidence / 'regression_rasters'
    raster.mkdir(exist_ok=True)
    pixels, peers = [], []
    for path in sorted(args.regression.glob('[0-9][0-9].json')):
        item = json.loads(path.read_text())
        before, after = item['before'], item['after']
        render = _renderer(after['source_design'], after['source_split'])
        images = []
        for arm, row in (('before', before), ('after', after)):
            dest = raster / f'{path.stem}_{arm}.png'
            render(row['output_design'], dest)
            with Image.open(dest) as im:
                images.append(im.convert('RGBA'))
        difference = ImageChops.difference(images[0].convert('RGB'), images[1].convert('RGB'))
        equal = images[0].size == images[1].size and images[0].tobytes() == images[1].tobytes()
        pixels.append(dict(pair_id=after['pair_id'], pixels_identical=equal,
                           difference_bbox=difference.getbbox(), document_identical=before['output_design'] == after['output_design']))
        _, units = _analyze(after['source_design'])
        bands = defaultdict(list)
        for unit in units:
            if unit.get('peer_ids') and unit.get('role') == 'tag':
                bands[unit['peer_ids']].append(unit)
        target = {e['id']: e for e in after['output_design']['elements']}
        for key, group in bands.items():
            boxes = sorted(_bounds([target[e['id']] for e in u['items']]) for u in group)
            if len(boxes) < 3:
                continue
            heights = [b[3]-b[1] for b in boxes]
            centers = [(b[1]+b[3])/2 for b in boxes]
            if max(centers)-min(centers) > min(heights)*.35:
                continue  # A wrapped/vertical group is not a horizontal peer row.
            gaps = [b[0]-a[2] for a, b in zip(boxes, boxes[1:])]
            peers.append(dict(pair_id=after['pair_id'], peer_ids=list(key), clear_gaps=gaps,
                              gap_asymmetry=max(gaps)-min(gaps), minimum_height=min(heights)))
    (evidence/'regression_pixels.json').write_text(json.dumps(dict(cases=len(pixels), identical=sum(p['pixels_identical'] for p in pixels), results=pixels), indent=2)+'\n')
    (evidence/'peer_gaps.json').write_text(json.dumps(dict(scope='Final realized horizontal peer rows, source-evidenced groups only', rows=peers), indent=2)+'\n')

    report = json.loads((args.ab/'report.json').read_text())
    cases = report['results']; cards = []; compact = []
    for i, baseline in enumerate(cases['0.6b']):
        render = _renderer(baseline['source_design'], baseline['source_split'])
        source_path = args.output/f'{i:02d}_source.png'
        render(baseline['source_design'], source_path)
        source_jpg = source_path.with_suffix('.jpg')
        with Image.open(source_path) as im:
            im.convert('RGB').save(source_jpg, quality=92, optimize=True)
        source_path.unlink()
        arms = {}
        for arm in ('0.6b', '1.7b'):
            row = cases[arm][i]
            assert row['pair_id'] == baseline['pair_id']
            path = args.output/f'{i:02d}_{arm}.png'
            render(row['output_design'], path)
            with Image.open(path) as im:
                im.convert('RGB').save(path.with_suffix('.jpg'), quality=92, optimize=True)
            path.unlink()
            arms[arm] = {k: row.get(k) for k in ('winner','defect_flags','qwen_family_win','geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae','original_raw_proposal_valid')}
            arms[arm]['generation_seconds'] = row['original_generation'].get('generation_seconds')
            (args.output/f'{i:02d}_{arm}_output.json').write_text(json.dumps(row['output_design'], indent=2)+'\n')
        compact.append(dict(pair_id=baseline['pair_id'], arms=arms))
        name = baseline['source_design'].get('name') or baseline['pair_id']
        detail = {arm: f"{arms[arm]['winner']} · raw IDs valid: {arms[arm]['original_raw_proposal_valid']} · flags: {arms[arm]['defect_flags']}" for arm in arms}
        cards.append(f'<article><h2>{html.escape(str(name))}</h2><div class="panels"><figure><figcaption>Original</figcaption><img loading="lazy" src="{source_jpg.name}" alt="Original"></figure><figure><figcaption>Resized <span class="arm-label">0.6B</span></figcaption><a class="output-link" href="{i:02d}_0.6b_output.json">Output JSON</a><img loading="lazy" class="resized" data-index="{i:02d}" src="{i:02d}_0.6b.jpg" alt="Resized"></figure></div><details><summary>Audit details</summary><p class="arm-detail" data-details="{html.escape(json.dumps(detail), quote=True)}">{html.escape(detail["0.6b"])}</p></details></article>')
    stats = {}
    for arm in cases:
        fonts = [r['font_mae'] for r in cases[arm] if r.get('font_mae') is not None]
        raw_fonts = [r['raw_relative_font_mae'] for r in cases[arm] if r.get('raw_relative_font_mae') is not None]
        stats[arm] = dict(report['summary'][arm], mean_dataset_font_mae=sum(fonts)/len(fonts), mean_raw_dataset_relative_font_mae=sum(raw_fonts)/len(raw_fonts))
    font_gate=stats['1.7b']['mean_dataset_font_mae']<=stats['0.6b']['mean_dataset_font_mae']
    small = dict(summary=stats, results=compact, development_gate_passed=report['development_gate_passed'], font_nonregression_passed=font_gate, broader_promotion_gate_passed=report['development_gate_passed'] and font_gate, default_changed=False,
                 scope='Fresh 20-case paired generation, same final selector; eight authored alternates and twelve dataset pairs. No Qwen-off arm. No 89/309 release sweep.')
    (evidence/'dense_ab_summary.json').write_text(json.dumps(small, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(small, indent=2)+'\n')
    script = '''const s=document.getElementById('arm');s.onchange=()=>{document.querySelectorAll('.resized').forEach(e=>{e.src=e.dataset.index+'_'+s.value+'.jpg';e.previousElementSibling.href=e.dataset.index+'_'+s.value+'_output.json'});document.querySelectorAll('.arm-label').forEach(e=>e.textContent=s.value.toUpperCase());document.querySelectorAll('.arm-detail').forEach(e=>e.textContent=JSON.parse(e.dataset.details)[s.value])};'''
    (args.output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fresh dense model comparison</title><style>'+STYLE+'</style><header><p><a href="../regression_suite/index.html">Unified 38-case regression viewer</a></p><h1>Fresh dense model comparison</h1><p>Qwen enabled in both arms. Original and Resized only. Changing the model here changes the displayed experiment, not the project default. Complete IDs do not guarantee a good layout; deterministic candidates may win.</p><p>Twenty reviewed development cases; eight dense alternate conversions and twelve dataset pairs. No human resized targets exist for the authored cases. See <a href="summary.json">measured results</a>.</p><label>Resized model <select id="arm"><option value="0.6b">0.6B</option><option value="1.7b">1.7B</option></select></label></header><main>'+''.join(cards)+'</main><script>'+script+'</script></html>')
    print(json.dumps(dict(regression_pixels_identical=sum(p['pixels_identical'] for p in pixels), peer_rows=len(peers), max_peer_gap_asymmetry=max((r['gap_asymmetry'] for r in peers), default=0), ab_summary=stats)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--regression', type=Path, default=ROOT/'data/qwen_lora/regression_suite/3e769527fb4f81df')
    p.add_argument('--ab', type=Path, default=ROOT/'data/qwen_lora/engine_upgrade/dense_engine_ab')
    p.add_argument('--output', type=Path, default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/dense_ab')
    build(p.parse_args())
