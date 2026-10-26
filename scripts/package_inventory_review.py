#!/usr/bin/env python3
"""Refresh current evidence in the existing compact Gemini packet."""
from pathlib import Path
import hashlib,html,json,shutil,sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PACK=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25'
RUN=ROOT/'data/qwen_lora/inventory_fixes_verified'

def jpeg(source,dest):
    with Image.open(source) as im:im.convert('RGB').save(dest,quality=92,optimize=True,progressive=True)

def main():
    from renderer.converter.convert import render_design_to_png as official_render
    from scripts.dataset_renderer import render_design_to_png as dataset_render
    reports={k:json.loads((RUN/k/'report.json').read_text()) for k in ('official','crello','crello_1p7b','heldout')}
    assert [len(reports[k]['results']) for k in ('official','crello','crello_1p7b','heldout')]==[18,12,12,89]
    rows={k:{r['pair_id']:r for r in d['results']} for k,d in reports.items()}
    gallery=PACK/'review/main_engine_qwen_outputs';manifest=json.loads((gallery/'manifest.json').read_text());cards=[]
    for item in manifest:
        arm='official' if item['image'].startswith('official_') else 'crello_1p7b' if '_1p7b.' in item['image'] else 'crello'
        row=rows[arm][item['pair_id']];dest=gallery/(Path(item['image']).stem+'.jpg');temp=dest.with_suffix('.render.png')
        if arm=='official':official_render(row['output_design'],ROOT/'brand_kit/assets',temp,max_size=1200)
        else:dataset_render(row['output_design'],ROOT/'data'/('crello_'+row['source_split']),temp,ROOT/'data/crello_fonts.pickle',max_size=1200,adjust_tight_text_widths=False)
        jpeg(temp,dest);temp.unlink();(gallery/item['json']).write_text(json.dumps(row['output_design'],indent=2)+'\n')
        item.update(image=dest.name,winner=row['winner'],hard_checks_passed=row['hard_checks_passed'],
                    flagged=row['flagged'],defect_flags=row['defect_flags'])
        title=html.escape(item.get('case',item['pair_id']));label=html.escape(arm)
        cards.append(f'<article><h3>{title} · {label}</h3><img loading="lazy" src="{dest.name}"><p>{html.escape(row["winner"])}</p><p>Flags: {html.escape(str(row["defect_flags"]))} · <a href="{item["json"]}">Selected JSON</a></p></article>')
    (gallery/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current engine outputs</title><style>body{font:15px system-ui;background:#101820;color:#eee;margin:24px}a{color:#9ed4ff}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:#182532;padding:16px;border-radius:12px}img{width:100%;height:420px;object-fit:contain}p{overflow-wrap:anywhere}@media(max-width:800px){main{grid-template-columns:1fr}}</style><h1>Current engine outputs · 42 renders</h1><p>Final inventory repairs, identical saved decoded model coordinates. JSON remains beside every image.</p><p><a href="../index.html">Review overview</a> · <a href="../../docs/INVENTORY_REPAIR_AUDIT.md">Repair audit</a></p><main>'+''.join(cards)+'</main>')
    # Portable paired worst-case gallery, including external dataset previews.
    original=RUN/'heldout/viewer';target=PACK/'review/inventory_worst10';target.mkdir(parents=True,exist_ok=True)
    entries=json.loads((original/'manifest.json').read_text())
    for i,entry in enumerate(entries,1):
        for key in ('source','reference','candidate','baseline'):
            ref=entry.get(key)
            if not ref:continue
            src=(original/ref).resolve();dest=target/f'{i:02d}_{key}.jpg'
            jpeg(src,dest);entry[key]=dest.name
    payload=json.dumps(entries,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>Current worst 10</title><style>body{font:15px system-ui;background:#101820;color:#eee;margin:20px}a{color:#9ed4ff}header{position:sticky;top:0;background:#101820;padding:12px}button{padding:10px;margin-right:10px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}img{width:100%;height:420px;object-fit:contain;background:#182532}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><header><a href="../index.html">Overview</a> · <button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="count"></span></header><h2 id="title"></h2><div class="grid"><figure><figcaption>Source</figcaption><img id="source"></figure><figure id="ref"><figcaption>Human dataset preview</figcaption><img id="reference"></figure><figure><figcaption>Prior engine</figcaption><img id="baseline"></figure><figure><figcaption>Current engine</figcaption><img id="candidate"></figure></div><pre id="metrics"></pre><script>const rows=__ROWS__;let i=0;function show(){const r=rows[i];document.getElementById('count').textContent=`${i+1}/${rows.length}`;document.getElementById('title').textContent=r.pair_id;for(const k of ['source','reference','baseline','candidate'])document.getElementById(k).src=r[k]||'';document.getElementById('ref').hidden=!r.reference;document.getElementById('metrics').textContent=JSON.stringify(r.candidate_metrics,null,2)}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}show()</script>'''.replace('__ROWS__',payload)
    (target/'review.html').write_text(page);(target/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    evidence=PACK/'evidence/inventory_repairs';evidence.mkdir(parents=True,exist_ok=True)
    shutil.copy2(RUN/'summary.json',evidence/'summary.json')
    compact_keys=('pair_id','source_split','target_canvas','winner','flagged','defect_flags','glyph_overflow',
                  'source_contrast_regressions','control_collisions','geometry_mae','raw_geometry_mae',
                  'before_defects','after_defects','contrast_dispatch_reexecuted','output_design')
    for name,d in reports.items():
        (evidence/(name+'.jsonl')).write_text(''.join(json.dumps({k:r[k] for k in compact_keys if k in r},ensure_ascii=False)+'\n' for r in d['results']))
        (evidence/(name+'_run_manifest.json')).write_text(json.dumps(d['manifest'],indent=2)+'\n')
    for path in ['engine/resize.py','engine/qwen_refine.py','engine/qwen_seed.py','engine/qwen_prompt.py',
                 'scripts/layout_defects.py','scripts/audit_inventory_fixes.py','scripts/finalize_inventory_audit.py',
                 'tests/test_inventory_repairs.py','tests/test_stress_metrics.py','pytest.ini',
                 'docs/IMPLEMENTATION.md','docs/INVENTORY_REPAIR_AUDIT.md','README.md']:
        dest=PACK/'code'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
    (PACK/'docs').mkdir(exist_ok=True);shutil.copy2(ROOT/'docs/INVENTORY_REPAIR_AUDIT.md',PACK/'docs/INVENTORY_REPAIR_AUDIT.md')
    (PACK/'README_FOR_GEMINI.md').write_text('''# Compact current-engine review

Start at `review/index.html`.

- `docs/INVENTORY_REPAIR_AUDIT.md`: implemented causes, disagreements with source assumptions, tests and exact replay results.
- `review/main_engine_qwen_outputs/index.html`: all 18 official formats and both 12-case adapter arms, rendered with the final engine.
- `review/inventory_worst10/review.html`: source, dataset preview, prior engine and current output for the worst 10 held-out comparisons.
- `evidence/inventory_repairs/summary.json` and cohort JSONL files: all 89 held-out outputs plus official and paired adapter outputs. Raw coordinates are reused to isolate engine repair and selector behavior. The last contrast-only change replays 31 possible dispatches; 58 unchanged dispatches are reused with source-hash certification.
- `code/`: current implementation and focused regression tests. No weights are included.

`review/prompt_ab_extreme_resizes/` and `evidence/prompt_ab_same_adapter_20261025/` retain the earlier controlled prompt experiment. Their recorded code hashes refer to the earlier engine, not this inventory revision. The old prompt remains active; no training or prompt promotion is performed here.

Passing hard structural and raster checks does not mean every aesthetic or source-axis review flag is resolved. See the per-case images and retained alignment counts.
''')
    (PACK/'GEMINI_REVIEW_PROMPT.md').write_text('''Review the current inventory repairs in docs/INVENTORY_REPAIR_AUDIT.md, then inspect the 42 current main-engine outputs and the worst-10 held-out comparisons. Compare actual source JSON and painted output before asserting that an element is misplaced. Check grouped ornaments, source-relative attachments, contrast, readable extreme reflow and remaining axis flags. Distinguish source-authored imperfections from newly introduced defects. The evidence covers all 89 held-out cases, all 18 official formats and both reviewed adapter arms, using identical saved model coordinates. Recommend generic source-evidenced constraints; do not propose template-ID-specific coordinates or treat passing hard checks as aesthetic perfection.
''')
    (PACK/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current Gemini review</title><style>body{font:19px system-ui;background:#101820;color:#eee;max-width:900px;margin:60px auto}a{color:#9ed4ff}li{margin:22px}</style><h1>Current engine review</h1><ul><li><a href="main_engine_qwen_outputs/index.html">42 current engine outputs</a></li><li><a href="inventory_worst10/review.html">89-case audit: worst 10 comparisons</a></li><li><a href="../docs/INVENTORY_REPAIR_AUDIT.md">Repairs, tests and results</a></li><li><a href="prompt_ab_extreme_resizes/review.html">Earlier prompt A/B extreme-resize evidence</a></li></ul>')
    manifest=[]
    for p in sorted(PACK.rglob('*')):
        if p.is_file() and p.name not in ('MANIFEST.sha256','.DS_Store'):
            manifest.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(PACK).as_posix()}')
    (PACK/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    print('Refreshed',PACK,'files',len(manifest))

if __name__=='__main__':main()
