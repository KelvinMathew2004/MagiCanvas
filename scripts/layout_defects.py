"""Shared exact structural/raster gates for adapter and official-sample stress runs."""
from collections import Counter


def assess_engine_layout(source,design,report):
    from engine.resize import _analyze,_unintended_text_overlaps,_source_occlusion_defects,_render_text_ink,_source_ink_overhang
    from engine.main import self_check
    from scripts.evaluate_qwen_engine import image_ratio_violations
    expected={e['id'] for e in source['elements']}
    counts=Counter(e.get('id') for e in design.get('elements',[]));actual=set(counts)
    missing=sorted(expected-actual);extra=sorted(actual-expected)
    duplicates=sorted(eid for eid,n in counts.items() if n>1)
    _,units=_analyze(source)
    overlaps=_unintended_text_overlaps(design.get('elements',[]),units)
    hero=_source_occlusion_defects(design.get('elements',[]),source)
    ratio=image_ratio_violations(source,design)
    visual=report.get('visual') or {}
    winner=next((r for r in visual.get('candidates',[]) if r.get('name')==report.get('winner')),{})
    source_contrast=visual.get('source_contrast') or {}
    weak=winner.get('low_contrast',[])
    regression=[eid for eid in weak if not source_contrast.get(eid,{}).get('low',True)]
    overflow={}
    originals={e['id']:e for e in source['elements']}
    for e in design.get('elements',[]):
        if e.get('type')!='text':continue
        ink=_render_text_ink(e)
        if ink is not None:
            bx,by=_source_ink_overhang(e,originals.get(e['id']))
            amount=max(0.,-ink[0]-bx,-ink[1]-by,ink[2]-e['width'],ink[3]-e['height'],
                       -e['x']-ink[0],-e['y']-ink[1],
                       e['x']+ink[2]-design['canvas']['width'],
                       e['y']+ink[3]-design['canvas']['height'])
            if amount>1.01:overflow[e['id']]=round(amount,3)
    structural=self_check(source,design)
    flags=[]
    if missing or extra or duplicates:flags.append('element_ids')
    if ratio:flags.append('image_aspect_distortion')
    if overlaps:flags.append('text_on_text')
    if hero:flags.append('text_hero_occlusion')
    if regression:flags.append('contrast_below_source_floor')
    if overflow:flags.append('glyph_overflow')
    if winner.get('invisible_text'):flags.append('invisible_text')
    if structural:flags.append('structural_check')
    return dict(flagged=bool(flags),defect_flags=flags,missing_ids=missing,extra_ids=extra,
                duplicate_ids=duplicates,image_ratio_violations=ratio,
                unintended_text_overlaps=overlaps,text_hero_occlusion=hero,
                source_contrast_regressions=regression,glyph_overflow=overflow,
                structural_errors=structural,layout_defect_cost=winner.get('layout_defect_cost'),
                low_contrast_count=len(weak),source_lockup_error=winner.get('source_lockup_error'),
                selected_raster_metrics=winner,source_contrast=source_contrast)


FORMATS=(('story',1080,1920),('portrait',1080,1350),('square',1080,1080),
         ('widescreen',1600,400),('leaderboard',728,90),('skyscraper',160,600))


def official_records(folder):
    import json
    return [dict(pair_id=f'official:{path.stem}__to__{name}',family_id=path.stem,
                 official_path=str(path.resolve()),target_canvas=dict(width=w,height=h))
            for path in sorted(folder.glob('*.json'))
            if 'source' in json.loads(path.read_text()) for name,w,h in FORMATS]


def build_outlier_viewer(rows,records,output,top=10):
    import json,os
    from pathlib import Path
    from scripts.dataset_renderer import render_design_to_png as dataset_render
    from renderer.converter.convert import render_design_to_png as official_render
    root=Path(__file__).resolve().parents[1];output.mkdir(parents=True,exist_ok=True)
    lookup={r['pair_id']:r for r in records};entries=[]
    flagged=[r for r in rows if r.get('flagged')]
    flagged.sort(key=lambda r:(bool(r.get('engine_error')),bool(r.get('missing_ids') or r.get('image_ratio_violations')),
                  len(r.get('defect_flags',[])),r.get('layout_defect_cost') or 0.,r.get('low_contrast_count',0)),reverse=True)
    for i,r in enumerate(flagged[:top],1):
        entry={'pair_id':r['pair_id'],'metrics':{k:v for k,v in r.items() if k not in
               ('source_design','output_design','selected_raster_metrics','source_contrast')}}
        if r.get('output_design'):
            official=bool(r.get('official_path'));assets=root/'brand_kit/assets' if official else root/'data'/f"crello_{r['source_split']}"
            def render(doc,path):
                return official_render(doc,assets,path) if official else dataset_render(doc,assets,path,root/'data/crello_fonts.pickle',adjust_tight_text_widths=False)
            for label,key in (('source','source_design'),('output','output_design')):
                png=output/f'{i:02d}_{label}.png';render(r[key],png);entry[label]=png.name
            record=lookup[r['pair_id']]
            if not official and record.get('target_split'):
                folder=root/'data'/f"crello_{record['target_split']}";tid=record['pair_id'].split('__to__',1)[1]
                meta=next((json.loads(x) for x in (folder/'manifest.jsonl').read_text().splitlines()
                           if json.loads(x).get('template_id')==tid),None)
                if meta:entry['reference']=os.path.relpath(folder/meta['preview'],output)
        entries.append(entry)
    payload=json.dumps(entries,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>Engine stress outliers</title>
<style>body{background:#0b111b;color:#eee;font:15px system-ui;margin:20px}header{position:sticky;top:0;background:#0b111b;padding:12px}button{padding:10px;margin-right:8px}.grid{display:flex;gap:15px}.grid figure{margin:0;flex:1;min-width:0}img{width:100%;height:65vh;object-fit:contain;background:#171f2b}pre{white-space:pre-wrap}</style>
<header><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="count"></span></header><h3 id="title"></h3><div class="grid"><figure><figcaption>Source JSON render</figcaption><img id="source"></figure><figure id="ref"><figcaption>Dataset reference</figcaption><img id="reference"></figure><figure><figcaption>Selected output</figcaption><img id="output"></figure></div><pre id="metrics"></pre>
<script>const rows=__ROWS__;let i=0;function show(){if(!rows.length){document.getElementById('title').textContent='No flagged outputs';return}const r=rows[i];document.getElementById('count').textContent=`Flagged outlier ${i+1}/${rows.length}`;document.getElementById('title').textContent=r.pair_id;for(const k of ['source','reference','output'])document.getElementById(k).src=r[k]||'';document.getElementById('ref').hidden=!r.reference;document.getElementById('metrics').textContent=JSON.stringify(r.metrics,null,2)}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});show()</script>'''.replace('__ROWS__',payload)
    (output/'review.html').write_text(html);(output/'manifest.json').write_text(json.dumps(entries,indent=2))
    return str(output/'review.html')
