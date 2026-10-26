"""Shared exact structural/raster gates for adapter and official-sample stress runs."""
from collections import Counter


def assess_engine_layout(source,design,report):
    from engine.resize import _analyze,_unintended_text_overlaps,_source_occlusion_defects,_render_text_ink,_source_ink_overhang,_unintended_control_overlaps
    from engine.main import self_check
    from scripts.evaluate_qwen_engine import image_ratio_violations
    expected={e['id'] for e in source['elements']}
    counts=Counter(e.get('id') for e in design.get('elements',[]));actual=set(counts)
    missing=sorted(expected-actual);extra=sorted(actual-expected)
    duplicates=sorted(eid for eid,n in counts.items() if n>1)
    _,units=_analyze(source)
    overlaps=_unintended_text_overlaps(design.get('elements',[]),units)
    controls=_unintended_control_overlaps(design.get('elements',[]),source)
    from engine.layout_contracts import issues
    contracts=issues(design.get('elements',[]),source,design['canvas']['width'],design['canvas']['height'],units)
    hero=_source_occlusion_defects(design.get('elements',[]),source)
    ratio=image_ratio_violations(source,design)
    visual=report.get('visual') or {}
    winner=next((r for r in visual.get('candidates',[]) if r.get('name')==report.get('winner')),{})
    source_contrast=visual.get('source_contrast') or {}
    weak=winner.get('low_contrast',[])
    regression=winner.get('contrast_regressions',
        [eid for eid in weak if not source_contrast.get(eid,{}).get('low',True)])
    overflow={}
    originals={e['id']:e for e in source['elements']}
    for e in design.get('elements',[]):
        if e.get('type')!='text':continue
        ink=_render_text_ink(e)
        if ink is not None:
            bx,by=_source_ink_overhang(e,originals.get(e['id']))
            amount=max(0.,-ink[0]-bx,-ink[1]-by,ink[2]-round(e['width']),ink[3]-round(e['height']),
                       -e['x']-ink[0],-e['y']-ink[1],
                       e['x']+ink[2]-design['canvas']['width'],
                       e['y']+ink[3]-design['canvas']['height'])
            if amount>.01:overflow[e['id']]=round(amount,3)
    structural=self_check(source,design)
    flags=[]
    if missing or extra or duplicates:flags.append('element_ids')
    if ratio:flags.append('image_aspect_distortion')
    if overlaps:flags.append('text_on_text')
    if controls:flags.append('control_collision')
    if hero:flags.append('text_hero_occlusion')
    flags.extend(sorted({d['kind'] for d in contracts}))
    if regression:flags.append('contrast_below_source_floor')
    if overflow:flags.append('glyph_overflow')
    if winner.get('invisible_text'):flags.append('invisible_text')
    if structural:flags.append('structural_check')
    return dict(flagged=bool(flags),defect_flags=flags,missing_ids=missing,extra_ids=extra,
                duplicate_ids=duplicates,image_ratio_violations=ratio,
                unintended_text_overlaps=overlaps,text_hero_occlusion=hero,control_collisions=controls,visual_contracts=contracts,
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


def build_ab_viewer(candidate_rows,comparison_rows,records,output,top=10,
                    candidate_label='Candidate',comparison_label='Baseline'):
    """Render a paired viewer for the worst engine A/B cases."""
    import json,os,math
    from pathlib import Path
    from scripts.dataset_renderer import render_design_to_png as dataset_render
    from renderer.converter.convert import render_design_to_png as official_render
    root=Path(__file__).resolve().parents[1];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    record_by_id={r['pair_id']:r for r in records}
    candidate_by_id={r['pair_id']:r for r in candidate_rows}
    comparison_by_id={r['pair_id']:r for r in comparison_rows}
    def number(value):
        try:
            value=float(value)
            return value if math.isfinite(value) else 0.
        except (TypeError,ValueError):return 0.
    def severity(row):
        if not row:return (1,1,1,1,1,1)
        return (bool(row.get('engine_error') or row.get('flagged')),
                bool(row.get('engine_error')),
                len(row.get('defect_flags',[])),
                number(row.get('layout_defect_cost')),
                number(row.get('low_contrast_count')),
                number(row.get('geometry_mae')))
    paired=[]
    for pair_id in set(candidate_by_id)|set(comparison_by_id):
        candidate=candidate_by_id.get(pair_id,{});comparison=comparison_by_id.get(pair_id,{})
        cs,se=severity(candidate),severity(comparison)
        regression=(cs[0]>se[0] or cs[1]>se[1] or cs[2]>se[2]
                    or cs[3]>se[3]+1e-9 or cs[4]>se[4]
                    or cs[5]>se[5]+1e-9)
        paired.append((pair_id,candidate,comparison,regression,max(cs,se)))
    paired.sort(key=lambda item:(item[4],item[3]),reverse=True)
    entries=[]
    for index,(pair_id,candidate,comparison,regression,_) in enumerate(paired[:top],1):
        record=record_by_id.get(pair_id,{})
        # The source is shared by both arms; keep one copy in the viewer.
        source=(candidate.get('source_design') or comparison.get('source_design'))
        if source is None:continue
        official=bool(record.get('official_path'))
        assets=root/'brand_kit/assets' if official else root/'data'/f"crello_{record.get('source_split','test')}"
        def render(doc,path):
            return (official_render(doc,assets,path) if official else
                    dataset_render(doc,assets,path,root/'data/crello_fonts.pickle',adjust_tight_text_widths=False))
        source_path=output/f'{index:02d}_source.png';render(source,source_path)
        cand_design=candidate.get('output_design');base_design=comparison.get('output_design')
        cand_path=output/f'{index:02d}_candidate.png';base_path=output/f'{index:02d}_baseline.png'
        if cand_design:render(cand_design,cand_path)
        if base_design:render(base_design,base_path)
        reference=None
        if not official and record.get('target_split'):
            folder=root/'data'/f"crello_{record['target_split']}";target_id=pair_id.split('__to__',1)[-1]
            manifest=folder/'manifest.jsonl'
            if manifest.is_file():
                for line in manifest.read_text().splitlines():
                    row=json.loads(line)
                    if row.get('template_id')==target_id and row.get('preview'):
                        reference=os.path.relpath(folder/row['preview'],output);break
        def compact(row):
            keys=('valid','geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae',
                  'raw_json_complete','raw_proposal_valid','hard_checks_passed','flagged','defect_flags',
                  'layout_defect_cost','low_contrast_count','source_lockup_error','source_contrast_regressions',
                  'unintended_text_overlaps','text_hero_occlusion','glyph_overflow','winner','engine_error')
            return {key:row[key] for key in keys if key in row}
        entries.append({'pair_id':pair_id,'source':source_path.name,
            'candidate':cand_path.name if cand_design else None,
            'baseline':base_path.name if base_design else None,'reference':reference,
            'candidate_metrics':compact(candidate),'baseline_metrics':compact(comparison),
            'candidate_regression':regression})
    payload=json.dumps(entries,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>Qwen adapter A/B review</title>
<style>body{background:#0b111b;color:#e8edf5;font:15px system-ui;margin:18px}header{position:sticky;top:0;z-index:2;background:#0b111b;padding:10px;border-bottom:1px solid #344252}button{padding:8px 14px;margin-right:8px;background:#1c2938;color:#e8edf5;border:1px solid #41536a;border-radius:7px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}.card{background:#16212d;border:1px solid #293a4b;border-radius:12px;padding:12px;min-width:0}.card h3{margin:3px 0 10px}.card img{width:100%;height:48vh;object-fit:contain;background:#0b111b;border-radius:7px}.metrics{white-space:pre-wrap;font:12px ui-monospace,monospace;max-height:26vh;overflow:auto}.regression{color:#ffb5a8}.ok{color:#a5dfb3}</style>
<header><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="count"></span></header><h2 id="title"></h2><div id="grid" class="grid"></div>
<script>const rows=__ROWS__;let i=0;const labels=__LABELS__;function show(){if(!rows.length){document.getElementById('title').textContent='No A/B layouts to display';return}const r=rows[i];document.getElementById('count').textContent=`Case ${i+1}/${rows.length}`;document.getElementById('title').textContent=r.pair_id;const cards=[['Source JSON',r.source],['Dataset target',r.reference], [labels.candidate,r.candidate], [labels.baseline,r.baseline]];document.getElementById('grid').innerHTML=cards.filter(x=>x[1]).map(x=>`<section class="card"><h3>${x[0]}</h3><img src="${x[1]}"></section>`).join('')+`<section class="card"><h3 class="${r.candidate_regression?'regression':'ok'}">${r.candidate_regression?'Candidate has a regression vs baseline':'No measured regression on this case'}</h3><h3>${labels.candidate}</h3><pre class="metrics">${JSON.stringify(r.candidate_metrics,null,2)}</pre><h3>${labels.baseline}</h3><pre class="metrics">${JSON.stringify(r.baseline_metrics,null,2)}</pre></section>`}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});show()</script>'''.replace('__ROWS__',payload).replace('__LABELS__',json.dumps({'candidate':candidate_label,'baseline':comparison_label}))
    (output/'review.html').write_text(html)
    (output/'manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    return str(output/'review.html')
