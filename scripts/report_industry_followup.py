#!/usr/bin/env python3
"""Publish all changed research outcomes, native audits and reference metrics."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean
from hashlib import sha256
import argparse,json,html,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    from scripts.evaluate_industry_followup import DEFAULT
    from scripts.report_training_free_upgrades import gold_metrics
    from scripts.refresh_regression_viewer import documents_differ
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty,STYLE
    from scripts.audit_upgrade_rasters import paint_stats,against_source
    from experiments.layout_upgrade.industry import rag
    from engine.resize import _text_info
    from PIL import Image,ImageDraw
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/industry_followup')
    a=p.parse_args();cases=json.loads((a.root/'cases.json').read_text());aliases=json.loads((a.root/'test_pair_tasks.json').read_text())
    files=[a.root/'results'/f'{i:03d}.json' for i in range(len(cases))]
    assert all(f.exists() for f in files),'Complete every predeclared request before reporting'
    rows=[json.loads(f.read_text()) for f in files];lookup={c['key']:(c,r) for c,r in zip(cases,rows)}
    summary=dict(tasks=len(rows),reference_pairs=len(aliases),qwen_enabled=True,fresh_inference=False,production_changed=False,
                 baseline='Exact fresh production replay from scale_followup, verified complete engine/input/code hashes',
                 scope='Consumed development cohorts; reference target coordinates used only in this report',arms={})
    comparisons=[]
    for arm in ('paragraph-balance','group-neighborhood'):
        changes=[];strata=defaultdict(lambda:dict(tasks=0,changed=0));refs=[]
        for i,(c,r) in enumerate(zip(cases,rows)):
            b,t=r['baseline'],r[arm];changed=documents_differ(b['output_design'],t['output_design'])
            ratio=b['target_canvas']['width']/b['target_canvas']['height']
            bucket='skyscraper' if ratio<=.45 else 'portrait' if ratio<.9 else 'square' if ratio<=1.35 else 'wide' if ratio<=4 else 'leaderboard'
            for key in (c['cohort'],bucket):strata[key]['tasks']+=1;strata[key]['changed']+=changed
            if changed:
                item=dict(index=i,pair_id=c['pair_id'],arm=arm,new_flags=sorted(set(t['defect_flags'])-set(b['defect_flags'])))
                changes.append(item);comparisons.append(item)
        for alias in aliases:
            c,r=lookup[alias['task_key']];b,t=r['baseline'],r[arm]
            refs.append(dict(pair_id=alias['pair_id'],family_id=alias['record']['family_id'],
                before=gold_metrics(b['source_design'],b['output_design'],alias['record']),
                after=gold_metrics(b['source_design'],t['output_design'],alias['record'])))
        valid=[r for r in refs if r['before']['geometry_mae'] is not None and r['after']['geometry_mae'] is not None]
        reference=dict(pairs=len(valid))
        for key in ('geometry_mae','font_mae'):
            data=[r for r in valid if r['before'][key] is not None and r['after'][key] is not None]
            reference['before_'+key]=mean(r['before'][key] for r in data)
            reference['after_'+key]=mean(r['after'][key] for r in data)
        reference.update(improved=sum(r['after']['geometry_mae']<r['before']['geometry_mae']-1e-5 for r in valid),
                         worsened=sum(r['after']['geometry_mae']>r['before']['geometry_mae']+1e-5 for r in valid))
        summary['arms'][arm]=dict(changed=len(changes),new_flag_cases=sum(bool(c['new_flags']) for c in changes),
            generated=sum(r[arm].get('generated',0) for r in rows),eligible=sum(r[arm].get('eligible',0) for r in rows),
            reasons=dict(Counter(r[arm]['reason'] for r in rows)),
            screening=dict(Counter(n['reason'] for r in rows for n in r[arm].get('screening',[]))),
            changes=changes,strata=dict(strata),reference=reference,reference_pairs=refs)
    decisions_path=a.root/'visual_decisions.json'
    decisions=json.loads(decisions_path.read_text()) if decisions_path.is_file() else {}
    summary['decisions']=decisions
    summary['report_script_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True);cache={};source_stats={};cards=[];native=[];tiles=[];published=[]
    for item in comparisons:
        i,arm=item['index'],item['arm'];r=rows[i];before=r['baseline'];trial=r[arm];source=before['source_design'];split=before.get('source_split','train')
        os.environ['MAGICANVAS_DATASET_RENDERING']='0' if before.get('official_path') else '1'
        render=_renderer(source,None if before.get('official_path') else split);stem=f'{i:03d}_{arm}';images={}
        for label,doc in [('source',source),('before',before['output_design']),('after',trial['output_design'])]:
            key=sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()
            if key not in cache:
                png=a.output/f'{stem}_{label}.png';_preview_nonempty(render,doc,png)
                with Image.open(png) as im:im.convert('RGB').save(png.with_suffix('.jpg'),quality=94,optimize=True)
                png.unlink();cache[key]=png.with_suffix('.jpg').name
            images[label]=cache[key]
        marker=a.root/'native'/f'{stem}.json';marker.parent.mkdir(exist_ok=True)
        if marker.exists():audit=json.loads(marker.read_text())
        else:
            key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
            if key not in source_stats:source_stats[key]=paint_stats(source,source,split)
            audit=dict(index=i,arm=arm,
                before=against_source(paint_stats(before['output_design'],source,split),source_stats[key]),
                after=against_source(paint_stats(trial['output_design'],source,split),source_stats[key]))
            save(marker,audit)
        native.append(audit)
        text_changes=[];old={e['id']:e for e in before['output_design']['elements']}
        for e in trial['output_design']['elements']:
            if e['type']=='text' and e!=old[e['id']]:
                o=old[e['id']];x,y=_text_info(o),_text_info(e)
                text_changes.append(dict(id=e['id'],before_lines=x[2],after_lines=y[2],
                    before_rag=rag(x),after_rag=rag(y),font_changed=e['font_size']!=o['font_size'],
                    changed_fields=[k for k in set(e)|set(o) if e.get(k)!=o.get(k)]))
        item.update(images=images,text_changes=text_changes)
        audit_path=f'{stem}_audit.json';save(a.output/audit_path,dict(comparison=item,baseline=before,experiment=trial,native=audit,
            provenance=json.loads((a.root/'experiment_manifest.json').read_text())))
        published.append(dict(case=item['pair_id'],arm=arm,source=images['source'],image=images['after'],
                              audit=audit_path,status='Explored, rejected for production; current baseline unchanged'))
        cards.append(f'''<article><h2>{html.escape(arm)} · {html.escape(item['pair_id'])}</h2>
<p>Research only — not a production output. All proposals are shown; no favorable subset.</p>
<label>Resized image <select onchange="this.closest('article').querySelector('[data-output]').src=this.value">
<option value="{images['after']}">Experimental resized</option><option value="{images['before']}">Current production</option></select></label>
<div class="pair"><section><h3>Original</h3><img src="{images['source']}"></section><section><h3>Resized</h3><img data-output src="{images['after']}"></section></div>
<details><summary>Audit details</summary><p>New flags: {html.escape(str(item['new_flags']))}. <a href="{audit_path}">Geometry and raster JSON</a></p></details></article>''')
        tile=Image.new('RGB',(1200,440),'#172531');draw=ImageDraw.Draw(tile);draw.text((12,10),f'{i} {arm} {item["pair_id"][:80]}',fill='white')
        for label,x in [('before',0),('after',600)]:
            im=Image.open(a.output/images[label]).convert('RGB');im.thumbnail((580,380));tile.paste(im,(x+(600-im.width)//2,50+(380-im.height)//2))
            draw.text((x+12,32),'Current production' if label=='before' else 'Experimental',fill='white')
        tiles.append(tile)
    for page,start in enumerate(range(0,len(tiles),3)):
        selected=tiles[start:start+3];sheet=Image.new('RGB',(1200,440*len(selected)),'#172531')
        for j,tile in enumerate(selected):sheet.paste(tile,(0,440*j))
        sheet.save(a.root/f'comparisons_{page:02d}.jpg',quality=94)
    summary['native']=dict(comparisons=len(native),new_invisible=[],new_source_floor_regressions=[],
        limitation='Legacy shared all-text mask; color/rotation exclusions and ownership ambiguities remain. Not a proof of aesthetics.')
    for r in native:
        for key,destination in [('invisible_or_unpainted','new_invisible'),('source_floor_regressions','new_source_floor_regressions')]:
            added=sorted(set(r['after'][key])-set(r['before'][key]))
            if added:summary['native'][destination].append(dict(index=r['index'],arm=r['arm'],ids=added))
    summary['comparisons']=comparisons;save(a.root/'report.json',summary)
    save(a.output/'summary.json',summary)
    save(a.output/'manifest.json',published)
    page=f'''<!doctype html><meta charset="utf-8"><title>Industry algorithm follow-up</title><style>{STYLE}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img{{width:100%;max-height:850px;object-fit:contain}}select{{font-size:18px}}@media(max-width:750px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>Industry algorithm follow-up</h1><p><a href="../main_engine_qwen_outputs/index.html">Current production viewer</a> · <a href="summary.json">Full 134-request report</a></p>
<p>Original / Resized only. Resized dropdown compares the research proposal with current production. Exact recorded Qwen predictions; no training or fresh inference. Promotion requires separate evidence.</p>{''.join(cards)}'''
    (a.output/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('arms','comparisons')},indent=2))
    for arm,r in summary['arms'].items():print(arm,json.dumps({k:v for k,v in r.items() if k not in ('changes','reference_pairs','strata')},indent=2))


if __name__=='__main__':main()
