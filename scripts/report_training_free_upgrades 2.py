#!/usr/bin/env python3
"""Report paired changes and render Original/Resized algorithm comparisons."""
from pathlib import Path
from collections import Counter
from statistics import mean,median
from hashlib import sha256
from copy import deepcopy
import argparse,json,sys,html,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def gold_metrics(source,design,record):
    from engine.qwen_prompt import decode_v2_completion,record_prompt_version
    from scripts.evaluate_qwen_adapter import measure
    gold=json.loads(record['messages'][-1]['content'])
    if gold.get('elements') and 'box' in gold['elements'][0]:gold=decode_v2_completion(gold,source,record['target_canvas'],record_prompt_version(record))
    return measure(design['elements'],gold,record['target_canvas'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/training_free_scale_v1')
    p.add_argument('--output',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/training_free_scale');a=p.parse_args()
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty,STYLE
    from PIL import Image,ImageDraw
    cases=json.loads((a.root/'cases.json').read_text());aliases=json.loads((a.root/'test_pair_tasks.json').read_text());results=[]
    for i,c in enumerate(cases):
        b=json.loads((a.root/'baseline'/f'{i:03d}.json').read_text())['baseline'];r=json.loads((a.root/'algorithms'/f'{i:03d}.json').read_text())
        results.append((c,b,r))
    arms=list(results[0][2]['arms']);summaries={};allchanges=[];bykey={c['key']:(b,r) for c,b,r in results}
    for arm in arms:
        changes=[];generated=rendered=0;rejections=Counter();after_flags=Counter();before_flags=Counter()
        for i,(c,b,r) in enumerate(results):
            v=r['arms'][arm];generated+=v['generated'];rendered+=v['rendered'];before_flags.update(b['defect_flags']);after_flags.update(v['defect_flags'])
            for item in v['candidates']:rejections[item.get('rejected','eligible')]+=1
            if v['changed']:
                x=dict(index=i,pair_id=c['pair_id'],cohort=c['cohort'],arm=arm,before_winner=b['winner'],winner=v['winner'],before_flags=b['defect_flags'],flags=v['defect_flags'],before_quality=r['baseline_quality'],quality=v['quality'])
                changes.append(x);allchanges.append(x)
        reference=[]
        for alias in aliases:
            b,r=bykey[alias['task_key']];source=b['source_design'];record=alias['record'];v=r['arms'][arm]
            before=gold_metrics(source,b['output_design'],record);after=gold_metrics(source,v['output_design'],record)
            shift=max((record['target_canvas']['width']/record['target_canvas']['height'])/(source['canvas']['width']/source['canvas']['height']),(source['canvas']['width']/source['canvas']['height'])/(record['target_canvas']['width']/record['target_canvas']['height']))
            reference.append(dict(pair_id=alias['pair_id'],family_id=record['family_id'],identity=all(source['canvas'][k]==record['target_canvas'][k] for k in ('width','height')),ratio_shift=shift,before=before,after=after))
        def stats(rows):
            valid=[r for r in rows if r['before']['geometry_mae'] is not None and r['after']['geometry_mae'] is not None]
            return dict(pairs=len(valid),families=len({r['family_id'] for r in valid}),before_geometry_mae=mean(r['before']['geometry_mae'] for r in valid) if valid else None,after_geometry_mae=mean(r['after']['geometry_mae'] for r in valid) if valid else None,
                geometry_improved=sum(r['after']['geometry_mae']<r['before']['geometry_mae']-1e-5 for r in valid),geometry_worsened=sum(r['after']['geometry_mae']>r['before']['geometry_mae']+1e-5 for r in valid),
                before_font_mae=mean(r['before']['font_mae'] for r in valid if r['before']['font_mae'] is not None),after_font_mae=mean(r['after']['font_mae'] for r in valid if r['after']['font_mae'] is not None))
        summaries[arm]=dict(cases=len(results),generated=generated,rendered=rendered,changed=len(changes),changes=changes,rejections=dict(rejections),baseline_flag_counts=dict(before_flags),flag_counts=dict(after_flags),
            before_flagged=sum(bool(b['defect_flags']) for _,b,_ in results),after_flagged=sum(bool(r['arms'][arm]['defect_flags']) for _,_,r in results),
            all_test=stats(reference),actual_resizes=stats([r for r in reference if not r['identity']]),extreme_resizes=stats([r for r in reference if not r['identity'] and r['ratio_shift']>=1.8]),reference_pairs=reference)
    a.output.mkdir(parents=True,exist_ok=True);cards=[];image_cache={};contact=[];paired=[]
    for n,change in enumerate(allchanges):
        i=change['index'];case,b,r=results[i];v=r['arms'][change['arm']];source=b['source_design'];render=_renderer(source,None if b.get('official_path') else b['source_split']);stem=f'{i:03d}_{change["arm"]}'
        paths={}
        for label,doc in [('source',source),('before',b['output_design']),('after',v['output_design'])]:
            key=sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()
            if key not in image_cache:
                png=a.output/f'{stem}_{label}.png';_preview_nonempty(render,doc,png);jpg=png.with_suffix('.jpg')
                with Image.open(png) as im:im.convert('RGB').save(jpg,quality=92,optimize=True)
                png.unlink();image_cache[key]=jpg.name
            paths[label]=image_cache[key]
        audit=dict(source=source,before=b,algorithm_result=v,provenance=json.loads((a.root/'algorithms_manifest.json').read_text()),case=case,change=change)
        (a.output/f'{stem}_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        cards.append(f'<article data-arm="{change["arm"]}"><h2>{html.escape(change["arm"])} · {html.escape(case["pair_id"])}</h2><p>Experimental, not promoted. <label>Resized image <select onchange="this.closest(\'article\').querySelector(\'[data-result]\').src=this.value"><option value="{paths["after"]}">Proposed alternative</option><option value="{paths["before"]}">Current production</option></select></label></p><div class="panels"><figure><figcaption>Original</figcaption><img src="{paths["source"]}" loading="lazy"></figure><figure><figcaption>Resized</figcaption><img data-result src="{paths["after"]}" loading="lazy"></figure></div><p>Before: {html.escape(change["before_winner"])} · after: {html.escape(change["winner"])}</p><p>Flags: {html.escape(str(change["flags"]))}. Selection-quality proxy {change["before_quality"]["cost"]:.2f} → {change["quality"]["cost"]:.2f}; this is not a human aesthetic score.</p><a href="{stem}_audit.json">Raw/intermediate audit</a></article>')
        contact.append((change,paths))
        paired.append(dict(case=case['pair_id'],arm=change['arm'],source=paths['source'],image=paths['after'],before=paths['before'],audit=f'{stem}_audit.json'))
    rows=''.join(f'<tr><td>{html.escape(arm)}</td><td>{s["generated"]}</td><td>{s["rendered"]}</td><td>{s["changed"]}</td><td>{s["all_test"]["before_geometry_mae"]:.4f} → {s["all_test"]["after_geometry_mae"]:.4f}</td></tr>' for arm,s in summaries.items())
    page='<!doctype html><meta charset="utf-8"><title>No-training scale comparison</title><style>'+STYLE+'td,th{padding:8px;border:1px solid #334552;text-align:left}</style><header><a href="../main_engine_qwen_outputs/index.html">Current production viewer</a><h1>No-training algorithms · 134 distinct resize tasks</h1><p>All 89 held-out reference pairs are covered (83 distinct source/target-size tasks) alongside the regression cohort. Qwen enabled, exact saved predictions, no new training or inference. Each algorithm competes independently against the retained current result. Only changed outputs appear below. Zero flags is not aesthetic perfection.</p><table><tr><th>Arm</th><th>Generated</th><th>Rendered</th><th>Changed</th><th>89-pair geometry MAE</th></tr>'+rows+'</table><p><a href="report.json">All 134 paired results and per-ratio reference metrics</a>. Main content stays Original / Resized; switch the resized image to inspect its current-production baseline.</p></header><main>'+''.join(cards)+'</main>'
    (a.output/'index.html').write_text(page)
    report=dict(tasks=len(results),test_pairs=len(aliases),unique_test_tasks=len({r['task_key'] for r in aliases}),qwen_enabled=True,fresh_inference=False,production_changed=False,arms=summaries,
        scope='Development replay using consumed held-out pairs; geometry reference is evaluation-only, never used for candidate generation or selection. Automated improvement needs visual inspection.',
        visual_review_pending=not (a.root/'visual_decisions.json').is_file())
    for dest in (a.root/'report.json',a.output/'report.json'):dest.write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(paired,indent=2)+'\n')
    sheet=Image.new('RGB',(1400,max(1,len(contact))*390),'#182532');draw=ImageDraw.Draw(sheet)
    for n,(c,paths) in enumerate(contact):
        draw.text((15,n*390+8),f'{c["index"]:03d} {c["arm"]} {c["pair_id"]}',fill='white')
        for col,label in enumerate(('before','after')):
            with Image.open(a.output/paths[label]) as im:
                im=im.convert('RGB');im.thumbnail((680,340));sheet.paste(im,(col*700+(700-im.width)//2,n*390+40+(340-im.height)//2))
            draw.text((col*700+8,n*390+26),label,fill='white')
    sheet.save(a.root/'changed_contact.jpg',quality=90)
    print(json.dumps({arm:{k:v for k,v in s.items() if k not in ('changes','reference_pairs')} for arm,s in summaries.items()},indent=2))


if __name__=='__main__':main()
