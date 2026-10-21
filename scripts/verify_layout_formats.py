"""Render and check all official samples in six formats; development tool only."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.resize import _adapt_with_diagnostics, _bounds, _text
from engine.main import self_check
from renderer.converter.convert import render_design_to_png
from renderer.converter.assets import visible_content_bounds

FORMATS=(('story',1080,1920),('square',1080,1080),('portrait',1080,1350),
         ('widescreen',1920,1080),('web-banner',1600,400),('leaderboard',728,90))
MUTABLE={'x','y','width','height','font_size','letter_spacing','corner_radius','stroke_width','fit','text_align','rotation'}


def verify(source,target,result,report):
    problems=self_check(source,result)
    if result['canvas']!=target:problems.append('Target canvas differs')
    if [e['id'] for e in source['elements']]!=[e['id'] for e in result['elements']]:problems.append('Element IDs/order differ')
    if not report['hard_constraints_passed']:problems.append('Hard placement constraints failed')
    for old,e in zip(source['elements'],result['elements']):
        if {k:v for k,v in old.items() if k not in MUTABLE}!={k:v for k,v in e.items() if k not in MUTABLE}:
            problems.append(f"{e['id']}: fixed properties changed")
        if old.get('rotation',0)!=e.get('rotation',0):
            group=[item for item in source['elements'] if item.get('group_id')==old.get('group_id')]
            corner_callout=(target['width']/target['height']<.68 and old.get('group_id') and
                any(item['type']=='text' for item in group) and
                all(item['y']<source['canvas']['height']*.25 for item in group) and
                (min(item['x'] for item in group)<source['canvas']['width']*.2 or
                 max(item['x']+item['width'] for item in group)>source['canvas']['width']*.8))
            if not corner_callout or e.get('rotation')!=0:
                problems.append(f"{e['id']}: unexpected rotation change")
        b=_bounds([e])
        if not all(math.isfinite(v) for v in b) or b[0]<-1e-5 or b[1]<-1e-5 or b[2]>target['width']+1e-5 or b[3]>target['height']+1e-5:
            problems.append(f"{e['id']}: visual/raw bounds failed")
        if e['type']=='image':
            visible=visible_content_bounds(e,ROOT/'brand_kit'/'assets')
            if visible[0]<-1e-5 or visible[1]<-1e-5 or visible[2]>target['width']+1e-5 or visible[3]>target['height']+1e-5:
                problems.append(f"{e['id']}: visible pixels are off-canvas")
        if e['type']=='image' and e.get('fit')!='cover' and old['width']*old['height']:
            if not math.isclose(e['width']/e['height'],old['width']/old['height'],rel_tol=1e-6):problems.append(f"{e['id']}: image proportions changed")
        if e['type']=='text':
            if e['font_size']<1:problems.append(f"{e['id']}: font below 1px")
            h,w,_=_text(e)
            if h>e['height']+.01 or w>e['width']+.01:problems.append(f"{e['id']}: text overflow")
    return problems


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output'/'format-verification')
    parser.add_argument('--require-vision',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.require_vision:
        from engine.vision import available
        if not available():raise RuntimeError('Offline vision model could not load')
    results=[];all_errors=[];repeat=None
    for path in sorted((ROOT/'input').glob('*.json')):
        source=json.loads(path.read_text())['source'];before=deepcopy(source)
        for name,w,h in FORMATS:
            target=dict(width=w,height=h)
            output,report=_adapt_with_diagnostics(source,target)
            problems=verify(before,target,output,report)
            if source!=before:problems.append('Source mutated')
            stem=f'{path.stem}-{name}'
            (args.output/(stem+'.json')).write_text(json.dumps(output,indent=2))
            (args.output/(stem+'-diagnostics.json')).write_text(json.dumps(report,indent=2))
            render_design_to_png(output,ROOT/'brand_kit'/'assets',args.output/(stem+'.png'))
            row=dict(task=path.stem,format=name,winner=report['winner'],seconds=report['seconds'],
                     vision=report['visual'].get('perceptual',{}),errors=problems)
            results.append(row);all_errors.extend(f'{stem}: {p}' for p in problems)
            print(stem,'PASS' if not problems else problems,report['winner'],report['seconds']['total'],flush=True)
            if name=='web-banner' and 'level2' in path.stem:repeat=(deepcopy(source),target,output)
    deterministic=True
    if repeat:
        source,target,previous=repeat
        deterministic=_adapt_with_diagnostics(source,target)[0]==previous
        if not deterministic:all_errors.append('Repeated core banner output differed')
    summary=dict(combinations=len(results),passed=sum(not r['errors'] for r in results),
                 deterministic=deterministic,vision_used=sum(r['vision'].get('used',False) for r in results),
                 errors=all_errors,results=results)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)
    return 1 if all_errors else 0


if __name__=='__main__':raise SystemExit(main())
