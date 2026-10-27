#!/usr/bin/env python3
"""Prepare real, unlabelled same-renderer comparisons; never invent approvals.

Human target JSON and controlled corruptions are both rendered locally. Fidelity
failures remain available for review but are ineligible for training. Sources
come from training families, independently of the 38-case regression cohort.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,sys,html,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def build(output,limit=60):
    from scripts.dataset_renderer import render_design_to_png
    from scripts.compare_crello_renders import compare_images
    from PIL import Image
    data=ROOT/'data/crello_train';index={r['template_id']:r for r in map(json.loads,(data/'manifest.jsonl').read_text().splitlines()) if r.get('status')=='exported'}
    old=[json.loads(s) for s in (ROOT/'data/siglip_negative_pairs/manifest.jsonl').read_text().splitlines() if s.strip()]
    parent={}
    def find(i):
        parent.setdefault(i,i)
        if parent[i]!=i:parent[i]=find(parent[i])
        return parent[i]
    for r in old:parent[find(r['source_id'])]=find(r['target_id'])
    buckets={}
    for r in old:buckets.setdefault(find(r['source_id']),[]).append(r)
    ordered=[];seen=set()
    while any(buckets.values()):
        for family in sorted(buckets):
            if not buckets[family]:continue
            r=buckets[family].pop(0)
            if r['target_id'] in seen:continue
            seen.add(r['target_id']);ordered.append(r)
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True);entries=[]
    for i,r in enumerate(ordered[:limit]):
        meta=index.get(r['target_id'])
        if not meta:continue
        target=json.loads((data/meta['design']).read_text());bad=deepcopy(target);w,h=target['canvas']['width'],target['canvas']['height']
        # A controlled negative need not be bad to every human; all comparisons
        # are unlabelled until the reviewer explicitly chooses a preference.
        for e in bad['elements']:
            if e['type']!='text' and e['width']*e['height']>=.80*w*h:continue
            scale=.54 if i%2==0 else .72
            e.update(x=w/2+(e['x']-w/2)*scale,y=h/2+(e['y']-h/2)*scale,width=e['width']*scale,height=e['height']*scale)
            if e['type']=='text':e['font_size']*=scale;e['letter_spacing']=e.get('letter_spacing',0)*scale
        paths=[output/'images'/f'{i:02d}_{kind}.png' for kind in ('target','candidate')];warnings=[]
        for doc,path in zip((target,bad),paths):warnings+=render_design_to_png(doc,data,path,ROOT/'data/crello_fonts.pickle',max_size=1100,adjust_tight_text_widths=False)
        preview=Path(r['positive_image']);fidelity={}
        if preview.exists():
            with Image.open(paths[0]) as a,Image.open(preview) as b:
                sx,sy=b.width/w,b.height/h
                regions=[(e['x']*sx,e['y']*sy,(e['x']+e['width'])*sx,(e['y']+e['height'])*sy) for e in target['elements'] if e['type']=='text']
                fidelity=compare_images(a,b,regions)
        eligible=not warnings and fidelity.get('global_ssim',0)>=.90 and fidelity.get('text_region_ssim_mean',0)>=.65 and fidelity.get('text_region_ssim_p10',0)>=.35
        item=dict(pair_id=r['source_id']+'__'+r['target_id'],source_id=r['source_id'],target_id=r['target_id'],family_id=find(r['source_id']),target_canvas=target['canvas'],positive_images=[str(paths[0].resolve())],engine_image=str(paths[1].resolve()),status='candidate_without_measured_defect' if eligible else 'review_only_fidelity_failure',renderer='dataset_renderer; adjust_tight_text_widths=False',candidate_provenance='controlled proportional foreground shrink; not a current engine result',fidelity=fidelity,warnings=warnings,review_hash=sha256(paths[0].read_bytes()+paths[1].read_bytes()).hexdigest())
        entries.append(item);print(i+1,'eligible',eligible,flush=True)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
    views=[dict(pair_id=e['pair_id'],a=str(Path(e['positive_images'][0]).relative_to(output.resolve())),b=str(Path(e['engine_image']).relative_to(output.resolve())),review_hash=e['review_hash'],eligible=e['status']=='candidate_without_measured_defect',family=e['family_id']) for e in entries]
    payload=json.dumps(views).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>SigLIP2 human preferences</title><style>body{font:17px system-ui;background:#101820;color:white;margin:24px}img{width:100%;height:65vh;object-fit:contain}#pair{display:grid;grid-template-columns:1fr 1fr}button{padding:12px;margin:6px}p{max-width:1000px}</style><h1>Human preference review — no automatic labels</h1><p>Both images use the same local renderer. One is the human-authored target; the other is a controlled scale corruption, not a fresh engine output. Choose your actual preference or skip. Fidelity failures cannot enter training even if labelled. Choices are saved only in this browser until you export JSONL.</p><p id="status"></p><div id="pair"><figure><figcaption>A · local target</figcaption><img id="a"></figure><figure><figcaption>B · candidate</figcaption><img id="b"></figure></div><button id="aWins">Prefer A</button><button id="bWins">Prefer B</button><button id="skip">Tie / skip</button><button id="prev">Previous</button><button id="next">Next</button><button id="export">Export reviewed labels</button><script>const rows=__ROWS__;let i=0;const key='magicanvas-siglip-reviewed-v1';let labels=JSON.parse(localStorage.getItem(key)||'{}');function show(){let r=rows[i];document.getElementById('a').src=r.a;document.getElementById('b').src=r.b;document.getElementById('status').textContent=`${i+1}/${rows.length} · ${r.pair_id} · ${r.eligible?'fidelity gate passed':'REVIEW ONLY: fidelity gate failed'} · choice: ${labels[r.pair_id]?.decision||'unreviewed'}`;}function vote(decision){const r=rows[i];labels[r.pair_id]={pair_id:r.pair_id,decision,review_hash:r.review_hash,reviewed_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(labels));i=Math.min(rows.length-1,i+1);show()}document.getElementById('aWins').onclick=()=>vote('accept');document.getElementById('bWins').onclick=()=>vote('prefer_candidate');document.getElementById('skip').onclick=()=>vote('skip');document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};document.getElementById('next').onclick=()=>{i=Math.min(rows.length-1,i+1);show()};document.getElementById('export').onclick=()=>{const blob=new Blob([Object.values(labels).filter(x=>rows.some(r=>r.pair_id===x.pair_id&&r.review_hash===x.review_hash)).map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{type:'application/x-ndjson'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='siglip-human-approvals.jsonl';a.click();URL.revokeObjectURL(a.href)};show();</script>'''.replace('__ROWS__',payload)
    (output/'review.html').write_text(page)
    print(json.dumps(dict(comparisons=len(entries),eligible=sum(e['status']=='candidate_without_measured_defect' for e in entries),labels_created=0,page=str(output/'review.html'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/siglip_human_review');p.add_argument('--limit',type=int,default=60);a=p.parse_args();build(a.output.resolve(),a.limit)
