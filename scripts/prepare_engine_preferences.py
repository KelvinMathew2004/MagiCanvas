#!/usr/bin/env python3
"""Unlabelled, randomized preferences between actual Qwen-enabled candidates.

Use training families only. Capture complete engine realizations without
altering their generation or selection; compare the winner with alternatives.
Neither side is automatically declared correct. These are not test-set labels.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch
import argparse, html, json, os, random, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def build(args):
    os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    os.environ['MAGICANVAS_DATASET_RENDERING']='1'
    os.environ['MAGICANVAS_DATASET_ASSETS_DIR']=str(ROOT/'data/crello_train')
    import torch
    torch.set_num_threads(1)
    import importlib
    engine=importlib.import_module('engine.resize')
    from scripts.dataset_renderer import render_design_to_png
    from engine import qwen_seed
    data=ROOT/'data/crello_train'
    indexes={r['template_id']:r for r in map(json.loads,(data/'manifest.jsonl').read_text().splitlines()) if r.get('status')=='exported'}
    inventory=[r for r in map(json.loads,(ROOT/'data/siglip_human_review/manifest.jsonl').read_text().splitlines()) if r['status']=='candidate_without_measured_defect']
    choices=[];seen=set()
    for r in inventory:
        if r['family_id'] in seen or r['source_id'] not in indexes:continue
        source=json.loads((data/indexes[r['source_id']]['design']).read_text())
        if not 5<=len(source['elements'])<=25:continue
        choices.append((r,source));seen.add(r['family_id'])
        if len(choices)==args.sources:break
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'images').mkdir(exist_ok=True);(args.output/'audits').mkdir(exist_ok=True)
    fingerprints={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
    rows=[];rng=random.Random(2710)
    for i,(r,source) in enumerate(choices):
        checkpoint=args.output/'audits'/f'{i:02d}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if saved['engine_sha256']!=fingerprints:raise ValueError('Engine changed; use a new output folder')
        else:
            capture=[];assemble=engine._assemble
            def collect(*a,**kw):
                doc=assemble(*a,**kw);capture.append(deepcopy(doc));return doc
            with patch.object(engine,'_assemble',collect):
                winner,diagnostics=engine._adapt_with_diagnostics(source,r['target_canvas'])
            if 'Qwen' not in str(diagnostics.get('qwen','')):
                raise RuntimeError('Expected Qwen-enabled proposal attempt')
            _,units=engine._analyze(source);ids={e['id'] for e in source['elements']}
            pool={}
            for doc in capture:
                if {e['id'] for e in doc['elements']}!=ids:continue
                fg=[e for e in doc['elements'] if e['id'] in {z['id'] for u in units for z in u['items']}]
                if not engine._physical_layout_safe(fg,units,source,winner['canvas']['width'],winner['canvas']['height']):continue
                key=json.dumps(doc,sort_keys=True)
                if key==json.dumps(winner,sort_keys=True):continue
                pool[key]=doc
            wb={e['id']:e for e in winner['elements']};w,h=winner['canvas']['width'],winner['canvas']['height']
            def distance(doc):
                return sum(abs(e[k]-wb[e['id']][k])/d for e in doc['elements'] for k,d in (('x',w),('y',h),('width',w),('height',h)))/len(ids)
            candidates=sorted(((distance(doc),doc) for doc in pool.values() if distance(doc)>.005),key=lambda x:x[0])
            positions=sorted({0,len(candidates)//2,len(candidates)-1}) if candidates else []
            selected=[candidates[j] for j in positions]
            saved=dict(engine_sha256=fingerprints,source=source,winner=winner,diagnostics=diagnostics,
                       alternatives=[dict(distance=d,design=doc) for d,doc in selected],family_id=r['family_id'])
            checkpoint.write_text(json.dumps(saved)+'\n')
        for j,alternative in enumerate(saved['alternatives']):
            pair=f"engine-v2-{i:02d}-{j:02d}";a=args.output/'images'/f'{pair}_incumbent.png';b=args.output/'images'/f'{pair}_alternative.png'
            warnings=[]
            for doc,path in ((saved['winner'],a),(alternative['design'],b)):
                warnings+=render_design_to_png(doc,data,path,ROOT/'data/crello_fonts.pickle',max_size=1100,adjust_tight_text_widths=False)
            swap=bool(rng.getrandbits(1))
            rows.append(dict(pair_id=pair,source_id=r['source_id'],target_id=r['target_id'],family_id=r['family_id'],positive_images=[str(a.resolve())],engine_image=str(b.resolve()),status='unreviewed_engine_candidate' if not warnings else 'review_only_renderer_warning',review_hash=sha256(a.read_bytes()+b.read_bytes()).hexdigest(),display_swapped=swap,warnings=warnings,candidate_provenance='Actual current Qwen-enabled engine winner versus a complete generated realization; no human-reference or synthetic scale corruption',geometry_distance=alternative['distance'],incumbent=saved['diagnostics']['winner'],qwen_status=saved['diagnostics'].get('qwen'),audit=str(checkpoint.resolve())))
        print(i+1,len(choices),'actual Qwen-enabled sources; pairs',len(rows),flush=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    payload=[]
    for r in rows:
        a,b=Path(r['positive_images'][0]).relative_to(args.output),Path(r['engine_image']).relative_to(args.output)
        if r['display_swapped']:a,b=b,a
        payload.append(dict(pair_id=r['pair_id'],a=str(a),b=str(b),swapped=r['display_swapped'],review_hash=r['review_hash'],eligible=not r['warnings']))
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual engine candidate preferences</title><style>body{font:17px system-ui;background:#101820;color:white;margin:24px}img{width:100%;height:65vh;object-fit:contain}#pair{display:grid;grid-template-columns:1fr 1fr}button{padding:12px;margin:6px}@media(max-width:700px){#pair{grid-template-columns:1fr}}a{color:#9ed4ff}</style><h1>Actual engine candidate preferences</h1><p>These are different arrangements generated by the current Qwen-enabled engine on training families. Neither panel is the original human reference. A/B order is randomized. Compare readability, alignment, grouping, spacing, visual hierarchy and use of space. Choose Tie / neither when both are poor or no preference is clear. This is not a guarantee that the engine winner is better.</p><p><a href="../preferences_occupancy/review.html">Earlier occupancy-only batch</a> · <a href="../regression_suite/index.html">Unified 38-case viewer</a></p><p id="status"></p><div id="pair"><figure><figcaption>A</figcaption><img id="a"></figure><figure><figcaption>B</figcaption><img id="b"></figure></div><button id="aWins">Prefer A</button><button id="bWins">Prefer B</button><button id="skip">Tie / neither</button><button id="prev">Previous</button><button id="next">Next</button><button id="export">Export reviewed labels</button><script>const rows=__ROWS__;let i=0;const key='magicanvas-siglip-engine-v2';let labels=JSON.parse(localStorage.getItem(key)||'{}');function show(){let r=rows[i];document.getElementById('a').src=r.a;document.getElementById('b').src=r.b;document.getElementById('status').textContent=`${i+1}/${rows.length} · ${r.eligible?'renderer checks passed':'REVIEW ONLY: renderer warning'} · choice: ${labels[r.pair_id]?.display_choice||'unreviewed'}`;}function vote(choice){const r=rows[i];let decision=choice==='skip'?'skip':((choice==='a')!==r.swapped?'accept':'prefer_candidate');labels[r.pair_id]={pair_id:r.pair_id,decision,display_choice:choice,review_hash:r.review_hash,reviewed_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(labels));i=Math.min(rows.length-1,i+1);show()}document.getElementById('aWins').onclick=()=>vote('a');document.getElementById('bWins').onclick=()=>vote('b');document.getElementById('skip').onclick=()=>vote('skip');document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};document.getElementById('next').onclick=()=>{i=Math.min(rows.length-1,i+1);show()};document.getElementById('export').onclick=()=>{const blob=new Blob([Object.values(labels).filter(x=>rows.some(r=>r.pair_id===x.pair_id&&r.review_hash===x.review_hash)).map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{type:'application/x-ndjson'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='siglip-engine-preferences-v2.jsonl';a.click();URL.revokeObjectURL(a.href)};show();</script>'''.replace('__ROWS__',json.dumps(payload).replace('</','<\\/'))
    if not rows:raise ValueError('No distinct complete candidates available')
    (args.output/'review.html').write_text(page)
    print(json.dumps(dict(sources=len(choices),comparisons=len(rows),labels_created=0,eligible=sum(not r['warnings'] for r in rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/siglip_engine_review_v2');p.add_argument('--sources',type=int,default=8);a=p.parse_args();a.output=a.output.resolve();build(a)
