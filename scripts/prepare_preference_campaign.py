#!/usr/bin/env python3
"""Mine hundreds of diverse actual-engine comparisons, with resumable provenance.

No engine modifications, Qwen-off arm, synthetic shrink negatives or labels.
New authored Level-3-like sources complement family-separated dataset sources.
"""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict,Counter
from copy import deepcopy
from unittest.mock import patch
import argparse,importlib,inspect,json,os,random,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.preference_campaign_sources import authored_sources,FORMATS
from scripts.preference_campaign_view import publish


def tasks_for_campaign():
    records=[json.loads(x) for x in (ROOT/'data/qwen_lora_expanded_v2/train.jsonl').read_text().splitlines()]
    forbidden=set();families=set()
    for split in ('test','validation'):
        for r in map(json.loads,(ROOT/f'data/qwen_lora_expanded_v2/{split}.jsonl').read_text().splitlines()):
            forbidden.update(r['pair_id'].replace('__to__','__').split('__'));families.add(r['family_id'])
    index={}
    for split in ('train','validation','test'):
        index[split]={r['template_id']:r for r in map(json.loads,(ROOT/f'data/crello_{split}/manifest.jsonl').read_text().splitlines()) if r.get('status')=='exported'}
    old_ids=set()
    for manifest in ('data/siglip_human_review/manifest.jsonl','data/siglip_engine_review_v2/manifest.jsonl'):
        for r in map(json.loads,(ROOT/manifest).read_text().splitlines()):old_ids.update((r['source_id'],r['target_id']))
    grouped=defaultdict(list)
    for r in records:
        ids=r['pair_id'].replace('__to__','__').split('__')
        if set(ids)&forbidden or r['family_id'] in families or set(ids)&old_ids:continue
        grouped[r['family_id']].append(r)
    dataset=[]
    # Hash ordering avoids selecting easy-looking outputs. Buckets below use
    # source structure before running the model, never resized human geometry.
    for family in sorted(grouped,key=lambda k:sha256(k.encode()).hexdigest()):
        for r in sorted(grouped[family],key=lambda k:k['pair_id']):
            sid=r['source_template_id'];split=r['source_split'];meta=index[split].get(sid)
            if not meta:continue
            source=json.loads((ROOT/f'data/crello_{split}'/meta['design']).read_text());n=len(source['elements'])
            if not 7<=n<=28 or sum(e['type']=='text' for e in source['elements'])<3:continue
            dataset.append(dict(source_id=sid,family_id=family,source_split=split,source=source,category='dataset',title=f'Dataset family / {sid}',provenance='Model-training family, excluded from Qwen validation/test and previous human batches',element_count=n));break
    # Round-robin sparse/middle/dense inputs before selecting target aspect.
    buckets=defaultdict(list)
    for item in dataset:buckets[min(2,(item['element_count']-7)//7)].append(item)
    ordered=[]
    while any(buckets.values()):
        for key in (2,1,0):
            if buckets[key]:ordered.append(buckets[key].pop(0))
    dataset=[]
    for i,item in enumerate(ordered):
        name,w,h=FORMATS[i%len(FORMATS)]
        dataset.append(dict(item,format=name,target_canvas=dict(width=w,height=h),task_id='dataset-'+item['source_id']+'-'+name))
    novel=[]
    for item in authored_sources():
        for name,w,h in item['targets']:
            novel.append(dict(item,format=name,target_canvas=dict(width=w,height=h),task_id=item['source_id']+'-'+name))
    tasks=[]
    while novel or dataset:
        if novel:tasks.append(novel.pop(0))
        for _ in range(2):
            if dataset:tasks.append(dataset.pop(0))
    return tasks


def distance(a,b):
    old={e['id']:e for e in a['elements']};w,h=a['canvas']['width'],a['canvas']['height'];total=0.
    for e in b['elements']:
        o=old[e['id']]
        total+=sum(abs(e[k]-o[k])/d for k,d in (('x',w),('y',h),('width',w),('height',h)))
        if e['type']=='text':total+=.05*abs(e['font_size']-o['font_size'])/max(1,o['font_size'])
    return total/max(1,len(old))


def generate(task,engine,fp):
    source=task['source'];capture={};measurements={};assemble=engine._assemble;visual=engine._visual_rerank;started=time.perf_counter()
    def collect(*a,**kw):
        doc=assemble(*a,**kw);caller=inspect.currentframe().f_back
        if caller.f_code.co_name=='_visual_rerank':
            candidate=caller.f_locals.get('candidate',{})
            if candidate.get('name'):capture[candidate['name']]=deepcopy(doc)
        return doc
    def collect_visual(*a,**kw):
        selected,report=visual(*a,**kw)
        for row in report.get('candidates',[]):measurements[row['name']]=deepcopy(row)
        return selected,report
    with patch.object(engine,'_assemble',collect),patch.object(engine,'_visual_rerank',collect_visual):winner,diagnostics=engine._adapt_with_diagnostics(source,task['target_canvas'])
    _,units=engine._analyze(source);ids={e['id'] for e in source['elements']};foreground={e['id'] for u in units for e in u['items']}
    pool={};key_w=json.dumps(winner,sort_keys=True);w,h=winner['canvas']['width'],winner['canvas']['height']
    for name,doc in capture.items():
        if doc['canvas']!=winner['canvas'] or len(doc['elements'])!=len(ids) or {e['id'] for e in doc['elements']}!=ids:continue
        key=json.dumps(doc,sort_keys=True)
        if key==key_w or distance(winner,doc)<.004:continue
        pool[key]=(doc,measurements.get(name,{}))
    from engine.macro_layout import quality
    def eligible(doc,metrics):
        return bool(metrics.get('physical_eligible',False) and not metrics.get('structural_errors') and not metrics.get('invisible_text') and not metrics.get('glyph_overflow'))
    winner_metrics=measurements.get(diagnostics['winner'],{})
    winner_safe=eligible(winner,winner_metrics)
    ranked=sorted(pool.values(),key=lambda item:(not eligible(*item),item[1].get('layout_defect_cost',1e9),distance(winner,item[0])))
    selected=[]
    if ranked:
        selected.append(ranked[0])
        safe=[item for item in ranked[1:] if eligible(*item)]
        if safe:selected.append(max(safe,key=lambda item:distance(selected[0][0],item[0])))
    pairs=[]
    if selected:pairs.append(((winner,winner_metrics),selected[0]))
    if len(selected)>1:pairs.extend((((winner,winner_metrics),selected[1]),(selected[0],selected[1])))
    return dict(task={k:v for k,v in task.items() if k!='source'},source=source,winner=winner,pairs=[dict(a=a,b=b,a_training_eligible=eligible(a,am),b_training_eligible=eligible(b,bm),a_quality=quality(a['elements'],source,w,h),b_quality=quality(b['elements'],source,w,h),a_metrics=am,b_metrics=bm) for (a,am),(b,bm) in pairs],diagnostics=diagnostics,engine_fingerprint=fp,wall_seconds=time.perf_counter()-started)


def run(args):
    os.environ.update(MAGICANVAS_QWEN_ENABLED='1',MAGICANVAS_DATASET_RENDERING='1',MAGICANVAS_CPU_THREADS='1')
    import torch
    torch.set_num_threads(1)
    engine=importlib.import_module('engine.resize')
    from scripts.dataset_renderer import render_design_to_png
    from PIL import Image
    tasks=tasks_for_campaign()[:args.max_sources];folder=args.output.resolve()
    for name in ('images','audits','sources'):(folder/name).mkdir(parents=True,exist_ok=True)
    fp={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'engine').glob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/preference_campaign_sources.py']}
    provenance=dict(engine_fingerprint=fp,task_hash=sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),adapter_sha256=sha256((ROOT/'models/qwen3-layout-lora-v2/adapter_model.safetensors').read_bytes()).hexdigest(),fresh_qwen_inference=True,qwen_enabled=True,labels_created=0,target_pairs=args.target_pairs,max_sources=args.max_sources)
    marker=folder/'run_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=provenance:raise ValueError('Campaign inputs changed; choose a new output directory')
    marker.write_text(json.dumps(provenance,indent=2)+'\n');rows=[];seen_pairs=set();unique=0
    for i,task in enumerate(tasks):
        checkpoint=folder/'audits'/(task['task_id']+'.json')
        if checkpoint.exists():saved=json.loads(checkpoint.read_text())
        else:
            saved=generate(task,engine,fp);checkpoint.write_text(json.dumps(saved)+'\n')
        source_path=folder/'sources'/(task['source_id']+'.png')
        assets=ROOT/f'data/crello_{task["source_split"]}'
        if not source_path.exists():render_design_to_png(task['source'],assets,source_path,ROOT/'data/crello_fonts.pickle',max_size=1100,adjust_tight_text_widths=False)
        rendered={}
        for j,pair in enumerate(saved['pairs']):
            if unique>=args.target_pairs:break
            paths=[];warnings=[]
            for side in ('a','b'):
                doc=pair[side];digest=sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest();path=folder/'images'/(digest[:24]+'.png')
                if digest not in rendered:
                    notes=render_design_to_png(doc,assets,path,ROOT/'data/crello_fonts.pickle',max_size=1100,adjust_tight_text_widths=False);rendered[digest]=(path,notes)
                path,notes=rendered[digest];paths.append(path);warnings+=notes
            if warnings:continue
            hashes=[sha256(p.read_bytes()).hexdigest() for p in paths];signature=tuple(sorted(hashes))
            if signature in seen_pairs or hashes[0]==hashes[1]:continue
            seen_pairs.add(signature);pid='campaign-v3-'+task['task_id']+f'-{j}'
            # Stable independent side randomization, unaffected by resume order.
            swapped=bool(int(sha256(pid.encode()).hexdigest()[:8],16)%2)
            target_id=task['source_id']+'-campaign-'+task['format']
            safe_a=pair.get('a_training_eligible',True);safe_b=pair.get('b_training_eligible',True)
            row=dict(pair_id=pid,source_id=task['source_id'],target_id=target_id,family_id=task['family_id'],source_split=task['source_split'],category=task['category'],format=task['format'],positive_images=[str(paths[0])],engine_image=str(paths[1]),status='unreviewed_engine_candidate' if safe_a or safe_b else 'review_only_constraint_failure',a_training_eligible=safe_a,b_training_eligible=safe_b,review_hash=sha256(paths[0].read_bytes()+paths[1].read_bytes()).hexdigest(),display_swapped=swapped,source_image=str(source_path.relative_to(folder)),a_quality=pair['a_quality'],b_quality=pair['b_quality'],candidate_provenance='Two complete actually rendered engine realizations from a Qwen-enabled run; rejected-candidate diagnostics never override production hard gates',audit=str(checkpoint),title=task['title'])
            rows.append(row);unique+=1
            if unique%20==0:
                repeat=deepcopy(row);repeat.update(pair_id=pid+'-repeat',reliability_repeat_of=pid,status='reliability_check',display_swapped=not swapped);rows.append(repeat)
        summary=dict(unique_comparisons=unique,reliability_checks=len(rows)-unique,completed_tasks=i+1,max_sources=len(tasks),target_pairs=args.target_pairs,families=len({r['family_id'] for r in rows}),ranking_comparisons=sum(r['status']=='unreviewed_engine_candidate' for r in rows),constraint_diagnostics=sum(r['status']=='review_only_constraint_failure' for r in rows),categories=dict(Counter(r['category'] for r in rows if not r.get('reliability_repeat_of'))),formats=dict(Counter(r['format'] for r in rows if not r.get('reliability_repeat_of'))),labels_created=0,complete=unique>=args.target_pairs or i+1==len(tasks))
        (folder/'manifest.jsonl.next').write_text(''.join(json.dumps(r)+'\n' for r in rows));(folder/'manifest.jsonl.next').replace(folder/'manifest.jsonl')
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        publish(folder,rows,f'{unique} distinct comparisons + {len(rows)-unique} consistency repeats ready. {i+1}/{len(tasks)} source/target runs processed. '+('Generation complete.' if summary['complete'] else 'Generation continues; reload to load more. Reviewed image pairs stay unchanged.'))
        print(json.dumps(dict(task=i+1,task_id=task['task_id'],pairs=unique,category=task['category'],seconds=round(saved['wall_seconds'],2),winner=saved['diagnostics']['winner'])),flush=True)
        if unique>=args.target_pairs:break
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/preferences_campaign');p.add_argument('--target-pairs',type=int,default=240);p.add_argument('--max-sources',type=int,default=110);run(p.parse_args())
