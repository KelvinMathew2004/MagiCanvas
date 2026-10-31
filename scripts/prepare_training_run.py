#!/usr/bin/env python3
"""Freeze current v2 supervision, audit actual labels, export a portable CUDA bundle.

No target line count is given to the model. No test example enters the bundle.
Native visibility warnings require independent per-element confirmation before
exclusion. Existing source invisibility is not called a new converter defect.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from functools import lru_cache
from tempfile import TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,math,os,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/training_next'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


@lru_cache(maxsize=1)
def sources():
    from scripts.evaluate_qwen_adapter import read_sources
    return read_sources()


def supervised_target(row,source):
    from engine.qwen_prompt import decode_v2_completion
    decoded=decode_v2_completion(json.loads(row['messages'][-1]['content']),source,row['target_canvas'],2)
    elements=decoded['elements'];updates={e['id']:e for e in elements}
    expected={e['id'] for e in source['elements']}
    if len(updates)!=len(elements) or set(updates)!=expected:raise ValueError('Supervision ID mismatch')
    target=deepcopy(source);target['canvas']=dict(row['target_canvas'])
    for e in target['elements']:
        if 'font_size' in updates[e['id']]:
            # Match runtime tracking scaling before judging label wrapping or
            # paint. The answer predicts font size, not new letter spacing.
            old=float(e.get('font_size',1))
            e['letter_spacing']=float(e.get('letter_spacing',0))*updates[e['id']]['font_size']/max(1e-6,old)
        e.update(updates[e['id']])
        for key in ('x','y','width','height'):
            if not math.isfinite(e[key]):raise ValueError('Nonfinite geometry')
        if e['width']<=0 or e['height']<=0:raise ValueError('Nonpositive label extent')
    return target


def visible_contribution(doc,eid,split):
    """Full-vs-leave-one-out and isolated intended mask; native dimensions."""
    import numpy as np
    from PIL import Image,ImageChops
    from scripts.dataset_renderer import render_design_to_png
    def render(d,path):
        d=deepcopy(d);d['elements']=[e for e in d['elements'] if min(e['width'],e['height'])>0]
        render_design_to_png(d,ROOT/f'data/crello_{split}',path,ROOT/'data/crello_fonts.pickle',
            max_size=max(d['canvas'].values()),adjust_tight_text_widths=False)
        with Image.open(path) as im:return im.convert('RGB').copy()
    with TemporaryDirectory() as folder:
        path=Path(folder)/'r.png';full=render(doc,path)
        without=render(dict(doc,elements=[e for e in doc['elements'] if e['id']!=eid]),path)
        e=deepcopy(next(e for e in doc['elements'] if e['id']==eid));e.update(color='#ffffff',opacity=1.,z_index=1)
        for run in e.get('style_runs',[]):run['color']='#ffffff'
        bg=dict(id='__audit_bg',type='shape',shape_type='rectangle',x=0,y=0,width=doc['canvas']['width'],
            height=doc['canvas']['height'],fill_color='#000000',z_index=-1000,opacity=1,rotation=0)
        mask=np.asarray(render(dict(doc,elements=[bg,e]),path).convert('L'))>12
        diff=np.asarray(ImageChops.difference(full,without).convert('L'))>12
        return dict(intended=int(mask.sum()),visible=int((mask&diff).sum()))


def audit(task):
    split,row,cache,environment=task
    os.environ['MAGICANVAS_DATASET_RENDERING']='1'
    source=sources()(row['source_split'],row['source_template_id']);target=supervised_target(row,source)
    key=sha256(json.dumps(dict(source=source,row=row,audit_version=1,environment=environment),sort_keys=True).encode()).hexdigest()
    path=Path(cache)/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    from engine.resize import _text_info
    from engine.raster_metrics import glyph_overflow
    from engine.nested_structure import unpainted_ids
    from scripts.audit_upgrade_rasters import paint_stats
    inactive=unpainted_ids(source)
    source_over=glyph_overflow(source,source['elements'],**source['canvas'])
    target_over=glyph_overflow(source,target['elements'],**target['canvas'])
    a=paint_stats(source,source,row['source_split']);b=paint_stats(target,source,row['source_split'])
    confirmed=[];ownership={};review=[]
    originals={e['id']:e for e in source['elements']}
    for e in target['elements']:
        if e['type']!='text' or e['id'] in inactive:continue
        eid=e['id'];old=originals[eid]
        # A shared-mask contrast/visibility result is a trigger, never proof.
        stat=b['text'].get(eid,{})
        if stat.get('ink_pixels',0)==0 or stat.get('contrast') is None or stat['contrast']<1.15:
            own=visible_contribution(target,eid,row['source_split'])
            prior=visible_contribution(source,eid,row['source_split'])
            ownership[eid]=dict(source=prior,target=own)
            if prior['visible']>=8 and own['visible']==0:confirmed.append(eid)
        normalized_new=target_over.get(eid,0)/max(1,e['font_size'])-source_over.get(eid,0)/max(1,old['font_size'])
        if normalized_new>.25:review.append(eid)
    result=dict(pair_id=row['pair_id'],split=split,audit_sha256=key,
        confirmed_new_invisible=confirmed,new_severe_overflow=review,ownership=ownership,
        target_lines={e['id']:_text_info(e)[2] for e in target['elements'] if e['type']=='text' and e['id'] not in inactive},
        source_overflow=source_over,target_overflow=target_over,
        native_screen=b,source_native_screen=a,
        quality_weight=.5 if review else 1.,exclude=bool(confirmed))
    save(path,result);return result


def audit_environment(rows):
    """Bind resumable native audits to the code, fonts and actual source assets."""
    files=[Path(__file__),ROOT/'scripts/dataset_renderer.py',ROOT/'scripts/audit_upgrade_rasters.py',
        ROOT/'data/crello_fonts.pickle']
    files+=sorted((ROOT/'engine').glob('*.py'))+sorted((ROOT/'renderer/converter').glob('*.py'))
    files+=sorted((ROOT/'brand_kit/fonts').glob('*'))
    assets=set()
    for row in rows:
        source=sources()(row['source_split'],row['source_template_id'])
        for e in source['elements']:
            if e.get('type')=='image' and e.get('asset_id'):
                assets.add(ROOT/f"data/crello_{row['source_split']}"/e['asset_id'])
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
            for p in sorted(set(files)|assets)}
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('Pillow','numpy','skia-python','cr-renderer')}
    return dict(files=hashes,versions=versions)


def prepare(workers):
    from transformers import AutoTokenizer
    from engine.qwen_prompt import messages_for
    data=ROOT/'data/qwen_lora_expanded_v2'
    rows={s:[json.loads(x) for x in (data/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation','test')}
    families={s:{r['family_id'] for r in rr} for s,rr in rows.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if families[a]&families[b]:raise ValueError('Family split leakage')
    targets=[r['target_template_id'] for r in rows['train']]
    if len(set(targets))!=len(targets):raise ValueError('Training target cap violated')
    model_specs=[dict(key='qwen3-0.6b',repo='Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca'),
        dict(key='qwen3-1.7b',repo='Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')]
    tokenizers=[AutoTokenizer.from_pretrained(ROOT/'models'/s['key'],local_files_only=True) for s in model_specs]
    for spec,tok in zip(model_specs,tokenizers):spec['tokenizer_sha256']=sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
    environment=audit_environment(rows['train']+rows['validation'])
    environment_key=sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
    cache=OUT/'audit'/environment_key
    save(cache/'environment.json',environment)
    tasks=[(s,r,str(cache),environment_key) for s in ('train','validation') for r in rows[s]];audits={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(audit,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();audits[(r['split'],r['pair_id'])]=r
            if i%25==0:print(f'Native supervision audit {i}/{len(tasks)}',flush=True)
    from training.train import encode
    bundle=OUT/'bundle';bundle.mkdir(parents=True,exist_ok=True)
    count={};sidecar=[];lengths={s:[] for s in ('train','validation')}
    for split in ('train','validation'):
        selected=[]
        for row in rows[split]:
            original=row;row=deepcopy(row);report=audits[(split,row['pair_id'])];sidecar.append(report)
            if split=='train' and report['exclude']:continue
            source=sources()(row['source_split'],row['source_template_id'])
            prompts=[]
            for tok in tokenizers:
                messages=messages_for(source,row['target_canvas'],version=2)
                prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                if len(source['elements'])>16 and len(tok(prompt,add_special_tokens=False)['input_ids'])>4096:
                    messages=messages_for(source,row['target_canvas'],version=2,
                        relation_token_counter=lambda value:len(tok(value,add_special_tokens=False)['input_ids']),relation_token_budget=900)
                prompts.append(messages)
            if prompts[0]!=prompts[1]:raise ValueError('Models need different canonical prompts; separate data comparison required')
            row['messages']=prompts[0]+[deepcopy(original['messages'][-1])]
            row['n_elements']=len(source['elements']);row['quality_weight']=report['quality_weight'] if split=='train' else 1.
            row['sampling_repeat']=3 if split=='train' and row['n_elements']>=18 else 1
            for tok in tokenizers:lengths[split].append(len(encode(row,tok)['input_ids']))
            selected.append(row)
        (bundle/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        count[split]=dict(input_rows=len(rows[split]),rows=len(selected),dense_unique=sum(r['n_elements']>=18 for r in selected),
            epoch_visits=sum(r['sampling_repeat'] for r in selected),downweighted=sum(r['quality_weight']<1 for r in selected),
            max_tokens=max(lengths[split]),excluded_confirmed_invisible=len(rows[split])-len(selected))
    import shutil
    for name in ('train.py','requirements.txt'):shutil.copy2(ROOT/'training'/name,bundle/name)
    save(OUT/'audit_summary.json',dict(splits=count,native_audited=len(audits),family_overlap=0,
        strict_unique_train_targets=True,line_counts='Metadata only, never prompt input or an altered output schema',
        audit='Actual supervised geometry rendered with source assets/styles. Shared masks trigger per-element verification; clipping warnings downweighted, not deleted.',
        audit_environment_sha256=environment_key,
        source_hashes={s:sha256((data/f'{s}.jsonl').read_bytes()).hexdigest() for s in rows}))
    save(OUT/'line_count_sidecars.json',sidecar)
    manifest=dict(models=model_specs,splits=count,prompt_version=2,max_length=6144,audit_environment_sha256=environment_key,
        recipe=dict(epochs=3,r=16,lora_alpha=32,learning_rate=1e-4,gradient_accumulation=16,
            dense_repeat=3,seed=3407,response_only=True,base='fresh pinned base, not old adapter',
            optimizer='adamw_torch',optimizer_foreach=False,precision='bf16 if natively supported, otherwise fp16',
            minimum_cuda_memory_gib=14,feasibility='Per-model forward/backward/AdamW preflight; no truncation or silent model skipping'),
        bundle_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file() and p.name!='manifest.json'})
    save(bundle/'manifest.json',manifest)
    with zipfile.ZipFile(OUT/'layout_training_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.iterdir()):z.write(p,p.name)
    print(json.dumps(dict(bundle=str(OUT/'layout_training_bundle.zip'),summary=count),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=2);a=p.parse_args();prepare(a.workers)
