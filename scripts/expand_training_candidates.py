#!/usr/bin/env python3
"""Add six pinned model packages and private source-PNG conditioning to the audit bundle."""
from pathlib import Path
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import argparse,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data/upgrade_exploration/deps'));sys.path.insert(0,str(ROOT))


def render_source(task):
    source,split,path=task
    from scripts.dataset_renderer import render_design_to_png
    if not Path(path).exists():
        d=deepcopy(source);d['elements']=[e for e in d['elements'] if min(e['width'],e['height'])>0]
        render_design_to_png(d,ROOT/f'data/crello_{split}',path,ROOT/'data/crello_fonts.pickle',max_size=768,adjust_tight_text_widths=False)
    return str(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2);a=parser.parse_args()
    from scripts.evaluate_qwen_adapter import read_sources
    from engine.qwen_prompt import messages_for
    from training.providers import components,prompt_text,tokenize
    from training.train import encode
    from PIL import Image
    root=ROOT/'data/training_next';bundle=root/'bundle'
    manifest=json.loads((bundle/'manifest.json').read_text())
    if not manifest.get('audit_environment_sha256'):raise ValueError('Complete current audited preparation first')
    rows={s:[json.loads(x) for x in (bundle/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation')}
    candidate_config=json.loads((ROOT/'config/training_candidates.json').read_text())
    specs=candidate_config['models'];get=read_sources();tasks={};images={}
    for split in rows:
        for row in rows[split]:
            source=get(row['source_split'],row['source_template_id'])
            key=sha256(json.dumps(dict(source=source,audit=manifest['audit_environment_sha256']),sort_keys=True).encode()).hexdigest()
            path=bundle/'images'/f'{key}.png';path.parent.mkdir(parents=True,exist_ok=True)
            tasks[key]=(source,row['source_split'],str(path));images[row['pair_id']]=str(path.relative_to(bundle))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(render_source,t) for t in tasks.values()]),1):
            f.result()
            if i%100==0:print('Source PNGs',i,len(tasks),flush=True)
    summary={}
    for spec in specs:
        print('Preflight provider',spec['repo'],flush=True)
        tokenizer,processor=components(spec,ROOT/'models'/spec['key'],local_files_only=True)
        spec['tokenizer_sha256']=sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        spec['dataset_dir']='datasets/'+spec['key'];folder=bundle/spec['dataset_dir'];folder.mkdir(parents=True,exist_ok=True)
        counts={}
        for split in rows:
            output=[];lengths=[]
            for row in rows[split]:
                r=deepcopy(row);source=get(r['source_split'],r['source_template_id']);image=None
                if processor is not None:
                    r['source_image']=images[r['pair_id']]
                    with Image.open(bundle/r['source_image']) as im:image=im.convert('RGB')
                messages=messages_for(source,r['target_canvas'],version=2)
                text=prompt_text(messages,tokenizer,processor,image)
                tokens=tokenize(text,tokenizer,processor,image)
                n=tokens['input_ids'].shape[-1] if processor is not None else len(tokens['input_ids'])
                if len(source['elements'])>16 and n>4096:
                    messages=messages_for(source,r['target_canvas'],version=2,relation_token_budget=900,
                        relation_token_counter=lambda s:len(tokenizer(s,add_special_tokens=False)['input_ids']))
                r['messages']=messages+[deepcopy(row['messages'][-1])]
                encoded=encode(r,tokenizer,processor=processor,image=image)
                r['token_length']=len(encoded['input_ids'])
                lengths.append(len(encoded['input_ids']));output.append(r)
                if image is not None:image.close()
            (folder/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
            counts[split]=dict(rows=len(output),maximum_tokens=max(lengths),response_mask_verified=True)
        summary[spec['key']]=dict(kind=spec['kind'],splits=counts)
    for name in ('train.py','providers.py','requirements.txt'):shutil.copy2(ROOT/'training'/name,bundle/name)
    # A changed audit snapshot gets new source-image cache keys. Do not ship
    # obsolete, unreferenced renders from an earlier preparation in the ZIP.
    active_images=set(images.values())
    for path in (bundle/'images').glob('*.png'):
        if str(path.relative_to(bundle)) not in active_images:path.unlink()
    manifest.update(models=specs,training_queue=candidate_config['training_queue'],queue_policy=candidate_config['queue_policy'],provider_preflight=summary,source_pngs=len(tasks),
        input_note='Text models use source JSON. VL uses the SAME JSON plus rendered SOURCE only; no human target image, test assets or fonts in the bundle.')
    manifest['recipe'].update(optimizer='adamw_torch',optimizer_foreach=False,
        precision='bf16 if natively supported, otherwise fp16',minimum_cuda_memory_gib=14,
        feasibility='Per-model forward/backward/AdamW preflight; no truncation or silent model skipping')
    manifest['bundle_hashes']={str(p.relative_to(bundle)):sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.rglob('*')) if p.is_file() and p!=bundle/'manifest.json'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'candidate_preflight.json').write_text(json.dumps(dict(models=summary,source_pngs=len(tasks),production_changed=False),indent=2)+'\n')
    with zipfile.ZipFile(root/'layout_training_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle))
    print(json.dumps(dict(models=len(specs),source_pngs=len(tasks),zip_bytes=(root/'layout_training_bundle.zip').stat().st_size,preflight=summary),indent=2))


if __name__=='__main__':main()
