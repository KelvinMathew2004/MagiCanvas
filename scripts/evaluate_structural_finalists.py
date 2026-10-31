#!/usr/bin/env python3
"""No new fitting: existing structural head on safety-gated algorithm finalists.

This is not an ablation over every original engine shortlist entry. The pool
is the retained incumbent and clean finalists of the six no-training arms.
"""
from pathlib import Path
from hashlib import sha256
from tempfile import TemporaryDirectory
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/training_free_scale_v1');a=p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    from engine.structural_ranker import features,score
    from engine.vision import _POSITIVE,_NEGATIVE,_letterbox_square
    from scripts.train_siglip_ranker import image_embedding
    from scripts.audit_upgrade_rasters import paint_stats
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty
    torch.set_num_threads(1);model=None;processor=None;text=None;source_cache={};cache={};results=[]
    headpath=ROOT/'data/chameleon_upgrade_audit/structural_ranker/candidate.json';head=json.loads(headpath.read_text());cases=json.loads((a.root/'cases.json').read_text());out=a.root/'structural_finalists';out.mkdir(exist_ok=True)
    for i,case in enumerate(cases):
        file=out/f'{i:03d}.json'
        if file.exists():results.append(json.loads(file.read_text()));continue
        before=json.loads((a.root/'baseline'/f'{i:03d}.json').read_text())['baseline'];arms=json.loads((a.root/'algorithms'/f'{i:03d}.json').read_text())['arms']
        options=[('incumbent',before['output_design'])]+[(arm,v['output_design']) for arm,v in arms.items() if v['changed'] and not v['defect_flags']]
        r=dict(pair_id=case['pair_id'],key=case['key'],considered=len(options)>1,pool_size=len(options),changed=False,winner='incumbent',output_design=before['output_design'],flags=before['defect_flags'],scores=[])
        if len(options)>1 and not before['defect_flags']:
            if model is None:
                model=AutoModel.from_pretrained(ROOT/'models/siglip',local_files_only=True).eval();processor=AutoProcessor.from_pretrained(ROOT/'models/siglip',local_files_only=True)
                with torch.inference_mode():
                    t=model.get_text_features(**processor(text=list(_POSITIVE)+list(_NEGATIVE),padding='max_length',return_tensors='pt'))
                    if not torch.is_tensor(t):t=t.pooler_output
                    text=torch.nn.functional.normalize(t.float(),dim=-1)
            source=before['source_design'];split=before['source_split'];render=_renderer(source,None if before.get('official_path') else split);key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
            if key not in source_cache:source_cache[key]=paint_stats(source,source,split)
            base=source_cache[key]['text']
            for name,doc in options:
                k=key+sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()
                if k not in cache:
                    stats=paint_stats(doc,source,split)['text'];margin=min(((v['contrast'] if v['contrast'] is not None else 1.)-min(3.,(base.get(eid,{}).get('contrast') or 1.)*.9) for eid,v in stats.items()),default=0.)/3
                    with TemporaryDirectory() as folder:
                        path=Path(folder)/'candidate.png';_preview_nonempty(render,doc,path)
                        with Image.open(path) as im:inputs=processor(images=_letterbox_square(im.convert('RGB')),return_tensors='pt')
                        with torch.inference_mode():
                            vector=image_embedding(model,inputs,torch);logits=vector@text.T*model.logit_scale.exp();zero=float((logits[:len(_POSITIVE)].mean()-logits[len(_POSITIVE):].mean()).item())
                    phi=features(source,doc,margin,zero);cache[k]=dict(score=score(phi,head),features=phi)
                r['scores'].append(dict(name=name,**cache[k]))
            # Predeclared margin; keep incumbent on small model preferences.
            best=max(r['scores'],key=lambda s:s['score']);original=r['scores'][0]['score']
            if best['name']!='incumbent' and best['score']>=original+.5:
                r.update(changed=True,winner=best['name'],output_design=dict(options)[best['name']])
        file.write_text(json.dumps(r)+'\n');results.append(r);print(i+1,len(cases),r['considered'],r['winner'],flush=True)
    report=dict(cases=len(results),considered=sum(r['considered'] for r in results),changed=sum(r['changed'] for r in results),training=False,head_sha256=sha256(headpath.read_bytes()).hexdigest(),
        scope=__doc__,production_changed=False,pairwise_score_margin=.5,results=[{k:v for k,v in r.items() if k!='output_design'} for r in results])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('Considered',report['considered'],'changed',report['changed'])


if __name__=='__main__':main()
