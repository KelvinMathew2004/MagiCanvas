#!/usr/bin/env python3
"""Compare fixed zero-shot critics on hash-verified validation preferences only.

No fitting, prompt sweep, test-label access, or automatic production activation.
The tiled score is predeclared: half global, half mean of three square crops.
NaFlex sees the native aspect ratio with its pinned 256-patch budget.
"""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
from time import perf_counter
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def tiles(image):
    w,h=image.size;side=min(w,h)
    if w==h:return [image]
    offsets=[round((max(w,h)-side)*f) for f in (0.,.5,1.)]
    return [image.crop((o,0,o+side,h) if w>h else (0,o,w,o+side)) for o in offsets]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['global','tiled','naflex'],required=True)
    p.add_argument('--output',type=Path,default=ROOT/'data/upgrade_exploration/critic_pilot_v1')
    p.add_argument('--all-retrospective',action='store_true',help='Score all existing reviewed pairs with fixed critics, without fitting. Consumes all folds; not a sealed promotion test.')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    from scripts.train_siglip_ranker import reviewed_pairs,family_splits,image_embedding
    from engine.vision import _POSITIVE,_NEGATIVE,_letterbox_square
    corpus=ROOT/'data/siglip_reviewed_corpus'
    items=reviewed_pairs(list(map(json.loads,(corpus/'manifest.jsonl').read_text().splitlines())),corpus/'approvals.jsonl')
    split=family_splits(items,with_test=True)
    selected=list(items) if a.all_retrospective else [i for i in items if split[i[0]['pair_id']]=='validation']
    torch.set_num_threads(1)
    modeldir=ROOT/'models'/('siglip2-naflex' if a.arm=='naflex' else 'siglip')
    start=perf_counter()
    model=AutoModel.from_pretrained(modeldir,local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
    with torch.inference_mode():
        inp=processor(text=list(_POSITIVE)+list(_NEGATIVE),padding='max_length',return_tensors='pt')
        text=model.get_text_features(**inp)
        if not torch.is_tensor(text):text=text.pooler_output
        text=torch.nn.functional.normalize(text.float(),dim=-1)
    cache={};records=[];cohorts=defaultdict(list)
    def score(path):
        key=sha256(Path(path).read_bytes()).hexdigest()
        if key in cache:return cache[key]
        with Image.open(path) as im:
            im=im.convert('RGB')
            images=[im] if a.arm=='naflex' else [_letterbox_square(im)]
            if a.arm=='tiled':images+=tiles(im)
            scores=[]
            for image in images:
                inputs=processor(images=image,return_tensors='pt')
                with torch.inference_mode():
                    vector=image_embedding(model,inputs,torch)
                    logits=vector@text.T*model.logit_scale.exp()
                    scores.append(float((logits[:len(_POSITIVE)].mean()-logits[len(_POSITIVE):].mean()).item()))
        value=scores[0] if a.arm!='tiled' else .5*scores[0]+.5*sum(scores[1:])/len(scores[1:])
        cache[key]=value;return value
    for row,preferred,other in selected:
        good,bad=score(preferred),score(other)
        cohort='campaign' if row.get('audit') else 'controlled-shrink'
        correct=good>bad;cohorts[cohort].append(correct)
        records.append(dict(pair_id=row['pair_id'],cohort=cohort,review_hash=row['review_hash'],
                            preferred_score=good,other_score=bad,correct=correct))
        print(a.arm,len(records),len(selected),correct,flush=True)
    report=dict(arm=a.arm,model=str(modeldir),split='all-retrospective' if a.all_retrospective else 'validation',training=False,test_scored=a.all_retrospective,
        scope='Fixed critics, all existing labels now consumed; retrospective descriptive comparison, no tuning and no sealed-generalization claim.' if a.all_retrospective else 'Validation-only fixed-critic pilot',
        cases=len(records),correct=sum(r['correct'] for r in records),
        cohorts={k:dict(cases=len(v),correct=sum(v)) for k,v in cohorts.items()},
        seconds=perf_counter()-start,device='cpu',production_changed=False,records=records,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=sha256((modeldir/'config.json').read_bytes()).hexdigest(),
        manifest_sha256=sha256((corpus/'manifest.jsonl').read_bytes()).hexdigest(),
        approvals_sha256=sha256((corpus/'approvals.jsonl').read_bytes()).hexdigest())
    (a.output/(a.arm+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
