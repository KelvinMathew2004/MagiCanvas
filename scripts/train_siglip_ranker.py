#!/usr/bin/env python3
"""Train a small SigLIP image-ranking head from reviewed local-render pairs.

The SigLIP backbone stays frozen. Only same-renderer pairs with an image-bound
human preference record are eligible. Either side may be preferred; a named
positive/engine field is a storage convention, not an automatic correctness label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def family_keys(items):
    """Connected IDs AND declared family ancestry survive partial edge sampling."""
    parent={}
    def find(value):
        parent.setdefault(value,value)
        if parent[value]!=value:parent[value]=find(parent[value])
        return parent[value]
    def join(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    for row,_,_ in items:
        a,b='template:'+row['source_id'],'template:'+row['target_id']
        join(a,b)
        if row.get('family_id'):join(a,'family:'+row['family_id'])
    return {row['pair_id']:find('template:'+row['source_id']) for row,_,_ in items}


def family_splits(items,with_test=False):
    keys=family_keys(items)
    result={}
    for row,_,_ in items:
        root=keys[row['pair_id']]
        bucket=int.from_bytes(hashlib.sha256(root.encode()).digest()[:4],"big")%100
        result[row["pair_id"]]=('test' if bucket<15 else 'validation' if bucket<30 else 'train') if with_test else ('validation' if bucket<20 else 'train')
    return result


def inconsistent_repeat_ids(rows, choices):
    """Quarantine ambiguous originals; repeats never become extra votes."""
    by_id={r['pair_id']:r for r in rows};conflicts=set()
    for row in rows:
        original=row.get('reliability_repeat_of')
        if not original or original not in by_id:continue
        first,repeat=choices.get(original),choices.get(row['pair_id'])
        if not first or not repeat:continue
        if (first.get('review_hash')!=by_id[original].get('review_hash') or
                repeat.get('review_hash')!=row.get('review_hash')):continue
        a=(first.get('decision'),bool(first.get('reject_both')))
        b=(repeat.get('decision'),bool(repeat.get('reject_both')))
        if a!=b:conflicts.add(original)
    return conflicts


def reviewed_pairs(rows, approvals):
    """Latest human choice owns each pair; stale images and revocations reject it."""
    choices={}
    if approvals.is_file():
        for line in approvals.read_text().splitlines():
            if line.strip():
                choice=json.loads(line)
                choices[choice.get('pair_id')]=choice
    ambiguous=inconsistent_repeat_ids(rows,choices)
    eligible=[]
    for row in rows:
        if row.get('pair_id') in ambiguous:continue
        decision=choices.get(row.get('pair_id'))
        if not decision or decision.get('decision') not in ('accept','prefer_candidate'):continue
        positive=next((p for p in (row.get('positive_images') or []) if p and Path(p).is_file()),None)
        negative=row.get('engine_image')
        if not positive or not negative or not Path(negative).is_file():continue
        if row.get('review_hash'):
            actual=hashlib.sha256(Path(positive).read_bytes()+Path(negative).read_bytes()).hexdigest()
            if decision.get('review_hash')!=row['review_hash'] or actual!=row['review_hash']:continue
        if row.get('status') not in {'accepted_negative','candidate_without_measured_defect','unreviewed_engine_candidate'}:continue
        side='b_training_eligible' if decision['decision']=='prefer_candidate' else 'a_training_eligible'
        if row.get(side) is False:continue
        if decision['decision']=='prefer_candidate':positive,negative=negative,positive
        eligible.append((row,positive,negative))
    return eligible


def readiness(items):
    """Fixed three-way family split, shared by pilots and release evaluation."""
    split_by_pair=family_splits(items,with_test=True)
    keys=family_keys(items)
    counts={s:sum(split_by_pair[r['pair_id']]==s for r,_,_ in items) for s in ('train','validation','test')}
    families={s:len({keys[r['pair_id']] for r,_,_ in items if split_by_pair[r['pair_id']]==s}) for s in counts}
    return dict(eligible_pairs=len(items),split_pairs=counts,family_counts=families,
                ready=len(items)>=20 and families['train']>=20 and families['validation']>=5 and families['test']>=5)


def image_embedding(model, inputs, torch):
    """Current Transformers returns a vision output; older releases a tensor."""
    output=model.get_image_features(**inputs)
    features=getattr(output,'pooler_output',output)
    return torch.nn.functional.normalize(features.float().reshape(-1),dim=0).detach().cpu()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=ROOT/"data"/"siglip_pairs"/"manifest.jsonl")
    parser.add_argument("--approvals",type=Path,default=ROOT/"data"/"siglip_pairs"/"approvals.jsonl",
                        help="JSONL human choices: accept, prefer_candidate, or skip; image hashes must match")
    parser.add_argument("--model",type=Path,default=ROOT/"models"/"siglip")
    parser.add_argument("--output",type=Path,default=ROOT/"models"/"siglip-ranker.pt")
    parser.add_argument("--epochs",type=int,default=60)
    parser.add_argument("--learning-rate",type=float,default=.03)
    parser.add_argument("--weight-decay",type=float,default=.01)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument('--pilot',action='store_true',help='Small diagnostic only; requires an output outside models/ and never passes production gate')
    parser.add_argument('--report',type=Path,help='Write measured split counts and training/validation/test results')
    parser.add_argument('--check-only',action='store_true',help='Validate label eligibility and report fixed family folds without loading models')
    parser.add_argument('--feature-cache',type=Path,default=ROOT/'data/siglip_feature_cache',help='Image/model-hash-bound frozen embedding cache')
    args=parser.parse_args()
    if args.epochs<1 or args.learning_rate<=0 or args.weight_decay<0:
        parser.error("epochs and learning rate must be positive; weight decay cannot be negative")
    if not args.manifest.is_file():
        raise SystemExit(f"No SigLIP pair manifest at {args.manifest}; generate and review pairs first.")
    provenance=dict(trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        approvals_sha256=hashlib.sha256(args.approvals.read_bytes()).hexdigest() if args.approvals.is_file() else None,
        seed=args.seed,test_scored=False,production_changed=False)
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    eligible=reviewed_pairs(rows,args.approvals)
    ready=readiness(eligible)
    if args.check_only:
        if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(ready,indent=2)+'\n')
        print(json.dumps(ready,indent=2));return
    if len(eligible)<20:
        raise SystemExit(f"Need at least 20 accepted same-domain pairs; found {len(eligible)}. No weights were trained.")
    if args.pilot and args.output.resolve().is_relative_to((ROOT/'models').resolve()):
        parser.error('Pilot output must stay outside models/; it cannot replace the production ranker')
    groups={"train":[],"validation":[],**({} if args.pilot else {'test':[]})}
    split_by_pair=family_splits(eligible,with_test=True)
    for item in eligible:
        split=split_by_pair[item[0]['pair_id']]
        if split in groups:groups[split].append(item)  # A pilot never consumes the reserved test labels.
    if len(groups["train"])<10 or not groups["validation"]:
        raise SystemExit("Family split produced too few train/validation pairs; collect more reviewed families.")
    keys=family_keys(eligible)
    family_counts={split:len({keys[item[0]['pair_id']] for item in values}) for split,values in groups.items()}
    if not args.pilot and (family_counts['train']<20 or family_counts['validation']<5 or family_counts['test']<5):
        raise SystemExit(f'Production ranking needs >=20 train, >=5 validation and >=5 untouched test families; found {family_counts}. Use --pilot only for an explicitly non-production experiment.')

    import torch
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    from engine.vision import _letterbox_square, _POSITIVE, _NEGATIVE
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model=AutoModel.from_pretrained(str(args.model),local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model),local_files_only=True)
    for parameter in model.parameters():parameter.requires_grad_(False)

    model_files=list(args.model.glob('*.safetensors'))+list(args.model.glob('*.json'))
    model_digest=hashlib.sha256()
    for path in sorted(model_files):
        model_digest.update(path.name.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):model_digest.update(chunk)
    model_key=model_digest.hexdigest()
    cache=args.feature_cache/model_key;cache.mkdir(parents=True,exist_ok=True)
    def embed(path):
        key=hashlib.sha256(Path(path).read_bytes()).hexdigest();cached=cache/(key+'.pt')
        if cached.is_file():return torch.load(cached,weights_only=True,map_location='cpu')
        with Image.open(path) as image:
            inputs=processor(images=_letterbox_square(image),return_tensors="pt")
        with torch.inference_mode():
            features=image_embedding(model,inputs,torch)
        torch.save(features,cached);return features

    def zero_shot_pair(positive_path, negative_path):
        # Match engine.vision's default no-context prompt scorer exactly; this
        # gives the learned head a validation baseline from the same backbone.
        with Image.open(positive_path) as image:
            positive_image=_letterbox_square(image)
        with Image.open(negative_path) as image:
            negative_image=_letterbox_square(image)
        positive_prompts=list(_POSITIVE)
        negative_prompts=list(_NEGATIVE)
        inputs=processor(text=positive_prompts+negative_prompts,
                         images=[positive_image,negative_image],
                         padding="max_length",return_tensors="pt")
        with torch.inference_mode():
            output=model(**inputs)
            logits=output.logits_per_image
            positive=logits[:,:len(positive_prompts)].mean(dim=1)
            negative=logits[:,len(positive_prompts):].mean(dim=1)
            quality=positive-negative
        return float((quality[0]-quality[1]).item())

    # Cache embeddings so fitting is small and repeatable.
    prepared={}
    test_items=groups.pop('test',[])
    def prepare_pairs(items,split):
        pairs=[]
        for row,pos,neg in items:
            pkey,nkey=str(Path(pos).resolve()),str(Path(neg).resolve())
            if pkey not in prepared:prepared[pkey]=embed(pkey)
            if nkey not in prepared:prepared[nkey]=embed(nkey)
            baseline=zero_shot_pair(pkey,nkey) if split!='train' else None
            pairs.append((prepared[pkey],prepared[nkey],baseline))
        print(f'Prepared {len(pairs)} {split} pairs / {family_counts[split]} families',flush=True)
        return pairs
    for split,items in list(groups.items()):groups[split]=prepare_pairs(items,split)
    dimension=groups["train"][0][0].numel()
    head=torch.nn.Linear(dimension,1,bias=False)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=0.)
    train=torch.stack([torch.stack(pair[:2]) for pair in groups["train"]])
    validation=torch.stack([torch.stack(pair[:2]) for pair in groups["validation"]])
    zero_shot=torch.tensor([pair[2] for pair in groups["validation"]],dtype=torch.float32)
    score_scale=.25
    best=None
    for epoch in range(args.epochs):
        head.train();optimizer.zero_grad(set_to_none=True)
        pos=head(train[:,0]).squeeze(-1);neg=head(train[:,1]).squeeze(-1)
        loss=torch.nn.functional.softplus(-(pos-neg)).mean()+args.weight_decay*head.weight.square().sum()
        loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():
            val_head_diff=(head(validation[:,0])-head(validation[:,1])).squeeze(-1)
            val_diff=zero_shot+score_scale*val_head_diff
            accuracy=float((val_diff>0).float().mean())
            zero_shot_accuracy=float((zero_shot>0).float().mean())
            val_loss=float(torch.nn.functional.softplus(-val_diff).mean())
        if best is None or val_loss<best[0]:
            best=(val_loss,accuracy,zero_shot_accuracy,epoch+1,
                  {k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
    head.load_state_dict(best[4])
    result=dict(pilot=args.pilot,train_pairs=len(groups['train']),validation_pairs=len(groups['validation']),family_counts=family_counts,best_epoch=best[3],combined_validation_pair_accuracy=best[1],zero_shot_validation_pair_accuracy=best[2],validation_loss=best[0])
    gate=best[1]>.60 and best[1]>best[2]
    if not args.pilot and gate:
        # Test labels are not consulted for fitting, epoch selection or thresholds.
        test_pairs=prepare_pairs(test_items,'test')
        test=torch.stack([torch.stack(pair[:2]) for pair in test_pairs]);baseline=torch.tensor([pair[2] for pair in test_pairs])
        with torch.no_grad():combined=baseline+score_scale*(head(test[:,0])-head(test[:,1])).squeeze(-1)
        result.update(test_pairs=len(test_items),combined_test_pair_accuracy=float((combined>0).float().mean()),zero_shot_test_pair_accuracy=float((baseline>0).float().mean()))
        provenance['test_scored']=True
        gate=gate and result['combined_test_pair_accuracy']>.60 and result['combined_test_pair_accuracy']>result['zero_shot_test_pair_accuracy']
    elif not args.pilot:
        result.update(test_pairs=len(test_items),test_not_scored_reason='Validation gate failed; reserved test outcomes remain unscored')
    result['production_gate_passed']=bool(gate and not args.pilot)
    result['provenance']=dict(provenance,model_sha256=model_key)
    result['saved']=bool(gate or args.pilot)
    result['provenance']['production_changed']=bool(result['saved'] and args.output.resolve().is_relative_to((ROOT/'models').resolve()))
    result['output']=str(args.output.resolve())
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    if not result['saved']:
        print(json.dumps(result,indent=2));return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":head.state_dict(),"dimension":dimension,
                "model_id":"google/siglip2-base-patch16-224",
                "feature_normalization":"l2","score_scale":score_scale,
                "loss":"pairwise Bradley-Terry softplus","train_pairs":len(groups["train"]),
                "validation_pairs":len(groups["validation"]),"best_epoch":best[3],
                "validation_pair_accuracy":best[1],"zero_shot_validation_pair_accuracy":best[2],
                "validation_loss":best[0],"model_sha256":model_key,
                "pilot":args.pilot,"production_gate_passed":result['production_gate_passed'],
                "heldout_results":result},args.output)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
