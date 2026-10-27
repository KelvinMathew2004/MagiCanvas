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


def family_splits(items):
    parent={}
    def find(value):
        parent.setdefault(value,value)
        if parent[value]!=value:parent[value]=find(parent[value])
        return parent[value]
    for row,_,_ in items:
        parent[find(row["source_id"])]=find(row["target_id"])
    result={}
    for row,_,_ in items:
        root=find(row["source_id"])
        bucket=int.from_bytes(hashlib.sha256(root.encode()).digest()[:4],"big")%100
        result[row["pair_id"]]="validation" if bucket<20 else "train"
    return result


def reviewed_pairs(rows, approvals):
    """Latest human choice owns each pair; stale images and revocations reject it."""
    choices={}
    if approvals.is_file():
        for line in approvals.read_text().splitlines():
            if line.strip():
                choice=json.loads(line)
                choices[choice.get('pair_id')]=choice
    eligible=[]
    for row in rows:
        decision=choices.get(row.get('pair_id'))
        if not decision or decision.get('decision') not in ('accept','prefer_candidate'):continue
        positive=next((p for p in (row.get('positive_images') or []) if p and Path(p).is_file()),None)
        negative=row.get('engine_image')
        if not positive or not negative or not Path(negative).is_file():continue
        if row.get('review_hash'):
            actual=hashlib.sha256(Path(positive).read_bytes()+Path(negative).read_bytes()).hexdigest()
            if decision.get('review_hash')!=row['review_hash'] or actual!=row['review_hash']:continue
        if row.get('status') not in {'accepted_negative','candidate_without_measured_defect','unreviewed_engine_candidate'}:continue
        if decision['decision']=='prefer_candidate':positive,negative=negative,positive
        eligible.append((row,positive,negative))
    return eligible


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
    args=parser.parse_args()
    if args.epochs<1 or args.learning_rate<=0 or args.weight_decay<0:
        parser.error("epochs and learning rate must be positive; weight decay cannot be negative")
    if not args.manifest.is_file():
        raise SystemExit(f"No SigLIP pair manifest at {args.manifest}; generate and review pairs first.")
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    eligible=reviewed_pairs(rows,args.approvals)
    if len(eligible)<20:
        raise SystemExit(f"Need at least 20 accepted same-domain pairs; found {len(eligible)}. No weights were trained.")
    groups={"train":[],"validation":[]}
    split_by_pair=family_splits(eligible)
    for item in eligible:groups[split_by_pair[item[0]["pair_id"]]].append(item)
    if len(groups["train"])<10 or not groups["validation"]:
        raise SystemExit("Family split produced too few train/validation pairs; collect more reviewed families.")

    import torch
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    from engine.vision import _letterbox_square, _POSITIVE, _NEGATIVE
    torch.manual_seed(args.seed)
    model=AutoModel.from_pretrained(str(args.model),local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model),local_files_only=True)
    for parameter in model.parameters():parameter.requires_grad_(False)

    def embed(path):
        with Image.open(path) as image:
            inputs=processor(images=_letterbox_square(image),return_tensors="pt")
        with torch.inference_mode():
            output=model(**inputs)
            features=getattr(output,"image_embeds",None)
            if features is None:
                features=model.get_image_features(**inputs)
        return torch.nn.functional.normalize(features.float().reshape(-1),dim=0)

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
    for split,items in groups.items():
        pairs=[]
        for row,pos,neg in items:
            pkey,nkey=str(Path(pos).resolve()),str(Path(neg).resolve())
            if pkey not in prepared:prepared[pkey]=embed(pkey)
            if nkey not in prepared:prepared[nkey]=embed(nkey)
            baseline=zero_shot_pair(pkey,nkey) if split=="validation" else None
            pairs.append((prepared[pkey],prepared[nkey],baseline))
        groups[split]=pairs
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
    if best[1] <= .60 or best[1] <= best[2]:
        print(json.dumps({"saved":False,"reason":"combined validation scorer did not exceed 0.60 accuracy and zero-shot accuracy",
                          "train_pairs":len(groups["train"]),"validation_pairs":len(groups["validation"]),
                          "best_epoch":best[3],"combined_validation_pair_accuracy":best[1],
                          "zero_shot_validation_pair_accuracy":best[2],"validation_loss":best[0],
                          "output_preserved":str(args.output)},indent=2))
        return
    head.load_state_dict(best[4])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":head.state_dict(),"dimension":dimension,
                "model_id":"google/siglip2-base-patch16-224",
                "feature_normalization":"l2","score_scale":score_scale,
                "loss":"pairwise Bradley-Terry softplus","train_pairs":len(groups["train"]),
                "validation_pairs":len(groups["validation"]),"best_epoch":best[3],
                "validation_pair_accuracy":best[1],"zero_shot_validation_pair_accuracy":best[2],
                "validation_loss":best[0]},args.output)
    print(json.dumps({"output":str(args.output.resolve()),"train_pairs":len(groups["train"]),
                      "validation_pairs":len(groups["validation"]),"best_epoch":best[3],
                      "validation_pair_accuracy":best[1],"zero_shot_validation_pair_accuracy":best[2],
                      "validation_loss":best[0]},indent=2))


if __name__=="__main__":main()
