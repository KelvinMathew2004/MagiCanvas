#!/usr/bin/env python3
"""Validate unique-target training data and emit optional dense sampling weights.

No duplication of target IDs, no human-target evaluation data, no automatic
training. A future trainer can consume these weights only after the fresh A/B
shows a persistent raw dense-layout deficit.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json


def prepare(input_file,output,boost=2.):
    rows=[json.loads(s) for s in input_file.read_text().splitlines() if s.strip()];seen=set();weights=[]
    for r in rows:
        if r.get('model_split','train')!='train':raise ValueError('Continuation accepts training records only')
        tid=r.get('target_template_id') or r['pair_id'].split('__to__')[-1]
        if tid in seen:raise ValueError('Duplicate target ID; clean/cap data before continuation')
        seen.add(tid)
        label=json.loads(r['messages'][-1]['content']);n=len(label.get('elements',[]))
        weights.append(dict(pair_id=r['pair_id'],target_template_id=tid,elements=n,weight=boost if 15<=n<=25 else 1.))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(dict(input_sha256=sha256(input_file.read_bytes()).hexdigest(),source=str(input_file),target_unique=True,response_only_loss_required=True,training_started=False,reason='Conditional curriculum, not a recommendation to retrain before the dense A/B',examples=weights),indent=2)+'\n')
    print(json.dumps(dict(examples=len(rows),dense_examples=sum(x['weight']>1 for x in weights),training_started=False,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('data/qwen_lora_expanded_v2/train.jsonl'));p.add_argument('--output',type=Path,default=Path('data/qwen_lora/engine_upgrade/dense_curriculum.json'));p.add_argument('--boost',type=float,default=2.);a=p.parse_args();prepare(a.input,a.output,a.boost)
