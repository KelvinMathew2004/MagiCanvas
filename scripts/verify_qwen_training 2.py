#!/usr/bin/env python3
"""CPU tokenizer/data preflight for the two layout adapter training runs."""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/qwen_lora_expanded_v2')
    parser.add_argument('--max-length',type=int,default=6144)
    parser.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora_expanded_v2/training_preflight.json')
    args=parser.parse_args();splits={s:[json.loads(x) for x in (args.data/f'{s}.jsonl').read_text().splitlines() if x]
                                      for s in ('train','validation','test')}
    targets=Counter(r['target_template_id'] for r in splits['train']);assert max(targets.values())==1
    families={s:{r['family_id'] for r in rows} for s,rows in splits.items()}
    for a,b in (('train','validation'),('train','test'),('validation','test')):assert not families[a]&families[b]
    report={'split_counts':{s:len(r) for s,r in splits.items()},'unique_train_targets':len(targets),
            'family_overlap':0,'max_length':args.max_length,'file_sha256':{s:hashlib.sha256((args.data/f'{s}.jsonl').read_bytes()).hexdigest() for s in splits},'models':{}}
    for model in ('qwen3-1.7b','qwen2.5-1.5b-instruct'):
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'models'/model,local_files_only=True);stats={}
        for split,rows in splits.items():
            lengths=[];supervised=[]
            for row in rows:
                prompt=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
                answer=row['messages'][-1]['content'];json.loads(answer)
                full=tokenizer(prompt+answer+tokenizer.eos_token,add_special_tokens=False)['input_ids']
                assert full[:len(prefix)]==prefix,(model,split,row['pair_id'],'token-boundary mismatch')
                labels=[-100]*len(prefix)+full[len(prefix):]
                assert len(full)<=args.max_length,(model,split,row['pair_id'],len(full))
                assert labels[-1]==tokenizer.eos_token_id and len(full)>len(prefix)
                lengths.append(len(full));supervised.append(len(full)-len(prefix))
            ordered=sorted(lengths);stats[split]={'rows':len(rows),'maximum_tokens':max(lengths),
                'p95_tokens':ordered[int(len(ordered)*.95)],'over_4096':sum(x>4096 for x in lengths),
                'minimum_response_tokens':min(supervised),'prefix_mismatches':0}
        report['models'][model]=stats
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
