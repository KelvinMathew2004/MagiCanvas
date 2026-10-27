#!/usr/bin/env python3
"""Prepare a score-independent, new-family sample and run actual local inference."""
from pathlib import Path
import json,argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int,choices=[0,1]);p.add_argument('--merge',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from scripts.evaluate_qwen_adapter import read_sources,run_engine_mode,summarize_engine
    if a.prepare:
        from transformers import AutoTokenizer
        from engine.qwen_prompt import messages_for
        from scripts.build_diverse_dataset_viewer import choose
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/qwen3-0.6b'),local_files_only=True)
        records=[json.loads(l) for l in (ROOT/'data/qwen_lora_expanded_v2/validation.jsonl').read_text().splitlines() if l.strip()]
        reviewed={r['family_id'] for r in map(json.loads,(ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines())}
        get=read_sources();eligible=[];tokens={}
        for r in records:
            if r['family_id'] in reviewed:continue
            source=get(r['source_split'],r['source_template_id']);messages=messages_for(source,r['target_canvas'],version=2)
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tokenizer(prompt)['input_ids'])
            if n>4096:continue
            tokens[r['pair_id']]=n;eligible.append(dict(r,source_design=source))
        selected,coverage=choose(eligible,[],12);ids={r['pair_id'] for r in selected};lookup={r['pair_id']:r for r in records};chosen=[lookup[r['pair_id']] for r in selected]
        (a.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chosen))
        (a.output/'selection.json').write_text(json.dumps(dict(coverage=coverage,pair_ids=[r['pair_id'] for r in chosen],prompt_tokens={i:tokens[i] for i in ids},basis='Family/aspect/complexity only; known test families excluded, runtime prompt <=4096 tokens; chosen before inference.'),indent=2)+'\n')
        print('Prepared',len(chosen),'new validation families');return
    records=[json.loads(l) for l in (a.output/'records.jsonl').read_text().splitlines()]
    if a.shard is not None:
        run_engine_mode(records[a.shard::2],read_sources(),ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',a.output/f'shard_{a.shard}.jsonl');return
    if a.merge:
        rows=[json.loads(l) for i in (0,1) for l in (a.output/f'shard_{i}.jsonl').read_text().splitlines()]
        by={r['pair_id']:r for r in rows};ordered=[by[r['pair_id']] for r in records]
        (a.output/'report.json').write_text(json.dumps(dict(results=ordered,records=records,summary=summarize_engine(ordered),fresh_model_inference=True,adapter='qwen3-layout-lora-v2',model='Qwen3-0.6B'),indent=2)+'\n')
        print(json.dumps(dict(cases=len(rows),errors=sum(bool(r.get('engine_error')) for r in rows),flagged=sum(bool(r.get('flagged')) for r in rows)),indent=2))
if __name__=='__main__':main()
