#!/usr/bin/env python3
"""Training-free native speculative-decoding feasibility and parity probe.

Loads the two matching existing LoRA packages. Target is 1.7B, draft is 0.6B.
No grammar, remote code, model training or production-default change.
"""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/upgrade_exploration/assisted_decode_v1');p.add_argument('--indices',type=int,nargs='+',default=[69,18,14]);a=p.parse_args();a.output.mkdir(exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from scripts.verify_qwen_adapter import verify
    from engine.runtime import select_runtime,synchronize
    from engine.qwen_prompt import messages_for
    from scripts.run_upgrade_model_pilot import generation_stop_ids
    from experiments.layout_upgrade.contracts import decode
    runtime=select_runtime(torch,'qwen');tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen3-1.7b',local_files_only=True)
    assert (ROOT/'models/qwen3-1.7b/tokenizer.json').read_bytes()==(ROOT/'models/qwen3-0.6b/tokenizer.json').read_bytes(),'Draft/target token vocabularies differ'
    def load(base,adapter):
        verify(ROOT/'models'/adapter,ROOT/'models'/base)
        model=AutoModelForCausalLM.from_pretrained(ROOT/'models'/base,local_files_only=True,dtype=runtime.dtype,attn_implementation='sdpa')
        model=PeftModel.from_pretrained(model,ROOT/'models'/adapter,local_files_only=True).merge_and_unload(safe_merge=True)
        return model.to(runtime.device).eval()
    target=load('qwen3-1.7b','qwen3-layout-lora-v2-expanded-1.7b');draft=load('qwen3-0.6b','qwen3-layout-lora-v2')
    draft.generation_config.num_assistant_tokens=4;draft.generation_config.num_assistant_tokens_schedule='constant';draft.generation_config.assistant_confidence_threshold=0.
    cases=json.loads((ROOT/'data/upgrade_exploration/training_free_scale_v1/cases.json').read_text());results=[]
    manifest=dict(script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),runtime=runtime.diagnostics(),target='qwen3-1.7b + matching expanded adapter',draft='qwen3-0.6b + matching v2 adapter',max_draft_tokens=4,do_sample=False,grammar=False,production_changed=False,scope='Native feasibility probe, not M5 Pro or Docker latency. All models resident in both timing arms; exact target token parity required.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for n,i in enumerate(a.indices):
        case=cases[i];source=case['cached']['source_design'];canvas=case['cached']['target_canvas']
        messages=messages_for(source,canvas,version=2,relation_token_counter=lambda s:len(tokenizer(s,add_special_tokens=False)['input_ids']),relation_token_budget=900)
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);inputs=tokenizer(prompt,return_tensors='pt').to(runtime.device);tokens=inputs['input_ids'].shape[1]
        budget=min(4096,max(512,96*len(source['elements'])+128),8192-tokens)
        r=dict(index=i,pair_id=case['pair_id'],source=source,target=canvas,prompt_tokens=tokens,output_budget=budget,arms={})
        order=('target','assisted') if n%2==0 else ('assisted','target')
        for mode in order:
            synchronize(torch,runtime.device);start=perf_counter()
            try:
                with torch.inference_mode():
                    output=target.generate(**inputs,assistant_model=draft if mode=='assisted' else None,max_new_tokens=budget,max_time=150.,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,eos_token_id=generation_stop_ids(tokenizer,target))
                synchronize(torch,runtime.device);elapsed=perf_counter()-start;generated=output[0,tokens:].tolist();response=tokenizer.decode(generated,skip_special_tokens=True);es,diag=decode(response,source,canvas)
                r['arms'][mode]=dict(seconds=elapsed,output_tokens=len(generated),token_ids=generated,response=response,raw_valid=diag['exact_ids'],json_complete=diag['json_complete'],model_elements=es,hit_token_limit=len(generated)>=budget)
            except Exception as e:r['arms'][mode]=dict(seconds=perf_counter()-start,error=f'{type(e).__name__}: {e}')
            print(case['pair_id'],mode,r['arms'][mode].get('seconds'),r['arms'][mode].get('error'),flush=True)
        r['exact_target_token_parity']=bool('token_ids' in r['arms']['target'] and r['arms']['target'].get('token_ids')==r['arms']['assisted'].get('token_ids'))
        (a.output/f'{n:02d}.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r)
    (a.output/'summary.json').write_text(json.dumps(dict(cases=len(results),exact_parity=sum(r['exact_target_token_parity'] for r in results),records=[{k:v for k,v in r.items() if k!='source'} for r in results]),indent=2)+'\n')


if __name__=='__main__':main()
