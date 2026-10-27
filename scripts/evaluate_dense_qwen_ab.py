#!/usr/bin/env python3
"""Fresh paired dense-cohort inference; raw metrics never include recovery credit."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def cohort():
    spec=json.loads((ROOT/'config/regression_review.json').read_text());reports={};cases=[]
    for case in spec['cases']:
        if case['cohort']=='official':continue
        path=case['prediction_report']
        if path not in reports:reports[path]=json.loads((ROOT/path).read_text())
        old=next(r for r in reports[path]['results'] if r['pair_id']==case['pair_id'])
        record=next((r for r in reports[path].get('records',[]) if r['pair_id']==case['pair_id']),None)
        if record is None and case.get('records_file'):
            record=next(r for r in map(json.loads,(ROOT/case['records_file']).read_text().splitlines()) if r['pair_id']==case['pair_id'])
        if record is None:record={k:old[k] for k in ('pair_id','source_split','family_id','target_canvas') if k in old}
        cases.append(dict(case=case,source=old['source_design'],record=record))
    return cases


def run(args):
    import torch
    from engine import qwen_seed
    from engine.resize import _analyze,_proportional
    cases=cohort();args.output.mkdir(parents=True,exist_ok=True)
    arms={'0.6b':('qwen3-0.6b','qwen3-layout-lora-v2'),'1.7b':('qwen3-1.7b','qwen3-layout-lora-v2-expanded-1.7b')}
    model,adapter=arms[args.arm];qwen_seed.MODEL_DIR=ROOT/'models'/model;qwen_seed.ADAPTER_DIR=ROOT/'models'/adapter
    fp=dict(arm=args.arm,adapter_sha256=sha256((qwen_seed.ADAPTER_DIR/'adapter_model.safetensors').read_bytes()).hexdigest(),files_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in ('engine/qwen_seed.py','engine/runtime.py','engine/proposal_completion.py','engine/qwen_prompt.py')},cases_sha256=sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),fresh_model_inference=True)
    manifest=args.output/(args.arm+'.manifest.json')
    if manifest.exists() and json.loads(manifest.read_text())!=fp:raise SystemExit('Fresh inference inputs changed: use a new output folder')
    manifest.write_text(json.dumps(fp,indent=2)+'\n');checkpoint=args.output/(args.arm+'.jsonl');done={r['pair_id']:r for r in map(json.loads,checkpoint.read_text().splitlines())} if checkpoint.exists() else {}
    os.environ['MAGICANVAS_QWEN_ENABLED']='1';os.environ['MAGICANVAS_DATASET_RENDERING']='1'
    for i,item in enumerate(cases):
        record=item['record'];source=item['source'];target=record['target_canvas'];pid=record['pair_id']
        if pid in done:continue
        _,units=_analyze(source);base=_proportional(units,target['width'],target['height'],source['canvas']['width'],source['canvas']['height']);diag={}
        started=time.perf_counter();proposal,status=qwen_seed.generate_seed(source,target,base,diag)
        raw=[dict(id=eid,**p) for eid,p in diag.get('model_elements',{}).items()]
        valid=bool(diag.get('json_complete') and not diag.get('missing_ids') and not diag.get('duplicate_ids') and not diag.get('unknown_ids') and len(raw)==len(source['elements']))
        row=dict(pair_id=pid,source_split=record['source_split'],family_id=record.get('family_id'),target_canvas=target,source_design=source,qwen_model_elements=raw,raw_proposal_valid=valid,raw_json_complete=diag.get('json_complete'),qwen_status=status,generation=diag,wall_seconds=time.perf_counter()-started,completion_candidate_available=proposal is not None,fresh_model_inference=True)
        with checkpoint.open('a') as f:f.write(json.dumps(row)+'\n')
        print(args.arm,i+1,len(cases),'raw_ids',len(raw),'/',len(source['elements']),'generation_seconds',round(diag.get('generation_seconds',0),2),flush=True)
    print('Fresh inference finished',args.arm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('0.6b','1.7b'),required=True);p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/dense_ab_fresh_v1');run(p.parse_args())
