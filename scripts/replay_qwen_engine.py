#!/usr/bin/env python3
"""Replay saved raw predictions through the current parser and engine.

The model's exact decoded coordinates are reused; token-generation latency is
excluded. The normal Qwen decoder, aspect clamps, font fitting, candidate search,
and raster selection still execute. This isolates post-processing ablations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def _prediction_queues(records,cached,completed=()):
    """Keep distinct target-template pairs separate even at equal canvas sizes."""
    from collections import defaultdict,deque
    by_pair={r['pair_id']:r for r in cached};queues=defaultdict(deque)
    for record in records:
        if record['pair_id'] in completed:continue
        row=by_pair[record['pair_id']]
        key=(tuple(e['id'] for e in row['source_design']['elements']),
             row['target_canvas']['width'],row['target_canvas']['height'])
        queues[key].append(row)
    return queues,by_pair


def replay(records, cached, adapter, model, checkpoint):
    import torch
    from engine import qwen_seed
    from engine.qwen_prompt import font_reference
    from scripts.evaluate_qwen_adapter import read_sources,run_engine_mode
    completed=set()
    if checkpoint and Path(checkpoint).is_file():
        completed={r['pair_id'] for r in map(json.loads,Path(checkpoint).read_text().splitlines()) if not r.get('engine_error')}
    queues,by_pair=_prediction_queues(records,cached,completed)
    original_generate=qwen_seed.generate_seed
    state={}

    class Tokenizer:
        eos_token_id=0
        def apply_chat_template(self,*args,**kwargs):return 'cached prediction replay'
        def __call__(self,text,return_tensors=None,**kwargs):
            return {'input_ids':torch.zeros((1,1),dtype=torch.long)} if return_tensors else {'input_ids':[0]*max(1,len(text)//3)}
        def decode(self,*args,**kwargs):return state['response']

    class Model:
        config=SimpleNamespace(max_position_embeddings=8192)
        def generate(self,**kwargs):return torch.zeros((1,2),dtype=torch.long)

    def load():return Model(),Tokenizer(),torch,'cpu',2
    load.cache_clear=lambda:None

    def generate(source,target,base_elements,diagnostics=None):
        row=queues[(tuple(e['id'] for e in source['elements']),target['width'],target['height'])].popleft()
        if not row.get('raw_proposal_valid',bool(row.get('official_path'))) or not row.get('qwen_model_elements'):
            raise ValueError(f"Cannot replay incomplete raw prediction: {row['pair_id']}")
        aliases={e['id']:f'e{i:03d}' for i,e in enumerate(source['elements'])}
        payload=[]
        for e in row['qwen_model_elements']:
            p={'id':aliases[e['id']], 'box':[e['x']/target['width']*1000,e['y']/target['height']*1000,
                    e['width']/target['width']*1000,e['height']/target['height']*1000]}
            if 'font_size' in e:p['font_scale']=e['font_size']/font_reference(target,2)*1000
            payload.append(p)
        state['response']=json.dumps({'elements':payload},separators=(',',':'))
        return original_generate(source,target,base_elements,diagnostics=diagnostics)

    with patch.object(qwen_seed,'_load',load),patch.object(qwen_seed,'generate_seed',generate):
        rows=run_engine_mode(records,read_sources(),Path(adapter),Path(model),checkpoint)
    for row in rows:
        if row.get('engine_error'):continue
        old=by_pair[row['pair_id']]
        expected={e['id']:e for e in old.get('qwen_model_elements',[])}
        actual={e['id']:e for e in row.get('qwen_model_elements',[])}
        if expected and actual:
            if set(expected)!=set(actual) or any(abs(expected[eid].get(k,0)-actual[eid].get(k,0))>1e-6
                    for eid in expected for k in ('x','y','width','height','font_size')):
                raise ValueError(f"Saved prediction changed for pair {row['pair_id']}")
        if (row.get('raw_geometry_mae') is not None and old.get('raw_geometry_mae') is not None
                and abs(row['raw_geometry_mae']-old['raw_geometry_mae'])>1e-8):
            raise ValueError(f"Raw coordinates changed while replaying {row['pair_id']}")
        row['raw_prediction_reused']=True
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=1)
    args=p.parse_args()
    prior=json.loads(args.report.read_text())
    records=[json.loads(l) for l in Path(prior['test_file']).read_text().splitlines() if l.strip()]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    files=['engine/resize.py','engine/qwen_refine.py','engine/qwen_seed.py','scripts/layout_defects.py','scripts/evaluate_qwen_adapter.py','scripts/replay_qwen_engine.py']
    fingerprint={'prediction_report_sha256':sha256(args.report.read_bytes()).hexdigest(),
                 'code_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    manifest_path=args.output.with_suffix('.run_manifest.json')
    if manifest_path.is_file() and json.loads(manifest_path.read_text())!=fingerprint:
        raise SystemExit('Replay inputs or code changed; use a new output path.')
    manifest_path.write_text(json.dumps(fingerprint,indent=2)+'\n')
    from scripts.evaluate_qwen_adapter import summarize_engine
    from scripts.layout_defects import build_ab_viewer
    report={k:deepcopy(prior[k]) for k in ('test_file','adapter_dir','model_dir','comparison_adapter_dir','comparison_model_dir','examples')}
    report['results']={}
    configurations=[('candidate','adapter_dir','model_dir'),('comparison','comparison_adapter_dir','comparison_model_dir')]
    if args.workers==2:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs={arm:pool.submit(replay,records,prior['results'][arm],prior[ak],prior[mk],
                                 args.output.with_suffix(f'.{arm}.jsonl')) for arm,ak,mk in configurations}
            for arm,job in jobs.items():report['results'][arm]=job.result()
    else:
        for arm,ak,mk in configurations:
            report['results'][arm]=replay(records,prior['results'][arm],prior[ak],prior[mk],
                                        args.output.with_suffix(f'.{arm}.jsonl'))
    for arm in report['results']:report[f'{arm}_engine']=summarize_engine(report['results'][arm])
    files=['engine/resize.py','engine/qwen_refine.py','engine/qwen_seed.py','scripts/layout_defects.py','scripts/replay_qwen_engine.py']
    report['run_manifest']={'prediction_report':str(args.report.resolve()),
        'prediction_report_sha256':sha256(args.report.read_bytes()).hexdigest(),
        'code_sha256':{s:sha256((ROOT/s).read_bytes()).hexdigest() for s in files},
        'raw_coordinates_reused':True,'generation_latency_excluded':True}
    report['metric_scope']='Current full engine; exact decoded model coordinates replayed from the recorded inference run'
    report['review_page']=build_ab_viewer(report['results']['candidate'],report['results']['comparison'],records,
            args.output.parent/'ab_viewer',12,candidate_label='Qwen3 1.7B · current engine',comparison_label='Qwen3 0.6B · current engine')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',args.output,flush=True)
    print(json.dumps({a:report[f'{a}_engine'] for a in ('candidate','comparison')},indent=2))

if __name__=='__main__':main()
