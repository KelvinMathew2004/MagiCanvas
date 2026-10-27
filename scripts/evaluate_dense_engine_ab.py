#!/usr/bin/env python3
"""Evaluate fresh saved model outputs through one frozen Qwen-enabled selector."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def job(task):
    arm,item,raw=task
    os.environ['MAGICANVAS_QWEN_ENABLED']='1';os.environ['MAGICANVAS_CPU_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    model,adapter=('qwen3-0.6b','qwen3-layout-lora-v2') if arm=='0.6b' else ('qwen3-1.7b','qwen3-layout-lora-v2-expanded-1.7b')
    row=replay([item['record']],[raw],ROOT/'models'/adapter,ROOT/'models'/model,None,source_for=lambda split,key:deepcopy(item['source']),allow_incomplete=True)[0]
    if row.get('engine_error'):raise RuntimeError(row['engine_error'])
    if not item['record'].get('messages'):
        for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):row[k]=None
    row['original_raw_proposal_valid']=raw['raw_proposal_valid'];row['original_generation']=raw['generation']
    return row


def main(args):
    from scripts.evaluate_dense_qwen_ab import cohort
    items=cohort();args.output.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'engine').glob('*.py'))+[ROOT/'scripts/layout_defects.py',Path(__file__).resolve()]
    fp=dict(files_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},raw_sha256={arm:sha256((args.raw/(arm+'.jsonl')).read_bytes()).hexdigest() for arm in ('0.6b','1.7b')},qwen_enabled=True,fresh_model_inference=False,raw_fresh_inference=True)
    marker=args.output/'run_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise SystemExit('A/B selector or predictions changed; choose a new output directory')
    marker.write_text(json.dumps(fp,indent=2)+'\n')
    tasks=[];results={arm:{} for arm in ('0.6b','1.7b')}
    for arm in results:
        rows={r['pair_id']:r for r in map(json.loads,(args.raw/(arm+'.jsonl')).read_text().splitlines())}
        if len(rows)!=20:raise ValueError('Fresh arm is incomplete')
        for i,item in enumerate(items):
            path=args.output/f'{arm}_{i:02d}.json'
            if path.exists():results[arm][i]=json.loads(path.read_text())
            else:tasks.append((arm,i,(arm,item,rows[item['record']['pair_id']])))
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(job,t):(arm,i) for arm,i,t in tasks}
        for future in as_completed(pending):
            arm,i=pending[future];r=future.result();results[arm][i]=r
            (args.output/f'{arm}_{i:02d}.json').write_text(json.dumps(r)+'\n');print(arm,i+1,r['winner'],r['defect_flags'],flush=True)
    summary={}
    for arm,rows in results.items():
        values=list(rows.values());maes=[r['geometry_mae'] for r in values if r['geometry_mae'] is not None]
        summary[arm]=dict(cases=len(values),original_raw_valid=sum(r['original_raw_proposal_valid'] for r in values),flagged=sum(bool(r['defect_flags']) for r in values),qwen_wins=sum(r['qwen_family_win'] for r in values),mean_dataset_geometry_mae=sum(maes)/len(maes) if maes else None,generation_seconds=sum(r['original_generation'].get('generation_seconds',0) for r in values))
    a,b=summary['1.7b'],summary['0.6b']
    gate=a['original_raw_valid']>=b['original_raw_valid'] and a['flagged']<=b['flagged'] and a['mean_dataset_geometry_mae']<=b['mean_dataset_geometry_mae'] and a['qwen_wins']>=b['qwen_wins']
    report=dict(summary=summary,development_gate_passed=gate,default_changed=False,release_gate='Full 89-case re-evaluation and visual review still required before promotion',results={arm:[rows[i] for i in range(20)] for arm,rows in results.items()},raw_inference_files=str(args.raw),selector_manifest=fp)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=ROOT/'data/qwen_lora/dense_ab_fresh_v1');p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade/dense_engine_ab');main(p.parse_args())
