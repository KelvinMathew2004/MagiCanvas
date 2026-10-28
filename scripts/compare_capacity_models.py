#!/usr/bin/env python3
"""Replay both freshly generated arms through one current selector and summarize."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def job(task):
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    arm,record,raw=task
    model,adapter={'0.6b':('qwen3-0.6b','qwen3-layout-lora-v2'),
                   '1.7b':('qwen3-1.7b','qwen3-layout-lora-v2-expanded-1.7b')}[arm]
    result=replay([record],[raw],ROOT/'models'/adapter,ROOT/'models'/model,None,
                  source_for=lambda *_:deepcopy(raw['source_design']),allow_incomplete=True)[0]
    if result.get('engine_error'):raise RuntimeError(result['engine_error'])
    result.update(original_raw_proposal_valid=raw['raw_proposal_valid'],original_generation=raw.get('generation',{}),original_wall_seconds=raw['runtime'])
    for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):result[k]=None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/prompt_capacity_ab')
    p.add_argument('--reuse-regression-run',type=Path,help='Reuse verified current-selector 0.6B results from the completed fixed-cohort replay.')
    args=p.parse_args()
    base=args.root.resolve();out=base/'selector';out.mkdir(exist_ok=True)
    reports={a:json.loads((base/a/'report.json').read_text()) for a in ('0.6b','1.7b')}
    assert all(len(r['results'])==13 for r in reports.values())
    assert [r['pair_id'] for r in reports['0.6b']['records']]==[r['pair_id'] for r in reports['1.7b']['records']]
    assert reports['0.6b']['manifest']['sources']==reports['1.7b']['manifest']['sources']
    for path in ('engine/qwen_prompt.py','engine/qwen_seed.py','engine/runtime.py'):
        assert reports['0.6b']['manifest']['engine'][path]==reports['1.7b']['manifest']['engine'][path],path
    files=list((ROOT/'engine').glob('*.py'))+[ROOT/'scripts/evaluate_qwen_adapter.py',ROOT/'scripts/replay_qwen_engine.py',Path(__file__).resolve()]
    fp=dict(engine={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},reports={a:sha256((base/a/'report.json').read_bytes()).hexdigest() for a in reports},qwen_enabled=True,fresh_inference=False)
    reusable={}
    current_engine={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
    # Fresh inference already executes the complete selector. Reuse it only
    # when every engine digest matches this comparison's selector snapshot.
    for arm,report in reports.items():
        if report['manifest']['engine']==current_engine:
            reusable[arm]={r['pair_id']:r for r in report['results']}
    if args.reuse_regression_run:
        run=args.reuse_regression_run.resolve();mf=json.loads((run/'run_manifest.json').read_text())
        assert all(mf['fingerprint']['files_sha256'].get(k)==v for k,v in fp['engine'].items() if k.startswith('engine/') or k in ('scripts/evaluate_qwen_adapter.py','scripts/replay_qwen_engine.py'))
        assert mf['fingerprint']['adapter']=='models/qwen3-layout-lora-v2'
        reusable['0.6b']={v['case']['pair_id']:v['after'] for v in (json.loads(p.read_text()) for p in run.glob('[0-9][0-9].json'))}
        fp['reused_regression_manifest']=sha256((run/'run_manifest.json').read_bytes()).hexdigest()
    marker=out/'manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise SystemExit('Selector changed; choose a new experiment root')
    marker.write_text(json.dumps(fp,indent=2)+'\n');tasks=[];results={a:{} for a in reports}
    for arm,r in reports.items():
        for i,(record,raw) in enumerate(zip(r['records'],r['results'])):
            path=out/f'{arm}_{i:02d}.json'
            if path.exists():results[arm][i]=json.loads(path.read_text())
            elif record['pair_id'] in reusable.get(arm,{}):
                result=deepcopy(reusable[arm][record['pair_id']])
                assert result['source_design']==raw['source_design'] and result['target_canvas']==raw['target_canvas']
                a={e['id']:e for e in result['qwen_model_elements']};b={e['id']:e for e in raw['qwen_model_elements']}
                assert set(a)==set(b) and all(abs(a[i].get(k,0)-b[i].get(k,0))<1e-6 for i in a for k in ('x','y','width','height','font_size'))
                result.update(original_raw_proposal_valid=raw['raw_proposal_valid'],original_generation=raw.get('generation',{}),original_wall_seconds=raw['runtime'],verified_selector_result_reused=True)
                for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):result[k]=None
                results[arm][i]=result;path.write_text(json.dumps(result)+'\n')
            else:tasks.append((arm,i,(arm,record,raw)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(job,t):(a,i) for a,i,t in tasks}
        for f in as_completed(pending):
            a,i=pending[f];r=f.result();results[a][i]=r
            (out/f'{a}_{i:02d}.json').write_text(json.dumps(r)+'\n');print(a,i+1,r['winner'],r['defect_flags'],flush=True)
    stats={}
    for arm in reports:
        rows=[results[arm][i] for i in range(13)]
        (out/f'{arm}_report.json').write_text(json.dumps(dict(records=reports[arm]['records'],results=rows,selector_manifest=fp,inference_manifest=reports[arm]['manifest'],fresh_model_inference=False,human_target_available=False),indent=2)+'\n')
        stats[arm]=dict(cases=len(rows),prompt_budget_refusals=sum('exceeds context budget' in r.get('qwen_status','') for r in rows),json_complete=sum(bool(r['original_generation'].get('json_complete')) for r in rows),raw_inventory_valid=sum(bool(r['original_raw_proposal_valid']) for r in rows),flagged=sum(bool(r['defect_flags']) for r in rows),qwen_wins=sum(r['qwen_family_win'] for r in rows),defect_cost=sum(r.get('layout_defect_cost',0) for r in rows),generation_seconds=sum(r['original_generation'].get('generation_seconds',0) for r in rows),fresh_wall_seconds=sum(r['original_wall_seconds'] for r in rows))
    comparison=[]
    for i in range(13):
        a,b=results['0.6b'][i],results['1.7b'][i]
        comparison.append(dict(pair_id=a['pair_id'],flags_06=a['defect_flags'],flags_17=b['defect_flags'],winner_06=a['winner'],winner_17=b['winner'],cost_06=a.get('layout_defect_cost'),cost_17=b.get('layout_defect_cost'),same_output=a['output_design']==b['output_design']))
    (out/'comparison.json').write_text(json.dumps(dict(summary=stats,comparisons=comparison,default_changed=False,scope='13 development stress tasks; fresh proposals, identical current selector; no human target MAE; wider release/visual gate still required'),indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
