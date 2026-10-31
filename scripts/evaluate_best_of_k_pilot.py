#!/usr/bin/env python3
"""Compare one decode with three decodes through the unchanged engine selector.

Requires already recorded inference; never uses reference MAE to choose a winner.
Only the experiment injects alternatives. Production remains one decode.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from collections import Counter
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def job(task):
    case,arms,model,out=task
    import os,torch,importlib
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    torch.set_num_threads(1)
    from scripts.replay_upgrade_pilots import replay_one
    resize=importlib.import_module('engine.resize')
    source,target=case['source'],case['target'];alternatives=[];singles=[]
    for index,raw in enumerate(arms):
        result=replay_one(case,raw,model,out,trace_repair=True)
        if result.get('engine_error'):raise ValueError(result['engine_error'])
        singles.append(result)
        if not index:continue
        stage=result.get('repair_stages',{}).get('realized_qwen_candidate')
        if stage is None:continue
        backgrounds,units=resize._analyze(source)
        background_ids={e['id'] for e in backgrounds}
        es=[e for e in stage['design']['elements'] if e['id'] not in background_ids]
        # Use the same physical gate as a production pre-container proposal.
        # Render checks still apply, even when this coarse gate passes.
        if not resize._physical_layout_safe(es,units,source,**target):continue
        # _energy's desired map is already available in the rank call. Fill
        # energy there so the experiment does not duplicate layout heuristics.
        alternatives.append(dict(name=f'qwen-sampled-{index}',elements=deepcopy(es),energy=0.))
    if alternatives:
        from unittest.mock import patch
        original=resize._visual_rerank
        def rank(candidates,bg,units,source,width,height,desired):
            for c in candidates:
                if c['name'].startswith('qwen-sampled-'):
                    c['energy']=resize._energy(c['elements'],units,source,width,height,desired)
            return original(candidates,bg,units,source,width,height,desired)
        with patch.object(resize,'_visual_rerank',rank):
            combined=replay_one(case,arms[0],model,out,extra_candidates=alternatives)
    else:combined=deepcopy(singles[0])
    return dict(pair_id=case['pair_id'],source=source,target=target,singles=singles,combined=combined,
        offered_samples=[c['name'] for c in alternatives],
        selected_output_changed=singles[0]['output_design']!=combined['output_design'],
        new_flags=sorted(set(combined['defect_flags'])-set(singles[0]['defect_flags'])),
        total_generation_seconds=sum(r.get('generation_seconds',0) for r in arms),
        baseline_generation_seconds=arms[0].get('generation_seconds'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/best_of_k_v1')
    p.add_argument('--greedy',type=Path,default=ROOT/'data/upgrade_exploration/native_pilot_v1/qwen06_text_grammar.jsonl')
    p.add_argument('--workers',type=int,default=2,choices=(1,2));a=p.parse_args()
    samples=[[json.loads(line) for line in (a.root/f'qwen06_text_grammar_sample{s}.jsonl').read_text().splitlines()] for s in (17,31)]
    greedy={r['pair_id']:r for r in map(json.loads,a.greedy.read_text().splitlines())}
    model=Path(json.loads((a.root/'qwen06_text_grammar_sample17.manifest.json').read_text())['model'])
    out=a.root/'selection';out.mkdir(exist_ok=True)
    files=list((ROOT/'engine').glob('*.py'))+[Path(__file__),ROOT/'scripts/replay_upgrade_pilots.py']
    fp=dict(files_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},
        greedy_sha256=sha256(a.greedy.read_bytes()).hexdigest(),sample_sha256=[sha256((a.root/f'qwen06_text_grammar_sample{s}.jsonl').read_bytes()).hexdigest() for s in (17,31)],
        qwen_enabled=True,production_changed=False,fresh_generation_during_selection=False,k=3,
        scope='Small development pilot; shared selector, primary pool retained, two sample siblings inside learned shortlist cap. No oracle selection.')
    marker=out/'manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise ValueError('Experiment changed; use a new output directory')
    marker.write_text(json.dumps(fp,indent=2)+'\n');tasks=[];results=[]
    for i,case in enumerate(samples[0]):
        arms=[greedy[case['pair_id']],case,next(r for r in samples[1] if r['pair_id']==case['pair_id'])]
        if any(r['source']!=case['source'] or r['target']!=case['target'] for r in arms):raise ValueError('Unmatched source/target')
        tasks.append((case,arms,model,out))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(job,tasks)):
            (out/f'{i:02d}.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
            print(result['pair_id'],'changed:',result['selected_output_changed'],'new flags:',result['new_flags'],flush=True)
    summary=dict(cases=len(results),k=3,changed=sum(r['selected_output_changed'] for r in results),
        before_flagged=sum(bool(r['singles'][0]['defect_flags']) for r in results),
        after_flagged=sum(bool(r['combined']['defect_flags']) for r in results),
        selected_sample=sum(r['combined']['winner'].startswith('qwen-sampled-') for r in results),
        added_flags=dict(Counter(f for r in results for f in r['new_flags'])),
        records=[{k:v for k,v in r.items() if k not in ('source','singles','combined')}|dict(before_winner=r['singles'][0]['winner'],winner=r['combined']['winner'],before_flags=r['singles'][0]['defect_flags'],flags=r['combined']['defect_flags']) for r in results])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
