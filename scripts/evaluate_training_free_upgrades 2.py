#!/usr/bin/env python3
"""Frozen, paired no-training algorithm comparison; no target-guided selection.

All 89 saved test proposals plus the current regression cohort, deduplicated by
source/target. Exact decoded Qwen coordinates remain fixed. No Qwen-off arm.
"""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
from time import perf_counter
import argparse,json,os,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'data/upgrade_exploration/training_free_scale_v1'


def engine_hashes():
    paths=list((ROOT/'engine').glob('*.py'))+list((ROOT/'renderer/converter').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def freeze(root):
    from scripts.refresh_regression_viewer import fingerprint,PACK
    from scripts.evaluate_qwen_adapter import read_sources
    current=PACK/'review/regression_suite';manifest=json.loads((current/'run_manifest.json').read_text())
    assert all(manifest['fingerprint'].get(k)==v for k,v in fingerprint(ROOT/'config/regression_review.json').items()),'Regression viewer stale'
    run=ROOT/'data/qwen_lora/regression_suite'/sha256(json.dumps(manifest['fingerprint'],sort_keys=True).encode()).hexdigest()[:16]
    # The recorded comparison arm is fixed in advance, not chosen by quality.
    rawpath=ROOT/'data/qwen_lora/prompt_ab_same_adapter_20261025/old_prompt.jsonl'
    provenance=rawpath.parent/'run_manifest.json'
    records=[json.loads(l) for l in (ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines()]
    raw={r['pair_id']:r for r in map(json.loads,rawpath.read_text().splitlines())};source_for=read_sources()
    cases=[];seen=set()
    def key(source,target):return sha256(json.dumps([source,target],sort_keys=True).encode()).hexdigest()
    # Preserve the user's regression predictions for overlapping tasks. This
    # avoids changing established review examples to another historical decode.
    for i,case in enumerate(manifest['cohort']['cases']):
        r=json.loads((run/f'{i:02d}.json').read_text());row=r['after'];source=row['source_design'];target=row['target_canvas'];k=key(source,target)
        if k in seen:continue
        seen.add(k);cases.append(dict(key=k,pair_id=case['pair_id'],cohort=case['cohort'],record=r['record'],cached=row,
                                     baseline_reuse=str(run/f'{i:02d}.json'),original_test_pair=case['pair_id'] in raw))
    for record in records:
        source=source_for(record['source_split'],record['pair_id'].split('__to__')[0]);target=record['target_canvas'];k=key(source,target)
        if k in seen:continue
        seen.add(k);r=raw[record['pair_id']]
        es=[{k:e[k] for k in ('id','x','y','width','height','font_size') if k in e} for e in r['raw_elements']]
        cached=dict(pair_id=record['pair_id'],source_design=source,target_canvas=target,source_split=record['source_split'],qwen_model_elements=es,
            raw_json_complete=r.get('raw_json_complete'),raw_proposal_valid=bool(r.get('raw_json_complete') and not r.get('raw_missing_ids') and not r.get('raw_extra_ids')),
            qwen_status='Recorded old-v2 prompt arm; exact decoded fields reused')
        cases.append(dict(key=k,pair_id=record['pair_id'],cohort='test-expanded',record=record,cached=cached,original_test_pair=True))
    aliases=[dict(pair_id=r['pair_id'],task_key=key(source_for(r['source_split'],r['pair_id'].split('__to__')[0]),r['target_canvas']),record=r) for r in records]
    assert all(r['task_key'] in seen for r in aliases)
    fp=dict(engine=engine_hashes(),qwen_enabled=True,fresh_inference=False,
        model=manifest['cohort']['model'],adapter=manifest['cohort']['adapter'],
        cases_sha256=sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
        recorded_test_run_sha256=sha256(provenance.read_bytes()).hexdigest(),
        test_raw_sha256=sha256(rawpath.read_bytes()).hexdigest(),
        regression_manifest_sha256=sha256((current/'run_manifest.json').read_bytes()).hexdigest(),
        baseline_scope='Current engine on exact saved predictions; current regression results reused only under matching fingerprints. No generation latency measurement.',
        counts=dict(Counter(c['cohort'] for c in cases)),requested_test_pairs_covered=len(aliases),distinct_test_tasks=len({r['task_key'] for r in aliases}))
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        assert json.loads((root/'manifest.json').read_text())==fp,'Inputs or engine changed; use a new root'
    else:
        (root/'manifest.json').write_text(json.dumps(fp,indent=2)+'\n');(root/'cases.json').write_text(json.dumps(cases)+'\n')
        (root/'test_pair_tasks.json').write_text(json.dumps(aliases)+'\n')
    return cases,fp


def job(task):
    index,case,phase,root,arms,model,adapter=task
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    out=root/phase;out.mkdir(exist_ok=True);path=out/f'{index:03d}.json'
    if path.exists():return index,json.loads(path.read_text())
    started=perf_counter()
    if phase=='baseline':
        if case.get('baseline_reuse'):
            row=deepcopy(json.loads(Path(case['baseline_reuse']).read_text())['after']);reuse=True
        else:
            from scripts.replay_qwen_engine import replay
            row=replay([case['record']],[case['cached']],adapter,model,None,
                       source_for=lambda *_:deepcopy(case['cached']['source_design']),allow_incomplete=True)[0];reuse=False
        result=dict(pair_id=case['pair_id'],key=case['key'],cohort=case['cohort'],baseline=row,reused_exact_current_result=reuse)
    else:
        before=json.loads((root/'baseline'/f'{index:03d}.json').read_text())['baseline']
        if before.get('engine_error'):result=dict(pair_id=case['pair_id'],error=before['engine_error'])
        else:
            from experiments.layout_upgrade.training_free import evaluate
            result=evaluate(case,before,arms,root)
    result['evaluation_seconds']=perf_counter()-started
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result)+'\n');temporary.replace(path)
    return index,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--phase',choices=('baseline','algorithms'),default='baseline');p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--arms',nargs='+',default=['micro-polish','whitespace','calm-zone','retrieval','dual-granularity'])
    p.add_argument('--indices',type=int,nargs='+');a=p.parse_args();cases,fp=freeze(a.root)
    if a.phase=='algorithms':
        files=list((ROOT/'experiments/layout_upgrade').glob('*.py'))+[Path(__file__)]
        exp=dict(files_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},arms=a.arms,
            incumbent_retained=True,reference_geometry_not_used_for_selection=True)
        marker=a.root/'algorithms_manifest.json'
        if marker.exists():assert json.loads(marker.read_text())==exp,'Algorithm code changed; use a separate root/output snapshot'
        else:marker.write_text(json.dumps(exp,indent=2)+'\n')
    selected=range(len(cases)) if a.indices is None else a.indices
    tasks=[(i,cases[i],a.phase,a.root,a.arms,ROOT/fp['model'],ROOT/fp['adapter']) for i in selected]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(job,t):t[0] for t in tasks}
        for future in as_completed(pending):
            i,r=future.result();print(a.phase,i+1,len(cases),r['pair_id'],r.get('error') or r.get('baseline',{}).get('winner') or {k:v['changed'] for k,v in r.get('arms',{}).items()},flush=True)
    assert fp['engine']==engine_hashes(),'Engine changed during experiment'
    print('Completed',a.phase,len(tasks),'of',len(cases),flush=True)


if __name__=='__main__':main()
