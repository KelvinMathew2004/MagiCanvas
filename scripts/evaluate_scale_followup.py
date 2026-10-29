#!/usr/bin/env python3
"""Paired 134-task development evaluation of growth, quotas and exact statistics.

Keeps decoded Qwen predictions fixed. Human target geometry is evaluation-only.
No experiment is enabled in production by this script.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
from time import perf_counter
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'data/upgrade_exploration/deps'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value)+'\n');temporary.replace(path)


def one_replay(case,manifest,arm,options):
    from unittest.mock import patch
    from scripts.replay_qwen_engine import replay
    from engine import raster_metrics
    from experiments.layout_upgrade.scale_followup import partition_statistics,quota_funnel
    import importlib
    resize=importlib.import_module('engine.resize');original=resize._visual_rerank
    stats_original=raster_metrics.glyph_statistics;timings=[];calls=[]
    def statistics(*args,**kwargs):
        # Alternate execution order on identical live patches; return the
        # incumbent implementation so profiling cannot change the baseline.
        if len(timings)%2:
            t=perf_counter();alternative=partition_statistics(*args,**kwargs);new_seconds=perf_counter()-t
            t=perf_counter();before=stats_original(*args,**kwargs);old_seconds=perf_counter()-t
        else:
            t=perf_counter();before=stats_original(*args,**kwargs);old_seconds=perf_counter()-t
            t=perf_counter();alternative=partition_statistics(*args,**kwargs);new_seconds=perf_counter()-t
        if before!=alternative:raise AssertionError('Exact order-statistics parity failed')
        timings.append(dict(active=before['ink_pixels'] if before else 0,sort_seconds=old_seconds,partition_seconds=new_seconds))
        return before
    def rank(candidates,backgrounds,units,source,width,height,desired):
        eligible=candidates;quota=None
        if arm=='quota-funnel':eligible,quota=quota_funnel(candidates,source,units,width,height,desired,options['candidate_budget'])
        winner,report=original(eligible,backgrounds,units,source,width,height,desired)
        calls.append(dict(inputs=len(candidates),quota=quota,render_cache=report.get('raster_cache',{}),
            rendered_candidates=len(report.get('candidates',[])),coverage=report.get('shortlist_coverage',{})))
        return winner,report
    started=perf_counter()
    with patch.object(resize,'_visual_rerank',rank),patch.object(raster_metrics,'glyph_statistics',statistics):
        row=replay([case['record']],[case['cached']],ROOT/manifest['adapter'],ROOT/manifest['model'],None,
            source_for=lambda *_:deepcopy(case['cached']['source_design']),allow_incomplete=True)[0]
    return dict(row=row,rerank_calls=calls,exact_statistics=timings,seconds=perf_counter()-started,
        timing_scope='Non-isolated replay with paired-statistics instrumentation in both arms; not generation latency or a deployment speed benchmark')


def growth(before,case,options):
    from experiments.layout_upgrade.scale_followup import container_growth
    from scripts.layout_defects import assess_engine_layout
    from engine.macro_layout import checks,quality
    import importlib
    resize=importlib.import_module('engine.resize')
    source=before['source_design'];doc=before['output_design'];w,h=(doc['canvas'][k] for k in ('width','height'))
    result=dict(changed=False,output_design=doc,defect_flags=before['defect_flags'],winner=before['winner'])
    if all(source['canvas'][k]==doc['canvas'][k] for k in ('width','height')):
        return dict(result,reason='Identity preserves authored layout')
    os.environ['MAGICANVAS_DATASET_RENDERING']='0' if before.get('official_path') else '1'
    backgrounds,units=resize._analyze(source);bgids={e['id'] for e in backgrounds}
    elements=[e for e in doc['elements'] if e['id'] not in bgids]
    proposed,diagnostic=container_growth(elements,source,w,h,**options)
    result['solver']=diagnostic
    if proposed is None:return dict(result,reason=diagnostic['reason'])
    candidate=dict(name='experimental-shared-scale-growth',elements=proposed,macro_locked=True,source_attached=True,
        energy=resize._energy(proposed,units,source,w,h,1.))
    proposal=resize._assemble(proposed,backgrounds,source,w,h)
    result['raw_proposal']=proposal
    preliminary=checks(source,proposal,units,{},candidate['name'])
    if preliminary['flagged']:return dict(result,reason='Proposal fails geometry',proposal_flags=preliminary['defect_flags'])
    winner,visual=resize._visual_rerank([candidate],backgrounds,units,source,w,h,1.)
    realized=resize._assemble(winner['elements'],winner.get('backgrounds',backgrounds),source,w,h)
    audit=assess_engine_layout(source,realized,dict(winner=winner['name'],visual=visual))
    verified=checks(source,realized,units,visual,winner['name'])
    q0=quality(elements,source,w,h);q1=quality(winner['elements'],source,w,h)
    result.update(realized_proposal=realized,proposal_flags=audit['defect_flags'],proposal_checks=verified,
        before_quality=q0,proposal_quality=q1,visual=visual)
    if audit['flagged'] or verified['flagged']:return dict(result,reason='Realized proposal fails raster/geometry')
    if q1['cost']>=q0['cost']-max(2.,q0['cost']*.15):return dict(result,reason='Insufficient source-only quality margin')
    if q1.get('content_union_share',0)<q0.get('content_union_share',0)*.90:
        return dict(result,reason='Content utilization loss')
    return dict(result,reason='Research replacement passes fixed gates; human review still required',changed=True,
        output_design=realized,defect_flags=audit['defect_flags'],winner=winner['name'])


def job(task):
    i,case,root,manifest,options=task;path=root/'paired'/f'{i:03d}.json'
    if path.exists():return i,json.loads(path.read_text())
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    result=dict(index=i,pair_id=case['pair_id'],cohort=case['cohort'],key=case['key'])
    # Alternate full replay order to avoid always warming one arm first.
    for arm in (('baseline','quota-funnel') if i%2==0 else ('quota-funnel','baseline')):
        result[arm]=one_replay(case,manifest,arm,options['quota-funnel'])
        if result[arm]['row'].get('engine_error'):
            save(path,result);return i,result
    result['container-growth']=growth(result['baseline']['row'],case,options['container-growth'])
    save(path,result);return i,result


def main():
    from scripts.evaluate_training_free_upgrades import freeze,engine_hashes
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/scale_followup_20261029')
    p.add_argument('--indices',type=int,nargs='+');p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();cases,manifest=freeze(a.root)
    config=ROOT/'config/algorithm_followup.json';options=json.loads(config.read_text())['arms']
    files=[Path(__file__),ROOT/'experiments/layout_upgrade/scale_followup.py',config]
    marker=dict(config=options,files_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},
        qwen_enabled=True,fresh_inference=False,production_changed=False,
        reference_used_only_for_reporting=True,tasks=len(cases))
    path=a.root/'followup_manifest.json'
    if path.exists():assert json.loads(path.read_text())==marker,'Experiment changed; use a new root'
    else:save(path,marker)
    ids=range(len(cases)) if a.indices is None else a.indices
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[(i,cases[i],a.root,manifest,options) for i in ids]
        for future in as_completed([pool.submit(job,t) for t in tasks]):
            i,r=future.result();b=r.get('baseline',{}).get('row',{});f=r.get('quota-funnel',{}).get('row',{})
            print(i,len(cases),r['pair_id'],'funnel:',b.get('winner')!=f.get('winner'),
                'growth:',r.get('container-growth',{}).get('changed'),
                b.get('engine_error') or f.get('engine_error') or '',flush=True)
    assert engine_hashes()==manifest['engine'],'Engine changed during comparison'
    print('Completed',len(tasks),'tasks',flush=True)


if __name__=='__main__':main()
