#!/usr/bin/env python3
"""Bounded source-only neighborhood and paragraph balancing on frozen requests."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'data/upgrade_exploration/industry_followup_20261029'
BASELINE=ROOT/'data/upgrade_exploration/scale_followup_20261029'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value)+'\n');temporary.replace(path)


def evaluate(before,arm,options):
    import importlib
    from engine.macro_layout import checks,quality
    from experiments.layout_upgrade.industry import paragraph_balance,group_neighbors
    from scripts.layout_defects import assess_engine_layout
    resize=importlib.import_module('engine.resize')
    source=before['source_design'];doc=before['output_design'];w,h=(doc['canvas'][k] for k in ('width','height'))
    backgrounds,units=resize._analyze(source);bgids={e['id'] for e in backgrounds}
    elements=[e for e in doc['elements'] if e['id'] not in bgids]
    result=dict(changed=False,output_design=doc,defect_flags=before['defect_flags'],winner=before['winner'],arm=arm)
    if all(source['canvas'][k]==doc['canvas'][k] for k in ('width','height')):
        return dict(result,reason='identity preserves authored layout')
    if arm=='paragraph-balance':
        proposal,note=paragraph_balance(elements,source,w,h,**options)
        proposals=[] if proposal is None else [dict(elements=proposal,kind=arm)]
    else:proposals,note=group_neighbors(elements,source,w,h,**options)
    result['generation']=note;result['generated']=len(proposals);result['screening']=[]
    q0=quality(elements,source,w,h);eligible=[]
    for p in proposals:
        q=quality(p['elements'],source,w,h)
        # No gold or learned-critic scores. Neighbors must improve the fixed
        # existing quality by a margin; balance must not worsen that cost.
        margin=options.get('minimum_quality_gain',0.)
        if q['cost']>q0['cost']-margin+1e-8:
            result['screening'].append(dict(kind=p['kind'],reason='quality margin',cost=q['cost']));continue
        candidate=deepcopy(doc);candidate['elements']=[deepcopy(e) for e in doc['elements'] if e['id'] in bgids]+p['elements']
        audit=checks(source,candidate,units,{},p['kind'])
        if audit['flagged']:
            result['screening'].append(dict(kind=p['kind'],reason='geometry/ink gate',flags=audit['defect_flags']));continue
        eligible.append((q['cost'],p,candidate))
    result['eligible']=len(eligible)
    if not eligible:return dict(result,reason='no qualifying proposal')
    _,p,proposal=min(eligible,key=lambda r:r[0]);result['raw_proposal']=proposal
    candidate=dict(name='experimental-'+arm,elements=p['elements'],macro_locked=True,source_attached=True,
        energy=resize._energy(p['elements'],units,source,w,h,1.))
    winner,visual=resize._visual_rerank([candidate],backgrounds,units,source,w,h,1.)
    realized=resize._assemble(winner['elements'],winner.get('backgrounds',backgrounds),source,w,h)
    audit=assess_engine_layout(source,realized,dict(winner=winner['name'],visual=visual))
    verified=checks(source,realized,units,visual,winner['name'])
    result.update(realized_proposal=realized,proposal_flags=audit['defect_flags'],proposal_checks=verified,
                  before_quality=q0,proposal_quality=quality(winner['elements'],source,w,h),visual=visual,kind=p['kind'])
    if audit['flagged'] or verified['flagged']:
        return dict(result,reason='realized raster/geometry rejection')
    # The normal realizer must not undo paragraph gain or violate the fixed
    # quality threshold. Keep all rejected intermediates for audit.
    if result['proposal_quality']['cost']>q0['cost']-options.get('minimum_quality_gain',0.)+1e-8:
        return dict(result,reason='post-realization quality margin')
    return dict(result,reason='passes fixed research gates; visual review required',changed=True,
                output_design=realized,defect_flags=audit['defect_flags'],winner=winner['name'])


def job(task):
    i,case,root,options=task;path=root/'results'/f'{i:03d}.json'
    if path.exists():return i,json.loads(path.read_text())
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    baseline=json.loads((BASELINE/'paired'/f'{i:03d}.json').read_text())['baseline']['row']
    os.environ['MAGICANVAS_DATASET_RENDERING']='0' if baseline.get('official_path') else '1'
    result=dict(index=i,pair_id=case['pair_id'],baseline=baseline)
    for arm,parameters in options.items():
        started=perf_counter();result[arm]=evaluate(baseline,arm,parameters)
        result[arm]['experiment_seconds']=perf_counter()-started
    save(path,result);return i,result


def main():
    from scripts.evaluate_training_free_upgrades import freeze,engine_hashes
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--indices',type=int,nargs='+');p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();cases,manifest=freeze(a.root)
    previous=json.loads((BASELINE/'manifest.json').read_text())
    assert previous==manifest,'Do not compare against a stale production baseline'
    assert json.loads((BASELINE/'cases.json').read_text())==cases
    config=ROOT/'config/industry_followup.json';options=json.loads(config.read_text())
    files=[Path(__file__),ROOT/'experiments/layout_upgrade/industry.py',config]
    marker=dict(config=options,files_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},
                baseline_manifest_sha256=sha256((BASELINE/'followup_manifest.json').read_bytes()).hexdigest(),
                baseline_row_sha256={f'{i:03d}':sha256((BASELINE/'paired'/f'{i:03d}.json').read_bytes()).hexdigest() for i in range(len(cases))},
                tasks=len(cases),fresh_inference=False,production_changed=False,qwen_enabled=True)
    path=a.root/'experiment_manifest.json'
    if path.exists():assert json.loads(path.read_text())==marker,'Experiment changed; use a new root'
    else:save(path,marker)
    ids=range(len(cases)) if a.indices is None else a.indices
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(job,(i,cases[i],a.root,options['arms'])) for i in ids]):
            i,r=future.result();print(i,len(cases),r['pair_id'],{a:(r[a].get('generated',0),r[a]['changed']) for a in options['arms']},flush=True)
    assert engine_hashes()==manifest['engine'],'Engine changed during experiment'


if __name__=='__main__':main()
