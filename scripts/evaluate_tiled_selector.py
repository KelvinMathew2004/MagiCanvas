#!/usr/bin/env python3
"""Paired full-engine tiled-SigLIP experiment on all 89 saved test pairs.

Keeps the real contextual score, visual selector and physical safety gates.
Source/target-identical tasks reuse a result, retaining all reference aliases.
"""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def tiled_scores(original,images,*args,**kwargs):
    from scripts.evaluate_critic_pilot import tiles
    images=list(images);expanded=[];counts=[]
    for image in images:
        local=tiles(image);expanded.extend([image,*local]);counts.append(len(local)+1)
    result=original(expanded,*args,**kwargs)
    if result is None:return None
    scores,note=result
    if len(scores)!=len(expanded):return None
    offset=0;combined=[]
    for count in counts:
        local=scores[offset:offset+count];offset+=count
        combined.append(.5*float(local[0])+.5*sum(float(s) for s in local[1:])/(count-1))
    return combined,note+'; experimental fixed half-global/half-three-tiles'


def job(task):
    i,case,root,manifest=task;file=root/'tiled_selector'/f'{i:03d}.json'
    if file.exists():return i,json.loads(file.read_text())
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    from unittest.mock import patch
    from engine import vision
    from scripts.replay_qwen_engine import replay
    original=vision.score_layouts
    with patch.object(vision,'score_layouts',lambda *a,**k:tiled_scores(original,*a,**k)):
        row=replay([case['record']],[case['cached']],ROOT/manifest['adapter'],ROOT/manifest['model'],None,
            source_for=lambda *_:deepcopy(case['cached']['source_design']),allow_incomplete=True)[0]
    before=json.loads((root/'baseline'/f'{i:03d}.json').read_text())['baseline']
    result=dict(pair_id=case['pair_id'],key=case['key'],before_winner=before['winner'],winner=row.get('winner'),
        changed=before['output_design']!=row.get('output_design'),baseline_flags=before['defect_flags'],flags=row.get('defect_flags'),
        new_flags=sorted(set(row.get('defect_flags',[]))-set(before['defect_flags'])),after=row)
    file.write_text(json.dumps(result)+'\n');return i,result


def main():
    from scripts.evaluate_training_free_upgrades import engine_hashes
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/training_free_scale_v1');a=p.parse_args()
    cases=json.loads((a.root/'cases.json').read_text());manifest=json.loads((a.root/'manifest.json').read_text());assert engine_hashes()==manifest['engine']
    aliases=json.loads((a.root/'test_pair_tasks.json').read_text());keys={r['task_key'] for r in aliases};tasks=[(i,c,a.root,manifest) for i,c in enumerate(cases) if c['key'] in keys]
    out=a.root/'tiled_selector';out.mkdir(exist_ok=True);fp=dict(engine=manifest['engine'],script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),pair_count=len(aliases),distinct_tasks=len(tasks),weights_fixed=[.5,.5],qwen_enabled=True,fresh_inference=False)
    mark=out/'manifest.json'
    if mark.exists():assert json.loads(mark.read_text())==fp
    else:mark.write_text(json.dumps(fp,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,t) for t in tasks]):
            i,r=future.result();print(i+1,len(cases),r['pair_id'],r['changed'],r['new_flags'],flush=True)
    assert manifest['engine']==engine_hashes()
    print('Finished',len(tasks),'distinct tasks /',len(aliases),'test pairs')


if __name__=='__main__':main()
