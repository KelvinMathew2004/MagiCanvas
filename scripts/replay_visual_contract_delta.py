#!/usr/bin/env python3
"""Replay every source that can dispatch the final control-iteration change.

For sources with no inferred controls, the only extra operation is repeated
source-ordered copy clearance, an idempotent forward pass. Certify the exact
previous code preimage and reuse those rows; re-execute all controlled sources.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD='''    _copy_clearance(result,source,width,height)
    _clear_controls(result,source,width,height)
    _control_visual_clearance(result,source,width,height)
    _control_peer_clearance(result,source,width,height)
'''
NEW='''    # A control cleared away from an image can enter the copy band. Alternate
    # these shared constraints until the short dependent chain settles.
    for _ in range(3):
        _control_visual_clearance(result,source,width,height)
        _copy_clearance(result,source,width,height)
        _clear_controls(result,source,width,height)
        _control_peer_clearance(result,source,width,height)
'''
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',choices=['official','crello','crello_1p7b','heldout'],required=True);a=p.parse_args()
    from engine.layout_contracts import controls,_copy_clearance
    from scripts.replay_qwen_engine import replay
    from scripts.evaluate_qwen_adapter import summarize_engine
    from copy import deepcopy
    names=[f'heldout_{c}' for c in 'abcd'] if a.cohort=='heldout' else [a.cohort]
    reports=[json.loads((a.prior/n/'report.json').read_text()) for n in names];rows=[r for d in reports for r in d['results']]
    content=(ROOT/'engine/layout_contracts.py').read_text();assert NEW in content
    preimage=sha256(content.replace(NEW,OLD).encode()).hexdigest()
    for d in reports:
        for name,value in d['manifest']['code_sha256'].items():
            expected=preimage if name=='engine/layout_contracts.py' else sha256((ROOT/name).read_bytes()).hexdigest()
            assert expected==value,(name,'unrelated code change cannot be certified')
    lookup={r['pair_id']:r for r in rows};records=[]
    if a.cohort=='official':
        records=[dict(pair_id=r['pair_id'],official_path=r['official_path'],target_canvas=r['target_canvas']) for r in rows]
    else:
        file='data/qwen_lora_expanded_v2/test.jsonl' if a.cohort=='heldout' else 'data/qwen_lora/ab_qwen3_1.7b_vs_0.6b_review_fixes/input_12.jsonl'
        records=[json.loads(l) for l in (ROOT/file).read_text().splitlines() if l.strip()];records=[r for r in records if r['pair_id'] in lookup]
    affected={r['pair_id'] for r in rows if controls(r['source_design'])}
    a.output.mkdir(parents=True,exist_ok=True)
    adapter=ROOT/'models/qwen3-layout-lora-v2';model=ROOT/'models/qwen3-0.6b'
    if a.cohort=='crello_1p7b':
        original=json.loads((ROOT/'data/qwen_lora/surgical_selector_delivery/report.json').read_text());adapter=Path(original['adapter_dir']);model=Path(original['model_dir'])
    changed=replay([r for r in records if r['pair_id'] in affected],[r for r in rows if r['pair_id'] in affected],adapter,model,a.output/'reexecuted.jsonl') if affected else []
    changes={r['pair_id']:r for r in changed};result=[]
    for row in rows:
        if row['pair_id'] in changes:r=changes[row['pair_id']];r['control_dispatch_reexecuted']=True
        else:
            r=deepcopy(row);before=deepcopy(r['output_design']['elements']);after=deepcopy(before)
            _copy_clearance(after,r['source_design'],r['target_canvas']['width'],r['target_canvas']['height'])
            for x,y in zip(before,after):
                assert max(abs(x[k]-y[k]) for k in ('x','y','width','height'))<1e-7
            r['control_dispatch_reexecuted']=False;r['unchanged_dispatch_reason']='No inferred controls; exact code-preimage certified; copy pass idempotent.'
        result.append(r)
    manifest=dict(cohort=a.cohort,cases=len(result),reexecuted=len(changed),certified_unchanged=len(result)-len(changed),prior_layout_contracts_sha256=preimage,code_sha256={n:sha256((ROOT/n).read_bytes()).hexdigest() for n in reports[0]['manifest']['code_sha256']},raw_coordinates_reused=True)
    (a.output/'report.json').write_text(json.dumps(dict(manifest=manifest,summary=summarize_engine(result),results=result),indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
