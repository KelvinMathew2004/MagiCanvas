#!/usr/bin/env python3
"""Apply one unchanged selector to fresh model outputs or frozen CP-SAT trials."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0,str(ROOT/'data/upgrade_exploration/deps'))


def replay_one(case, raw, model_path, output, solver=False, trace_repair=False, extra_candidates=()):
    from scripts.replay_qwen_engine import replay
    from engine import macro_layout
    # Reuse each production cohort's renderer/asset mode and reference record.
    # Brand-like typography alone is not evidence that a dataset-backed design
    # should switch to the official asset resolver.
    from scripts.refresh_regression_viewer import load_cohort
    spec=load_cohort(ROOT/'config/regression_review.json')
    definition=next(c for c in spec['cases'] if c['pair_id']==case['pair_id'])
    prior=json.loads((ROOT/definition['prediction_report']).read_text())
    old=next(r for r in prior['results'] if r['pair_id']==case['pair_id'])
    record=next((r for r in prior.get('records',[]) if r['pair_id']==case['pair_id']),None)
    if record is None and definition.get('records_file'):
        record=next(r for r in map(json.loads,(ROOT/definition['records_file']).read_text().splitlines()) if r['pair_id']==case['pair_id'])
    if record is None:
        record={k:old[k] for k in ('pair_id','source_split','target_canvas','family_id') if k in old}
    types={e['id']:e['type'] for e in case['source']['elements']}
    # The engine parser ignores font_scale on non-text layers. Preserve the
    # literal payload in inference JSON, but compare its actual parser fields.
    parsed=[{k:v for k,v in e.items() if k!='font_size' or types.get(e['id'])=='text'}
            for e in raw.get('qwen_model_elements',[])]
    cached=dict(pair_id=case['pair_id'], source_design=case['source'], source_split=record['source_split'],
                target_canvas=case['target'], qwen_model_elements=parsed,
                raw_proposal_valid=raw.get('raw_proposal_valid',False),
                qwen_status=raw.get('qwen_status','Experimental cached proposal'))
    marker=output/'prompt_contract';marker.mkdir(exist_ok=True)
    (marker/'magicanvas_prompt_version.txt').write_text('2\n')
    original=macro_layout.candidates;generated=[]
    def augmented(source,width,height,proposal=None,*,tight=None,extras=True):
        result=original(source,width,height,proposal,tight=tight,extras=extras)
        if tight is None and extras:
            from experiments.layout_upgrade.packing import candidates
            for granularity in (True,False):
                added=candidates(source,width,height,proposal,seconds=.3,tight=granularity)
                generated.extend(dict(name=c['name'],solver=c['solver']) for c in added);result+=added
        return result
    context=patch.object(macro_layout,'candidates',augmented if solver else original)
    from contextlib import ExitStack
    # Resolve the module, not the package's exported resize function.
    import importlib
    resize_module=importlib.import_module('engine.resize')
    from engine import qwen_seed
    original_seed=qwen_seed.generate_seed;original_rank=resize_module._visual_rerank;stages={}
    def seed_trace(*args,**kwargs):
        result=original_seed(*args,**kwargs)
        if result[0] is not None:stages['parsed_aspect_font_repair']=deepcopy(result[0])
        return result
    def rank_trace(candidates,*args,**kwargs):
        if extra_candidates and not stages.get('sample_pool_injected'):
            # Experiment only: keep the incumbent pool and its primary learned
            # representative, then offer one realized proposal per extra decode
            # inside the existing six-learned-candidate shortlist allowance.
            # Normal raster/hard-rule selection decides; no target labels enter.
            candidates=list(candidates)
            first=next((i for i,c in enumerate(candidates) if c['name'].startswith('qwen-')),len(candidates)-1)
            candidates[first+1:first+1]=deepcopy(list(extra_candidates))
            stages['sample_pool_injected']=True
        if 'realized_qwen_candidate' not in stages:
            priorities=('qwen-container-locked-font-fitted','qwen-pre-container-font-fitted','qwen-original')
            primary=next((c for name in priorities for c in candidates if c['name']==name),None)
            if primary:
                stages['realized_qwen_candidate']=deepcopy(primary['elements'])
                stages['realized_qwen_name']=primary['name']
        return original_rank(candidates,*args,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(context)
        if trace_repair:
            stack.enter_context(patch.object(qwen_seed,'generate_seed',seed_trace))
        if trace_repair or extra_candidates:
            stack.enter_context(patch.object(resize_module,'_visual_rerank',rank_trace))
        result=replay([record],[cached],marker,model_path,None,
                      source_for=lambda *_:deepcopy(case['source']),allow_incomplete=True)[0]
    if trace_repair:
        from scripts.layout_defects import assess_engine_layout
        result['repair_stages']={}
        for name,es in stages.items():
            if not isinstance(es,list):continue
            bg_ids={e['id'] for e in case['source']['elements']} - {e['id'] for e in es}
            # Seed/candidate lists omit engine-owned backgrounds. Reconstruct
            # those through the engine's ordinary background assembly policy.
            backgrounds,_=resize_module._analyze(case['source'])
            doc=resize_module._assemble(es,backgrounds,case['source'],**case['target'])
            result['repair_stages'][name]=dict(design=doc,defects=assess_engine_layout(case['source'],doc,{}),
                contrast_evaluated=False,omitted_from_leaf_list=sorted(bg_ids),name=stages.get('realized_qwen_name') if name=='realized_qwen_candidate' else None)
    result['actual_fresh_model_validity']=raw.get('raw_proposal_valid')
    result['actual_fresh_json_complete']=raw.get('raw_json_complete')
    result['solver_generated']=generated
    result['model_generation_seconds']=raw.get('generation_seconds')
    result['actual_model_tuned']=raw.get('tuned')
    # Only dataset tasks with actual human reference messages have MAE.
    if not record.get('messages'):
        for key in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):result[key]=None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/model_pilot_v1')
    p.add_argument('--arm',required=True,help='Fresh model checkpoint stem, or cpsat for frozen proposals')
    p.add_argument('--tag',default='',help='Separate output revision without overwriting prior evidence')
    p.add_argument('--trace-repair',action='store_true',help='Audit parsed seed and realized learned candidate without changing selection')
    p.add_argument('--indices',nargs='+',type=int,default=[5,14,18,49,30,40])
    args=p.parse_args();out=args.root/'replay'/(args.arm+('_'+args.tag if args.tag else ''));out.mkdir(parents=True,exist_ok=True)
    from scripts.layout_defects import assess_engine_layout
    from experiments.layout_upgrade.contracts import raw_document
    solver=args.arm=='cpsat'
    if solver:
        viewer=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/regression_suite'
        manifest=json.loads((viewer/'manifest.json').read_text());rows=[]
        for i in args.indices:
            entry=manifest[i];a=json.loads((viewer/entry['audit']).read_text());d=a['diagnostics']
            rows.append(dict(pair_id=entry['pair_id'],cohort=entry['cohort'],source=a['source'],
                source_split=a.get('source_split') or 'train',target=a['target'],
                qwen_model_elements=d['raw_model_elements'],raw_proposal_valid=d.get('original_raw_proposal_valid'),
                raw_json_complete=d.get('original_raw_json_complete'),qwen_status=d.get('qwen_status'),tuned=True,
                baseline_design=a['design'],baseline_winner=d['winner'],baseline_flags=a.get('defects',{}).get('defect_flags',[])))
        model=ROOT/'models/qwen3-0.6b'
    else:
        rows=[json.loads(s) for s in (args.root/(args.arm+'.jsonl')).read_text().splitlines()]
        model=Path(json.loads((args.root/(args.arm+'.manifest.json')).read_text())['model'])
    fp=dict(engine={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in (ROOT/'engine').glob('*.py')},
            experiment={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'experiments/layout_upgrade/packing.py']},
            rows_sha256=sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),solver=solver,qwen_enabled=True,
            fresh_inference=False,trace_repair=args.trace_repair,scope='Frozen prediction repair/selection; actual inference provenance retained')
    marker=out/'manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise ValueError('Replay inputs/code changed; use a new experiment root')
    marker.write_text(json.dumps(fp,indent=2)+'\n');results=[]
    for i,raw in enumerate(rows):
        file=out/f'{i:02d}.json'
        if file.exists():result=json.loads(file.read_text())
        else:
            try:
                result=replay_one(raw,raw,model,out,solver,args.trace_repair)
                if not result.get('engine_error'):
                    os.environ['MAGICANVAS_DATASET_RENDERING']='0' if result.get('official_path') else '1'
                    document=raw_document(raw['source'],raw['target'],raw.get('qwen_model_elements',[]))
                    before=assess_engine_layout(raw['source'],document,{})
                    before.update(contrast_evaluated=False,low_contrast_count=None)
                    result.update(raw_defects=before)
                if solver:
                    control=replay_one(raw,raw,model,out,False)
                    result.update(control=control,baseline_flags=control.get('defect_flags',[]),
                                  baseline_winner=control.get('winner'),
                                  baseline_unchanged=result.get('output_design')==control.get('output_design'),
                                  published_baseline_unchanged=control.get('output_design')==raw['baseline_design'])
            except Exception as exc:
                result=dict(pair_id=raw['pair_id'],engine_error=f'{type(exc).__name__}: {exc}',defect_flags=['engine_error'])
            file.write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        print(args.arm,i+1,len(rows),result.get('winner'),result.get('defect_flags'),flush=True)
    summary=dict(arm=args.arm,cases=len(results),qwen_enabled=True,fresh_model_inference=False,
         raw_valid=sum(bool(r.get('actual_fresh_model_validity')) for r in results),
         learned_winners=sum(bool(r.get('qwen_family_win')) for r in results),
         selected_flags=dict(Counter(f for r in results for f in r.get('defect_flags',[]))),
         raw_flags=dict(Counter(f for r in results for f in r.get('raw_defects',{}).get('defect_flags',[]))),
         flagged=sum(bool(r.get('defect_flags')) for r in results),
         engine_errors=[r.get('engine_error') for r in results if r.get('engine_error')],
         changed=sum(not r.get('baseline_unchanged',True) for r in results) if solver else None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
