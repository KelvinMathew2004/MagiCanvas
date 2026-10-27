#!/usr/bin/env python3
"""Re-evaluate a saved model prediction against current repairs and selector.

No inference is rerun: paired outputs use identical decoded coordinates. This
is an engine regression audit, not a model or prompt A/B experiment.
"""
from pathlib import Path
import argparse,json,sys,os
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['official','crello','heldout'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--arm',choices=['candidate','comparison'],default='comparison')
    p.add_argument('--pair-contains')
    p.add_argument('--pair-list',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from scripts.replay_qwen_engine import replay
    from scripts.evaluate_qwen_adapter import summarize_engine,read_sources
    from scripts.compare_qwen_prompt_variants import defects as base_defects
    def defects(source,design,report):
        result=base_defects(source,design,report)
        from engine.resize import _unintended_control_overlaps
        controls=_unintended_control_overlaps(design['elements'],source)
        result['overlap']['count']+=len(controls)
        result['overlap']['details'].extend(controls)
        src={e['id']:e for e in source['elements']}
        stretched=[e['id'] for e in design['elements'] if e['id'] in src
                   and src[e['id']].get('shape_type')=='ellipse'
                   and abs(src[e['id']]['width']-src[e['id']]['height'])<=2
                   and abs(e['width']/max(1.,e['height'])-1)>.05]
        result['stretch']['count']+=len(stretched);result['stretch']['elements'].extend(stretched)
        return result
    from scripts.layout_defects import build_ab_viewer
    if args.kind=='official':
        prior=json.loads((ROOT/'data/qwen_lora/surgical_selector_delivery/official/report.json').read_text())
        cached=prior['results']['candidate']
        records=[dict(pair_id=r['pair_id'],official_path=r['official_path'],target_canvas=r['target_canvas']) for r in cached]
    elif args.kind=='crello':
        prior=json.loads((ROOT/'data/qwen_lora/surgical_selector_delivery/report.json').read_text())
        cached=prior['results'][args.arm]
        available=[json.loads(l) for l in (ROOT/'data/qwen_lora/ab_qwen3_1.7b_vs_0.6b_review_fixes/input_12.jsonl').read_text().splitlines() if l.strip()]
        lookup={r['pair_id']:r for r in available};records=[lookup[r['pair_id']] for r in cached]
    else:
        records=[json.loads(l) for l in (ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines() if l.strip()]
        path=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/evidence/prompt_ab_same_adapter_20261025/old_prompt.jsonl'
        old={r['pair_id']:r for r in map(json.loads,path.read_text().splitlines())};get_source=read_sources();cached=[]
        for r in records:
            v=old[r['pair_id']];source=get_source(r['source_split'],r['pair_id'].split('__to__')[0])
            cached.append(dict(pair_id=r['pair_id'],source_design=source,target_canvas=r['target_canvas'],
                               qwen_model_elements=v['raw_elements'],raw_proposal_valid=True,
                               output_design={'canvas':r['target_canvas'],'elements':v['final_elements']},
                               before_defects=v['final_defects'],winner=v['winner']))
    if args.pair_list:
        selected=set(json.loads(args.pair_list.read_text()))
        records=[r for r in records if r['pair_id'] in selected];cached=[r for r in cached if r['pair_id'] in selected]
    if args.pair_contains:
        selected={r['pair_id'] for r in records if args.pair_contains in r['pair_id']}
        records=[r for r in records if r['pair_id'] in selected];cached=[r for r in cached if r['pair_id'] in selected]
    if args.offset:records=records[args.offset:];cached=cached[args.offset:]
    if args.limit:records=records[:args.limit];cached=cached[:args.limit]
    paths=['engine/source_structure.py','engine/clean_zones.py','engine/structural_reflow.py','engine/layout_contracts.py','engine/resize.py','engine/qwen_refine.py','engine/qwen_seed.py','scripts/layout_defects.py']
    manifest={'kind':args.kind,'arm':args.arm if args.kind=='crello' else 'comparison','cases':len(records),'pair_ids':[r['pair_id'] for r in records],'code_sha256':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in paths},'raw_coordinates_reused':True}
    mp=args.output/'run_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise SystemExit('Code changed: select a new output directory.')
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    adapter=Path(prior['adapter_dir']) if args.kind=='crello' and args.arm=='candidate' else ROOT/'models/qwen3-layout-lora-v2'
    model=Path(prior['model_dir']) if args.kind=='crello' and args.arm=='candidate' else ROOT/'models/qwen3-0.6b'
    rows=replay(records,cached,adapter,model,args.output/'results.jsonl')
    before={r['pair_id']:r for r in cached};totals={stage:{} for stage in ('before','after')}
    for row in rows:
        if row.get('engine_error'):continue
        old=before[row['pair_id']]
        bd=defects(row['source_design'],old['output_design'],{})
        ad=defects(row['source_design'],row['output_design'],{})
        row['before_defects']=bd;row['after_defects']=ad
        for stage,d in [('before',bd),('after',ad)]:
            for kind,v in d.items():
                target=totals[stage].setdefault(kind,dict(incidents=0,outputs=0))
                target['incidents']+=v['count'];target['outputs']+=int(v['count']>0)
    report=dict(manifest=manifest,summary=summarize_engine(rows),defects=totals,results=rows)
    report['viewer']=build_ab_viewer(rows,cached,records,args.output/'viewer',min(18,len(records)),
                      candidate_label='Current fixes · same decoded Qwen',comparison_label='Prior engine · same decoded Qwen')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary':report['summary'],'defects':totals},indent=2),flush=True)

if __name__=='__main__':main()
