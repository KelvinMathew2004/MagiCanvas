#!/usr/bin/env python3
"""Paired, family-aware reporting for the completed recovered-adapter A/B.

No target coordinates enter generation or selector decisions. An incomplete
run can be inspected with --progress but cannot publish a final decision.
"""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean,median
from hashlib import sha256
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def load(path,allow_incomplete_tail=False):
    rows={}
    if path.exists():
        data=path.read_text();lines=data.splitlines()
        if allow_incomplete_tail and data and not data.endswith('\n'):
            lines=lines[:-1]  # A writer may be midway through its final append.
        for line in lines:
            if line.strip():
                r=json.loads(line);rows[r['pair_id']]=r
    return rows


def pair_stats(before,after,ids):
    result=dict(tasks=len(ids),before_flagged=sum(bool(before[k].get('flagged')) for k in ids),
        after_flagged=sum(bool(after[k].get('flagged')) for k in ids),
        before_qwen_wins=sum(bool(before[k].get('qwen_family_win')) for k in ids),
        after_qwen_wins=sum(bool(after[k].get('qwen_family_win')) for k in ids),
        before_flag_types=dict(Counter(f for k in ids for f in before[k].get('defect_flags',[]))),
        after_flag_types=dict(Counter(f for k in ids for f in after[k].get('defect_flags',[]))))
    attempted=[k for k in ids if before[k]['source_design']['canvas']!=before[k]['target_canvas']]
    result.update(generation_tasks=len(attempted),before_raw_ids_valid=sum(bool(before[k].get('raw_proposal_valid')) for k in attempted),
        after_raw_ids_valid=sum(bool(after[k].get('raw_proposal_valid')) for k in attempted),
        before_json_complete=sum(bool(before[k].get('raw_json_complete')) for k in attempted),
        after_json_complete=sum(bool(after[k].get('raw_json_complete')) for k in attempted))
    for key in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):
        eligible=[k for k in ids if before[k].get(key) is not None and after[k].get(key) is not None
                  and (not key.startswith('raw_') or before[k].get('raw_proposal_valid') and after[k].get('raw_proposal_valid')
                       and before[k].get('geometry_mae') is not None and after[k].get('geometry_mae') is not None)]
        result[key]=dict(common_pairs=len(eligible),
            before=mean(before[k][key] for k in eligible) if eligible else None,
            after=mean(after[k][key] for k in eligible) if eligible else None,
            before_median=median(before[k][key] for k in eligible) if eligible else None,
            after_median=median(after[k][key] for k in eligible) if eligible else None,
            improved=sum(after[k][key]<before[k][key]-1e-5 for k in eligible),
            worsened=sum(after[k][key]>before[k][key]+1e-5 for k in eligible))
    times=[]
    for key in attempted:
        a,b=before[key].get('generation',{}),after[key].get('generation',{})
        if a.get('generation_seconds') is not None and b.get('generation_seconds') is not None:
            times.append((a['generation_seconds'],b['generation_seconds']))
    result['generation_timing']=dict(common_pairs=len(times),before_median=median(a for a,b in times) if times else None,
        after_median=median(b for a,b in times) if times else None,
        scope='Observed local wall time; independent MPS arms overlap and baseline handoff suspension can be included. Not an isolated generation-speed benchmark, speedup claim or Docker latency claim')
    def inventory(rows):
        failures=Counter();types=Counter();missing=0
        for k in attempted:
            r=rows[k];g=r.get('generation',{});source={e.get('id'):e for e in r['source_design'].get('elements',[])}
            for reason,key in [('missing_ids','missing_ids'),('duplicate_ids','duplicate_ids'),('unknown_ids','unknown_ids')]:
                if g.get(key):failures[reason]+=1
            if g.get('hit_token_limit'):failures['output_token_limit']+=1
            if not r.get('raw_json_complete'):failures['incomplete_json_or_no_proposal']+=1
            for eid in g.get('missing_ids',[]):
                missing+=1;types[source.get(eid,{}).get('type','unknown')]+=1
        return dict(tasks_by_reason=dict(failures),missing_element_instances=missing,missing_by_source_type=dict(types))
    result['raw_inventory_diagnostics']=dict(before=inventory(before),after=inventory(after))
    result['prediction_collection']=dict(
        before_saved_fresh_predictions_replayed=sum(bool(before[k].get('raw_prediction_reused')) for k in ids),
        after_saved_fresh_predictions_replayed=sum(bool(after[k].get('raw_prediction_reused')) for k in ids),
        note='Saved decoded values retain original model completion diagnostics; final outputs are always computed under the reported engine snapshot.')
    result['candidate_realization_failures']=dict(
        before=sum(str(before[k].get('qwen_status','')).startswith('Qwen decoded; realization failed:') for k in ids),
        after=sum(str(after[k].get('qwen_status','')).startswith('Qwen decoded; realization failed:') for k in ids))
    for name,key in [('glyph_overflow_instances','glyph_overflow'),('source_contrast_regressions','source_contrast_regressions'),
                     ('text_overlap_instances','unintended_text_overlaps'),('control_collision_instances','control_collisions')]:
        result[name]=dict(before=sum(len(before[k].get(key,[])) for k in ids),after=sum(len(after[k].get(key,[])) for k in ids))
    return result


def family_interval(before,after,ids,key,seed=3407,repeats=2000):
    import numpy as np
    families=defaultdict(list)
    for k in ids:
        if before[k].get(key) is not None and after[k].get(key) is not None:
            families[before[k].get('family_id') or k].append(after[k][key]-before[k][key])
    if not families:return None
    values=[np.asarray(v) for v in families.values()];rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        chosen=rng.integers(0,len(values),len(values));sample=np.concatenate([values[i] for i in chosen]);samples.append(float(sample.mean()))
    return dict(families=len(values),pairs=sum(map(len,values)),mean_delta=float(np.concatenate(values).mean()),
                interval95=[float(v) for v in np.quantile(samples,[.025,.975])],resamples=repeats,
                scope='Descriptive family-cluster bootstrap on consumed development references; not unseen-test significance')


def main():
    from scripts.evaluate_trained_layouts import promotion
    from scripts.refresh_regression_viewer import documents_differ
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--progress',action='store_true')
    a=p.parse_args();before=load(a.run/'baseline.jsonl',a.progress);after=load(a.run/'audited-0.6b-recovered.jsonl',a.progress)
    if a.progress:
        print(json.dumps(dict(baseline=len(before),candidate=len(after),expected_per_arm=148,
             baseline_errors=[k for k,r in before.items() if r.get('engine_error')],candidate_errors=[k for k,r in after.items() if r.get('engine_error')]),indent=2));return
    if len(before)!=148 or before.keys()!=after.keys():raise SystemExit('Need complete matching 148-task arms before final reporting')
    if any(r.get('engine_error') for r in list(before.values())+list(after.values())):raise SystemExit('Resolve engine errors before final reporting')
    refs=[k for k in before if k.startswith('reference:')];review=[k for k in before if k.startswith('review:')]
    assert len(refs)==89 and len(review)==59
    report=dict(qwen_enabled=True,fresh_inference=True,production_changed=False,reference=pair_stats(before,after,refs),
                review=pair_stats(before,after,review),all=pair_stats(before,after,list(before)),
                promotion=promotion(list(before.values()),list(after.values())),strata={},changes=[],
                family_bootstrap={key:family_interval(before,after,refs,key) for key in ('geometry_mae','font_mae')},
                evaluator_manifest_sha256=sha256((a.run/'manifest.json').read_bytes()).hexdigest(),
                arms_sha256={name:sha256((a.run/(name+'.jsonl')).read_bytes()).hexdigest()
                    for name in ('baseline','audited-0.6b-recovered')},
                report_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    parallel=a.run/'parallel_arm_provenance.json'
    if parallel.is_file():report['throughput_orchestration']=json.loads(parallel.read_text())
    timing=a.run/'timing_limitations.json'
    if timing.is_file():report['timing_limitations']=json.loads(timing.read_text())
    groups=defaultdict(list)
    for k,r in before.items():
        w,h=r['target_canvas']['width'],r['target_canvas']['height'];ar=w/h
        bucket='skyscraper' if ar<=.45 else 'portrait' if ar<.9 else 'square' if ar<=1.35 else 'wide' if ar<=4 else 'leaderboard'
        groups['aspect:'+bucket].append(k)
        groups['density:'+('18-plus' if len(r['source_design']['elements'])>=18 else 'under-18')].append(k)
        sw,sh=r['source_design']['canvas']['width'],r['source_design']['canvas']['height']
        shift=max(ar/(sw/sh),(sw/sh)/ar)
        groups['aspect-change:'+('under-1.35' if shift<1.35 else '1.35-to-3' if shift<3 else '3-plus')].append(k)
        if not k.startswith('reference:'):
            from scripts.enhance_adapter_viewer import cohort
            groups['review:'+cohort(k,r)].append(k)
        changed=documents_differ(r['output_design'],after[k]['output_design'])
        if changed or r['winner']!=after[k]['winner'] or r['defect_flags']!=after[k]['defect_flags']:
            report['changes'].append(dict(pair_id=k,geometry_changed=changed,before_winner=r['winner'],after_winner=after[k]['winner'],
                new_flags=sorted(set(after[k]['defect_flags'])-set(r['defect_flags'])),removed_flags=sorted(set(r['defect_flags'])-set(after[k]['defect_flags'])),
                before_geometry=r.get('geometry_mae'),after_geometry=after[k].get('geometry_mae'),
                before_font=r.get('font_mae'),after_font=after[k].get('font_mae')))
    report['strata']={key:pair_stats(before,after,ids) for key,ids in groups.items()}
    (a.run/'paired_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    pointer=ROOT/'data/training_next/latest_completed_evaluation.json'
    pointer.write_text(json.dumps(dict(run=str(a.run.resolve()),
        manifest_sha256=report['evaluator_manifest_sha256'],
        analysis_sha256=sha256((a.run/'paired_analysis.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('changes','strata')},indent=2))


if __name__=='__main__':main()
