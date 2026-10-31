#!/usr/bin/env python3
"""Native raster audit and reference-only evaluation for experimental changes."""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
from statistics import mean
import argparse,json,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sign_probability(wins,losses):
    n=wins+losses
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/training_free_scale_v1');a=p.parse_args()
    from scripts.audit_upgrade_rasters import paint_stats,against_source
    from scripts.build_upgrade_pilot_review import flaws
    from scripts.layout_defects import assess_engine_layout
    from scripts.report_training_free_upgrades import gold_metrics
    cases=json.loads((a.root/'cases.json').read_text());aliases=json.loads((a.root/'test_pair_tasks.json').read_text());bykey={c['key']:i for i,c in enumerate(cases)}
    source_cache={};records=[];out=a.root/'native_audit';out.mkdir(exist_ok=True)
    for i,case in enumerate(cases):
        file=out/f'{i:03d}.json'
        if file.exists():records.append(json.loads(file.read_text()));continue
        before=json.loads((a.root/'baseline'/f'{i:03d}.json').read_text())['baseline'];arms=json.loads((a.root/'algorithms'/f'{i:03d}.json').read_text())['arms']
        options={k:v['output_design'] for k,v in arms.items() if v['changed']}
        tile=a.root/'tiled_selector'/f'{i:03d}.json'
        if tile.exists():
            t=json.loads(tile.read_text())
            if t['changed']:options['tiled-selector']=t['after']['output_design']
        structural=a.root/'structural_finalists'/f'{i:03d}.json'
        if structural.exists():
            t=json.loads(structural.read_text())
            if t['changed']:options['structural-finalists']=t['output_design']
        if not options:continue
        source=before['source_design'];key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest();split=before['source_split']
        if key not in source_cache:source_cache[key]=paint_stats(source,source,split)
        r=dict(index=i,pair_id=case['pair_id'],source=source_cache[key],before=against_source(paint_stats(before['output_design'],source,split),source_cache[key]),alternatives={})
        for arm,doc in options.items():
            r['alternatives'][arm]=dict(native=against_source(paint_stats(doc,source,split),source_cache[key]),
                geometry_flaw_instances=flaws(assess_engine_layout(source,doc,{}),source))
        file.write_text(json.dumps(r,indent=2)+'\n');records.append(r);print('native',i+1,len(cases),list(options),flush=True)
    extra={}
    for arm,folder in [('tiled-selector','tiled_selector'),('structural-finalists','structural_finalists')]:
        comparison=[];changes=[]
        for alias in aliases:
            i=bykey[alias['task_key']];case=cases[i];before=json.loads((a.root/'baseline'/f'{i:03d}.json').read_text())['baseline'];file=a.root/folder/f'{i:03d}.json'
            if not file.exists():continue
            t=json.loads(file.read_text());doc=t['after']['output_design'] if folder=='tiled_selector' else t['output_design']
            comparison.append(dict(pair_id=alias['pair_id'],family_id=alias['record']['family_id'],before=gold_metrics(before['source_design'],before['output_design'],alias['record']),after=gold_metrics(before['source_design'],doc,alias['record'])))
        for f in (a.root/folder).glob('[0-9]*.json'):
            t=json.loads(f.read_text())
            if t['changed']:changes.append(dict(index=int(f.stem),pair_id=t['pair_id'],winner=t['winner'],new_flags=t.get('new_flags',[])))
        extra[arm]=dict(pairs=len(comparison),families=len({r['family_id'] for r in comparison}),changed_tasks=len(changes),changes=changes,
            before_geometry_mae=mean(r['before']['geometry_mae'] for r in comparison),after_geometry_mae=mean(r['after']['geometry_mae'] for r in comparison),
            geometry_improved=sum(r['after']['geometry_mae']<r['before']['geometry_mae']-1e-5 for r in comparison),geometry_worsened=sum(r['after']['geometry_mae']>r['before']['geometry_mae']+1e-5 for r in comparison),reference_pairs=comparison)
    critics={p.stem:json.loads(p.read_text()) for p in (ROOT/'data/upgrade_exploration/critic_retrospective_all').glob('*.json')}
    pairs={}
    for arm in ('tiled','naflex'):
        baseline={r['pair_id']:r for r in critics['global']['records']};wins=losses=0
        for r in critics[arm]['records']:
            old=baseline[r['pair_id']]
            wins+=r['correct'] and not old['correct'];losses+=old['correct'] and not r['correct']
        pairs[arm]=dict(wins=wins,losses=losses,paired_exact_sign_probability=sign_probability(wins,losses),scope='Descriptive pair-level test; family dependence and exploratory multiple comparisons remain.')
    report=dict(changed_task_native_audits=len(records),native_records=records,extra_selector_arms=extra,
        critics={k:{n:v for n,v in r.items() if n!='records'} for k,r in critics.items()},critic_paired_comparisons=pairs,
        scope='Native audits cover every changed experimental output. Unchanged outputs share their exact baseline pixels. Reference coordinates used only after selection. No new human labels or fitting.')
    (a.root/'decision_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='native_records'},indent=2))


if __name__=='__main__':main()
