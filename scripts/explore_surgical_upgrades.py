#!/usr/bin/env python3
"""Measure six surgical follow-ups without changing production or datasets."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from collections import Counter
from statistics import mean
import argparse,json,os,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'data/upgrade_exploration/training_free_scale_v1'
OUT=ROOT/'data/upgrade_exploration/surgical_followup_v1'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def diagnostic(index):
    from engine.resize import _text_info,_render_text_ink,_set_font_size
    from engine.nested_structure import unpainted_ids
    from experiments.layout_upgrade.surgical import font_slack,dual_envelope,envelope_violation,topology,hierarchy_band
    from scripts.layout_defects import assess_engine_layout
    case=json.loads((BASE/'cases.json').read_text())[index]
    before=json.loads((BASE/'baseline'/f'{index:03d}.json').read_text())['baseline']
    source,doc=before['source_design'],before['output_design'];w,h=(doc['canvas'][k] for k in ('width','height'))
    os.environ['MAGICANVAS_DATASET_RENDERING']='0' if before.get('official_path') else '1'
    by={e['id']:e for e in source['elements']};inactive=unpainted_ids(source);raw={e['id']:e for e in before.get('qwen_model_elements',[])};texts=[]
    for e in doc['elements']:
        if e['type']!='text' or e['id'] in inactive:continue
        s=by[e['id']];r=raw.get(e['id'],{});sf=float(s.get('font_size',1));current=float(e['font_size'])
        record=dict(id=e['id'],source_lines=_text_info(s)[2],selected_lines=_text_info(e)[2],
            source_font=sf,selected_font=current,raw_font=r.get('font_size'),box_width_scale=e['width']/max(1.,s['width']),
            local_font_scale=current/max(1.,sf),explicit_breaks='\n' in s.get('content',''))
        if r.get('font_size'):
            p=deepcopy(e);_set_font_size(p,r['font_size']);record['lines_at_raw_font']=_text_info(p)[2]
            record['selected_vs_raw_font_ratio']=current/r['font_size']
        texts.append(record)
    candidate,notes=font_slack(doc['elements'],source,w,h)
    repaired=deepcopy(doc);repaired['elements']=candidate
    defects=assess_engine_layout(source,repaired,{}) if notes else None
    raw_es,raw_notes=font_slack(doc['elements'],source,w,h,{eid:e['font_size'] for eid,e in raw.items() if e.get('font_size')})
    raw_doc=deepcopy(doc);raw_doc['elements']=raw_es
    raw_flags=assess_engine_layout(source,raw_doc,{})['defect_flags'] if raw_notes else []
    hierarchy,hn=hierarchy_band(doc['elements'],source);hd=deepcopy(doc);hd['elements']=hierarchy
    hf=assess_engine_layout(source,hd,{})['defect_flags'] if hn else []
    bounds=[]
    for e in doc['elements']:
        v=envelope_violation(e,w,h)
        if v>0:bounds.append(dict(id=e['id'],pixels=v,rotation=e.get('rotation',0)))
    clamped=dual_envelope(doc['elements'],w,h)
    result=dict(index=index,pair_id=case['pair_id'],texts=texts,boundary_excess=bounds,
        dual_clamp_changed=sum(a!=b for a,b in zip(doc['elements'],clamped)),
        after_dual_max_excess=max((envelope_violation(e,w,h) for e in clamped),default=0),
        literal_topology=topology(source,doc),owned_topology=topology(source,doc,True),
        hierarchy_band=dict(notes=hn,changed=bool(hn),output_design=hd if hn else None,flags=hf),
        raw_font_slack=dict(notes=raw_notes,changed=bool(raw_notes),output_design=raw_doc if raw_notes else None,flags=raw_flags),
        font_slack=dict(notes=notes,changed=bool(notes),output_design=repaired if notes else None,
                        flags=defects['defect_flags'] if defects else []))
    save(OUT/'diagnostics'/f'{index:03d}.json',result)
    return dict(index=index,pair_id=case['pair_id'],font_changes=len(notes),hierarchy_changes=len(hn),bounds=len(bounds))


def topology_replay():
    from experiments.layout_upgrade.surgical import topology
    cases=json.loads((BASE/'cases.json').read_text());records=[]
    for i,case in enumerate(cases):
        before=json.loads((BASE/'baseline'/f'{i:03d}.json').read_text())['baseline']
        old=json.loads((BASE/'algorithms'/f'{i:03d}.json').read_text())['arms']
        for arm in ('retrieval','dual-granularity'):
            r=old[arm]
            if not r['changed']:continue
            literal=topology(before['source_design'],r['output_design']);owned=topology(before['source_design'],r['output_design'],True)
            q0=json.loads((BASE/'algorithms'/f'{i:03d}.json').read_text())['baseline_quality'];q1=r['quality']
            occupancy=q1['content_union_share']/max(1e-6,q0['content_union_share'])
            native=BASE/'native_audit'/f'{i:03d}.json';floor=False
            if native.exists():floor=bool(json.loads(native.read_text())['before']['source_floor_regressions'])
            records.append(dict(index=i,pair_id=case['pair_id'],arm=arm,literal=literal,owned=owned,
                occupancy_ratio=occupancy,baseline_native_floor_regression=floor,
                # A contrast trigger never bypasses the topology gate.
                literal_trigger_pass=literal['passes'] and (floor or occupancy>1.25),
                owned_trigger_pass=owned['passes'] and (floor or occupancy>1.25)))
    save(OUT/'topology.json',dict(scope='Retrospective filter of 14 previously selected experimental changes, not a new full candidate search; no gold used for eligibility.',records=records,
        literal_survivors=sum(r['literal_trigger_pass'] for r in records),owned_survivors=sum(r['owned_trigger_pass'] for r in records)))


def critic_pairs():
    from PIL import Image
    from scripts.train_siglip_ranker import reviewed_pairs
    from experiments.layout_upgrade.surgical import gated_critic
    folder=ROOT/'data/siglip_reviewed_corpus';items=reviewed_pairs(list(map(json.loads,(folder/'manifest.jsonl').read_text().splitlines())),folder/'approvals.jsonl')
    globals={r['pair_id']:r for r in json.loads((ROOT/'data/upgrade_exploration/critic_retrospective_all/global.json').read_text())['records']}
    tiles={r['pair_id']:r for r in json.loads((ROOT/'data/upgrade_exploration/critic_retrospective_all/tiled.json').read_text())['records']}
    records=[]
    for row,preferred,other in items:
        with Image.open(preferred) as im:size=im.size
        with Image.open(other) as im:other_size=im.size
        a,b=globals[row['pair_id']],tiles[row['pair_id']]
        gs=[a['preferred_score'],a['other_score']];ts=[b['preferred_score'],b['other_score']]
        scores,ids=gated_critic(gs,ts,size)
        if other_size!=size:ids=[];scores=gs
        records.append(dict(pair_id=row['pair_id'],size=size,cohort=a['cohort'],activated=bool(ids),
            global_correct=a['correct'],full_tiled_correct=b['correct'],gated_correct=scores[0]>scores[1],global_margin=abs(gs[0]-gs[1])))
    save(OUT/'gated_critic.json',dict(pairs=len(records),global_correct=sum(r['global_correct'] for r in records),full_tiled_correct=sum(r['full_tiled_correct'] for r in records),
        gated_correct=sum(r['gated_correct'] for r in records),activated=sum(r['activated'] for r in records),
        scope='Fixed 2.2 aspect / 0.15 margin; cached critic scores, all labels consumed. Pair study, not full engine/latency proof.',records=records))


def target_audit(native_count):
    from scripts.evaluate_qwen_adapter import read_sources
    from engine.resize import _text_info
    from engine.raster_metrics import glyph_overflow
    from engine.nested_structure import unpainted_ids
    from engine.main import self_check
    from scripts.audit_upgrade_rasters import paint_stats
    get=read_sources();rows={};pair_counts={};sourcefiles={}
    for split in ('train','validation'):
        file=ROOT/f'data/qwen_lora_expanded_v2/{split}.jsonl';pairs=list(map(json.loads,file.read_text().splitlines()));pair_counts[split]=len(pairs);sourcefiles[str(file.relative_to(ROOT))]=sha256(file.read_bytes()).hexdigest()
        for pair in pairs:
            tid=pair['pair_id'].split('__to__')[1];key=(pair['target_split'],tid)
            entry=rows.setdefault(key,dict(target_id=tid,split=pair['target_split'],model_splits=[],pair_ids=[]))
            if split not in entry['model_splits']:entry['model_splits'].append(split)
            entry['pair_ids'].append(pair['pair_id'])
    records=[]
    for n,(key,meta) in enumerate(sorted(rows.items())):
        path=OUT/'target_geometry'/f'{meta["split"]}_{meta["target_id"]}.json'
        if path.exists():r=json.loads(path.read_text());records.append(r);continue
        doc=get(*key);os.environ['MAGICANVAS_DATASET_RENDERING']='1';inactive=unpainted_ids(doc)
        r=dict(**meta,element_count=len(doc['elements']),unpainted_ids=sorted(inactive),
            lines={e['id']:_text_info(e)[2] for e in doc['elements'] if e['type']=='text' and e['id'] not in inactive},
            overflow=glyph_overflow(doc,doc['elements'],doc['canvas']['width'],doc['canvas']['height']),
            structural_errors=self_check(doc,doc),content_sha256=sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest())
        save(path,r);records.append(r)
        if n%50==0:print('target geometry',n+1,len(rows),flush=True)
    # Predeclared mixed strata: half deterministic hash sample, half flagged
    # geometry. No filtering or training-file rewrite is done by this audit.
    ordered=sorted(records,key=lambda r:sha256((r['split']+r['target_id']).encode()).hexdigest())
    selected=ordered[:native_count//2];selected+= [r for r in ordered if r not in selected and (r['overflow'] or r['structural_errors'])][:native_count-len(selected)]
    selected+=[r for r in ordered if r not in selected][:native_count-len(selected)]
    native=[]
    for meta in selected:
        path=OUT/'target_native'/f'{meta["split"]}_{meta["target_id"]}.json'
        if path.exists():native.append(json.loads(path.read_text()));continue
        doc=get(meta['split'],meta['target_id']);r=dict(target_id=meta['target_id'],split=meta['split'],stats=paint_stats(doc,doc,meta['split']))
        r['visibility_suspects']=[eid for eid,v in r['stats']['text'].items() if v['ink_pixels']==0 or v['contrast'] is None or v['contrast']<1.15]
        r['contrast_suspects']=[eid for eid,v in r['stats']['text'].items() if v['contrast'] is not None and v['contrast']<v['threshold']*.97]
        save(path,r);native.append(r);print('target native',len(native),len(selected),flush=True)
    save(OUT/'target_audit.json',dict(pair_counts=pair_counts,unique_targets=len(records),geometry_audited=len(records),
        overflow_targets=sum(bool(r['overflow']) for r in records),structural_targets=sum(bool(r['structural_errors']) for r in records),
        native_audited=len(native),visibility_suspect_targets=sum(bool(r['visibility_suspects']) for r in native),
        contrast_suspect_targets=sum(bool(r['contrast_suspects']) for r in native),source_files=sourcefiles,
        scope='Full train/validation unique-target geometry and line-count sidecars; stratified native subset only. Global difference masks are screening signals, not per-element proof. No automatic deletion or prompt-schema change.',
        native_sample=[dict(split=r['split'],target_id=r['target_id']) for r in selected]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('diagnostics','targets','policies'),required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--native-targets',type=int,default=32);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    provenance=dict(baseline_manifest_sha256=sha256((BASE/'manifest.json').read_bytes()).hexdigest(),
        source_attachment_sha256=sha256(Path('/Users/kelvinmathew/.codex/attachments/ce7a30ed-8b51-4f38-9b4d-6c14d97a8921/Pasted text.txt').read_bytes()).hexdigest(),
        experiment_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'experiments/layout_upgrade/surgical.py',ROOT/'scripts/audit_target_glyph_ownership.py',ROOT/'scripts/report_surgical_upgrades.py']},
        executed_engine_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'engine').glob('*.py'))},
        production_changed=False,scope='Frozen outputs from the recorded scale baseline, analyzed with current default-equivalent measurement code. No fresh generation.')
    save(OUT/'manifest.json',provenance)
    if a.phase=='targets':target_audit(a.native_targets)
    elif a.phase=='policies':topology_replay();critic_pairs()
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        cases=json.loads((BASE/'cases.json').read_text())
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(diagnostic,i) for i in range(len(cases))]):print(json.dumps(f.result()),flush=True)
        records=[json.loads(f.read_text()) for f in sorted((OUT/'diagnostics').glob('*.json'))]
        texts=[t for r in records for t in r['texts']]
        save(OUT/'diagnostic_summary.json',dict(tasks=len(records),text_elements=len(texts),
            extra_source_line_count=sum(t['selected_lines']>t['source_lines'] for t in texts),
            raw_font_available=sum(t.get('raw_font') is not None for t in texts),
            shrunk_more_than_25pct_vs_raw=sum(t.get('selected_vs_raw_font_ratio',1)<.75 for t in texts),
            slack_changed_tasks=sum(r['font_slack']['changed'] for r in records),slack_changed_elements=sum(len(r['font_slack']['notes']) for r in records),
            slack_flagged_tasks=sum(bool(r['font_slack']['flags']) for r in records if r['font_slack']['changed']),
            raw_slack_changed_tasks=sum(r['raw_font_slack']['changed'] for r in records),
            raw_slack_changed_elements=sum(len(r['raw_font_slack']['notes']) for r in records),
            raw_slack_flagged_tasks=sum(bool(r['raw_font_slack']['flags']) for r in records if r['raw_font_slack']['changed']),
            hierarchy_changed_tasks=sum(r['hierarchy_band']['changed'] for r in records),
            hierarchy_changed_elements=sum(len(r['hierarchy_band']['notes']) for r in records),
            hierarchy_flagged_tasks=sum(bool(r['hierarchy_band']['flags']) for r in records if r['hierarchy_band']['changed']),
            raw_strict_boundary_instances=sum(len(r['boundary_excess']) for r in records),
            boundary_instances_over_01px=sum(x['pixels']>.01 for r in records for x in r['boundary_excess']),
            baseline_literal_topology_pass=sum(r['literal_topology']['passes'] for r in records),
            baseline_owned_topology_pass=sum(r['owned_topology']['passes'] for r in records)))


if __name__=='__main__':
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[var]='1'
    import torch;torch.set_num_threads(1)
    main()
