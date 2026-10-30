#!/usr/bin/env python3
"""Count physical defects in literal raw proposals and selected layouts.

No missing IDs are filled and no coordinates/fonts are fitted in the raw arm.
Source tracking is derived proportionally from the predicted font as it is at
runtime. Contrast uses the independent native shared-mask diagnostic, whose
ambiguities and unmeasured rotated labels remain explicit.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def raw_design(row):
    from experiments.layout_upgrade.contracts import raw_document
    source=row['source_design'];doc=raw_document(source,row['target_canvas'],row.get('qwen_model_elements',[]))
    originals={e['id']:e for e in source['elements']}
    for e in doc['elements']:
        old=originals[e['id']]
        if e['type']=='text':e['letter_spacing']=float(old.get('letter_spacing',0))*e['font_size']/max(1e-6,old['font_size'])
    return doc


def category_counts(records, stage):
    """Cases per measured failure class, not an exhaustive aesthetic score."""
    groups=dict(stretch={'image_aspect_distortion','art_aspect_distortion'},
        misalignment={'control_label_axis','primary_axis'},
        overlap={'text_on_text','control_collision','text_hero_occlusion',
                 'text_art_occlusion','image_on_image'},
        text_fit={'glyph_overflow'})
    counts={name:sum(bool(set(r[stage+'_flags']) & flags) for r in records)
            for name,flags in groups.items()}
    # Use the exact self-check's off-canvas diagnostic, not all structural errors:
    # missing IDs are a different defect and must never be counted as bounds.
    counts['out_of_bounds']=sum(any(' is off-canvas ' in issue
        for issue in r[stage+'_details']['structural_errors']) for r in records)
    return counts


def main():
    from scripts.report_recovered_adapter import load
    from scripts.layout_defects import assess_engine_layout
    from scripts.audit_upgrade_rasters import paint_stats,against_source
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    arms={name:load(a.run/(name+'.jsonl'),True) for name in ('baseline','audited-0.6b-recovered')}
    keys=[k for k in arms['baseline'] if k in arms['audited-0.6b-recovered']]
    folder=a.run/'stage_defects';folder.mkdir(exist_ok=True);source_cache={};rows=[]
    signature=sha256((a.run/'manifest.json').read_bytes()+Path(__file__).read_bytes()+
        (ROOT/'scripts/audit_upgrade_rasters.py').read_bytes()).hexdigest()
    for key in keys:
        for arm,data in arms.items():
            row=data[key]
            if row.get('engine_error'):continue
            source=row['source_design']
            if source['canvas']==row['target_canvas']:continue
            payload=dict(signature=signature,source=source,target=row['target_canvas'],proposal=row.get('qwen_model_elements'),
                output=row['output_design'],generation=row.get('generation'),arm=arm)
            digest=sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();path=folder/(digest+'.json')
            if path.is_file():rows.append(json.loads(path.read_text()));continue
            os.environ['MAGICANVAS_DATASET_RENDERING']='0' if row.get('official_path') else '1'
            try:
                if not row.get('qwen_model_elements'):
                    result=dict(pair_id=key,arm=arm,no_raw_proposal=True)
                else:
                    raw=raw_design(row);physical=assess_engine_layout(source,raw,{})
                    source_key=sha256(json.dumps(dict(source=source,split=row['source_split']),sort_keys=True).encode()).hexdigest()
                    if source_key not in source_cache:source_cache[source_key]=paint_stats(source,source,row['source_split'])
                    contrast=against_source(paint_stats(raw,source,row['source_split']),source_cache[source_key])
                    result=dict(pair_id=key,arm=arm,raw_flags=physical['defect_flags'],
                        raw_details={k:physical[k] for k in ('missing_ids','extra_ids','duplicate_ids','image_ratio_violations','unintended_text_overlaps','structural_errors','glyph_overflow')},
                        raw_contrast=contrast,final_flags=row['defect_flags'],
                        final_details=dict(structural_errors=row.get('structural_errors',[])),
                        raw_ids_valid=row['raw_proposal_valid'],final_exact_ids=row['exact_ids'])
            except Exception as e:result=dict(pair_id=key,arm=arm,audit_error=f'{type(e).__name__}: {e}')
            if not result.get('audit_error'):path.write_text(json.dumps(result,indent=2)+'\n')
            rows.append(result);print('Stage audit',len(rows),'/',2*len(keys),arm,key,result.get('audit_error',''),flush=True)
    common_exact={key for key in keys if all(arms[arm][key].get('raw_proposal_valid') for arm in arms)}
    totals={}
    for arm in arms:
        group=[r for r in rows if r['arm']==arm];valid=[r for r in group if 'raw_flags' in r]
        totals[arm]=dict(attempted=len(group),no_proposal=sum(r.get('no_raw_proposal',False) for r in group),
            audit_errors=sum('audit_error' in r for r in group),measured=len(valid),
            raw_cases_by_type=dict(Counter(flag for r in valid for flag in set(r['raw_flags']))),
            final_cases_by_type=dict(Counter(flag for r in valid for flag in set(r['final_flags']))),
            raw_source_contrast_regression_cases=sum(bool(r['raw_contrast']['source_floor_regressions']) for r in valid),
            raw_unmeasured_rotated_labels=sum(len(r['raw_contrast']['unmeasured_rotated_or_color']) for r in valid),
            categories=dict(raw=category_counts(valid,'raw'),final=category_counts(valid,'final')))
        comparable=[r for r in valid if r['pair_id'] in common_exact]
        totals[arm]['both_raw_inventories_complete']=dict(cases=len(comparable),
            raw=category_counts(comparable,'raw'),final=category_counts(comparable,'final'))
    report=dict(signature=signature,matching_tasks=len(keys),complete=len(keys)==148 and all(
        not r.get('engine_error') for data in arms.values() for r in data.values()) and not any('audit_error' in r for r in rows),totals=totals,records=rows,
        scope='Raw proposals are literal decoded geometry, with missing IDs left missing. Fewer raw overlaps when text was omitted is not a quality gain: inspect inventory counts and the common-exact-inventory subset. Final means the engine-selected output, which can be deterministic, not necessarily a repair of the raw proposal. Categories count measured cases; misalignment covers existing label/primary-axis contracts, not every possible aesthetic axis. Bounds use the existing self-check tolerance. Contrast is diagnostic shared-mask evidence, not an individual-glyph guarantee. This report never selects or modifies a candidate.')
    (a.run/'stage_defect_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(matching_tasks=len(keys),complete=report['complete'],totals=totals),indent=2))


if __name__=='__main__':main()
