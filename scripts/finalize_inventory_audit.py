#!/usr/bin/env python3
"""Merge controlled inventory replays and refresh the compact review evidence."""
from pathlib import Path
import hashlib,json,shutil,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from scripts.evaluate_qwen_adapter import summarize_engine
    from scripts.layout_defects import build_ab_viewer
    initial=ROOT/'data/qwen_lora/inventory_fixes_final'
    final=ROOT/'data/qwen_lora/inventory_fixes_verified'
    prior=json.loads((initial/'heldout_before_final_gate.json').read_text())
    original={r['pair_id']:r for r in prior['results']}
    expected=[]
    for r in original.values():
        sc=r.get('source_contrast',{})
        if any(any(not sc.get(eid,{}).get('low',True) for eid in c.get('low_contrast',[]))
               and not c.get('invisible_text') and 'contrast-repaired' not in c['name']
               for c in r.get('rendered_candidates',[])):
            expected.append(r['pair_id'])
    changed={}
    for p in sorted(final.glob('heldout_changed_*/report.json')):
        d=json.loads(p.read_text())
        for r in d['results']:
            assert r['pair_id'] not in changed
            changed[r['pair_id']]=r
    assert set(changed)==set(expected),(len(changed),len(expected))
    # Certify conditional reuse: no other inference code may have changed.
    current=(ROOT/'engine/qwen_refine.py').read_text();at=current.index('def whole_lockup_contrast_variants(')
    old=current[:at]+current[at:].replace('from engine.resize import _physical_layout_safe,_unintended_text_overlaps,_rgba,_luminance,_text_visible_bounds','from engine.resize import _feasible,_unintended_text_overlaps,_rgba,_luminance,_text_visible_bounds',1).replace('if not _physical_layout_safe(candidate,units,source,width,height):continue',"if (not _feasible(candidate,units,width,height,strict_margins=False,\n                                  gap_factor=.004,allow_text_visual_overlap=True)\n                        or _unintended_text_overlaps(candidate,units)):continue",1)
    assert hashlib.sha256(old.encode()).hexdigest()==prior['reconstructed_prior_qwen_refine_sha256']
    resize_code=(ROOT/'engine/resize.py').read_text()
    start=resize_code.index('def _simple_centered_stack_source(')
    end=resize_code.index('def _source_geometry_safety(',start)
    previous_resize=(resize_code[:start]+resize_code[end:]).replace(
        '    _simple_stack_axis_error(out,units,source,width,apply=True)\n','',1).replace(
        "objective_cost=lambda e:(e['layout_defect_cost']+e['cost']+25.*e.get('source_relationship_error',0.)/max(1.,ratio_shift)\n                             +60.*_simple_stack_axis_error(e['candidate']['elements'],units,source,width))",
        "objective_cost=lambda e:e['layout_defect_cost']+e['cost']+25.*e.get('source_relationship_error',0.)/max(1.,ratio_shift)",1)
    resize_before_stack=hashlib.sha256(previous_resize.encode()).hexdigest()
    assert resize_before_stack==prior['prior_manifest']['code_sha256']['engine/resize.py']
    from engine.resize import _analyze,_simple_centered_stack_source
    assert not any(_simple_centered_stack_source(_analyze(r['source_design'])[1],r['source_design']) for r in original.values())
    paths=['engine/resize.py','engine/qwen_refine.py','engine/qwen_seed.py','scripts/layout_defects.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    for p in paths:
        if p not in ('engine/qwen_refine.py','engine/resize.py'):assert hashes[p]==prior['prior_manifest']['code_sha256'][p]
    for p in final.glob('*/report.json'):
        checkpoint=json.loads(p.read_text());old_hashes=checkpoint['manifest']['code_sha256']
        if old_hashes!=hashes:
            assert all(old_hashes[k]==hashes[k] for k in hashes if k!='engine/resize.py'),p
            assert old_hashes['engine/resize.py']==resize_before_stack,p
            assert not any(_simple_centered_stack_source(_analyze(r['source_design'])[1],r['source_design']) for r in checkpoint['results']),p
    rows=[]
    records=[json.loads(l) for l in (ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines() if l.strip()]
    for record in records:
        pid=record['pair_id'];r=changed.get(pid,original[pid])
        r['contrast_dispatch_reexecuted']=pid in changed
        rows.append(r)
    assert len(rows)==89 and not any(r.get('engine_error') for r in rows)
    totals={stage:{} for stage in ('before','after')}
    for r in rows:
        for stage,key in [('before','before_defects'),('after','after_defects')]:
            for kind,value in r[key].items():
                t=totals[stage].setdefault(kind,dict(incidents=0,outputs=0))
                t['incidents']+=value['count'];t['outputs']+=int(value['count']>0)
    baseline_path=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/evidence/prompt_ab_same_adapter_20261025/old_prompt.jsonl'
    baseline_rows={r['pair_id']:r for r in map(json.loads,baseline_path.read_text().splitlines())}
    baselines=[dict(pair_id=r['pair_id'],source_design=r['source_design'],target_canvas=r['target_canvas'],
               output_design=dict(canvas=r['target_canvas'],elements=baseline_rows[r['pair_id']]['final_elements']),
               winner=baseline_rows[r['pair_id']]['winner']) for r in rows]
    out=final/'heldout';out.mkdir(exist_ok=True)
    report=dict(manifest={'code_sha256':hashes,'cases':89,'raw_coordinates_reused':True,
        'conditional_replay':{'full_initial_replays':89,'contrast_dispatch_reexecuted':len(changed),
            'unchanged_dispatch_results_reused':89-len(changed),
            'only_changed_function':'whole_lockup_contrast_variants',
            'prior_qwen_refine_sha256':prior['reconstructed_prior_qwen_refine_sha256']},
        'source_stack_dispatch_certificate':{'eligible_sources':0,'verified_sources':89,'reconstructed_prior_resize_sha256':resize_before_stack}},
        summary=summarize_engine(rows),defects=totals,results=rows)
    report['viewer']=build_ab_viewer(rows,baselines,records,out/'viewer',10,
        candidate_label='Current engine · same raw Qwen',comparison_label='Prior engine · same raw Qwen')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    groups={'heldout':report}
    for label in ('official','crello','crello_1p7b'):
        groups[label]=json.loads((final/label/'report.json').read_text())
        if label!='official':
            assert not any(_simple_centered_stack_source(_analyze(r['source_design'])[1],r['source_design']) for r in groups[label]['results'])
            groups[label]['manifest']['source_stack_dispatch_certificate']={'eligible_sources':0,'verified_sources':len(groups[label]['results']),'reconstructed_prior_resize_sha256':resize_before_stack}
            (final/label/'report.json').write_text(json.dumps(groups[label],indent=2)+'\n')
    summary={k:{'summary':d['summary'],'defects':d['defects']} for k,d in groups.items()}
    (final/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    links=[('89 held-out cases · worst 10','heldout/viewer/review.html'),('18 official formats','official/viewer/review.html'),
           ('12 Crello cases · 0.6B','crello/viewer/review.html'),('12 Crello cases · 1.7B','crello_1p7b/viewer/review.html')]
    index='<!doctype html><meta charset="utf-8"><title>Verified engine repairs</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;background:#101820;color:#eee}a{color:#9ed4ff}li{margin:18px}pre{white-space:pre-wrap;font-size:13px}</style><h1>Engine repair comparisons</h1><p>Identical saved model coordinates. Current repairs and selector; sources and prior outputs shown beside each result.</p><ul>'
    index+=''.join(f'<li><a href="{url}">{label}</a></li>' for label,url in links)+'</ul><pre>'+json.dumps(summary,indent=2)+'</pre>'
    (final/'index.html').write_text(index)
    append='\n## Verified replay results\n\n'
    append+='The full initial engine replay covered all 89 held-out cases. A contrast-only change replaced the unit-gap gate inside `whole_lockup_contrast_variants`; reconstruction verifies the previous source hash exactly. All 31 cases that could dispatch that function were re-executed, and the other 58 reuse their unchanged dispatch results. Both reviewed adapter arms were replayed after that contrast fix. A subsequent brand-stack axis correction has a source-only dispatch that is disabled for every held-out and reviewed Crello source; the recorded predicate checks and reconstructed prior resize hash certify unchanged results. All 18 official formats were re-executed with that final correction.\n\n'
    append+='| Cohort | Cases | Flagged final outputs | Before → after alignment incidents | Before → after glyph-fit incidents |\n| --- | ---: | ---: | ---: | ---: |\n'
    for k,d in groups.items():
        rows_=d['results'];df=d['defects'];append+=f"| {k} | {len(rows_)} | {sum(bool(r.get('flagged')) for r in rows_)} | {df['before']['misalignment']['incidents']} → {df['after']['misalignment']['incidents']} | {df['before']['text_fit']['incidents']} → {df['after']['text_fit']['incidents']} |\n"
    append+='\nThe frame-axis counter is intentionally not a clean-layout gate: it also counts intentional reflows and frame shifts used to center painted labels. Its remaining flags are exposed for review, not removed to manufacture a zero. See `data/qwen_lora/inventory_fixes_verified/index.html` and `summary.json`.\n'
    doc=ROOT/'docs/archive/INVENTORY_REPAIR_AUDIT.md';s=doc.read_text().split('\n## Verified replay results')[0];doc.write_text(s+append)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
