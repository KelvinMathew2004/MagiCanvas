#!/usr/bin/env python3
"""Replay the fixed Qwen-enabled regression cohort and publish one paired viewer.

Saved raw predictions isolate engine changes. This is not fresh model inference
or a generation-speed benchmark. Inputs, code and adapter are fingerprinted;
partial or mismatched runs never replace the public viewer.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PACK=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25'


def load_cohort(config):
    spec=json.loads(Path(config).read_text());spec['fixed_cases']=len(spec['cases'])
    supplement=spec.get('supplemental_config')
    if supplement:spec['cases']+=json.loads((ROOT/supplement).read_text())['cases']
    return spec


def fingerprint(config):
    files=sorted((ROOT/'engine').glob('*.py'))+[ROOT/'scripts/publish_capacity_comparison.py',ROOT/'scripts/build_focus_reflow_viewer.py',ROOT/'scripts/layout_defects.py',ROOT/'scripts/evaluate_qwen_adapter.py',ROOT/'scripts/replay_qwen_engine.py',Path(__file__).resolve(),config]
    spec=load_cohort(config)
    comparison=ROOT/'data/prompt_capacity_ab/selector/1.7b_report.json'
    if comparison.is_file():files.append(comparison)
    if spec.get('supplemental_config'):files.append(ROOT/spec['supplemental_config'])
    files += [ROOT/spec['adapter']/name for name in ('adapter_config.json','adapter_model.safetensors','magicanvas_prompt_version.txt')]
    files += sorted({ROOT/c['prediction_report'] for c in spec['cases']})
    files += sorted({ROOT/c['records_file'] for c in spec['cases'] if c.get('records_file')})
    files += sorted((ROOT/'renderer/converter').glob('*.py'))
    files += sorted((ROOT/'engine/data').glob('*.json'))
    files += [ROOT/'data/crello_fonts.pickle',ROOT/spec['model']/'config.json']
    files += sorted((ROOT/'tests/fixtures/novel_level3_sources').glob('n*.json'))
    files += sorted((ROOT/'tests/fixtures/fresh_level3_sources').glob('f*.json'))
    files += sorted((ROOT/'tests/fixtures/user_level3_sources').glob('u*.json'))
    files += sorted((ROOT/'input').glob('*level*.json'))
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    # Model weights stay local; record size/mtime plus configuration, without
    # hashing/loading a multi-GB base on each refresh.
    weights={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted((ROOT/spec['model']).glob('*.safetensors'))}
    return dict(files_sha256=hashes,model=spec['model'],model_weights=weights,adapter=spec['adapter'],layout_experiments=dict(
        macro_library=os.environ.get('MAGICANVAS_EXPERIMENTAL_MACRO_LIBRARY')=='1',
        vision_tiled=os.environ.get('MAGICANVAS_SIGLIP_TILED','').strip().lower() in ('1','true','yes','on')))


def _job(task):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    from unittest.mock import patch
    spec,case,row,record,compare,baseline=task
    def source_for(split,key):return deepcopy(row['source_design'])
    kwargs=dict(source_for=source_for,allow_incomplete=True)
    before=deepcopy(baseline['after']) if baseline else None
    if compare:
        # Exact pre-change ablation, same engine/selector/decoded model values.
        # Only this change's new candidate-axis contracts are disabled.
        with patch('engine.source_structure.copy_axis_contracts',return_value=[]):
            before=replay([record],[row],ROOT/spec['adapter'],ROOT/spec['model'],None,**kwargs)[0]
    after=replay([record],[row],ROOT/spec['adapter'],ROOT/spec['model'],None,**kwargs)[0]
    for result in (before,after):
        if result is None:continue
        if result.get('engine_error'):raise RuntimeError(result['engine_error'])
        if not record.get('messages'):
            for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):result[k]=None
        # Reconstructed replay JSON is not evidence of original completion.
        result['original_raw_json_complete']=row.get('raw_json_complete')
        result['original_raw_proposal_valid']=row.get('raw_proposal_valid')
    return dict(case=case,record=record,before=before,after=after)


def documents_differ(a,b):
    """Ignore sub-micro-pixel roundoff in the review filter, retain exact audits."""
    from numbers import Real
    from math import isclose
    if isinstance(a,Real) and isinstance(b,Real):return not isclose(a,b,rel_tol=0.,abs_tol=1e-6)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()!=b.keys() or any(documents_differ(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)!=len(b) or any(documents_differ(x,y) for x,y in zip(a,b))
    return a!=b


def publish(run,packet,spec):
    from scripts.build_focus_reflow_viewer import build,STYLE
    stage=packet/'review/regression_suite.next'
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True)
    records=run/'render_records';records.mkdir(exist_ok=True)
    results=[]
    for i,case in enumerate(spec['cases']):
        result=json.loads((run/f'{i:02d}.json').read_text());results.append(result)
        row=result['after'];source=row['source_design'];cohort=case['cohort']
        title=(source.get('name') or next((e.get('content','')[:65] for e in source['elements'] if e['type']=='text'),'Design'))
        if cohort in ('official','alternate'):title=case['pair_id'].replace('__to__',' → ')
        audit=dict(name=title,source=source,design=row['output_design'],target=row['target_canvas'],source_split=row['source_split'] if row['source_split']!='official' else None,source_note=case.get('source_note'),
            frozen_model_prediction=True,diagnostics=dict(winner=row['winner'],pair_id=case['pair_id'],qwen_enabled=True,
                original_raw_json_complete=row.get('original_raw_json_complete'),original_raw_proposal_valid=row.get('original_raw_proposal_valid'),
                raw_model_elements=row.get('qwen_model_elements',[]),qwen_status=row.get('qwen_status')),defects={k:row.get(k) for k in ('defect_flags','layout_defect_cost','structural_errors','glyph_overflow','source_lockup_error','geometry_mae')})
        (records/f'{i:02d}.json').write_text(json.dumps(audit,indent=2)+'\n')
    rows=build(records,stage,'Unified engine regression review')
    labels=dict(official='Official levels 1–3 · six formats each',alternate='Independently authored Level 3 difficulty',test='Complex held-out dataset',validation='Complex validation dataset',fresh='New independent designs · fresh Qwen predictions recorded',supplied='Supplied complex designs · disclosed asset-backed test copies',skyscraper='Portrait → 160×600 skyscraper stress tests')
    cards=[]
    for entry,case,result in zip(rows,spec['cases'],results):
        before=result.get('before');after=result['after'];source=after['source_design']
        changed=bool(before and documents_differ(before['output_design'],after['output_design']))
        added=sorted(set(after['defect_flags'])-set(before['defect_flags'])) if before else []
        entry.update(pair_id=case['pair_id'],cohort=case['cohort'],changed=changed,new_flags=added,
            before_winner=before.get('winner') if before else None)
        dimensions=f"{source['canvas']['width']} × {source['canvas']['height']} → {after['target_canvas']['width']} × {after['target_canvas']['height']}"
        change_note=f"Changed from baseline · {entry['before_winner']} → {entry['winner']}" if changed else 'Unchanged from baseline' if before else 'No baseline supplied'
        diagnostic=f'<p>{dimensions} · {html.escape(change_note)}</p><p>Automated flags: {html.escape(str(entry["defect_flags"]))} · new types: {html.escape(str(added))}</p>'
        note=f'<p class="muted">{html.escape(case["source_note"])}</p>' if case.get('source_note') else ''
        cards.append(f'<article data-changed="{str(changed).lower()}" data-flagged="{str(bool(after["defect_flags"])).lower()}" data-new-flags="{str(bool(added)).lower()}" data-cohort="{case["cohort"]}" data-pair-id="{html.escape(case["pair_id"],quote=True)}"><p class="muted">{labels[case["cohort"]]}</p><h2>{html.escape(entry["case"])}</h2>'+diagnostic+note+'<div class="panels">'+''.join(
            f'<figure data-panel="{label}"><figcaption>{label}</figcaption><a href="{entry[key]}"><img loading="lazy" src="{entry[key]}" alt="{label}"></a></figure>' for label,key in [('Original','source'),('Resized','image')])+f'</div><details><summary>Audit details</summary><p data-selection>Qwen enabled · 0.6B · selected {html.escape(entry["winner"])} · automated flags {html.escape(str(entry["defect_flags"]))}</p><p data-qwen-status>{html.escape(str(result["after"].get("qwen_status","")))}</p><p data-render-notes>Render notes: {html.escape(str(entry.get("render_warnings",{})))}</p><a data-output href="{entry["json"]}">Output JSON</a> · <a data-audit href="{entry["audit"]}">Diagnostics and raw prediction</a></details></article>')
    opts=''.join(f'<option value="{k}">{v}</option>' for k,v in labels.items())
    from collections import Counter
    counts=Counter(c['cohort'] for c in spec['cases'])
    overview=' · '.join(f'{n} {html.escape(labels[k])}' for k,n in counts.items())
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unified MagiCanvas review</title><style>'+STYLE+'</style><header><h1>Unified engine regression review</h1><p>'+overview+'. Qwen enabled in every case. Original and Resized only.</p><p class="muted">Development cohort, not a blind benchmark. Same saved raw predictions, adapter and selector in all replays. Intermediate diagnostics remain in audit files. Zero automated flags does not mean aesthetic perfection. Supplied designs use disclosed test-copy artwork/fonts; they are not confirmed hidden-test references.</p><p>Engine snapshot '+run.name+' · <a href="run_manifest.json">Provenance</a> · <a href="summary.json">Before/after evidence</a></p><label>Show <select id="group"><option value="">All conversions</option>'+opts+'</select></label> <label>Review filter <select id="condition"><option value="">All cases</option><option value="changed">Changed outputs</option><option value="flagged">Flagged outputs</option><option value="newFlags">New flag types</option></select></label><p class="muted">Open either image for native-resolution inspection. Only Original and Resized appear; raw coordinates and measured defects remain under Audit details.</p></header><main>'+''.join(cards)+'</main><script>const s=document.getElementById("group");s.value=new URLSearchParams(location.search).get("group")||"";const f=document.getElementById("condition");function update(){document.querySelectorAll("article[data-cohort]").forEach(c=>c.hidden=(!!s.value&&c.dataset.cohort!==s.value)||(!!f.value&&c.dataset[f.value]!=="true"))}s.onchange=update;f.onchange=update;update()</script></html>'
    (stage/'index.html').write_text(page);(stage/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    from scripts.publish_capacity_comparison import augment
    augment(stage,run)
    comparison=json.loads((run/'run_manifest.json').read_text()).get('baseline',{})
    summary=dict(cases=len(results),fixed_cases=spec.get('fixed_cases',len(results)),supplemental_cases=len(results)-spec.get('fixed_cases',len(results)),qwen_enabled=True,fresh_model_inference=False,ablation=comparison.get('scope','No baseline supplied'),results=[])
    for r in results:
        a,b=r['after'],r['before'];old={e['id']:e for e in b['output_design']['elements']} if b else {}
        delta=max((abs(float(e.get(k,0))-float(old[e['id']].get(k,0))) for e in a['output_design']['elements'] if e['id'] in old for k in ('x','y','width','height','font_size')),default=0)
        summary['results'].append(dict(pair_id=r['case']['pair_id'],cohort=r['case']['cohort'],before_winner=b.get('winner') if b else None,winner=a['winner'],max_geometry_change=delta,
            before_flags=b.get('defect_flags') if b else None,flags=a['defect_flags'],before_cost=b.get('layout_defect_cost') if b else None,cost=a.get('layout_defect_cost'),
            before_geometry_mae=b.get('geometry_mae') if b else None,geometry_mae=a.get('geometry_mae')))
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ('run_manifest.json','summary.json'):shutil.copy2(run/name,stage/name)
    # Publish one completed cohort. Previous complete images stay available if
    # computation/rendering fails; partial runs are never called current.
    target=packet/'review/regression_suite'
    backup=target.with_name('regression_suite.previous')
    if backup.exists():shutil.rmtree(backup)
    if target.exists():target.rename(backup)
    stage.rename(target)
    gallery=packet/'review/main_engine_qwen_outputs';gallery.mkdir(exist_ok=True)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current unified review</title><script>location.replace("../regression_suite/index.html"+location.search)</script><a href="../regression_suite/index.html">Open all original/resized conversions</a>')
    (packet/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>MagiCanvas review</title><h1>Current engine review</h1><p><a href="regression_suite/index.html">All original/resized conversions</a></p><p>Official, alternate and complex dataset cases share one frozen, Qwen-enabled engine snapshot. See LATEST_UPDATE.md for scope and limitations.</p>')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/regression_review.json')
    p.add_argument('--packet',type=Path,default=PACK)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--compare-before',action='store_true',help='Disable only the new copy-axis contract in the baseline; both arms have Qwen enabled.')
    p.add_argument('--baseline-run',type=Path,help='Frozen earlier run for all before/after comparisons; Qwen remains enabled.')
    p.add_argument('--check',action='store_true',help='Check whether the published viewer matches current code and frozen inputs.')
    args=p.parse_args();args.config=args.config.resolve();spec=load_cohort(args.config);fp=fingerprint(args.config)
    if args.baseline_run and args.compare_before:p.error('Choose baseline-run or compare-before, not both')
    baseline_fp=sha256((args.baseline_run/'run_manifest.json').read_bytes()).hexdigest() if args.baseline_run else None
    fp.update(compare_before=args.compare_before,qwen_enabled=True,fresh_model_inference=False,baseline_manifest_sha256=baseline_fp)
    signature=sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest()[:16]
    run=ROOT/'data/qwen_lora/regression_suite'/signature
    if args.check:
        current=args.packet/'review/regression_suite/run_manifest.json'
        ok=current.is_file() and all(json.loads(current.read_text())['fingerprint'].get(k)==fp.get(k) for k in ('files_sha256','model','model_weights','adapter','layout_experiments'))
        print('CURRENT' if ok else 'STALE: run scripts/refresh_regression_viewer.py');return 0 if ok else 1
    run.mkdir(parents=True,exist_ok=True)
    (run/'run_manifest.json').write_text(json.dumps(dict(fingerprint=fp,cohort=spec,baseline=dict(path=str(args.baseline_run) if args.baseline_run else None,scope='Frozen prior engine vs current, same saved Qwen predictions' if args.baseline_run else 'Same current engine with only candidate-axis contracts disabled vs enabled' if args.compare_before else 'No baseline supplied'),scope='Frozen raw Qwen predictions through the complete current engine; no Qwen-off arm or fresh generation timing.'),indent=2)+'\n')
    reports={};gold_records={};tasks=[]
    for i,case in enumerate(spec['cases']):
        if (run/f'{i:02d}.json').exists():continue
        path=case['prediction_report']
        if path not in reports:reports[path]=json.loads((ROOT/path).read_text())
        report=reports[path];row=next(r for r in report['results'] if r['pair_id']==case['pair_id'])
        record=next((r for r in report.get('records',[]) if r['pair_id']==case['pair_id']),None)
        if record is None and case.get('records_file'):
            gold_path=case['records_file']
            if gold_path not in gold_records:
                gold_records[gold_path]={r['pair_id']:r for r in map(json.loads,(ROOT/gold_path).read_text().splitlines()) if r}
            record=gold_records[gold_path][case['pair_id']]
        if record is None:
            record={k:row[k] for k in ('pair_id','source_split','target_canvas','family_id') if k in row}
        baseline_path=args.baseline_run/f'{i:02d}.json' if args.baseline_run else None
        baseline=json.loads(baseline_path.read_text()) if baseline_path and baseline_path.exists() else None
        if baseline and baseline['case']['pair_id']!=case['pair_id']:raise ValueError('Baseline cohort/order mismatch')
        tasks.append((i,(spec,case,row,record,args.compare_before,baseline)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(_job,t):i for i,t in tasks}
        for future in as_completed(pending):
            i=pending[future];result=future.result()
            (run/f'{i:02d}.json').write_text(json.dumps(result)+'\n')
            print(f'Completed {i+1}/{len(spec["cases"])} {result["case"]["pair_id"]}',flush=True)
    if fingerprint(args.config)!= {k:v for k,v in fp.items() if k not in ('compare_before','qwen_enabled','fresh_model_inference','baseline_manifest_sha256')}:
        raise SystemExit('Inputs or code changed during the run; nothing published. Rerun with a consistent snapshot.')
    summary=publish(run,args.packet,spec)
    print(json.dumps(dict(cases=summary['cases'],run=str(run),viewer=str(args.packet/'review/regression_suite/index.html'))))
    return 0


if __name__=='__main__':raise SystemExit(main())
