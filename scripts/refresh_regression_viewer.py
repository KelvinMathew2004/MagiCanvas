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


def fingerprint(config):
    files=sorted((ROOT/'engine').glob('*.py'))+[ROOT/'scripts/layout_defects.py',ROOT/'scripts/evaluate_qwen_adapter.py',ROOT/'scripts/replay_qwen_engine.py',Path(__file__).resolve(),config]
    spec=json.loads(config.read_text())
    files += [ROOT/spec['adapter']/name for name in ('adapter_config.json','adapter_model.safetensors','magicanvas_prompt_version.txt')]
    files += sorted({ROOT/c['prediction_report'] for c in spec['cases']})
    files += sorted({ROOT/c['records_file'] for c in spec['cases'] if c.get('records_file')})
    files += sorted((ROOT/'renderer/converter').glob('*.py'))
    files += [ROOT/'data/crello_fonts.pickle',ROOT/spec['model']/'config.json']
    files += sorted((ROOT/'tests/fixtures/novel_level3_sources').glob('n*.json'))
    files += sorted((ROOT/'input').glob('*level*.json'))
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    # Model weights stay local; record size/mtime plus configuration, without
    # hashing/loading a multi-GB base on each refresh.
    weights={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted((ROOT/spec['model']).glob('*.safetensors'))}
    return dict(files_sha256=hashes,model=spec['model'],model_weights=weights,adapter=spec['adapter'])


def _job(task):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    from unittest.mock import patch
    spec,case,row,record,compare=task
    def source_for(split,key):return deepcopy(row['source_design'])
    kwargs=dict(source_for=source_for,allow_incomplete=True)
    before=None
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
        audit=dict(name=title,source=source,design=row['output_design'],target=row['target_canvas'],source_split=row['source_split'] if row['source_split']!='official' else None,
            frozen_model_prediction=True,diagnostics=dict(winner=row['winner'],pair_id=case['pair_id'],qwen_enabled=True,
                original_raw_json_complete=row.get('original_raw_json_complete'),original_raw_proposal_valid=row.get('original_raw_proposal_valid'),
                raw_model_elements=row.get('qwen_model_elements',[])),defects={k:row.get(k) for k in ('defect_flags','layout_defect_cost','structural_errors','glyph_overflow','source_lockup_error','geometry_mae')})
        (records/f'{i:02d}.json').write_text(json.dumps(audit,indent=2)+'\n')
    rows=build(records,stage,'Unified engine regression review')
    labels=dict(official='Official levels 1–3 · six formats each',alternate='Independently authored Level 3 difficulty',test='Complex held-out dataset',validation='Complex validation dataset')
    cards=[]
    for entry,case in zip(rows,spec['cases']):
        entry.update(pair_id=case['pair_id'],cohort=case['cohort'])
        cards.append(f'<article data-cohort="{case["cohort"]}"><p class="muted">{labels[case["cohort"]]}</p><h2>{html.escape(entry["case"])}</h2><div class="panels">'+''.join(
            f'<figure><figcaption>{label}</figcaption><a href="{entry[key]}"><img loading="lazy" src="{entry[key]}" alt="{label}"></a></figure>' for label,key in [('Original','source'),('Resized','image')])+f'</div><details><summary>Audit details</summary><p>Qwen enabled · selected {html.escape(entry["winner"])} · automated flags {html.escape(str(entry["defect_flags"]))}</p><a href="{entry["json"]}">Output JSON</a> · <a href="{entry["audit"]}">Diagnostics and raw prediction</a></details></article>')
    opts=''.join(f'<option value="{k}">{v}</option>' for k,v in labels.items())
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unified MagiCanvas review</title><style>'+STYLE+'</style><header><h1>Unified engine regression review</h1><p>18 official conversions · 8 alternate conversions · 12 complex dataset cases. Qwen enabled in every case. Original and Resized only.</p><p class="muted">Fixed development cohort, not a blind benchmark. Same saved raw predictions, adapter and selector in all replays. Intermediate diagnostics remain in audit files. Zero automated flags does not mean aesthetic perfection.</p><p>Engine snapshot '+run.name+' · <a href="run_manifest.json">Provenance</a> · <a href="summary.json">Before/after evidence</a></p><label>Show <select id="group"><option value="">All 38 conversions</option>'+opts+'</select></label></header><main>'+''.join(cards)+'</main><script>const s=document.getElementById("group");s.value=new URLSearchParams(location.search).get("group")||"";function update(){document.querySelectorAll("article[data-cohort]").forEach(c=>c.hidden=!!s.value&&c.dataset.cohort!==s.value)}s.onchange=update;update()</script></html>'
    (stage/'index.html').write_text(page);(stage/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary=dict(cases=len(results),qwen_enabled=True,fresh_model_inference=False,ablation='Same current engine with only candidate-axis contracts disabled vs enabled',results=[])
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
    (packet/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>MagiCanvas review</title><h1>Current engine review</h1><p><a href="regression_suite/index.html">All 38 original/resized conversions</a></p><p>Official, alternate and complex dataset cases share one frozen, Qwen-enabled engine snapshot. See LATEST_UPDATE.md for scope and limitations.</p>')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/regression_review.json')
    p.add_argument('--packet',type=Path,default=PACK)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--compare-before',action='store_true',help='Disable only the new copy-axis contract in the baseline; both arms have Qwen enabled.')
    p.add_argument('--check',action='store_true',help='Check whether the published viewer matches current code and frozen inputs.')
    args=p.parse_args();spec=json.loads(args.config.read_text());fp=fingerprint(args.config)
    fp.update(compare_before=args.compare_before,qwen_enabled=True,fresh_model_inference=False)
    signature=sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest()[:16]
    run=ROOT/'data/qwen_lora/regression_suite'/signature
    if args.check:
        current=args.packet/'review/regression_suite/run_manifest.json'
        ok=current.is_file() and json.loads(current.read_text())['fingerprint']['files_sha256']==fp['files_sha256']
        print('CURRENT' if ok else 'STALE: run scripts/refresh_regression_viewer.py');return 0 if ok else 1
    run.mkdir(parents=True,exist_ok=True)
    (run/'run_manifest.json').write_text(json.dumps(dict(fingerprint=fp,cohort=spec,scope='Frozen raw Qwen predictions through the complete current engine; no Qwen-off arm or fresh generation timing.'),indent=2)+'\n')
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
        tasks.append((i,(spec,case,row,record,args.compare_before)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(_job,t):i for i,t in tasks}
        for future in as_completed(pending):
            i=pending[future];result=future.result()
            (run/f'{i:02d}.json').write_text(json.dumps(result)+'\n')
            print(f'Completed {i+1}/{len(spec["cases"])} {result["case"]["pair_id"]}',flush=True)
    if fingerprint(args.config)!= {k:v for k,v in fp.items() if k not in ('compare_before','qwen_enabled','fresh_model_inference')}:
        raise SystemExit('Inputs or code changed during the run; nothing published. Rerun with a consistent snapshot.')
    summary=publish(run,args.packet,spec)
    print(json.dumps(dict(cases=summary['cases'],run=str(run),viewer=str(args.packet/'review/regression_suite/index.html'))))
    return 0


if __name__=='__main__':raise SystemExit(main())
