#!/usr/bin/env python3
"""Replay actual human-rejected campaign tasks, preserving the voted images.

Exact saved Qwen values isolate engine changes. These are reviewed development
outliers, not unseen tests; no inference, new approvals or ranker training.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json, os, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def job(task):
    os.environ.update(MAGICANVAS_QWEN_ENABLED='1',MAGICANVAS_CPU_THREADS='1')
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    key,saved=task;source=saved['source'];meta=saved['task'];pid=meta['source_id']+'__to__'+meta['format']
    record=dict(pair_id=pid,source_split=meta.get('source_split','train'),family_id=meta['family_id'],target_canvas=meta['target_canvas'])
    cached=dict(pair_id=pid,source_design=source,target_canvas=meta['target_canvas'],qwen_model_elements=saved['diagnostics']['qwen_diagnostics']['model_elements'])
    row=replay([record],[cached],ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',None,source_for=lambda *a:deepcopy(source),allow_incomplete=True)[0]
    if row.get('engine_error'):raise RuntimeError(row['engine_error'])
    for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):row[k]=None
    return row


def main(args):
    from scripts.build_focus_reflow_viewer import build
    manifest=[json.loads(l) for l in (args.campaign/'manifest.jsonl').read_text().splitlines()]
    by={r['pair_id']:r for r in manifest};labels=[json.loads(l) for l in args.labels.read_text().splitlines()]
    selected={}
    for vote in labels:
        if not vote.get('reject_both') or vote['pair_id'].endswith('-repeat'):continue
        item=by[vote['pair_id']]
        if vote['review_hash']!=item['review_hash']:raise ValueError('Label/image provenance mismatch')
        audit=Path(item['audit']);selected[audit.stem]=(audit,item)
    args.output.mkdir(parents=True,exist_ok=True)
    fp=dict(engine={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')},
            labels_sha256=sha256(args.labels.read_bytes()).hexdigest(),audits={key:sha256(p.read_bytes()).hexdigest() for key,(p,_) in selected.items()},qwen_enabled=True,fresh_inference=False)
    marker=args.output/'run_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise ValueError('Replay snapshot changed; choose a new directory')
    marker.write_text(json.dumps(fp,indent=2)+'\n');results={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={}
        for key,(path,_) in selected.items():
            target=args.output/(key+'.json')
            if target.exists():results[key]=json.loads(target.read_text())
            else:tasks[pool.submit(job,(key,json.loads(path.read_text())))]=key
        for future in as_completed(tasks):
            key=tasks[future];row=future.result();results[key]=row
            (args.output/(key+'.json')).write_text(json.dumps(row)+'\n')
            print(key,row['winner'],row['defect_flags'],flush=True)
    records=args.output/'render_records';records.mkdir(exist_ok=True);summary=[]
    from engine.macro_layout import quality
    for key,(path,item) in selected.items():
        saved=json.loads(path.read_text());row=results[key];old=saved['winner'];new=row['output_design'];s=row['source_design'];w,h=(new['canvas'][k] for k in ('width','height'))
        before=quality(old['elements'],s,w,h);after=quality(new['elements'],s,w,h)
        original={e['id']:e for e in old['elements']}
        delta=max((abs(e.get(k,0)-original[e['id']].get(k,0)) for e in new['elements'] for k in ('x','y','width','height','font_size')),default=0.)
        summary.append(dict(task=key,format=item['format'],max_geometry_change=delta,before_winner=saved['diagnostics']['winner'],winner=row['winner'],
            changed=old!=new,flags=row['defect_flags'],before_quality=before,quality=after,
            learned_macro_rendered=[c['name'] for c in row.get('macro_reflow',{}).get('candidates',[]) if c['name'].startswith('qwen-macro')]))
        record=dict(name=item.get('title',key),source=s,design=new,target=new['canvas'],source_split=item['source_split'],
            frozen_model_prediction=True,diagnostics=dict(winner=row['winner'],raw_model_elements=row.get('qwen_model_elements'),macro_reflow=row.get('macro_reflow')),
            defects=dict(defect_flags=row['defect_flags']))
        (records/(key+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    report=dict(scope='Human neither-good campaign tasks; frozen decoded Qwen replay, development evidence only. Preferences and their images unchanged.',tasks=len(summary),results=summary)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    build(records,args.viewer,'Human-rejected campaign designs · current engine')
    print(json.dumps(dict(tasks=len(summary),changed=sum(r['changed'] for r in summary),flagged=sum(bool(r['flags']) for r in summary))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--campaign',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/preferences_campaign')
    p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade/measured_reviewed_outliers')
    p.add_argument('--viewer',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/reviewed_outliers')
    main(p.parse_args())
