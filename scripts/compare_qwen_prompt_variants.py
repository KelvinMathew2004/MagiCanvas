#!/usr/bin/env python3
"""Paired old/new prompt audit with identical model, adapter and engine selector.

Raw model defects are measured before resize.py candidate generation/repair;
final defects are measured on the selected output. The pre-change prompt is
loaded from git HEAD and snapshotted with the result for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT_DATA=ROOT/'data/qwen_lora_expanded_v2/test.jsonl'
DEFAULT_OUTPUT=ROOT/'data/qwen_lora/prompt_ab_same_adapter'
MISALIGNMENT_THRESHOLD=.004


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def load_old_prompt():
    source=subprocess.check_output(['git','show','HEAD:engine/qwen_prompt.py'],cwd=ROOT,text=True)
    module=ModuleType('engine._qwen_prompt_prechange')
    module.__package__='engine'
    exec(compile(source,'git:HEAD:engine/qwen_prompt.py','exec'),module.__dict__)
    return source,module


def records_and_sources(path):
    from scripts.evaluate_qwen_adapter import read_sources
    records=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(records)!=89:
        raise ValueError(f'Expected the 89-row held-out test split, found {len(records)} in {path}')
    return records,read_sources()


def _raw_design(source,canvas,model_rows):
    rows={row['id']:row for row in model_rows or []}
    elements=[]
    for original in source.get('elements',[]):
        proposal=rows.get(original['id'])
        if proposal is None:continue
        elements.append({**original,**proposal})
    return {'canvas':dict(canvas),'elements':elements}


def _outside_ids(design,width,height):
    from engine.resize import _rotated_bounds
    result=[]
    for e in design.get('elements',[]):
        b=_rotated_bounds(e)
        if (e['x'] < -1e-5 or e['y'] < -1e-5 or e['x']+e['width']>width+1e-5
                or e['y']+e['height']>height+1e-5 or b[0]<-1e-5 or b[1]<-1e-5
                or b[2]>width+1e-5 or b[3]>height+1e-5):
            result.append(e['id'])
    return sorted(set(result))


def defects(source,design,report,missing_ids=()):
    from scripts.layout_defects import assess_engine_layout
    from scripts.mine_qwen_batch_defects import _source_axis_alignment
    metrics=assess_engine_layout(source,design,report)
    axes=_source_axis_alignment(source,design.get('elements',[]))
    bad_axis_groups=[g for g in axes['clusters'] if g['max_error']>MISALIGNMENT_THRESHOLD]
    bad_axis_members=sorted({eid for group in bad_axis_groups for eid in group['ids']})
    stretch=list(metrics.get('image_ratio_violations') or [])
    overlap=list(metrics.get('unintended_text_overlaps') or [])
    overlap+=list(metrics.get('text_hero_occlusion') or [])
    out=_outside_ids(design,design['canvas']['width'],design['canvas']['height'])
    fit=metrics.get('glyph_overflow') or {}
    return {
        'stretch':{'count':len(stretch),'elements':stretch},
        'misalignment':{'count':len(bad_axis_groups),'elements':bad_axis_members,
                        'groups':bad_axis_groups,'mean_normalized_error':axes['mean_normalized_error'],
                        'threshold':MISALIGNMENT_THRESHOLD},
        'overlap':{'count':len(overlap),'details':overlap},
        'out_of_bounds':{'count':len(out),'elements':out},
        'text_fit':{'count':len(fit),'elements':fit},
        'incomplete_ids':{'count':len(missing_ids),'elements':sorted(missing_ids)},
    }


def _load_checkpoint(path):
    rows={}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row=json.loads(line);rows[row['pair_id']]=row
    return rows


def _run_arm(name,records,source_for,prompt_module,checkpoint):
    import engine.qwen_seed as qseed
    import engine.resize as resize
    completed=_load_checkpoint(checkpoint)
    qseed.messages_for=prompt_module.messages_for
    qseed.font_reference=prompt_module.font_reference
    os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    os.environ['MAGICANVAS_QWEN_PROMPT_VERSION']='2'
    for index,record in enumerate(records,1):
        pair=record['pair_id']
        if pair in completed:
            print(f'{name} {index}/{len(records)}: checkpointed',flush=True)
            continue
        source_id=pair.split('__to__',1)[0]
        source=source_for(record.get('source_split','test'),source_id)
        canvas=record['target_canvas']
        try:
            design,engine_report=resize._adapt_with_diagnostics(source,canvas)
            diag=engine_report.get('qwen_diagnostics') or {}
            raw_design=_raw_design(source,canvas,diag.get('model_elements',[]))
            generation=diag.get('generation') or {}
            raw_defects=defects(source,raw_design,
                                {'winner':'raw','visual':{'candidates':[{'name':'raw'}]}},
                                generation.get('missing_ids',[]))
            final_defects=defects(source,design,engine_report)
            row={
                'pair_id':pair,'family_id':record.get('family_id'),
                'prompt_variant':name,'winner':engine_report.get('winner'),
                'proposal_received':bool(diag.get('proposal_received')),
                'raw_json_complete':generation.get('json_complete'),
                'raw_missing_ids':generation.get('missing_ids',[]),
                'raw_extra_ids':generation.get('unknown_ids',[]),
                'raw_defects':raw_defects,'final_defects':final_defects,
                'raw_elements':raw_design.get('elements',[]),
                'final_elements':design.get('elements',[]),
                'target_canvas':canvas,
            }
        except Exception as exc:
            row={'pair_id':pair,'family_id':record.get('family_id'),
                 'prompt_variant':name,'engine_error':f'{type(exc).__name__}: {exc}',
                 'raw_defects':{},'final_defects':{},'raw_elements':[],'final_elements':[],
                 'target_canvas':canvas}
        completed[pair]=row
        with checkpoint.open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
        print(f'{name} {index}/{len(records)}: {row.get("winner",row.get("engine_error"))}',flush=True)
    return [completed[record['pair_id']] for record in records]


def _summary(rows,stage):
    categories=('stretch','misalignment','overlap','out_of_bounds','text_fit','incomplete_ids')
    result={'outputs':len(rows),'engine_errors':sum(bool(r.get('engine_error')) for r in rows),
            'categories':{}}
    for category in categories:
        per_case=[r.get(stage,{}).get(category,{}) for r in rows]
        result['categories'][category]={
            'outputs_affected':sum(bool(x.get('count')) for x in per_case),
            'total_incidents':sum(int(x.get('count',0)) for x in per_case),
        }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--adapter-dir',type=Path,default=ROOT/'models/qwen3-layout-lora-v2')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models/qwen3-0.6b')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    records,source_for=records_and_sources(args.input)
    marker=args.adapter_dir/'magicanvas_prompt_version.txt'
    if not marker.is_file() or marker.read_text().strip()!='2':
        raise SystemExit(f'This paired audit requires the prompt-v2 adapter marker at {marker}')
    if not args.adapter_dir.is_dir() or not args.model_dir.is_dir():
        raise SystemExit('Adapter or base-model directory is missing')
    old_source,old_module=load_old_prompt()
    import engine.qwen_prompt as new_module
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'old_prompt_snapshot.py').write_text(old_source)
    manifest={
        'input':str(args.input.resolve()),'input_sha256':sha256(args.input),
        'adapter':str(args.adapter_dir.resolve()),'adapter_sha256':sha256(args.adapter_dir/'adapter_model.safetensors'),
        'base_model':str(args.model_dir.resolve()),
        'base_weights':{p.name:sha256(p) for p in sorted(args.model_dir.glob('*.safetensors'))},
        'engine_resize_sha256':sha256(ROOT/'engine/resize.py'),
        'engine_qwen_seed_sha256':sha256(ROOT/'engine/qwen_seed.py'),
        'old_prompt_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
        'new_prompt_sha256':sha256(ROOT/'engine/qwen_prompt.py'),
        'old_prompt_revision':'git HEAD:engine/qwen_prompt.py',
        'selector':'same current engine._adapt_with_diagnostics for both prompt variants',
        'generation':'do_sample=False; prompt-v2 adapter; Qwen3-0.6B base',
    }
    manifest_path=args.output_dir/'run_manifest.json'
    if args.resume and manifest_path.is_file() and json.loads(manifest_path.read_text())!=manifest:
        raise SystemExit('Inputs, adapter, model, prompt, or selector code changed; use a new output directory.')
    if args.resume and not manifest_path.is_file():
        raise SystemExit('No matching run manifest exists for resume.')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    old_path=args.output_dir/'old_prompt.jsonl';new_path=args.output_dir/'new_prompt.jsonl'
    if not args.resume:
        for path in (old_path,new_path):
            if path.exists():path.unlink()
    import engine.qwen_seed as qseed
    qseed.MODEL_DIR=args.model_dir.resolve();qseed.ADAPTER_DIR=args.adapter_dir.resolve()
    qseed._load.cache_clear()
    # Load identical weights once; only the source prompt function changes.
    loaded=qseed._load()
    if loaded is None:raise SystemExit('Could not load the local Qwen model/adapter; check qwen_seed._load requirements.')
    old_rows=_run_arm('old',records,source_for,old_module,old_path)
    new_rows=_run_arm('new',records,source_for,new_module,new_path)
    summary={'manifest':manifest,
        'old_prompt':{'before_repair':_summary(old_rows,'raw_defects'),
                      'after_repair':_summary(old_rows,'final_defects')},
        'new_prompt':{'before_repair':_summary(new_rows,'raw_defects'),
                      'after_repair':_summary(new_rows,'final_defects')},
        'paired_deltas_new_minus_old':{},
    }
    for label in ('before_repair','after_repair'):
        old_summary=summary['old_prompt'][label]['categories']
        new_summary=summary['new_prompt'][label]['categories']
        summary['paired_deltas_new_minus_old'][label]={k:{
            'outputs_affected':new_summary[k]['outputs_affected']-old_summary[k]['outputs_affected'],
            'total_incidents':new_summary[k]['total_incidents']-old_summary[k]['total_incidents'],
        } for k in old_summary}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    qseed._load.cache_clear()


if __name__=='__main__':main()

