#!/usr/bin/env python3
"""Compare Qwen-off and Qwen-primary engine results on a held-out JSONL split."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SPLIT_DIRS={name:ROOT/'data'/f'crello_{name}' for name in ('train','validation','test')}


def index_designs(folder):
    rows={}
    for line in (folder/'manifest.jsonl').read_text().splitlines():
        if line.strip():
            row=json.loads(line)
            if row.get('status')=='exported': rows[row['template_id']]=row
    return rows


def geometry_metrics(elements,targets,canvas):
    predicted={e['id']:e for e in elements}; expected={e['id']:e for e in targets}
    ids=sorted(predicted.keys() & expected.keys())
    if not ids: return {'matched':0,'geometry_mae':None,'font_relative_mae':None}
    w,h=canvas['width'],canvas['height']; errors=[]; fonts=[]
    for eid in ids:
        a,b=predicted[eid],expected[eid]
        errors.extend(abs(float(a.get(k,0))-float(b.get(k,0)))/denom
                      for k,denom in (('x',w),('y',h),('width',w),('height',h)))
        if 'font_size' in a and 'font_size' in b:
            fonts.append(abs(float(a['font_size'])-float(b['font_size']))/max(1.,float(b['font_size'])))
    return {'matched':len(ids),'geometry_mae':sum(errors)/len(errors),
            'font_relative_mae':sum(fonts)/len(fonts) if fonts else None}


def image_ratio_violations(source,output):
    out={e['id']:e for e in output['elements']}; sw,sh=source['canvas']['width'],source['canvas']['height']
    violations=[]
    for original in source['elements']:
        if original.get('type')!='image': continue
        if original.get('fit')=='cover': continue
        full=(original['x']<=sw*.02 and original['y']<=sh*.02 and
              original['x']+original['width']>=sw*.98 and original['y']+original['height']>=sh*.98)
        behind=all(original.get('z_index',0)<=t.get('z_index',0)
                   for t in source['elements'] if t.get('type')=='text')
        if full and behind and not original.get('group_id'): continue
        e=out.get(original['id'])
        if not e: continue
        a=float(original['width'])/max(1e-8,float(original['height']))
        b=float(e['width'])/max(1e-8,float(e['height']))
        if abs(b/a-1)>.05: violations.append(original['id'])
    return violations


def run_one(row,indexes):
    from engine.main import self_check
    from engine.qwen_prompt import decode_v2_completion
    import engine.resize as resize
    import engine.qwen_seed as qwen_seed
    source_id,target_id=row['pair_id'].split('__to__',1)
    source_split,target_split=row['source_split'],row['target_split']
    source_row=indexes[source_split][source_id]; target_row=indexes[target_split][target_id]
    source=json.loads((SPLIT_DIRS[source_split]/source_row['design']).read_text())
    target=json.loads((SPLIT_DIRS[target_split]/target_row['design']).read_text())
    canvas=row['target_canvas']; gold=json.loads(row['messages'][-1]['content'])
    if gold.get('elements') and 'box' in gold['elements'][0]:
        gold=decode_v2_completion(gold,source,canvas)
    expected=gold['elements']
    os.environ['MAGICANVAS_QWEN_ENABLED']='0'
    off,off_report=resize._adapt_with_diagnostics(source,canvas)
    captured=[]; original_generate=qwen_seed.generate_seed
    def capture(*args,**kwargs):
        result=original_generate(*args,**kwargs)
        if result[0] is not None: captured.append(result[0])
        return result
    qwen_seed.generate_seed=capture
    os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    try: on,on_report=resize._adapt_with_diagnostics(source,canvas)
    finally: qwen_seed.generate_seed=original_generate
    qdiag=on_report.get('qwen_diagnostics',{})
    target_geometry=geometry_metrics(expected,expected,canvas)
    def assess(output,report):
        visual=report.get('visual',{})
        chosen=report.get('winner')
        visual_candidates=visual.get('candidates',[])
        warnings=next((c.get('warnings',[]) for c in visual_candidates if c.get('name')==chosen),[])
        return {'self_check':self_check(source,output),
                'image_aspect_violations':image_ratio_violations(source,output),
                'renderer_warnings':warnings}
    return {'pair_id':row['pair_id'],'canvas':canvas,'elements':len(source['elements']),
            'qwen_proposal_received':qdiag.get('proposal_received',False),
            'qwen_status':qdiag.get('status'),'winner':on_report.get('winner'),
            'raw_qwen':geometry_metrics(captured[-1],expected,canvas) if captured else None,
            'qwen_on':geometry_metrics(on['elements'],expected,canvas),
            'qwen_off':geometry_metrics(off['elements'],expected,canvas),
            'qwen_on_checks':assess(on,on_report),'qwen_off_checks':assess(off,off_report)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/qwen_lora/test.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/qwen_engine_evaluation.json')
    p.add_argument('--limit',type=int)
    p.add_argument('--resume',action='store_true',help='continue from the adjacent .progress.jsonl file')
    a=p.parse_args()
    os.environ['MAGICANVAS_DATASET_RENDERING']='1'
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    if a.limit: rows=rows[:a.limit]
    if rows:
        prompt_version=2 if 'box' in json.loads(rows[0]['messages'][-1]['content'])['elements'][0] else 1
        if prompt_version==2:
            import engine.qwen_seed as qwen_seed
            marker=qwen_seed.ADAPTER_DIR/'magicanvas_prompt_version.txt'
            adapter_version=int(marker.read_text().strip()) if marker.is_file() else 1
            if adapter_version!=2:
                raise SystemExit(f"Held-out JSONL requires prompt v2, but installed adapter is v{adapter_version}. Train/install the v2 adapter first.")
        os.environ['MAGICANVAS_QWEN_PROMPT_VERSION']=str(prompt_version)
    indexes={split:index_designs(folder) for split,folder in SPLIT_DIRS.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    progress=a.output.with_suffix(a.output.suffix+'.progress.jsonl')
    results=[]
    if a.resume and progress.is_file():
        results=[json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    else:
        progress.write_text('',encoding='utf-8')
    completed={r['pair_id'] for r in results}
    remaining=[row for row in rows if row['pair_id'] not in completed]
    for i,row in enumerate(remaining,1):
        result=run_one(row,indexes); results.append(result)
        with progress.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(f"{len(results)}/{len(rows)} {row['pair_id']} · winner={result['winner']} · proposal={result['qwen_proposal_received']}",flush=True)
    def mean(key,split):
        vals=[r[split][key] for r in results if r[split].get(key) is not None]
        return sum(vals)/len(vals) if vals else None
    summary={'examples':len(results),
             'mean_geometry_mae':{'qwen_off':mean('geometry_mae','qwen_off'),'qwen_on':mean('geometry_mae','qwen_on'),'raw_qwen':mean('geometry_mae','raw_qwen')},
             'mean_font_relative_mae':{'qwen_off':mean('font_relative_mae','qwen_off'),'qwen_on':mean('font_relative_mae','qwen_on'),'raw_qwen':mean('font_relative_mae','raw_qwen')},
             'qwen_on_id_or_bounds_failures':sum(bool(r['qwen_on_checks']['self_check']) for r in results),
             'qwen_on_image_aspect_violation_count':sum(len(r['qwen_on_checks']['image_aspect_violations']) for r in results),
             'qwen_on_renderer_warning_count':sum(len(r['qwen_on_checks']['renderer_warnings']) for r in results),
             'qwen_winners':sum(r['winner']=='qwen-proposal' for r in results),
             'qwen_proposals_received':sum(r['qwen_proposal_received'] for r in results)}
    payload={'input':str(a.input.resolve()),'summary':summary,'results':results}
    temp=a.output.with_suffix(a.output.suffix+'.tmp')
    temp.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    temp.replace(a.output)
    print(json.dumps({'output':str(a.output.resolve()),'summary':summary},indent=2))

if __name__=='__main__': main()
