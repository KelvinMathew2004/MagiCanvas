#!/usr/bin/env python3
"""Infer independent new sources with Qwen enabled; preserve raw audit records."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(args):
    os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    from scripts.evaluate_qwen_adapter import run_engine_mode
    fixtures={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'tests/fixtures/fresh_level3_sources').glob('f*.json'))}
    records=[dict(pair_id=f'fresh:{key}__to__{t["name"]}',family_id='fresh:'+key,source_split='train',target_canvas={k:t[k] for k in ('width','height')}) for key,f in fixtures.items() for t in f['targets']]
    if args.limit:records=records[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    fp=dict(engine={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')},sources={k:sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in fixtures.items()},environment={k:v for k,v in os.environ.items() if k.startswith(('MAGICANVAS_','PYTORCH_'))},fresh_inference=True)
    marker=args.output/'run_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise SystemExit('Snapshot changed; use a new output folder')
    marker.write_text(json.dumps(fp,indent=2)+'\n')
    rows=run_engine_mode(records,lambda split,key:deepcopy(fixtures[key.split(':',1)[1]]['source']),ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',args.output/'inference.jsonl')
    for r in rows:
        for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):r[k]=None
    report=dict(records=records,results=rows,fresh_model_inference=True,human_target_available=False,manifest=fp)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),errors=sum(bool(r.get('engine_error')) for r in rows),flagged=sum(bool(r.get('defect_flags')) for r in rows),qwen_wins=sum(r.get('qwen_family_win',False) for r in rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);main(p.parse_args())
