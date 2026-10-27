#!/usr/bin/env python3
"""Compare frozen model coordinates across a source-structure engine revision."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import json,sys,argparse,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=0);p.add_argument('--off',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 # A resumed cohort can be requested while a long audit chain is running.
 # Serialize writers to the same checkpoint, without blocking other cohorts.
 lock=(a.output/'.audit.lock').open('a')
 import fcntl
 fcntl.flock(lock,fcntl.LOCK_EX)
 from scripts.replay_qwen_engine import replay
 from scripts.evaluate_qwen_adapter import summarize_engine
 from scripts.layout_defects import assess_engine_layout
 from engine.resize import _adapt_with_diagnostics
 prior=json.loads(a.prior.read_text());cached=prior['results'][a.offset:];cached=cached[:a.limit] if a.limit else cached
 records={r['pair_id']:r for split in ('test','validation') for r in map(json.loads,(ROOT/f'data/qwen_lora_expanded_v2/{split}.jsonl').read_text().splitlines())}
 records.update({r['pair_id']:r for r in prior.get('records',prior.get('manifest',{}).get('records',[]))})
 chosen=[];bank={}
 for r in cached:
  key=r['pair_id'].split('__to__')[0];bank[key]=r['source_design'];meta=deepcopy(records.get(r['pair_id'],{}));meta.update(pair_id=r['pair_id'],source_split=r.get('source_split','train'),target_canvas=r['target_canvas'])
  if r.get('official_path'):meta['official_path']=r['official_path']
  chosen.append(meta)
 files=sorted((ROOT/'engine').glob('*.py'))
 manifest=dict(prior_sha256=sha256(a.prior.read_bytes()).hexdigest(),code_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},offset=a.offset,limit=a.limit,raw_coordinates_reused=True,fresh_model_inference=False)
 mp=a.output/'run_manifest.json'
 if mp.exists() and json.loads(mp.read_text())!=manifest:raise SystemExit('Frozen code changed. Use another output directory.')
 mp.write_text(json.dumps(manifest,indent=2)+'\n')
 adapter=ROOT/'models'/('qwen3-layout-lora-v2-expanded-1.7b' if '1p7b' in str(a.prior) else 'qwen3-layout-lora-v2')
 model=ROOT/'models'/('qwen3-1.7b' if '1p7b' in str(a.prior) else 'qwen3-0.6b')
 rows=replay(chosen,cached,adapter,model,a.output/'results.jsonl',source_for=lambda split,key:deepcopy(bank[key]),allow_incomplete=all(r['pair_id'].startswith('authored:') for r in chosen))
 old={r['pair_id']:r for r in cached}
 for r in rows:
  before=old[r['pair_id']]
  if r['pair_id'].startswith('authored:'):
   for key in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):r[key]=None
  r['comparison']=dict(prior_winner=before['winner'],prior_flags=before['defect_flags'],geometry_identical=r['output_design']==before['output_design'],prior_geometry_mae=before.get('geometry_mae'))
  if a.off:
   os.environ['MAGICANVAS_QWEN_ENABLED']='0';os.environ['MAGICANVAS_DATASET_RENDERING']='1'
   output,diag=_adapt_with_diagnostics(r['source_design'],r['target_canvas']);r['qwen_off']=dict(pair_id=r['pair_id'],output_design=output,winner=diag['winner'],**assess_engine_layout(r['source_design'],output,diag))
 report=dict(summary=summarize_engine(rows),records=chosen,results=rows,manifest=manifest,fresh_model_inference=False,prediction_origin='Exact coordinates from the earlier frozen inference; current grouping/packing and selection.')
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
