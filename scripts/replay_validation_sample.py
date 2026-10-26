#!/usr/bin/env python3
"""Re-evaluate a fixed validation sample with identical saved model coordinates."""
from pathlib import Path
import argparse,json,sys
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapter-dir',type=Path,default=ROOT/'models/qwen3-layout-lora-v2')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/qwen3-0.6b')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from scripts.replay_qwen_engine import replay
    from scripts.evaluate_qwen_adapter import summarize_engine
    prior=json.loads(a.prior.read_text());files=['engine/source_structure.py','engine/resize.py','engine/layout_contracts.py','engine/qwen_refine.py','engine/qwen_seed.py','engine/clean_zones.py','engine/structural_reflow.py','scripts/layout_defects.py']
    manifest=dict(prior_sha256=sha256(a.prior.read_bytes()).hexdigest(),code_sha256={f:sha256((ROOT/f).read_bytes()).hexdigest() for f in files},raw_coordinates_reused=True,model_dir=str(a.model_dir),adapter_dir=str(a.adapter_dir))
    mp=a.output/'run_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise SystemExit('Code or inputs changed: select another output folder.')
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    rows=replay(prior['records'],prior['results'],a.adapter_dir,a.model_dir,a.output/'results.jsonl')
    (a.output/'report.json').write_text(json.dumps(dict(results=rows,records=prior['records'],summary=summarize_engine(rows),manifest=manifest,fresh_model_inference=False,raw_coordinates_reused=True,prediction_origin='Earlier fresh local validation inference; the current engine replays the exact same coordinates.'),indent=2)+'\n')
    print(json.dumps(summarize_engine(rows),indent=2))
if __name__=='__main__':main()
