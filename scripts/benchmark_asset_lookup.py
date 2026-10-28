#!/usr/bin/env python3
"""Non-production ablation of request-local asset-root lookup memoization.

Same saved Qwen proposals and exact selector; never changes engine files.
Warm timings are paired; imports/token sampling are excluded. Background jobs
can affect wall time, so this is diagnostic evidence rather than a speed promise.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from hashlib import sha256
import argparse,importlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(args):
    os.environ['MAGICANVAS_QWEN_ENABLED']='1';os.environ['MAGICANVAS_CPU_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    from scripts.build_focus_reflow_viewer import _renderer
    from PIL import Image
    engine=importlib.import_module('engine.resize');original=engine._asset_root;uncached=engine._uncached_asset_root
    item=json.loads((ROOT/'data/qwen_lora/regression_suite/3e769527fb4f81df'/f'{args.case:02d}.json').read_text());row=item['after'];record=item['record']
    args.output.mkdir(parents=True,exist_ok=True);results=[]
    def run(mode):
        calls=misses=0
        def root(element):
            nonlocal calls
            calls+=1
            return original(element)
        def discover(element):
            nonlocal misses
            misses+=1
            return uncached(element)
        start=time.perf_counter()
        with patch.dict(os.environ,{'MAGICANVAS_ASSET_LOOKUP_CACHE':'1' if mode=='memoized' else '0'}),patch.object(engine,'_asset_root',root),patch.object(engine,'_uncached_asset_root',discover):
            out=replay([record],[row],ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',None,source_for=lambda s,k:deepcopy(row['source_design']),allow_incomplete=True)[0]
        return dict(mode=mode,wall_seconds=time.perf_counter()-start,calls=calls,misses=misses,winner=out['winner'],flags=out['defect_flags'],design=out['output_design'])
    warmup=run('baseline');outputs=[]
    for mode in ('baseline','memoized','memoized','baseline'):
        result=run(mode);outputs.append(result['design']);results.append({k:v for k,v in result.items() if k!='design'});print(json.dumps(results[-1]),flush=True)
    identical=all(doc==outputs[0] for doc in outputs)
    render=_renderer(row['source_design'],row['source_split']);hashes=[]
    for i,doc in enumerate(outputs):
        path=args.output/f'{i}.png';render(doc,path)
        with Image.open(path) as image:hashes.append(sha256(image.convert('RGBA').tobytes()).hexdigest())
    report=dict(pair_id=row['pair_id'],warmup_seconds=warmup['wall_seconds'],results=results,documents_identical=identical,pixels_identical=len(set(hashes))==1,production_change=True,cache_toggle='MAGICANVAS_ASSET_LOOKUP_CACHE',scope='Two alternating warm pairs, same saved Qwen/selector; actual production cache toggled, true uncached discovery calls counted. Imports and generation excluded; concurrent preference jobs may affect wall time. Asset roots immutable within each request.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=int,default=1);p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade/asset_lookup_benchmark');main(p.parse_args())
