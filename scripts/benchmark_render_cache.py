#!/usr/bin/env python3
"""Isolate raster cache/concurrency timings; exact output pixels must agree."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,os,sys,time,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(output):
    from engine.render_cache import DiagnosticRenderCache
    from renderer.converter.convert import render_design_to_png
    from PIL import Image
    source=json.loads((ROOT/'input/chameleon_sample_level3_advanced.json').read_text())['source'];docs=[]
    for i in range(6):
        doc=deepcopy(source)
        for e in doc['elements']:
            if e['type']=='text':e['x']+=i*.1
        docs += [doc,dict(canvas=doc['canvas'],elements=[e for e in doc['elements'] if e['type']!='text'])]
    docs += deepcopy(docs[:4]);reports=[];reference=None
    for workers in (1,2):
        os.environ['MAGICANVAS_RENDER_WORKERS']=str(workers)
        with tempfile.TemporaryDirectory() as directory:
            started=time.perf_counter()
            with DiagnosticRenderCache(render_design_to_png,ROOT/'brand_kit/assets') as cache:
                for i in range(0,len(docs),2):cache.render_many([(d,Path(directory)/f'{j}.png') for j,d in enumerate(docs[i:i+2],i)])
                pixel_hashes=[]
                for i in range(len(docs)):
                    with Image.open(Path(directory)/f'{i}.png') as im:pixel_hashes.append(sha256(im.convert('RGB').tobytes()).hexdigest())
                report=dict(workers=workers,wall_seconds=time.perf_counter()-started,stats=cache.stats,pixel_hashes=pixel_hashes)
            if reference is None:reference=pixel_hashes
            assert pixel_hashes==reference,'Concurrent render changed pixels'
            reports.append(report)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(dict(reports=reports,default_workers=1,scope='Raster-only diagnostic microbenchmark, not total model/engine latency'),indent=2)+'\n');print(json.dumps(reports,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade/render_benchmark.json');run(p.parse_args().output)
