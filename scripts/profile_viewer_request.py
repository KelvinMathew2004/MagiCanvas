#!/usr/bin/env python3
"""Profile the real preview path, including Qwen; never substitute cached boxes."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch
import argparse,cProfile,json,os,pstats,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(args):
    os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    os.environ['MAGICANVAS_PREVIEW_CACHE']='0'
    import importlib
    engine=importlib.import_module('engine.resize')
    from viewer.server import adapt_preview
    args.output.mkdir(parents=True,exist_ok=True)
    task=json.loads(args.input.read_text());task['target_canvas']=dict(width=args.width,height=args.height)
    fingerprints={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
    original=engine._adapt_with_diagnostics;results=[]
    for index in range(args.requests):
        captured={}
        def capture(*a,**kw):
            output,diagnostics=original(*a,**kw);captured.update(diagnostics);return output,diagnostics
        profile=cProfile.Profile();start=time.perf_counter()
        with patch.object(engine,'_adapt_with_diagnostics',capture):
            profile.enable();r=adapt_preview(deepcopy(task));profile.disable()
        elapsed=time.perf_counter()-start;profile.dump_stats(args.output/f'{index}.pstats');stats=pstats.Stats(profile)
        top=[dict(file=f,line=l,function=n,calls=t,own_seconds=o,cumulative_seconds=c) for (f,l,n),(p,t,o,c,callers) in sorted(stats.stats.items(),key=lambda item:-item[1][3])[:45]]
        row=dict(request=index+1,cold_process=index==0,wall_seconds=elapsed,preview_seconds=r['seconds'],engine_seconds=captured.get('seconds'),
            generation=captured.get('qwen_diagnostics',{}).get('generation'),qwen_status=captured.get('qwen'),winner=captured.get('winner'),
            output_sha256=sha256(json.dumps(r['design'],sort_keys=True).encode()).hexdigest(),preview_timings=r.get('timings'),top_functions=top)
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='top_functions'}),flush=True)
    report=dict(scope='Actual adapt_preview path, Qwen enabled, fresh deterministic token generation each request. Includes profiling overhead and preview renders; excludes browser/network/animation. Not a controlled CPU-vs-MPS or Docker benchmark.',
        engine_sha256=fingerprints,results=results,input=str(args.input),target=task['target_canvas'],environment={k:v for k,v in os.environ.items() if k.startswith('MAGICANVAS_')})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'input/chameleon_sample_level3_advanced.json')
    p.add_argument('--width',type=int,default=728);p.add_argument('--height',type=int,default=90);p.add_argument('--requests',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
