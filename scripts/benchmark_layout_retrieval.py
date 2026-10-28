#!/usr/bin/env python3
"""Measure lookup/assignment separately from packing and render verification."""
from pathlib import Path
from statistics import median
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(args):
    from engine.macro_layout import composites
    from engine.retrieve import plans,candidates,library
    results=[]
    for path in sorted(args.run.glob('[0-9][0-9].json')):
        row=json.loads(path.read_text())['after'];source=row['source_design'];w,h=(row['target_canvas'][k] for k in ('width','height'))
        blocks=composites(source,tight=True)['blocks'];durations=[]
        for _ in range(6):
            start=time.perf_counter();p=plans(source,blocks,w,h);durations.append(1000*(time.perf_counter()-start))
        start=time.perf_counter();cs=candidates(source,blocks,w,h);realize=1000*(time.perf_counter()-start)
        results.append(dict(case=path.stem,blocks=len(blocks),near_plans=len(p),realized=len(cs),cold_lookup_ms=durations[0],warm_lookup_median_ms=median(durations[1:]),realization_including_lookup_ms=realize))
    report=dict(scope='Current fixed 38 sources; standalone geometry lookup/assignment and measured packing. Excludes rendering, model inference and native safety verification.',library_entries=len(library()),results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cases=len(results),plans=sum(r['near_plans'] for r in results),realized=sum(r['realized'] for r in results),median_lookup_ms=median(r['warm_lookup_median_ms'] for r in results),max_lookup_ms=max(r['warm_lookup_median_ms'] for r in results),median_realize_ms=median(r['realization_including_lookup_ms'] for r in results))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'data/chameleon_upgrade_audit/retrieval_benchmark.json');main(p.parse_args())
