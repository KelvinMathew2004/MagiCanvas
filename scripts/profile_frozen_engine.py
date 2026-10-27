#!/usr/bin/env python3
"""Profile a frozen-prediction engine case; excludes real Qwen token sampling."""
from pathlib import Path
from copy import deepcopy
import argparse,cProfile,pstats,json,sys,os,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(output):
    os.environ['MAGICANVAS_CPU_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    prior=json.loads((ROOT/'data/qwen_lora/peer_edge_alignment/official/report.json').read_text())
    row=next(r for r in prior['results'] if 'level3_advanced__to__leaderboard' in r['pair_id']);record=next(r for r in prior['records'] if r['pair_id']==row['pair_id'])
    profile=cProfile.Profile();started=time.perf_counter();profile.enable()
    result=replay([record],[row],ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',None,source_for=lambda s,k:deepcopy(row['source_design']))[0]
    profile.disable();wall=time.perf_counter()-started;output.mkdir(parents=True,exist_ok=True);profile.dump_stats(output/'engine.pstats')
    stats=pstats.Stats(profile);top=[]
    for (file,line,name),(primitive,total,own,cumulative,callers) in sorted(stats.stats.items(),key=lambda r:-r[1][3])[:40]:
        top.append(dict(file=file,line=line,function=name,calls=total,own_seconds=own,cumulative_seconds=cumulative))
    report=dict(pair_id=row['pair_id'],wall_seconds=wall,token_generation_included=False,cold_process=True,cpu_threads=1,profiling_overhead_included=True,winner=result['winner'],flags=result['defect_flags'],top_functions=top,scope='One frozen-prediction, cold-process development profile; not fresh inference or an M5 Docker measurement')
    (output/'profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade/profile');main(p.parse_args().output)
