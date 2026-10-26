#!/usr/bin/env python3
"""Run real Qwen inference on synthetic extreme-source reverse-resize fixtures."""
from pathlib import Path
import json,sys,html,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay-report',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from scripts.evaluate_qwen_adapter import run_engine_mode,read_sources,summarize_engine
    from renderer.converter.convert import render_design_to_png
    from PIL import Image
    records=[]
    for path in sorted((ROOT/'tests/fixtures/level3_reverse').glob('*.json')):
        for name,w,h in [('square',1080,1080),('portrait',1080,1350)]:
            records.append(dict(pair_id=f'synthetic:{path.stem}__to__{name}',family_id='synthetic_level3_reverse',official_path=str(path),target_canvas=dict(width=w,height=h)))
    if args.replay_report:
        from scripts.replay_qwen_engine import replay
        cached=json.loads(args.replay_report.read_text())['results']
        rows=replay(records,cached,ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',args.output/'results.jsonl')
    else:
        rows=run_engine_mode(records,read_sources(),ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',args.output/'results.jsonl')
    cards=[];manifest=[]
    for i,r in enumerate(rows,1):
        if r.get('engine_error'):continue
        entry={k:r.get(k) for k in ['pair_id','winner','defect_flags','hard_checks_passed','target_canvas']}
        for label,doc in [('source',r['source_design']),('output',r['output_design'])]:
            png=args.output/f'{i:02d}_{label}.png';jpg=png.with_suffix('.jpg');render_design_to_png(doc,ROOT/'brand_kit/assets',png,max_size=1200)
            with Image.open(png) as im:im.convert('RGB').save(jpg,quality=92,optimize=True)
            png.unlink();entry[label]=jpg.name
        name=f'{i:02d}_output.json';(args.output/name).write_text(json.dumps(r['output_design'],indent=2)+'\n');entry['json']=name
        (args.output/f'{i:02d}_source.json').write_text(json.dumps(r['source_design'],indent=2)+'\n')
        manifest.append(entry);cards.append(f'<article><h2>{html.escape(r["pair_id"])}</h2><div><figure><figcaption>Synthetic authored source</figcaption><img src="{entry["source"]}"></figure><figure><figcaption>Main engine + 0.6B adapter</figcaption><img src="{entry["output"]}"></figure></div><p>{html.escape(r["winner"])} · Flags: {html.escape(str(r["defect_flags"]))} · <a href="{name}">Output JSON</a> · <a href="{i:02d}_source.json">Source JSON</a></p></article>')
    (args.output/'report.json').write_text(json.dumps(dict(summary=summarize_engine(rows),records=records,results=rows),indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Synthetic Level 3 reverse resizes</title><style>body{font:16px system-ui;background:#101820;color:#eee;margin:24px}a{color:#9ed4ff}article{background:#182532;padding:20px;margin-bottom:25px;border-radius:12px}article div{display:grid;grid-template-columns:1fr 1fr;gap:15px}figure{margin:0}img{width:100%;height:600px;object-fit:contain}p{overflow-wrap:anywhere}@media(max-width:800px){article div{grid-template-columns:1fr}}</style><h1>Synthetic Level 3 · extreme-source reverse resize</h1><p>Four independently authored test sources, two targets each. These are not official hackathon samples or training data. Real installed 0.6B inference and the normal selector; no target reference. All original Level 3 content/IDs and adversarial missing asset are retained.</p>'+''.join(cards))
    print(json.dumps(dict(cases=len(rows),flagged=sum(bool(r.get('flagged')) for r in rows),errors=sum(bool(r.get('engine_error')) for r in rows)),indent=2))
if __name__=='__main__':main()
