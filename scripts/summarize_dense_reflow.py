#!/usr/bin/env python3
"""Check snapshot identity and package measured reflow evidence, without inference."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(args):
    from PIL import Image,ImageOps,ImageDraw
    from engine.macro_layout import quality
    from scripts.build_focus_reflow_viewer import _renderer
    current={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
    for folder,key in ((args.attachments,None),(args.dense,'files_sha256'),(args.reviewed,'engine')):
        manifest=json.loads((folder/'run_manifest.json').read_text());hashes=manifest if key is None else manifest[key]
        if any(hashes.get(p)!=digest for p,digest in current.items()):raise ValueError(f'Stale engine snapshot: {folder}')
    args.output.mkdir(parents=True,exist_ok=True);campaign=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/preferences_campaign'
    attachment=[];panels=[]
    for key in ('pref-new-00-00-leaderboard','pref-new-00-01-leaderboard','pref-new-00-00-widescreen','pref-new-00-01-widescreen'):
        old=json.loads((ROOT/'data/qwen_lora/engine_upgrade/attachment_engine_replay'/(key+'.json')).read_text())
        row=json.loads((args.attachments/(key+'.json')).read_text());s=row['source_design'];w,h=(row['target_canvas'][k] for k in ('width','height'))
        attachment.append(dict(task=key,before_winner=old['winner'],winner=row['winner'],before_flags=old['defect_flags'],flags=row['defect_flags'],
            before_quality=quality(old['output_design']['elements'],s,w,h),quality=quality(row['output_design']['elements'],s,w,h),raw_ids_complete=row['raw_proposal_valid']))
        panels.append((key,old['output_design'],row['output_design'],s,row['source_split']))
    def contact(entries,path,labels):
        sheet=Image.new('RGB',(1400,330*len(entries)),'#17232e');draw=ImageDraw.Draw(sheet)
        for i,(key,a,b,src,split) in enumerate(entries):
            render=_renderer(src,split)
            for col,doc in enumerate((a,b)):
                dest=args.output/(path.stem+f'_{i}_{col}.png');render(doc,dest)
                with Image.open(dest) as im:thumb=ImageOps.contain(im.convert('RGB'),(680,285))
                sheet.paste(thumb,(col*700+(700-thumb.width)//2,i*330+35+(285-thumb.height)//2));draw.text((col*700+8,i*330+8),key+' / '+labels[col],fill='white')
                dest.unlink()
        sheet.save(path,quality=86,optimize=True)
    contact(panels,args.output/'dense_reflow_attachments.jpg',('Before','After'))
    reviewed=json.loads((args.reviewed/'summary.json').read_text());entries=[]
    for r in reviewed['results']:
        row=json.loads((args.reviewed/(r['task']+'.json')).read_text());entries.append((r['task'],row['source_design'],row['output_design'],row['source_design'],row['source_split']))
    contact(entries,args.output/'dense_reflow_reviewed.jpg',('Original','Resized'))
    dense=json.loads((args.dense/'report.json').read_text());model={}
    for arm,rows in dense['results'].items():
        macro=[c for row in rows for c in row.get('macro_reflow',{}).get('candidates',[]) if c['name'].startswith('qwen-macro-band-flow')]
        model[arm]=dict(dense['summary'][arm],learned_macro_rendered=len(macro),learned_macro_wins=sum(r['winner'].startswith('qwen-macro-band-flow') for r in rows),
            learned_macro_flags=dict(Counter(f for c in macro for f in c['flags'])))
    regression=json.loads((args.regression/'summary.json').read_text())
    summary=dict(scope='Current engine, exact saved Qwen proposals. Reviewed development cases, not new independent tests or inference latency.',engine_sha256=current,
        regression=dict(cases=regression['cases'],winner_flips=sum(r['before_winner']!=r['winner'] for r in regression['results']),flagged=sum(bool(r['flags']) for r in regression['results']),
            changed=sum(r['max_geometry_change']>1e-4 for r in regression['results'])),attachments=attachment,
        reviewed=dict(tasks=reviewed['tasks'],formats=dict(Counter(r['format'] for r in reviewed['results'])),changed=sum(r.get('max_geometry_change',0)>1e-4 for r in reviewed['results']),
            winner_flips=sum(r['before_winner']!=r['winner'] for r in reviewed['results']),flagged=sum(bool(r['flags']) for r in reviewed['results']),results=reviewed['results']),models=model)
    (args.output/'dense_reflow_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('engine_sha256','reviewed','attachments')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('regression','attachments','dense','reviewed'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade')
    main(p.parse_args())
