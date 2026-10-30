#!/usr/bin/env python3
"""Render current-versus-experimental selectors; public cards stay two-panel."""
from pathlib import Path
from PIL import Image,ImageDraw
import argparse,json,html,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty,STYLE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/training_free_scale_v1');a=p.parse_args()
    output=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/training_free_selectors';output.mkdir(exist_ok=True);cards=[];contact=[];rows=[]
    cases=json.loads((a.root/'cases.json').read_text())
    for folder in ('tiled_selector','structural_finalists'):
        for p in sorted((a.root/folder).glob('[0-9]*.json')):
            r=json.loads(p.read_text())
            if not r['changed']:continue
            i=int(p.stem);b=json.loads((a.root/'baseline'/p.name).read_text())['baseline'];source=b['source_design'];doc=r['after']['output_design'] if folder=='tiled_selector' else r['output_design'];render=_renderer(source,None if b.get('official_path') else b['source_split']);stem=f'{folder}_{i:03d}';paths={}
            for label,value in [('source',source),('before',b['output_design']),('after',doc)]:
                png=output/f'{stem}_{label}.png';_preview_nonempty(render,value,png);jpg=png.with_suffix('.jpg')
                with Image.open(png) as im:im.convert('RGB').save(jpg,quality=93,optimize=True)
                png.unlink();paths[label]=jpg.name
            audit=dict(case=cases[i],before=b,experiment=r);(output/f'{stem}_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            cards.append(f'<article><h2>{folder} · {html.escape(r["pair_id"])}</h2><label>Resized image <select onchange="this.closest(\'article\').querySelector(\'[data-result]\').src=this.value"><option value="{paths["after"]}">Experimental selector</option><option value="{paths["before"]}">Current production</option></select></label><div class="panels"><figure><figcaption>Original</figcaption><img src="{paths["source"]}" loading="lazy"></figure><figure><figcaption>Resized</figcaption><img data-result src="{paths["after"]}" loading="lazy"></figure></div><p>Experimental, not promoted. Winner {html.escape(r["winner"])}</p><a href="{stem}_audit.json">Full audit</a></article>')
            contact.append((folder,i,r,paths));rows.append(dict(arm=folder,index=i,pair_id=r['pair_id'],paths=paths,audit=f'{stem}_audit.json'))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Selector scale comparison</title><style>'+STYLE+'</style><header><a href="../training_free_scale/index.html">Algorithm comparison</a><h1>Paired selector changes</h1><p>Qwen enabled with saved predictions. Tiled scoring reruns the complete engine on 83 unique tasks covering all 89 test pairs. The existing structural head compares only clean finalists from the six algorithm arms and the incumbent; no new fitting. These are consumed development cases, not a sealed benchmark. Zero flags does not imply aesthetic perfection.</p></header><main>'+''.join(cards)+'</main>')
    (output/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    for page,start in enumerate(range(0,len(contact),3)):
        batch=contact[start:start+3];sheet=Image.new('RGB',(1400,390*len(batch)),'#182532');draw=ImageDraw.Draw(sheet)
        for n,(folder,i,r,paths) in enumerate(batch):
            draw.text((15,n*390+8),f'{folder} {i:03d} {r["pair_id"]}',fill='white')
            for col,label in enumerate(('before','after')):
                with Image.open(output/paths[label]) as im:
                    im=im.convert('RGB');im.thumbnail((680,340));sheet.paste(im,(700*col+(700-im.width)//2,n*390+40+(340-im.height)//2))
                draw.text((700*col+10,n*390+26),label,fill='white')
        sheet.save(a.root/f'selector_page_{page}.jpg',quality=94)
    print(len(contact),'changes rendered',output/'index.html')


if __name__=='__main__':main()
