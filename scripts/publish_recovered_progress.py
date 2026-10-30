#!/usr/bin/env python3
"""Preview matching fresh A/B rows without mislabeling a partial run as final."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import argparse,json,html,sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from scripts.report_recovered_adapter import load
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty,STYLE
    from scripts.refresh_regression_viewer import PACK,documents_differ
    from PIL import Image,ImageDraw
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=PACK/'review/recovered_adapter_progress');a=p.parse_args()
    before=load(a.run/'baseline.jsonl',True);after=load(a.run/'audited-0.6b-recovered.jsonl',True)
    snapshot=sha256((a.run/'manifest.json').read_bytes()).hexdigest()
    common=[k for k in before if k in after and before[k].get('output_design') and after[k].get('output_design')]
    a.output.mkdir(parents=True,exist_ok=True);cards=[];manifest=[];tiles=[]
    for i,key in enumerate(common):
        b,t=before[key],after[key];source=b['source_design'];split=b.get('source_split','train')
        os.environ['MAGICANVAS_DATASET_RENDERING']='0' if b.get('official_path') else '1'
        render=_renderer(source,None if b.get('official_path') else split);paths={}
        for label,doc in [('source',source),('before',b['output_design']),('after',t['output_design'])]:
            digest=sha256(json.dumps(dict(snapshot=snapshot,split=split,official=bool(b.get('official_path')),doc=doc),sort_keys=True).encode()).hexdigest()[:20];image=a.output/f'{digest}.jpg'
            if not image.exists():
                png=image.with_suffix('.png');_preview_nonempty(render,doc,png)
                with Image.open(png) as im:im.convert('RGB').save(image,quality=94,optimize=True)
                png.unlink()
            paths[label]=image.name
        audit=f'{i:03d}_audit.json';(a.output/audit).write_text(json.dumps(dict(baseline=b,candidate=t),ensure_ascii=False)+'\n')
        changed=documents_differ(b['output_design'],t['output_design'])
        manifest.append(dict(index=i,pair_id=key,paths=paths,audit=audit,changed=changed))
        cards.append(f'''<article><h2>{html.escape(key)}</h2><p>Changed geometry: {changed}. New adapter winner: {html.escape(t['winner'])}.</p>
<label>Resized <select onchange="this.closest('article').querySelector('[data-output]').src=this.value"><option value="{paths['after']}">Recovered 0.6B</option><option value="{paths['before']}">Current champion</option></select></label>
<div class="pair"><section><h3>Original</h3><img src="{paths['source']}"></section><section><h3>Resized</h3><img data-output src="{paths['after']}"></section></div>
<details><summary>Audit details</summary><p>Champion flags: {html.escape(str(b['defect_flags']))}; candidate: {html.escape(str(t['defect_flags']))}. <a href="{audit}">Raw/final JSON</a></p></details></article>''')
        if changed:
            tile=Image.new('RGB',(1200,470),'#172531');draw=ImageDraw.Draw(tile);draw.text((10,8),f'{i} {key[:110]}',fill='white')
            for label,x in [('before',0),('after',600)]:
                with Image.open(a.output/paths[label]) as im:
                    im=im.convert('RGB');im.thumbnail((580,405));tile.paste(im,(x+(600-im.width)//2,50+(405-im.height)//2))
                draw.text((x+10,30),'Current champion' if label=='before' else 'Recovered adapter',fill='white')
            tiles.append(tile)
    for page,start in enumerate(range(0,len(tiles),3)):
        items=tiles[start:start+3];sheet=Image.new('RGB',(1200,470*len(items)),'#172531')
        for j,im in enumerate(items):sheet.paste(im,(0,470*j))
        sheet.save(a.output/f'changes_{page:02d}.jpg',quality=94)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    title=f'Fresh adapter A/B — {len(common)}/148 matching tasks'
    page=f'''<!doctype html><meta charset="utf-8"><title>{title}</title><style>{STYLE}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img{{width:100%;max-height:850px;object-fit:contain}}select{{font-size:18px}}@media(max-width:750px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>{title}</h1><p>IN PROGRESS — partial inspection only, no promotion or final gain claim. Recorded snapshot {html.escape(a.run.name)}. Both models enabled. Predictions were generated for this adapter comparison; resumed decoded predictions may be replayed through the recorded engine correction. <a href="../main_engine_qwen_outputs/index.html">Current production</a>.</p>{''.join(cards)}'''
    (a.output/'index.html').write_text(page)
    print(json.dumps(dict(baseline=len(before),candidate=len(after),matching=len(common),changed=sum(r['changed'] for r in manifest),viewer=str(a.output/'index.html'))))


if __name__=='__main__':main()
