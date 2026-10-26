#!/usr/bin/env python3
"""Render every changed result against the frozen prior and disclose unchanged hashes."""
from pathlib import Path
from hashlib import sha256
import json,html,sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 from PIL import Image,ImageOps,ImageDraw
 from scripts.dataset_renderer import render_design_to_png as dataset
 from renderer.converter.convert import render_design_to_png as official
 changes=[];unchanged=[];allrows=[]
 def equivalent(a,b):
  if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):return abs(a-b)<=1e-8
  if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
  if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
  return a==b
 for cohort in ('novel_designs','novel_validation','heldout','official','crello','crello_1p7b','reverse_level3'):
  path=a.run/cohort/'report.json'
  if not path.exists():continue
  current=json.loads(path.read_text())['results'];before={r['pair_id']:r for r in json.loads((a.prior/cohort/'report.json').read_text())['results']}
  for r in current:
   old=before[r['pair_id']]
   def digest(doc):return sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   prior_hash=digest(old['output_design']);new_hash=digest(r['output_design'])
   item=dict(pair_id=r['pair_id'],cohort=cohort,prior_winner=old['winner'],winner=r['winner'],prior_flags=old['defect_flags'],flags=r['defect_flags'],prior_hash=prior_hash,output_hash=new_hash,numerically_equivalent=equivalent(old['output_design'],r['output_design']),macro=r.get('macro_reflow',{}))
   allrows.append(item)
   if prior_hash==new_hash:unchanged.append(item);continue
   index=len(changes)+1;pixels=[];sheet=Image.new('RGB',(1500,700),'#182532');draw=ImageDraw.Draw(sheet)
   for j,(label,doc) in enumerate([('Source',r['source_design']),('Before / same model coordinates',old['output_design']),('Current engine',r['output_design'])]):
    png=a.output/f'{index:03d}_{j}.png'
    if r.get('official_path'):official(doc,ROOT/'brand_kit/assets',png,max_size=1100)
    else:dataset(doc,ROOT/f'data/crello_{r["source_split"]}',png,ROOT/'data/crello_fonts.pickle',max_size=1100,adjust_tight_text_widths=False)
    with Image.open(png) as im:
     pixels.append(sha256(im.convert('RGB').tobytes()).hexdigest())
     tile=ImageOps.contain(im.convert('RGB'),(480,640));sheet.paste(tile,(j*500+(500-tile.width)//2,35+(640-tile.height)//2))
    png.unlink();draw.text((j*500+10,10),label,fill='white')
   item['render_pixels_identical']=pixels[1]==pixels[2]
   item['render_pixel_scope']='Matched review render; maximum dimension 1100px'
   item['image']=f'{index:03d}_comparison.jpg';sheet.save(a.output/item['image'],quality=89,optimize=True)
   item['json']=f'{index:03d}_output.json';(a.output/item['json']).write_text(json.dumps(r['output_design'],indent=2)+'\n');changes.append(item)
 (a.output/'changes.json').write_text(json.dumps(dict(changed=changes,unchanged=unchanged,total=len(allrows),material_changes=sum(not r['numerically_equivalent'] for r in changes),roundoff_only=sum(r['numerically_equivalent'] for r in changes)),indent=2)+'\n')
 cards=[]
 for r in changes:
  cards.append(f'<article><h2>{html.escape(r["cohort"])} · {html.escape(r["pair_id"])}</h2><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a><p>{html.escape(r["prior_winner"])} → {html.escape(r["winner"])}</p><p>Roundoff only: {r['numerically_equivalent']} · Render pixels identical: {r['render_pixels_identical']}</p><p>Flags {html.escape(str(r["prior_flags"]))} → {html.escape(str(r["flags"]))} · <a href="{r["json"]}">Output JSON</a></p><details><summary>Admission and quality metrics</summary><pre>{html.escape(json.dumps(r["macro"],indent=2))}</pre></details></article>')
 (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Structural reflow changes</title><style>body{font:16px system-ui;background:#101820;color:#eee;margin:24px}a{color:#9ed4ff}article{background:#182532;padding:18px;margin:25px 0}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Every changed output · frozen-model comparison</h1><p>'+str(len(changes))+' changed; '+str(len(unchanged))+' retain identical canonical design hashes. All changed outputs are included, not just improvements. Human target previews are never engine inputs.</p>'+''.join(cards))
 print(json.dumps(dict(total=len(allrows),changed=len(changes),identical=len(unchanged),material_changes=sum(not r['numerically_equivalent'] for r in changes),roundoff_only=sum(r['numerically_equivalent'] for r in changes)),indent=2))
if __name__=='__main__':main()
