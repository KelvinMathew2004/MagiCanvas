#!/usr/bin/env python3
"""Keep the reported outputs directly comparable with current renders."""
from pathlib import Path
import json,html,shutil,argparse
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25'
MAIN=PACK/'review/main_engine_qwen_outputs';DEST=PACK/'review/visual_contract_fixes'
SELECT=[(0,'Title / image-button clearance'),(5,'Product / copy separation'),(11,'Ribbon body centering'),(12,'Readable padded button'),(21,'Inset foreground card'),(32,'Logo before hero'),(33,'Copy / CTA clearance'),(36,'Shared primary center axis'),(31,'Canvas breathing room'),(41,'Rotated SUNSET SALE label'),(16,'Source bottom-edge attachment')]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-before',action='store_true');a=p.parse_args()
    DEST.mkdir(exist_ok=True);items=json.loads((MAIN/'manifest.json').read_text())
    if a.capture_before:
        for n,(i,_) in enumerate(SELECT,1):
            target=DEST/f'{n:02d}_before.jpg'
            if not target.exists():shutil.copy2(MAIN/items[i]['image'],target)
        return
    cards=[]
    for n,(i,label) in enumerate(SELECT,1):
        item=items[i];cards.append(f'<article><h2>{n}. {html.escape(label)}</h2><div><figure><figcaption>Reported output</figcaption><img src="{n:02d}_before.jpg"></figure><figure><figcaption>Current output</figcaption><img src="../main_engine_qwen_outputs/{item["image"]}"></figure></div><p>{html.escape(item["winner"])} · Flags: {html.escape(str(item["defect_flags"]))} · <a href="../main_engine_qwen_outputs/{item["json"]}">Current JSON</a></p></article>')
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reported visual-contract repairs</title><style>body{font:17px system-ui;background:#101820;color:#eee;margin:25px}a{color:#9ed4ff}article{background:#182532;padding:20px;margin:25px 0;border-radius:12px}article div{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%;height:600px;object-fit:contain}p{overflow-wrap:anywhere}@media(max-width:800px){article div{grid-template-columns:1fr}}</style><h1>Reported defects · before / after</h1><p>Same decoded model predictions; current source-inferred repair and selector. <a href="../main_engine_qwen_outputs/index.html">All current outputs</a> · <a href="../../docs/SOURCE_ATTACHMENT_REVIEW.md">Implementation and limitations</a></p>'+''.join(cards))
if __name__=='__main__':main()
