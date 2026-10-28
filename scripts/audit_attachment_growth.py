#!/usr/bin/env python3
"""Frozen, source-evidenced audit of caption backing growth; no fresh inference.

Example IDs select review data only. The engine has no knowledge of them.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from hashlib import sha256
import json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    torch.set_num_threads(1)
    from engine import layout_contracts as contracts
    from engine import macro_layout
    candidates=macro_layout.candidates
    from engine.resize import _analyze,_assemble,_unintended_control_overlaps
    from scripts.dataset_renderer import render_design_to_png
    from PIL import Image,ImageDraw,ImageOps
    folder=ROOT/'data/qwen_lora/engine_upgrade/attachment_growth';folder.mkdir(exist_ok=True)
    campaign=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/preferences_campaign'
    results=[];images=[]
    for edition in ('00','01'):
        for aspect,name in [('leaderboard','macro-media-split-0.28-2-1'),('widescreen','macro-band-reflow-2x1-1-center')]:
            saved=json.loads((campaign/f'audits/pref-new-00-{edition}-{aspect}.json').read_text());source=saved['source'];w=saved['task']['target_canvas']['width'];h=saved['task']['target_canvas']['height']
            arms={}
            for arm in ('before','after'):
                # Disable only the new contact-edge anchoring, preserving every
                # other current engine/font/aspect policy in both arms.
                manager=patch.object(contracts,'_attached_growth',lambda panel,original,proposed,*a:proposed) if arm=='before' else patch.object(contracts,'_attached_growth',contracts._attached_growth)
                captured={};original_check=macro_layout.checks
                def capture_check(src,doc,*a,**kw):
                    candidate_name=a[-1] if a and isinstance(a[-1],str) else kw.get('name')
                    captured[candidate_name]=deepcopy(doc)
                    return original_check(src,doc,*a,**kw)
                with manager,patch.object(macro_layout,'checks',capture_check):
                    candidates(source,w,h)
                    doc=captured[name]
                    path=folder/f'{edition}_{aspect}_{arm}.png'
                    warnings=render_design_to_png(doc,ROOT/'data/crello_train',path,ROOT/'data/crello_fonts.pickle',max_size=1600,adjust_tight_text_widths=False)
                    row=dict(control_overlaps=_unintended_control_overlaps(doc['elements'],source),issues=contracts.issues(doc['elements'],source,w,h,_analyze(source)[1]),warnings=warnings,image=str(path),doc=doc)
                    arms[arm]=row
            results.append(dict(source=source['elements'][0]['id'],aspect=aspect,candidate=name,arms=arms));images.append(arms)
    report=dict(scope='Same macro candidate and renderer, only contact-preserving backing growth disabled/enabled. Not a final selector experiment or fresh model inference.',engine_fingerprint={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')},results=results)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    sheet=Image.new('RGB',(1600,4*270),'#17232e');draw=ImageDraw.Draw(sheet)
    for j,arms in enumerate(images):
        for i,arm in enumerate(('before','after')):
            with Image.open(arms[arm]['image']) as image:thumb=ImageOps.contain(image.convert('RGB'),(780,225))
            sheet.paste(thumb,(i*800+(800-thumb.width)//2,j*270+32+(225-thumb.height)//2));draw.text((i*800+10,j*270+8),f'{arm} / {len(arms[arm]["control_overlaps"])} control collisions',fill='white')
    sheet.save(folder/'contact.jpg');print(json.dumps([{k:r[k] for k in ('source','aspect','candidate')}|{a:dict(collisions=len(r['arms'][a]['control_overlaps']),issues=r['arms'][a]['issues']) for a in ('before','after')} for r in results],indent=2))


if __name__=='__main__':main()
