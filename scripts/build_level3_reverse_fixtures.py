#!/usr/bin/env python3
"""Author synthetic Level-3-complexity wide/tall sources using official assets.

These are test fixtures, not official tasks, resized engine outputs, or training
pairs. Every original Level 3 ID, string, asset, group and adversarial missing
asset is retained. Geometry is independently authored for each source regime.
"""
from pathlib import Path
from copy import deepcopy
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from engine.resize import _calibrate_text_boxes,_center_control_labels
    original=json.loads((ROOT/'input/chameleon_sample_level3_advanced.json').read_text())
    destination=ROOT/'tests/fixtures/level3_reverse';destination.mkdir(parents=True,exist_ok=True)
    for name,w,h in [('leaderboard_copy_lead',1456,180),('leaderboard_product_lead',1456,180),('skyscraper_320',320,1200),('skyscraper_240',240,900)]:
        source=deepcopy(original['source']);source['canvas']=dict(width=w,height=h);by={e['id']:e for e in source['elements']}
        def place(eid,x,y,width,height,font=None,align=None):
            e=by[eid];e.update(x=float(x),y=float(y),width=float(width),height=float(height))
            if font is not None:e['font_size']=float(font)
            if align is not None:e['text_align']=align
        place('bg',0,0,w,h)
        if w>h:
            product_first=name.endswith('product_lead');copy_x=390 if product_first else 190
            place('logo',18,48,135,50);place('product',165 if product_first else 1125,7,150,150)
            place('headline',copy_x,13,620,78,21,'left')
            place('subhead',copy_x,94,620,43,11,'left')
            for i,(x,bw) in enumerate([(copy_x,172),(copy_x+184,210),(copy_x+406,166)],1):
                place(f'badge{i}_bg',x,145,bw,24);place(f'badge{i}_label',x,148,bw,20,9)
            action_x=1040 if product_first else 840
            place('cta_bg',action_x,66,260,42);place('cta_label',action_x,76,260,24,15)
            place('ribbon_bg',1280,19,155,29);place('ribbon_label',1280,24,155,20,10)
            place('holiday_watermark',1320,86,110,45)
            place('legal_text',835,154,605,15,7,'center')
        else:
            scale=w/320
            def tall(eid,x,y,bw,bh,font=None):place(eid,x*scale,y*scale,bw*scale,bh*scale,font*scale if font else None)
            tall('logo',80,28,160,50);tall('product',60,112,200,200)
            tall('headline',22,350,276,230,23);tall('subhead',26,600,268,135,13)
            for i,(y,bw) in enumerate([(760,245),(810,270),(860,225)],1):
                tall(f'badge{i}_bg',(320-bw)/2,y,bw,36);tall(f'badge{i}_label',(320-bw)/2,y+7,bw,24,11)
            tall('cta_bg',30,930,260,56);tall('cta_label',30,946,260,30,15)
            tall('ribbon_bg',156,88,145,28);tall('ribbon_label',156,94,145,20,10)
            tall('holiday_watermark',208,1030,85,40);tall('legal_text',22,1110,276,65,9)
        source['elements'],_=_calibrate_text_boxes(source['elements'],source,w,h)
        _center_control_labels(source['elements'],source,w,h)
        fixture=dict(task_id='synthetic_'+name,difficulty='synthetic_level3_stress',
                     notes=__doc__,source=source,target_canvas=dict(width=1080,height=1080))
        (destination/(name+'.json')).write_text(json.dumps(fixture,indent=2)+'\n')
    print(destination)
if __name__=='__main__':main()
