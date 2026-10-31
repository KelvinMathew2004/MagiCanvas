#!/usr/bin/env python3
"""Validate training-target screening with isolated masks and leave-one-out.

Never delete a training row on the basis of a shared all-text difference mask.
This independent native-size diagnostic does not alter engine contrast scores.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/upgrade_exploration/surgical_followup_v1'


def inspect(meta):
    import numpy as np
    from PIL import Image,ImageChops
    from tempfile import TemporaryDirectory
    from scripts.evaluate_qwen_adapter import read_sources
    from scripts.dataset_renderer import render_design_to_png
    from engine.nested_structure import unpainted_ids
    from engine.raster_metrics import glyph_statistics
    from engine.resize import _rgba
    doc=read_sources()(meta['split'],meta['target_id']);inactive=unpainted_ids(doc);size=max(doc['canvas'].values())
    def render(design,path):
        d=deepcopy(design);d['elements']=[e for e in d['elements'] if min(e['width'],e['height'])>0]
        render_design_to_png(d,ROOT/('data/crello_'+meta['split']),path,ROOT/'data/crello_fonts.pickle',max_size=size,adjust_tight_text_widths=False)
        with Image.open(path) as im:return im.convert('RGB').copy()
    rows=[]
    with TemporaryDirectory() as folder:
        path=Path(folder)/'paint.png';full=render(doc,path)
        under=render(dict(doc,elements=[e for e in doc['elements'] if e['type']!='text']),path);back=np.asarray(under)
        black=dict(id='__mask_background',type='shape',shape_type='rectangle',x=0.,y=0.,width=doc['canvas']['width'],height=doc['canvas']['height'],fill_color='#000000',z_index=-1000,rotation=0,opacity=1.)
        for e in doc['elements']:
            if e['type']!='text' or e['id'] in inactive:continue
            color=_rgba(e.get('color',''))
            if not color:rows.append(dict(id=e['id'],unmeasured='color unavailable'));continue
            glyph=deepcopy(e);glyph.update(color='#ffffff',opacity=1.,z_index=1)
            for run in glyph.get('style_runs',[]):
                if 'color' in run:run['color']='#ffffff'
            isolated=render(dict(doc,elements=[black,glyph]),path)
            mask=np.asarray(isolated.convert('L'));intended=int((mask>12).sum())
            without=render(dict(doc,elements=[item for item in doc['elements'] if item['id']!=e['id']]),path)
            contribution=np.asarray(ImageChops.difference(full,without).convert('L'))
            pixels=int(((contribution>12)&(mask>12)).sum())
            stat=glyph_statistics(back,mask,color,e.get('opacity',1)) if intended else None
            rows.append(dict(id=e['id'],rotation=e.get('rotation',0),intended_mask_pixels=intended,
                visible_contribution_pixels=pixels,visible_fraction=pixels/max(1,intended),
                contrast_lower_quintile=stat['lower_quintile'] if stat else None,
                suspect_unpainted=intended==0,suspect_no_visible_contribution=intended>0 and pixels==0))
    return dict(**meta,method='Native isolated white glyph mask plus per-element leave-one-out render; contrast uses existing engine gamma convention, not a WCAG certification.',text=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    from concurrent.futures import ProcessPoolExecutor,as_completed
    audit=json.loads((OUT/'target_audit.json').read_text());folder=OUT/'target_owned_masks';folder.mkdir(exist_ok=True)
    records=[];tasks=[]
    for r in audit['native_sample']:
        f=folder/f'{r["split"]}_{r["target_id"]}.json'
        if f.exists():records.append(json.loads(f.read_text()))
        else:tasks.append(r)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(inspect,r) for r in tasks]):
            r=future.result();(folder/f'{r["split"]}_{r["target_id"]}.json').write_text(json.dumps(r,indent=2)+'\n');records.append(r);print('owned masks',len(records),len(audit['native_sample']),flush=True)
    text=[e for r in records for e in r['text'] if 'unmeasured' not in e]
    summary=dict(targets=len(records),text_elements=len(text),zero_intended_masks=sum(e['suspect_unpainted'] for e in text),
        zero_visible_contribution=sum(e['suspect_no_visible_contribution'] for e in text),
        contrast_below_115=sum(e['contrast_lower_quintile'] is not None and e['contrast_lower_quintile']<1.15 for e in text),
        rotated_measured=sum(abs(e['rotation'])>1 for e in text),
        scope='Same 32 preselected targets, validation diagnostic only; visible antialiased/color contribution is not a binary human readability judgment. No dataset deletions.')
    (OUT/'target_owned_masks.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)


if __name__=='__main__':main()
