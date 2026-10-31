#!/usr/bin/env python3
"""Native-size raw/final glyph contrast, without editing candidate geometry."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def paint_stats(doc,source,split):
    import numpy as np
    from PIL import Image,ImageChops
    from engine.resize import _rgba
    from engine.raster_metrics import glyph_statistics
    from engine.nested_structure import unpainted_ids
    fonts={e.get('font_family','') for e in source['elements'] if e['type']=='text'}
    def render(design,path):
        copy=deepcopy(design);copy['elements']=[e for e in copy['elements'] if min(e['width'],e['height'])>0]
        size=max(copy['canvas'].values())
        if fonts and not fonts<={'Solstice Sans','Solstice Serif',''}:
            from scripts.dataset_renderer import render_design_to_png
            return render_design_to_png(copy,ROOT/('data/crello_'+split),path,ROOT/'data/crello_fonts.pickle',max_size=size,adjust_tight_text_widths=False)
        from renderer.converter.convert import render_design_to_png
        return render_design_to_png(copy,ROOT/'brand_kit/assets',path,fonts_dir=ROOT/'brand_kit/fonts',max_size=size)
    with TemporaryDirectory() as folder:
        full,under=Path(folder)/'full.png',Path(folder)/'under.png'
        warnings=render(doc,full)
        backing=deepcopy(doc);backing['elements']=[e for e in backing['elements'] if e['type']!='text'];render(backing,under)
        with Image.open(full) as a,Image.open(under) as b:
            a,b=a.convert('RGB'),b.convert('RGB');pixels=np.asarray(b);mask=np.asarray(ImageChops.difference(a,b).convert('L'))
    result={};excluded=unpainted_ids(source);skipped=[]
    for e in doc['elements']:
        if e['type']!='text' or e['id'] in excluded or not e.get('content','').strip():continue
        color=_rgba(e.get('color',''))
        if not color or abs(e.get('rotation',0))>1:skipped.append(e['id']);continue
        x,y=max(0,round(e['x'])),max(0,round(e['y']));r,b=min(pixels.shape[1],round(e['x']+e['width'])),min(pixels.shape[0],round(e['y']+e['height']))
        stat=glyph_statistics(pixels[y:b,x:r],mask[y:b,x:r],color,e.get('opacity',1)) if r>x and b>y else None
        result[e['id']]=dict(contrast=stat['lower_quintile'] if stat else None,
            ink_pixels=stat['ink_pixels'] if stat else 0,
            threshold=3. if e['font_size']>=24 or e.get('font_weight')=='bold' and e['font_size']>=18 else 4.5)
    return dict(text=result,unmeasured_rotated_or_color=skipped,renderer_warnings=warnings)


def against_source(stats,original):
    low=[];regressed=[];invisible=[]
    for eid,s in stats['text'].items():
        reference=original['text'].get(eid,{});base=reference.get('contrast')
        threshold=min(s['threshold'],max(1.,base*.9)) if base is not None else s['threshold']
        value=s['contrast']
        if value is None or value<1.15:invisible.append(eid)
        if value is None or value<threshold*.97:
            low.append(eid)
            if base is not None and base>=reference['threshold']*.97:regressed.append(eid)
    return dict(low_contrast=low,source_floor_regressions=regressed,invisible_or_unpainted=invisible,**stats)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();raw={r['pair_id']:r for r in map(json.loads,(a.root/(a.arm+'.jsonl')).read_text().splitlines())}
    folder=a.root/'replay'/a.arm;output=folder/'native_rasters.json';records=[];cache={}
    from experiments.layout_upgrade.contracts import raw_document
    for file in sorted(folder.glob('[0-9]*.json')):
        row=json.loads(file.read_text())
        if not row.get('output_design'):continue
        r=raw[row['pair_id']];source=r['source'];split=r['source_split']
        key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
        try:
            if key not in cache:cache[key]=paint_stats(source,source,split)
            before=paint_stats(raw_document(source,r['target'],r.get('qwen_model_elements',[])),source,split)
            after=paint_stats(row['output_design'],source,split)
            record=dict(pair_id=row['pair_id'],raw=against_source(before,cache[key]),selected=against_source(after,cache[key]))
        except Exception as e:record=dict(pair_id=row['pair_id'],error=f'{type(e).__name__}: {e}')
        records.append(record);print(a.arm,len(records),'raster error',record.get('error'),flush=True)
    report=dict(arm=a.arm,native_resolution=True,source_supported_floor=True,production_changed=False,records=records,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_sha256=sha256((a.root/(a.arm+'.jsonl')).read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
