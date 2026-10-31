"""Isolated no-training alternatives. No automatic production activation.

Candidates move complete source-owned blocks. Geometry references are never
consulted during generation or selection. Every alternative is rendered and
uses the production checks, prominence guard and replacement margin.
"""
from copy import deepcopy
from math import sqrt,hypot
from pathlib import Path
from hashlib import sha256
import json,os,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'data/upgrade_exploration/deps'))


def source_blocks(source,elements):
    from engine.macro_layout import composites
    by={e['id']:e for e in elements};spec=composites(source,tight=True)
    return [[by[e['id']] for e in b if e['id'] in by] for b in spec['blocks']]


def translated(elements,changes):
    result=deepcopy(elements)
    for e in result:
        dx,dy=changes.get(e['id'],(0.,0.));e['x']+=dx;e['y']+=dy
    return result


def micro_polish(elements,source,width,height):
    """Bounded continuous rigid-block translations, no leaf/font distortion."""
    from scipy.optimize import minimize
    from engine.macro_layout import bounds,composites,center_evidence
    blocks=source_blocks(source,elements)
    if not 2<=len(blocks)<=16:return []
    by={e['id']:e for e in source['elements']};boxes=np.array([bounds(b) for b in blocks],dtype=float)
    oldboxes=np.array([bounds([by[e['id']] for e in b]) for b in blocks],dtype=float)
    scale=np.array([width,height,width,height]);b=boxes/scale
    axes=[];pairs=[]
    def overlap(a,b):return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    for i in range(len(b)):
        for j in range(i):
            pairs.append((i,j,overlap(b[i],b[j])))
            for axis,dim in ((0,width),(1,height)):
                for fraction in (0.,.5,1.):
                    p=oldboxes[i,axis]+fraction*(oldboxes[i,axis+2]-oldboxes[i,axis]);q=oldboxes[j,axis]+fraction*(oldboxes[j,axis+2]-oldboxes[j,axis])
                    x=boxes[i,axis]+fraction*(boxes[i,axis+2]-boxes[i,axis]);y=boxes[j,axis]+fraction*(boxes[j,axis+2]-boxes[j,axis])
                    if abs(p-q)<=4 and abs(x-y)<=.03*dim:axes.append((i,j,axis,fraction))
    excluded={e['id'] for e in source['elements']} - {e['id'] for block in blocks for e in block}
    sb=bounds([e for e in source['elements'] if e['id'] not in excluded])
    center_axes=[axis for axis,d in ((0,source['canvas']['width']),(1,source['canvas']['height']))
        if center_evidence(source,excluded) and abs((sb[axis]+sb[axis+2])/2-d/2)<.08*d
        and abs((b[:,axis].min()+b[:,axis+2].max())/2-.5)<.15]
    limits=[]
    for box in b:
        for axis in (0,1):
            low=max(-.08,-box[axis]);high=min(.08,1.-box[axis+2])
            if low>high:return []
            limits.append((low,high))
    def objective(v):
        offsets=v.reshape(-1,2);moved=b+np.tile(offsets,(1,2));cost=float((offsets**2).sum())*.5
        for i,j,axis,f in axes:
            a=moved[i,axis]+f*(moved[i,axis+2]-moved[i,axis]);c=moved[j,axis]+f*(moved[j,axis+2]-moved[j,axis]);cost+=40*(a-c)**2
        for i,j,allowed in pairs:cost+=1e4*max(0.,overlap(moved[i],moved[j])-allowed)**2
        for axis in center_axes:cost+=10*((moved[:,axis].min()+moved[:,axis+2].max())/2-.5)**2
        return cost
    zero=np.zeros(len(b)*2);fit=minimize(objective,zero,method='L-BFGS-B',bounds=limits,options=dict(maxiter=30,ftol=1e-10,maxls=12))
    if not np.isfinite(fit.x).all() or objective(fit.x)>=objective(zero)-1e-8:return []
    if np.max(np.abs(fit.x))<1e-5:return []
    changes={e['id']:(offset[0]*width,offset[1]*height) for block,offset in zip(blocks,fit.x.reshape(-1,2)) for e in block}
    return [dict(name='experimental-rigid-micro-polish',elements=translated(elements,changes),macro_locked=True,source_attached=True,
                 optimization=dict(success=bool(fit.success),iterations=int(fit.nit),before=objective(zero),after=objective(fit.x)))]


def whitespace(elements,source,width,height):
    """Compare conservative free rectangles with distance-transform proposals."""
    from engine.macro_layout import bounds,edge_strips
    from engine.free_space import free_rectangles,place
    from engine.layout_descriptors import describe
    from scipy.ndimage import distance_transform_edt,maximum_filter
    blocks=source_blocks(source,elements);by={e['id']:e for e in source['elements']}
    desc=describe(source,[[by[e['id']] for e in b] for b in blocks]);result=[]
    for i,(block,d) in enumerate(zip(blocks,desc)):
        if d['role'] not in ('tag','footer','art') or d['area']>.08:continue
        old=bounds([by[e['id']] for e in block]);sw,sh=(source['canvas'][k] for k in ('width','height'))
        # Edge attachments and container-owned children are never floating.
        if old[0]<=2 or old[1]<=2 or old[2]>=sw-2 or old[3]>=sh-2:continue
        b=bounds(block);occupied=[bounds(bs) for j,bs in enumerate(blocks) if j!=i]
        def contact(a,c):return max(0.,min(a[2],c[2])-max(a[0],c[0]))*max(0.,min(a[3],c[3])-max(a[1],c[1]))
        source_contacts=any(contact(old,bounds([by[e['id']] for e in bs]))>0 for j,bs in enumerate(blocks) if j!=i)
        if source_contacts or not any(contact(b,c)>1 for c in occupied):continue
        anchor=((old[0]+old[2])/2/sw*width,(old[1]+old[3])/2/sh*height);bw,bh=b[2]-b[0],b[3]-b[1]
        rects=free_rectangles(width,height,occupied,cells=64,margin=.045*min(width,height),padding=.015*min(width,height))
        positions=[];p=place((bw,bh),rects,anchor,(width,height),scale_cap=1.)
        if p and p['scale']>=1.-1e-6:positions.append(('rectangle',p['x'],p['y']))
        cells=64;mask=np.ones((cells,cells),dtype=bool)
        for l,t,r,v in occupied:
            mask[max(0,int(t/height*cells)):min(cells,int(np.ceil(v/height*cells))),max(0,int(l/width*cells)):min(cells,int(np.ceil(r/width*cells)))]=False
        distance=distance_transform_edt(mask,sampling=(height/cells,width/cells));peaks=np.argwhere((distance==maximum_filter(distance,size=5)) & (distance>0))
        options=[];margin=.045*min(width,height)
        for row,col in peaks:
            x=(col+.5)/cells*width-bw/2;y=(row+.5)/cells*height-bh/2;box=(x,y,x+bw,y+bh)
            if x<margin or y<margin or x+bw>width-margin or y+bh>height-margin:continue
            if any(contact(box,c)>1e-6 for c in occupied):continue
            options.append((hypot(x+bw/2-anchor[0],y+bh/2-anchor[1]),x,y))
        if options:_,x,y=min(options);positions.append(('distance',x,y))
        for method,x,y in positions:
            changes={e['id']:(x-b[0],y-b[1]) for e in block}
            result.append(dict(name=f'experimental-whitespace-{method}-{i}',elements=translated(elements,changes),macro_locked=True,source_attached=True))
        if len(result)>=2:break
    return result[:2]


def calm_zone(elements,source,width,height,render):
    """Bounded whole-copy translations scored on real background color/texture.

    This is a low-level raster proposal, not semantic subject recognition.
    Real glyph-mask contrast and attachments are checked by the engine later.
    """
    from tempfile import TemporaryDirectory
    from PIL import Image
    from engine.resize import _analyze,_rgba
    from engine.macro_layout import bounds
    bg,_=_analyze(source)
    if not any(e['type']=='image' for e in bg):return []
    from engine.resize import _assemble
    background=_assemble(elements,bg,source,width,height);background['elements']=[e for e in background['elements'] if e['type']!='text']
    with TemporaryDirectory() as folder:
        path=Path(folder)/'back.png';render(background,path)
        with Image.open(path) as im:im=im.convert('RGB');im.thumbnail((256,256));pixels=np.asarray(im,dtype=float)/255.
    ih,iw=pixels.shape[:2];luma=pixels@np.array([.2126,.7152,.0722]);blocks=source_blocks(source,elements);options=[]
    for i,block in enumerate(blocks):
        if any(e['type']=='image' for e in block):continue
        texts=[e for e in block if e['type']=='text' and e.get('content','').strip()]
        if not texts:continue
        b=bounds(block)
        if (b[2]-b[0])*(b[3]-b[1])>.5*width*height:continue
        def proxy(dx,dy):
            cost=0.
            for e in texts:
                x=max(0,int((e['x']+dx)/width*iw));r=min(iw,int(np.ceil((e['x']+dx+e['width'])/width*iw)))
                y=max(0,int((e['y']+dy)/height*ih));v=min(ih,int(np.ceil((e['y']+dy+e['height'])/height*ih)))
                patch=luma[y:v,x:r]
                if patch.size==0:return float('inf')
                color=_rgba(e.get('color','#000000'))
                if color is None:return float('inf')
                c=np.array(color[:3],dtype=float)/255.;gray=float(c@np.array([.2126,.7152,.0722]))
                contrast=(np.maximum(gray,patch)+.05)/(np.minimum(gray,patch)+.05)
                cost+=float(1./max(1.,np.percentile(contrast,20)))+float(np.std(patch))
            return cost/len(texts)
        old=proxy(0,0);moves=[]
        for dx in (-.10*width,-.05*width,0.,.05*width,.10*width):
            for dy in (-.10*height,-.05*height,0.,.05*height,.10*height):
                if dx==dy==0:continue
                if b[0]+dx<.03*width or b[2]+dx>.97*width or b[1]+dy<.03*height or b[3]+dy>.97*height:continue
                value=proxy(dx,dy)+.08*hypot(dx/width,dy/height)
                if value<old-.05:moves.append((value,dx,dy))
        if moves:
            value,dx,dy=min(moves);options.append((old-value,i,dx,dy))
    results=[]
    for gain,i,dx,dy in sorted(options,reverse=True)[:2]:
        results.append(dict(name=f'experimental-near-calm-{i}',elements=translated(elements,{e['id']:(dx,dy) for e in blocks[i]}),macro_locked=True,source_attached=True,raster_proposal_gain=gain))
    return results


def retrieved(source,width,height,library):
    from engine.retrieve import candidates
    from engine.macro_layout import composites,_prepared,_ambient_layers,clamp_edge_strips,quality
    prepared=_prepared(source);spec=composites(prepared,tight=True)
    old=os.environ.get('MAGICANVAS_LAYOUT_LIBRARY');os.environ['MAGICANVAS_LAYOUT_LIBRARY']=str(library)
    try:cs=candidates(prepared,spec['blocks'],width,height)
    finally:
        if old is None:os.environ.pop('MAGICANVAS_LAYOUT_LIBRARY',None)
        else:os.environ['MAGICANVAS_LAYOUT_LIBRARY']=old
    for c in cs:
        c['elements']+=_ambient_layers(source,width,height,spec.get('ambient_ids',[]))+[deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
        clamp_edge_strips(c['elements'],source,width,height)
    return sorted(cs,key=lambda c:quality(c['elements'],source,width,height)['cost'])[:2]


def evaluate(case,before,arms,root):
    import importlib
    from time import perf_counter
    from engine.macro_layout import quality,checks,composites,candidates,bounds
    from scripts.build_focus_reflow_viewer import _renderer
    from scripts.layout_defects import assess_engine_layout
    from scripts.evaluate_qwen_adapter import measure
    from engine.qwen_prompt import decode_v2_completion,record_prompt_version
    resize=importlib.import_module('engine.resize');source=before['source_design'];doc=before['output_design'];w,h=(doc['canvas'][k] for k in ('width','height'))
    os.environ['MAGICANVAS_DATASET_RENDERING']='0' if before.get('official_path') else '1'
    render=_renderer(source,None if before.get('official_path') else before['source_split']);bg,units=resize._analyze(source);bgids={e['id'] for e in bg};es=[e for e in doc['elements'] if e['id'] not in bgids]
    q=quality(es,source,w,h);desired=sqrt(w*h/(source['canvas']['width']*source['canvas']['height']))
    oldvisual=dict(used=True,candidates=[dict(before.get('selected_raster_metrics') or {},name=before['winner'])]);oldcheck=checks(source,doc,units,oldvisual,before['winner'])
    result=dict(pair_id=case['pair_id'],key=case['key'],cohort=case['cohort'],baseline_winner=before['winner'],baseline_flags=before['defect_flags'],baseline_quality=q,arms={})
    # Identity requests intentionally preserve source composition.
    identity=all(source['canvas'][k]==doc['canvas'][k] for k in ('width','height'))
    for arm in arms:
        start=perf_counter();entry=dict(generated=0,rendered=0,changed=False,candidates=[],output_design=doc,winner=before['winner'],defect_flags=before['defect_flags'],geometry_mae=before.get('geometry_mae'),quality=q)
        if identity:entry['skip']='identity request';result['arms'][arm]=entry;continue
        if arm=='micro-polish':cs=micro_polish(es,source,w,h)
        elif arm=='whitespace':cs=whitespace(es,source,w,h)
        elif arm=='calm-zone':cs=calm_zone(es,source,w,h,render)
        elif arm=='retrieval':cs=retrieved(source,w,h,root/'expanded_library.json')
        elif arm=='dual-granularity':
            shift=max((w/h)/(source['canvas']['width']/source['canvas']['height']),(source['canvas']['width']/source['canvas']['height'])/(w/h))
            cs=[]
            if shift>=1.8:
                seen=set()
                for mode in (True,False):
                    sig=tuple(sorted(tuple(sorted(e['id'] for e in b)) for b in composites(source,tight=mode)['blocks']))
                    if sig in seen:continue
                    seen.add(sig)
                    pool=candidates(source,w,h,before.get('qwen_model_elements'),tight=mode,extras=False)
                    for c in sorted(pool,key=lambda c:quality(c['elements'],source,w,h)['cost'])[:1]:c['name']='experimental-dual-'+str(mode)+'-'+c['name'];cs.append(c)
        elif arm=='cpsat':
            import sys
            sys.path.insert(0,str(root.parent/'deps'))
            from experiments.layout_upgrade.packing import candidates as packed
            cs=packed(source,w,h,before.get('qwen_model_elements'),seconds=.1,tight=True)
        else:raise ValueError(arm)
        entry['generated']=len(cs);best=None
        for c in cs:
            c['energy']=resize._energy(c['elements'],units,source,w,h,desired)
            candidate_doc=resize._assemble(c['elements'],bg,source,w,h)
            pre=checks(source,candidate_doc,units,{},c['name']);row=dict(name=c['name'],geometry_flags=pre['defect_flags'])
            entry['candidates'].append(row)
            if pre['flagged']:row['rejected']='physical geometry';continue
            winner,visual=resize._visual_rerank([c],bg,units,source,w,h,desired);entry['rendered']+=1
            newdoc=resize._assemble(winner['elements'],winner.get('backgrounds',bg),source,w,h)
            exact=assess_engine_layout(source,newdoc,dict(winner=winner['name'],visual=visual))
            gate=checks(source,newdoc,units,visual,winner['name']);score=quality(winner['elements'],source,w,h)
            row.update(flags=exact['defect_flags'],gate=gate,quality=score,selected_name=winner['name'])
            # Same strict clean-replacement principle as production, without
            # weakening a clipping/contrast/ownership gate to get a win.
            if exact['flagged'] or gate['flagged']:row['rejected']='rendered defects';continue
            if not oldcheck['flagged'] and score['cost']>=q['cost']-max(2.,q['cost']*.15):row['rejected']='insufficient quality margin';continue
            if not oldcheck['flagged'] and score.get('content_union_share',0)<.90*q.get('content_union_share',0):row['rejected']='occupied content shrank';continue
            if arm=='retrieval' and any(q.get(k,0)>0 and score.get(k,0)<.70*q[k] for k in ('headline_font','hero_area_share')):row['rejected']='retrieval prominence loss';continue
            hero_lost=q.get('hero_area_share',0)==0 or score.get('hero_area_share',0)<.90*q['hero_area_share']
            if not oldcheck['flagged'] and hero_lost and score.get('headline_font',0)<.90*q.get('headline_font',0):row['rejected']='combined primary prominence loss';continue
            row['eligible']=True
            if best is None or score['cost']<best[0]:best=(score['cost'],newdoc,winner,exact,score)
        if best:
            _,newdoc,winner,exact,score=best
            entry.update(output_design=newdoc,winner=winner['name'],defect_flags=exact['defect_flags'],quality=score,changed=newdoc!=doc)
            if case['record'].get('messages'):
                gold=json.loads(case['record']['messages'][-1]['content'])
                if gold.get('elements') and 'box' in gold['elements'][0]:gold=decode_v2_completion(gold,source,doc['canvas'],record_prompt_version(case['record']))
                entry['geometry_mae']=measure(newdoc['elements'],gold,doc['canvas'])['geometry_mae']
        entry['evaluation_seconds']=perf_counter()-start;result['arms'][arm]=entry
    return result
