"""Source-inferred visual contracts shared by repair, ranking and diagnostics.

These contracts have no template IDs or target reference coordinates. They
separate compact text controls, complete foreground cards, primary reading
axes and an edge photograph's visible subject from generic layer overlap.
"""
from copy import deepcopy
from functools import lru_cache
from math import ceil
import json


@lru_cache(maxsize=512)
def _image_regions(path):
    """Return an opaque light control core and a border-isolated photo subject."""
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            im=im.convert('RGBA');im.thumbnail((768,768));p=np.asarray(im)
        h,w=p.shape[:2];alpha=p[:,:,3]>210
        light=(p[:,:,:3].min(axis=2)>232)&alpha
        core=(0.,0.,1.,1.)
        def longest(mask):
            best=(0,0);start=None
            for i,v in enumerate(list(mask)+[False]):
                if v and start is None:start=i
                if not v and start is not None:
                    if i-start>best[1]-best[0]:best=(start,i)
                    start=None
            return best
        # A ribbon's tails are not the label's body. Find the common opaque
        # light rectangle through the middle, rather than the whole alpha bbox.
        if light.mean()>.25:
            y0,y1=longest(light[:,int(.3*w):max(int(.3*w)+1,int(.7*w))].mean(axis=1)>.94)
            if y1-y0>=.25*h:
                x0,x1=longest(light[y0:y1].mean(axis=0)>.94)
                if x1-x0>=.3*w:core=(x0/w,y0/h,x1/w,y1/h)
        border=np.concatenate((p[0,:,:3],p[-1,:,:3],p[:,0,:3],p[:,-1,:3]))
        subject=None
        # Only isolate objects against a genuinely plain light field. A normal
        # photograph, gradient or intentionally full-bleed texture fails this.
        if (border.min(axis=1)>238).mean()>.94:
            mask=(p[:,:,:3].min(axis=2)<220)&(p[:,:,3]>100)
            yy,xx=np.nonzero(mask)
            if len(xx) and .025<mask.mean()<.6:
                bbox=(float(xx.min()/w),float(yy.min()/h),float((xx.max()+1)/w),float((yy.max()+1)/h))
                if bbox[2]-bbox[0]<.85:subject=bbox
        return core,subject
    except (ImportError,OSError,ValueError):return (0.,0.,1.,1.),None


def control_body(element):
    if element.get('type')=='image' and element.get('asset_id'):
        from engine.resize import _asset_root
        core,_=_image_regions(str(_asset_root(element)/element['asset_id']))
    else:core=(0.,0.,1.,1.)
    x,y,w,h=(float(element[k]) for k in ('x','y','width','height'))
    return (x+core[0]*w,y+core[1]*h,x+core[2]*w,y+core[3]*h)


@lru_cache(maxsize=512)
def _controls_cached(payload):
    from engine.resize import _text_visible_bounds,_overlap,_rotated_bounds
    source=json.loads(payload);sw,sh=source['canvas']['width'],source['canvas']['height'];items=source['elements'];pairs=[]
    for panel in items:
        if (panel.get('type') not in ('shape','image') or float(panel.get('opacity',1.))<.9
                or panel['width']*panel['height']>.17*sw*sh
                or panel['width']/max(1.,panel['height'])<1.6
                or abs(float(panel.get('rotation',0.)))>.5):continue
        pb=_rotated_bounds(panel);labels=[]
        for text in items:
            if text.get('type')!='text' or not text.get('content','').strip() or len(text['content'])>80:continue
            if text.get('z_index',0)<panel.get('z_index',0):continue
            if panel.get('group_id') and text.get('group_id')!=panel['group_id']:continue
            tb=_text_visible_bounds(text);area=max(1.,(tb[2]-tb[0])*(tb[3]-tb[1]))
            if _overlap(pb,tb)/area>=.96:labels.append(text)
        if len(labels)==1:pairs.append((panel['id'],labels[0]['id']))
    # A small nested asset should not claim a label already owned by an
    # explicit group. Otherwise choose the tightest actual backing.
    chosen={};by={e['id']:e for e in items}
    for pid,tid in pairs:
        if tid not in chosen or by[pid]['width']*by[pid]['height']<by[chosen[tid]]['width']*by[chosen[tid]]['height']:chosen[tid]=pid
    return tuple((pid,tid) for tid,pid in chosen.items())


def controls(source):
    return _controls_cached(json.dumps(source,sort_keys=True,separators=(',',':')))


def center_control_labels(elements,source,width,height):
    """Center painted labels on the actual control body, including ribbon cores."""
    from engine.resize import _render_text_ink
    out={e['id']:e for e in elements}
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        panel,label=out[pid],out[tid];body=control_body(panel);ink=_render_text_ink(label)
        if ink is None:continue
        label['x']=(body[0]+body[2]-ink[0]-ink[2])/2
        label['y']=(body[1]+body[3]-ink[1]-ink[3])/2
        label['x']=max(0.,min(width-label['width'],label['x']))
        label['y']=max(0.,min(height-label['height'],label['y']))


def _grow_controls(elements,source,width,height):
    from engine.resize import _set_font_size,_render_text_ink,_text_info
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']}
    reference=min(width/source['canvas']['width'],height/source['canvas']['height']);margin=.04*min(width,height)
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        panel,label=out[pid],out[tid];original=src[tid]
        floor=max(original['font_size']*reference*.85,min(14.,min(width,height)*.06))
        desired=max(label['font_size'],floor)
        probe=deepcopy(label);_set_font_size(probe,desired)
        natural=_text_info(probe,1e9)[1]
        probe.update(width=natural+4.,height=max(probe['height'],desired*2.5))
        ink=_render_text_ink(probe)
        if ink is None:continue
        padding=max(3.,desired*.55)
        old_body=control_body(panel);fx=(old_body[2]-old_body[0])/max(1.,panel['width']);fy=(old_body[3]-old_body[1])/max(1.,panel['height'])
        wanted_w=max(panel['width'],(ink[2]-ink[0]+2*padding)/max(.2,fx))
        wanted_h=max(panel['height'],(ink[3]-ink[1]+max(4.,desired*.45))/max(.2,fy))
        if panel['type']=='image':
            aspect=src[pid]['width']/src[pid]['height'];wanted_w=max(wanted_w,wanted_h*aspect);wanted_h=wanted_w/aspect
        if wanted_w>width-2*margin or wanted_h>height*.3:continue
        cx,cy=panel['x']+panel['width']/2,panel['y']+panel['height']/2
        panel.update(width=wanted_w,height=wanted_h,x=min(width-margin-wanted_w,max(margin,cx-wanted_w/2)),
                     y=min(height-margin-wanted_h,max(margin,cy-wanted_h/2)))
        body=control_body(panel)
        _set_font_size(label,desired)
        label['width']=max(natural+3.,min(label['width'],body[2]-body[0]-2*padding))
        label['height']=max(ceil(ink[3])+2.,min(label['height'],wanted_h))
    center_control_labels(elements,source,width,height)


def edge_scene(source,width,height):
    """Reserve the source side of an object-on-plain-field edge photograph."""
    from engine.resize import _edge_backdrop,_asset_root
    for e in source['elements']:
        edge=_edge_backdrop(e,source)
        if edge not in ('left','right') or not e.get('asset_id'):continue
        _,subject=_image_regions(str(_asset_root(e)/e['asset_id']))
        if subject is None:continue
        margin=.04*min(width,height);gap=.025*min(width,height);slot=.52*width
        if edge=='left':copy_left=width-margin-slot;limit=copy_left-gap
        else:copy_left=margin;limit=width-(copy_left+slot)-gap
        fraction=subject[2] if edge=='left' else 1-subject[0]
        aspect=e['width']/e['height']
        w=min(width*.68,limit/max(.1,fraction),height*.86*aspect);h=w/aspect
        photo=deepcopy(e);photo.update(rotation=0.,width=w,height=h,x=0. if edge=='left' else width-w,y=(height-h)/2)
        box=(photo['x']+subject[0]*w,photo['y']+subject[1]*h,photo['x']+subject[2]*w,photo['y']+subject[3]*h)
        return dict(photo=photo,subject=box,copy=(copy_left,margin,copy_left+slot,height-margin),edge=edge)
    return None


def _scene_reflow(elements,source,width,height):
    from engine.resize import _raw_bounds,_set_font_size
    scene=edge_scene(source,width,height)
    if scene is None:return
    source_by={e['id']:e for e in source['elements']};members=[e for e in elements if e['id']!=scene['photo']['id']]
    if not members:return
    original=[source_by[e['id']] for e in members];sb=_raw_bounds(original);slot=scene['copy']
    factor=min((slot[2]-slot[0])/max(1.,sb[2]-sb[0]),(slot[3]-slot[1])/max(1.,sb[3]-sb[1]))
    top=(height-(sb[3]-sb[1])*factor)/2
    for e in members:
        src=source_by[e['id']]
        e.update(x=slot[0]+(src['x']-sb[0])*factor,y=top+(src['y']-sb[1])*factor,
                 width=src['width']*factor,height=src['height']*factor,rotation=src.get('rotation',0.))
        if e['type']=='text':_set_font_size(e,src['font_size']*factor)


def _card_gutters(elements,source,units,width,height):
    from engine.qwen_refine import build_container_tree
    from engine.resize import _raw_bounds,_set_font_size
    out={e['id']:e for e in elements};sw,sh=source['canvas']['width'],source['canvas']['height'];tree=build_container_tree(source,units)
    for node in tree['children']:
        ids=[i for i in node['element_ids'] if i in out]
        if len(ids)<2:continue
        old=[tree['source_elements'][i] for i in ids];sb=_raw_bounds(old);area=(sb[2]-sb[0])*(sb[3]-sb[1])
        if not any(e['type']!='text' and e['width']*e['height']>=.65*area for e in old):continue
        members=[out[i] for i in ids];box=_raw_bounds(members);g=.04*min(width,height)
        left=g if sb[0]>.02*sw else 0.;right=width-g if sb[2]<.98*sw else width
        top=g if sb[1]>.02*sh else 0.;bottom=height-g if sb[3]<.98*sh else height
        factor=min(1.,(right-left)/max(1.,box[2]-box[0]),(bottom-top)/max(1.,box[3]-box[1]))
        nw,nh=(box[2]-box[0])*factor,(box[3]-box[1])*factor
        x=min(right-nw,max(left,(box[0]+box[2]-nw)/2));y=min(bottom-nh,max(top,(box[1]+box[3]-nh)/2))
        for e in members:
            e.update(x=x+(e['x']-box[0])*factor,y=y+(e['y']-box[1])*factor,width=e['width']*factor,height=e['height']*factor)
            if e['type']=='text':_set_font_size(e,e['font_size']*factor)


def _primary_axis(elements,source,units,width,height,apply=False):
    from engine.resize import _bounds,_text_visible_bounds
    if width/height>=1.8:return []
    sw=source['canvas']['width'];out={e['id']:e for e in elements};candidates=[]
    for u in units:
        if u.get('semantic') not in ('logo','hero','headline','subheading','body','cta'):continue
        if abs((u['bounds'][0]+u['bounds'][2])/2-sw/2)>.02*sw:continue
        es=[out[e['id']] for e in u['items'] if e['id'] in out]
        if not es:continue
        box=_text_visible_bounds(es[0]) if len(es)==1 and es[0]['type']=='text' else _bounds(es)
        delta=width/2-(box[0]+box[2])/2
        if abs(delta)<=.18*width:candidates.append((es,delta))
    if len(candidates)<2:return []
    issues=[]
    for es,delta in candidates:
        if abs(delta)>.004*width:issues.append(dict(kind='primary_axis',ids=[e['id'] for e in es],amount=abs(delta)))
        if apply:
            for e in es:e['x']+=delta
    return issues


def _clear_controls(elements,source,width,height):
    from engine.resize import _text_visible_bounds,_overlap
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']};gap=max(3.,min(width,height)*.012)
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        panel,label=out[pid],out[tid]
        others=[e for e in elements if e['type']=='text' and e['id']!=tid]
        for text in sorted(others,key=lambda e:e['y']):
            ink=_text_visible_bounds(text);pb=control_body(panel)
            horizontal=min(ink[2],pb[2])-max(ink[0],pb[0])
            if horizontal<=1:continue
            original=src[text['id']];source_panel=src[pid]
            # A primary text block outside a control is never another label.
            if original['y']<source_panel['y']:
                required=ink[3]+gap-panel['y']
                if required<=0:continue
                if panel['y']+panel['height']+required<=height-.03*min(width,height):
                    panel['y']+=required;label['y']+=required
                else:
                    dy=panel['y']-gap-ink[3]
                    if text['y']+dy>=.03*min(width,height):text['y']+=dy
            elif original['y']>=source_panel['y']+source_panel['height']:
                dy=panel['y']+panel['height']+gap-ink[1]
                if dy>0 and text['y']+text['height']+dy<=height:text['y']+=dy


def _brand_order(elements,source,units,width,height,apply=False):
    from engine.resize import _raw_bounds,_set_font_size
    if width/height<2.5:return []
    out={e['id']:e for e in elements}
    logos=[u for u in units if u.get('semantic')=='logo'];heroes=[u for u in units if u.get('semantic')=='hero']
    if len(logos)!=1 or len(heroes)!=1:return []
    a,b=logos[0],heroes[0]
    if a['bounds'][1]>=b['bounds'][1]:return []
    la=[out[e['id']] for e in a['items']];ha=[out[e['id']] for e in b['items']];lb,hb=_raw_bounds(la),_raw_bounds(ha)
    if lb[0]<=hb[0] or min(lb[3],hb[3])<=max(lb[1],hb[1]):return []
    issues=[dict(kind='brand_order',ids=[e['id'] for e in la+ha])]
    if apply:
        ids={e['id'] for e in la+ha};gap=max(3.,.012*height);left=.04*height
        other_left=min((e['x'] for e in elements if e['id'] not in ids and e['type']=='text' and e['x']>min(lb[0],hb[0])),default=width*.55)
        right=max(other_left-gap,max(lb[2],hb[2]));available=right-left-gap
        factor=min(1.,available/max(1.,lb[2]-lb[0]+hb[2]-hb[0]));cursor=left
        for members,box in ((la,lb),(ha,hb)):
            cy=(box[1]+box[3])/2;newh=(box[3]-box[1])*factor
            for e in members:
                e.update(x=cursor+(e['x']-box[0])*factor,y=cy-newh/2+(e['y']-box[1])*factor,width=e['width']*factor,height=e['height']*factor)
                if e['type']=='text':_set_font_size(e,e['font_size']*factor)
            cursor+=(box[2]-box[0])*factor+gap
    return issues


def repair(elements,source,units,width,height):
    result=deepcopy(elements)
    _scene_reflow(result,source,width,height)
    _card_gutters(result,source,units,width,height)
    _primary_axis(result,source,units,width,height,apply=True)
    _grow_controls(result,source,width,height)
    _clear_controls(result,source,width,height)
    _brand_order(result,source,units,width,height,apply=True)
    center_control_labels(result,source,width,height)
    return result


def issues(elements,source,width,height,units=None):
    from engine.resize import _text_visible_bounds,_overlap,_analyze,_raw_bounds
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']};result=[];gap=max(3.,min(width,height)*.012)
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        p,t=out[pid],out[tid];body=control_body(p);ink=_text_visible_bounds(t);size=t['font_size']
        ref=min(width/source['canvas']['width'],height/source['canvas']['height'])
        floor=max(src[tid]['font_size']*ref*.85,min(14.,min(width,height)*.06))
        if size<floor*.94:result.append(dict(kind='control_readability',ids=[pid,tid],amount=floor-size))
        pad=min((ink[0]-body[0]),body[2]-ink[2]);expected=max(2.,size*.35)
        if pad<expected-.5:result.append(dict(kind='control_padding',ids=[pid,tid],amount=expected-pad))
        if abs((ink[1]+ink[3]-body[1]-body[3])/2)>max(1.,.02*p['height']):result.append(dict(kind='control_label_axis',ids=[pid,tid]))
        for e in elements:
            if e['type']!='text' or e['id']==tid:continue
            eb=_text_visible_bounds(e);cross=min(eb[2],body[2])-max(eb[0],body[0])
            if cross<=1:continue
            old=src[e['id']];panel=src[pid]
            if old['y']<panel['y'] and eb[1]<body[3] and eb[3]+gap>body[1]+.5:
                result.append(dict(kind='text_control_clearance',ids=[e['id'],pid],amount=eb[3]+gap-body[1]))
            elif old['y']>=panel['y']+panel['height'] and eb[3]>body[1] and eb[1]<body[3]+gap-.5:
                result.append(dict(kind='text_control_clearance',ids=[e['id'],pid],amount=body[3]+gap-eb[1]))
    units=units if units is not None else _analyze(source)[1]
    result.extend(_primary_axis(elements,source,units,width,height))
    result.extend(_brand_order(elements,source,units,width,height))
    scene=edge_scene(source,width,height)
    if scene:
        for e in elements:
            if e['type']=='text':
                box=_text_visible_bounds(e)
                if _overlap(box,scene['subject'])>1:result.append(dict(kind='background_subject_overlap',ids=[e['id'],scene['photo']['id']]))
    return result
