"""Source-inferred visual contracts shared by repair, ranking and diagnostics.

These contracts have no template IDs or target reference coordinates. They
separate compact text controls, complete foreground cards, primary reading
axes and an edge photograph's visible subject from generic layer overlap.
"""
from copy import deepcopy
from functools import lru_cache
from math import ceil,radians,cos,sin
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
                or panel['width']/max(1.,panel['height'])<1.6):continue
        pb=_rotated_bounds(panel);labels=[]
        for text in items:
            if text.get('type')!='text' or not text.get('content','').strip():continue
            if text.get('z_index',0)<panel.get('z_index',0):continue
            if panel.get('group_id') and text.get('group_id')!=panel['group_id']:continue
            tb=_text_visible_bounds(text);area=max(1.,(tb[2]-tb[0])*(tb[3]-tb[1]))
            if _overlap(pb,tb)/area>=.96:labels.append(text)
        if len(labels)==1 and len(labels[0]['content'])<=80:pairs.append((panel['id'],labels[0]['id']))
    # A small nested asset should not claim a label already owned by an
    # explicit group. Otherwise choose the tightest actual backing.
    chosen={};by={e['id']:e for e in items}
    for pid,tid in pairs:
        if tid not in chosen or by[pid]['width']*by[pid]['height']<by[chosen[tid]]['width']*by[chosen[tid]]['height']:chosen[tid]=pid
    return tuple((pid,tid) for tid,pid in chosen.items())


def controls(source):
    return _controls_cached(json.dumps(source,sort_keys=True,separators=(',',':')))


def control_font_floor(source,tid,width,height):
    from engine.source_structure import groups
    original=next(e for e in source['elements'] if e['id']==tid)
    scaled=original['font_size']*min(width/source['canvas']['width'],height/source['canvas']['height'])
    # Date/price annotations on illustrated panels are not action buttons.
    # Their source hierarchy supplies the floor; CTA/badge labels retain the
    # stronger minimum used for small interactive controls.
    interactive=groups(source)['semantics'].get(tid) in ('cta','badge')
    return max(scaled,min(14.,min(width,height)*.06)) if interactive else scaled


def center_control_labels(elements,source,width,height):
    """Center painted labels on the actual control body, including ribbon cores."""
    from engine.resize import _render_text_ink
    out={e['id']:e for e in elements}
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        panel,label=out[pid],out[tid];body=control_body(panel);ink=_render_text_ink(label)
        if ink is None:continue
        angle=radians(float(panel.get('rotation',0.)))
        cx,cy=panel['x']+panel['width']/2,panel['y']+panel['height']/2
        dx,dy=(body[0]+body[2])/2-cx,(body[1]+body[3])/2-cy
        target_x,target_y=cx+dx*cos(angle)-dy*sin(angle),cy+dx*sin(angle)+dy*cos(angle)
        label['rotation']=float(panel.get('rotation',0.))
        dx,dy=(ink[0]+ink[2]-label['width'])/2,(ink[1]+ink[3]-label['height'])/2
        label['x']=target_x-label['width']/2-dx*cos(angle)+dy*sin(angle)
        label['y']=target_y-label['height']/2-dx*sin(angle)-dy*cos(angle)
        label['x']=max(0.,min(width-label['width'],label['x']))
        label['y']=max(0.,min(height-label['height'],label['y']))


def _grow_controls(elements,source,width,height):
    from engine.resize import _set_font_size,_render_text_ink,_text_info
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']}
    reference=min(width/source['canvas']['width'],height/source['canvas']['height']);margin=.04*min(width,height)
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        panel,label=out[pid],out[tid];original=src[tid]
        floor=control_font_floor(source,tid,width,height)
        desired=max(label['font_size'],floor)
        probe=deepcopy(label);_set_font_size(probe,desired)
        natural=_text_info(probe,1e9)[1]
        probe.update(width=natural+4.,height=max(probe['height'],desired*2.5))
        ink=_render_text_ink(probe)
        if ink is None:continue
        source_body=control_body(src[pid]);source_ink=_render_text_ink(original)
        source_ink_world=(original['x']+source_ink[0],original['y']+source_ink[1],original['x']+source_ink[2],original['y']+source_ink[3]) if source_ink else source_body
        authored_pad=max(3.,min(source_ink_world[0]-source_body[0],source_body[2]-source_ink_world[2]))
        padding=max(3.,min(desired*.55,authored_pad*desired/max(1.,original['font_size'])))
        old_body=control_body(panel);fx=(old_body[2]-old_body[0])/max(1.,panel['width']);fy=(old_body[3]-old_body[1])/max(1.,panel['height'])
        wanted_w=max(panel['width'],(ink[2]-ink[0]+2*padding)/max(.2,fx))
        wanted_h=max(panel['height'],(ink[3]-ink[1]+max(4.,desired*.45))/max(.2,fy))
        if panel['type']=='shape':
            wanted_h=max(src[pid]['height']*desired/max(1.,original['font_size']),
                         (ink[3]-ink[1]+max(4.,desired*.45))/max(.2,fy))
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
        if not any(e['type']!='text' and e['width']*e['height']>=.65*area
                   and any(t['type']=='text' and t.get('z_index',0)>e.get('z_index',0)
                           and t['x']>=e['x'] and t['y']>=e['y']
                           and t['x']+t['width']<=e['x']+e['width']+1
                           and t['y']+t['height']<=e['y']+e['height']+1 for t in old) for e in old):continue
        members=[out[i] for i in ids]
        # A real containing card is a lockup, not a bag of leaves. Recover
        # authored internal offsets before applying the common gutter scale.
        backing=max((e for e in old if e['type']!='text'),key=lambda e:e['width']*e['height'])
        target=out[backing['id']]
        scale=min(target['width']/backing['width'],target['height']/backing['height'])
        x0=target['x']+(sb[0]-backing['x'])*scale;y0=target['y']+(sb[1]-backing['y'])*scale
        source_by={e['id']:e for e in old}
        for e in members:
            authored=source_by[e['id']]
            e.update(x=x0+(authored['x']-sb[0])*scale,y=y0+(authored['y']-sb[1])*scale,
                     width=authored['width']*scale,height=authored['height']*scale)
            if e['type']=='text':_set_font_size(e,authored['font_size']*scale)
        box=_raw_bounds(members);g=.04*min(width,height)
        left=g if sb[0]>.02*sw else 0.;right=width-g if sb[2]<.98*sw else width
        top=g if sb[1]>.02*sh else 0.;bottom=height-g if sb[3]<.98*sh else height
        factor=min(1.,(right-left)/max(1.,box[2]-box[0]),(bottom-top)/max(1.,box[3]-box[1]))
        nw,nh=(box[2]-box[0])*factor,(box[3]-box[1])*factor
        x=min(right-nw,max(left,(box[0]+box[2]-nw)/2));y=min(bottom-nh,max(top,(box[1]+box[3]-nh)/2))
        for e in members:
            e.update(x=x+(e['x']-box[0])*factor,y=y+(e['y']-box[1])*factor,width=e['width']*factor,height=e['height']*factor)
            if e['type']=='text':_set_font_size(e,e['font_size']*factor)


def _primary_axis(elements,source,units,width,height,apply=False):
    """Shared canvas axes belong to functional groups, never their suffixes."""
    from engine.resize import _bounds,_text_visible_bounds
    from engine.source_structure import groups
    if width/height>=1.8:return []
    sw=source['canvas']['width'];out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']}
    semantics={e['id']:u.get('semantic') for u in units for e in u['items']};candidates=[]
    def painted(es):
        boxes=[_text_visible_bounds(e) if e['type']=='text' else _bounds([e]) for e in es]
        return (min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))
    for ids in groups(source)['members']:
        if not any(semantics.get(i) in ('logo','hero','headline','subheading','body','cta') for i in ids):continue
        if not all(i in out for i in ids):continue
        before=painted([src[i] for i in ids])
        if abs((before[0]+before[2])/2-sw/2)>.025*sw:continue
        es=[out[i] for i in ids];box=painted(es);delta=width/2-(box[0]+box[2])/2
        if abs(delta)<=.18*width or width/height<.85:candidates.append((es,delta))
    if len(candidates)<2:return []
    issues=[]
    for es,delta in candidates:
        if abs(delta)>.004*width:issues.append(dict(kind='primary_axis',ids=[e['id'] for e in es],amount=abs(delta)))
        if apply and all(e['x']+delta>=0 and e['x']+e['width']+delta<=width for e in es):
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


def _source_edge_attachments(elements,source,width,height):
    """Retain visible foreground edge contact in a comparable aspect regime."""
    from engine.resize import _bounds,_text_visual_coverage
    sw,sh=source['canvas']['width'],source['canvas']['height']
    if max(width/height,sw/sh)/min(width/height,sw/sh)>=1.4:return
    out={e['id']:e for e in elements};originals={e['id']:e for e in source['elements']}
    for original in source['elements']:
        if original['type']!='image' or original['width']*original['height']>.65*sw*sh:continue
        sb=_bounds([original])
        if abs(sb[3]-sh)>max(2.,.004*sh):continue
        e=out.get(original['id'])
        if e is None:continue
        box=_bounds([e]);dy=height-box[3]
        if abs(dy)<.5:continue
        probe=dict(e,y=e['y']+dy)
        if probe['y']<0:continue
        # A source attachment is weaker than readable copy. Skip the anchor
        # when reflow makes its old edge incompatible with the new copy.
        blocked=False
        for t in elements:
            if t['type']!='text':continue
            old=_text_visual_coverage(originals[t['id']],original)
            if _text_visual_coverage(t,probe)>max(old+.2,.25):blocked=True;break
        if not blocked:
            group=e.get('group_id')
            for member in elements:
                if member['id']==e['id'] or (group is not None and member.get('group_id')==group):member['y']+=dy


def _control_visual_clearance(elements,source,width,height):
    """Keep source-disjoint compact controls clear of a brand/hero image."""
    from engine.resize import _bounds,_overlap
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']}
    pairs=controls(source);owners={p for p,_ in pairs};sw,sh=source['canvas']['width'],source['canvas']['height']
    gap=max(3.,.012*min(width,height))
    for pid,tid in pairs:
        if pid not in out or tid not in out:continue
        panel,label=out[pid],out[tid]
        for e in elements:
            old=src[e['id']]
            if e['type']!='image' or e['id'] in owners or old['width']*old['height']>.5*sw*sh:continue
            if _overlap(_bounds([src[pid]]),_bounds([old]))>1:continue
            pb,eb=_bounds([panel]),_bounds([e])
            if _overlap(pb,eb)<=1:continue
            if width<height:
                dy=eb[3]+gap-pb[1]
                if pb[3]+dy<=height-gap:panel['y']+=dy;label['y']+=dy
            else:
                dx=eb[2]+gap-pb[0]
                if pb[2]+dx<=width-gap:panel['x']+=dx;label['x']+=dx


def _control_peer_clearance(elements,source,width,height):
    """Growing readable labels must not make neighboring controls collide."""
    from engine.resize import _overlap
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']}
    pairs=[(out[p],out[t]) for p,t in controls(source) if p in out and t in out]
    pairs.sort(key=lambda pair:(pair[0]['y'],pair[0]['x']))
    gap=max(3.,min(width,height)*.012)
    for i,(a,_) in enumerate(pairs):
        ab=(a['x'],a['y'],a['x']+a['width'],a['y']+a['height'])
        for b,label in pairs[i+1:]:
            sa,sb=src[a['id']],src[b['id']]
            if _overlap((sa['x'],sa['y'],sa['x']+sa['width'],sa['y']+sa['height']),
                        (sb['x'],sb['y'],sb['x']+sb['width'],sb['y']+sb['height']))>1:continue
            bb=(b['x'],b['y'],b['x']+b['width'],b['y']+b['height'])
            if _overlap(ab,bb)<=0:continue
            dx=abs((ab[0]+ab[2]-bb[0]-bb[2])/2)/max(a['width'],b['width'])
            dy=abs((ab[1]+ab[3]-bb[1]-bb[3])/2)/max(a['height'],b['height'])
            if dy>=dx:
                shift=ab[3]+gap-b['y']
                if bb[3]+shift<=height-gap:b['y']+=shift;label['y']+=shift
            else:
                shift=ab[2]+gap-b['x']
                if bb[2]+shift<=width-gap:b['x']+=shift;label['x']+=shift


def _copy_clearance(elements,source,width,height):
    """Axis projection must not turn separate source copy into intersecting ink."""
    from engine.resize import _text_visible_bounds,_overlap
    src={e['id']:e for e in source['elements']}
    labels={tid for _,tid in controls(source)}
    copy=[e for e in elements if e['type']=='text' and e['id'] not in labels]
    copy.sort(key=lambda e:src[e['id']]['y'])
    gap=max(3.,min(width,height)*.012)
    for i,a in enumerate(copy):
        ab=_text_visible_bounds(a);sa=_text_visible_bounds(src[a['id']])
        for b in copy[i+1:]:
            sb=_text_visible_bounds(src[b['id']]);bb=_text_visible_bounds(b)
            if sa[3]>sb[1]+1 or min(ab[2],bb[2])<=max(ab[0],bb[0]):continue
            dy=ab[3]+gap-bb[1]
            if dy>0 and bb[3]+dy<=height-gap:b['y']+=dy


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


def _new_art_collision(text,art,old_text,old_art):
    from engine.resize import _text_visual_coverage_stats
    authored,source_peak,_=_text_visual_coverage_stats(old_text,old_art)
    if authored>.05:return None
    coverage,peak,pixels=_text_visual_coverage_stats(text,art)
    # Source-supported intersections (e.g. lettering over a decorative frame)
    # can shift between lines after wrapping. Use the aggregate regression
    # floor for those; the stricter line check detects newly introduced contact.
    new_line_contact=(source_peak<=.01 and pixels>=max(8,.02*text.get('font_size',12)**2)
                      and peak>max(.01,source_peak+.005))
    if coverage>max(.02,authored+.01) or new_line_contact:
        return dict(kind='text_art_occlusion',ids=[text['id'],art['id']],coverage=coverage,line_coverage=peak,new_line_contact=new_line_contact)
    return None


def art_occlusions(elements,source):
    """New glyph intersections with opaque foreground art, in either paint order."""
    from engine.resize import _analyze,_overlap,_bounds
    background={e['id'] for e in _analyze(source)[0]};src={e['id']:e for e in source['elements']};defects=[]
    for text in elements:
        if text['type']!='text':continue
        for art in elements:
            if art['type'] not in ('image','shape') or art['id'] in background or float(art.get('opacity',1))<.9:continue
            if text.get('group_id') is not None and text.get('group_id')==art.get('group_id'):continue
            old_text,old_art=src[text['id']],src[art['id']]
            if _overlap(_bounds([text]),_bounds([art]))<=0:continue
            collision=_new_art_collision(text,art,old_text,old_art)
            if collision:defects.append(collision)
    return defects


def image_collisions(elements,source):
    from engine.resize import _analyze,_bounds,_overlap
    background={e['id'] for e in _analyze(source)[0]};src={e['id']:e for e in source['elements']};defects=[]
    images=[e for e in elements if e['type']=='image' and e['id'] not in background]
    for i,a in enumerate(images):
        for b in images[:i]:
            if a.get('group_id') is not None and a.get('group_id')==b.get('group_id'):continue
            if _overlap(_bounds([src[a['id']]]),_bounds([src[b['id']]]))>1:continue
            overlap=_overlap(_bounds([a]),_bounds([b]))
            if overlap>1:defects.append(dict(kind='image_on_image',ids=[a['id'],b['id']],area=overlap))
    return defects


def _separate_art(elements,source,width,height,new_contacts_only=False):
    from engine.resize import _bounds,_text_visible_bounds
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']};pad=.025*min(width,height)
    for defect in art_occlusions(elements,source):
        if new_contacts_only and not defect.get('new_line_contact'):continue
        text,art=(out[i] for i in defect['ids'])
        if art.get('group_id') is not None:continue
        tb=_text_visible_bounds(text);ab=_bounds([art]);old=src[art['id']];sw,sh=source['canvas']['width'],source['canvas']['height']
        # Transparent padding is not the subject. Move the painted art only
        # far enough to clear the glyphs, rather than overshooting by its frame.
        probes=[(old['x']/sw*width,old['y']/sh*height),(art['x']+tb[2]+pad-ab[0],art['y']),
                (art['x']+tb[0]-pad-ab[2],art['y']),(art['x'],art['y']+tb[3]+pad-ab[1]),
                (art['x'],art['y']+tb[1]-pad-ab[3])]
        valid=[]
        for x,y in probes:
            candidate=dict(art,x=min(width-art['width']-pad,max(pad,x)),y=min(height-art['height']-pad,max(pad,y)))
            if candidate['x']<0 or candidate['y']<0:continue
            if all(_new_art_collision(t,candidate,src[t['id']],old) is None for t in elements if t['type']=='text'):
                valid.append(candidate)
        if valid:
            chosen=min(valid,key=lambda e:abs(e['x']-art['x'])+abs(e['y']-art['y']));art.update(chosen)


def _pin_corners(elements,source,units,width,height):
    from engine.resize import _callout_block,_fit_raw_block,_raw_bounds,_shift
    out={e['id']:e for e in elements};sw,sh=source['canvas']['width'],source['canvas']['height'];margin=.045*min(width,height)
    for u in units:
        if not u.get('corner_callout') or u['role']!='decoration':continue
        from engine.resize import _set_font_size,_render_text_ink,_text_info
        block=_callout_block(u,width,height,sw,sh)
        panels=[e for e in block if e['type']=='shape'];labels=[e for e in block if e['type']=='text']
        if len(panels)==1 and len(labels)==1:
            p,t=panels[0],labels[0];original=next(e for e in u['items'] if e['type']=='text')
            desired=max(original['font_size']*min(width/sw,height/sh),min(14.,min(width,height)*.06))
            _set_font_size(t,desired);natural=_text_info(t,1e9)[1];t.update(width=natural+4.,height=desired*2.)
            ink=_render_text_ink(t)
            if ink:
                p.update(x=0.,y=0.,width=max(natural+desired*1.4,ink[2]-ink[0]+desired*1.4),height=ink[3]-ink[1]+desired*.8)
                t.update(x=(p['width']-t['width'])/2,y=(p['height']-t['height'])/2)
        block=_fit_raw_block(block,width-2*margin if width/height<=.4 else width*.30,height*.45);b=_raw_bounds(block)
        x=width-margin-b[2]+b[0] if sum(u['bounds'][::2])/2>sw/2 else margin
        for e in _shift(block,x-b[0],margin-b[1]):
            if e['id'] in out:out[e['id']].update(e)


def _footer_caps(elements,units,width):
    from engine.resize import _raw_bounds,_set_font_size
    out={e['id']:e for e in elements}
    for u in units:
        if u['role']!='footer' or not any(e['type']=='image' for e in u['items']):continue
        members=[out[e['id']] for e in u['items'] if e['id'] in out];l,t,r,b=_raw_bounds(members)
        scale=min(1.,.14*width/max(1,r-l))
        for e in members:
            e.update(x=l+(e['x']-l)*scale,y=t+(e['y']-t)*scale,width=e['width']*scale,height=e['height']*scale)
            if e['type']=='text':_set_font_size(e,e['font_size']*scale)


def _spread_story_copy(elements,source,units,width,height):
    from engine.resize import _bounds,_raw_bounds
    if height/width<1.5:return
    heroes=[u for u in units if u.get('semantic')=='hero'];copy=[u for u in units if u['role'] in ('copy','action')]
    if len(heroes)!=1 or not 1<=len(copy)<=4:return
    out={e['id']:e for e in elements};hero=_bounds([out[e['id']] for e in heroes[0]['items'] if e['id'] in out])
    copy.sort(key=lambda u:(u['role']=='action',u['bounds'][1]));blocks=[[out[e['id']] for e in u['items'] if e['id'] in out] for u in copy]
    if any(not b for b in blocks):return
    bounds=_raw_bounds([e for b in blocks for e in b]);font=max((e.get('font_size',0) for e in blocks[0]),default=24)
    if bounds[3]-bounds[1]>=height*.15 or bounds[1]-hero[3]<=height*.3:return
    gaps=[(.85 if copy[i+1]['role']=='action' else .65)*font for i in range(len(blocks)-1)]
    total=sum(_raw_bounds(b)[3]-_raw_bounds(b)[1] for b in blocks)+sum(gaps)
    cursor=min(height-width*.05-total,(hero[3]+height-total)/2)
    for i,block in enumerate(blocks):
        b=_raw_bounds(block)
        for e in block:e['y']+=cursor-b[1]
        cursor+=b[3]-b[1]+(gaps[i] if i<len(gaps) else 0)


def repair(elements,source,units,width,height):
    result=deepcopy(elements)
    _scene_reflow(result,source,width,height)
    _card_gutters(result,source,units,width,height)
    _source_edge_attachments(result,source,width,height)
    _primary_axis(result,source,units,width,height,apply=True)
    _grow_controls(result,source,width,height)
    # A control cleared away from an image can enter the copy band. Alternate
    # these shared constraints until the short dependent chain settles.
    for _ in range(3):
        _control_visual_clearance(result,source,width,height)
        _copy_clearance(result,source,width,height)
        _clear_controls(result,source,width,height)
        _control_peer_clearance(result,source,width,height)
    _separate_art(result,source,width,height)
    _brand_order(result,source,units,width,height,apply=True)
    _footer_caps(result,units,width)
    _spread_story_copy(result,source,units,width,height)
    _pin_corners(result,source,units,width,height)
    center_control_labels(result,source,width,height)
    return result


def issues(elements,source,width,height,units=None):
    from engine.resize import _text_visible_bounds,_overlap,_analyze,_raw_bounds
    out={e['id']:e for e in elements};src={e['id']:e for e in source['elements']};result=[];gap=max(3.,min(width,height)*.012)
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        p,t=out[pid],out[tid];body=control_body(p);ink=_text_visible_bounds(t);size=t['font_size']
        ref=min(width/source['canvas']['width'],height/source['canvas']['height'])
        floor=control_font_floor(source,tid,width,height)
        if size<floor*.94:result.append(dict(kind='control_readability',ids=[pid,tid],amount=floor-size))
        pad=min((ink[0]-body[0]),body[2]-ink[2])
        source_body=control_body(src[pid]);source_ink=_text_visible_bounds(src[tid])
        authored_pad=max(3.,min(source_ink[0]-source_body[0],source_body[2]-source_ink[2]))
        expected=max(2.,min(size*.35,authored_pad*size/max(1.,src[tid]['font_size'])))
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
    from engine.qwen_refine import build_container_tree
    tree=build_container_tree(source,units);sw,sh=source['canvas']['width'],source['canvas']['height']
    for node in tree['children']:
        ids=[i for i in node['element_ids'] if i in out]
        if len(ids)<2:continue
        old=[tree['source_elements'][i] for i in ids];sb=_raw_bounds(old);area=(sb[2]-sb[0])*(sb[3]-sb[1])
        if not any(e['type']!='text' and e['width']*e['height']>=.65*area
                   and any(t['type']=='text' and t.get('z_index',0)>e.get('z_index',0)
                           and t['x']>=e['x'] and t['y']>=e['y']
                           and t['x']+t['width']<=e['x']+e['width']+1
                           and t['y']+t['height']<=e['y']+e['height']+1 for t in old) for e in old):continue
        box=_raw_bounds([out[i] for i in ids]);g=.04*min(width,height)
        gaps=(box[0],box[1],width-box[2],height-box[3])
        authored=(sb[0]>.02*sw,sb[1]>.02*sh,sb[2]<.98*sw,sb[3]<.98*sh)
        if any(needed and gap_<g-.5 for needed,gap_ in zip(authored,gaps)):
            result.append(dict(kind='foreground_gutter',ids=ids))
    scene=edge_scene(source,width,height)
    if scene:
        for e in elements:
            if e['type']=='text':
                box=_text_visible_bounds(e)
                if _overlap(box,scene['subject'])>1:result.append(dict(kind='background_subject_overlap',ids=[e['id'],scene['photo']['id']]))
    result.extend(art_occlusions(elements,source))
    result.extend(image_collisions(elements,source))
    from engine.geometry_policy import aspect_issues
    result.extend(aspect_issues(elements,source))
    from engine.source_structure import containment_issues
    result.extend(containment_issues(elements,source))
    return result
