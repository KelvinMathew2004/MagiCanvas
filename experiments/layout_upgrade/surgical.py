"""Source-evidenced follow-up experiments; never activated by production.

No target coordinates, template IDs or content strings are consulted here.
Every geometry proposal still needs the normal raster/contract verifier.
"""
from copy import deepcopy
from math import cos,sin,radians


def envelopes(element):
    x,y,w,h=(float(element[k]) for k in ('x','y','width','height'))
    angle=radians(float(element.get('rotation',0)))
    rx=(w*abs(cos(angle))+h*abs(sin(angle)))/2
    ry=(w*abs(sin(angle))+h*abs(cos(angle)))/2
    cx,cy=x+w/2,y+h/2
    return (x,y,x+w,y+h),(cx-rx,cy-ry,cx+rx,cy+ry)


def envelope_violation(element,width,height):
    return max(0.,*(v for b in envelopes(element) for v in (-b[0],-b[1],b[2]-width,b[3]-height)))


def dual_envelope(elements,width,height):
    """Geometric prototype only: simultaneous raw and rotated containment.

    A production final-output mutation is unsafe: this may detach children or
    change rendered clipping. Source bleed/ambient paint needs separate policy.
    """
    from engine.resize import _set_font_size
    result=deepcopy(elements)
    for e in result:
        if min(e['width'],e['height'])<=0:continue
        raw,rotated=envelopes(e);cx=(raw[0]+raw[2])/2;cy=(raw[1]+raw[3])/2
        rx=max(e['width'],rotated[2]-rotated[0])/2
        ry=max(e['height'],rotated[3]-rotated[1])/2
        scale=min(1.,width/(2*rx),height/(2*ry))
        e['width']*=scale;e['height']*=scale
        if e.get('type')=='text':_set_font_size(e,e['font_size']*scale)
        rx*=scale;ry*=scale
        cx=min(max(cx,rx),width-rx);cy=min(max(cy,ry),height-ry)
        e['x']=max(0.,cx-e['width']/2);e['y']=max(0.,cy-e['height']/2)
    return result


def font_slack(elements,source,width,height,preferred_fonts=None):
    """Restore a small box-width-derived type deficit with <=18% extra width.

    Preserve the existing alignment anchor, explicit breaks and source line
    count. Do not shrink existing type, cross a backing or invent free space.
    Already equivalent calibration exists; this tests the remaining headroom.
    """
    from engine.resize import _text_info,_render_text_ink,_set_font_size,_overlap
    from engine.nested_structure import unpainted_ids
    result=deepcopy(elements);src={e['id']:e for e in source['elements']};inactive=unpainted_ids(source);notes=[]
    for e in result:
        s=src[e['id']]
        if e['type']!='text' or e['id'] in inactive or abs(e.get('rotation',0))>.1:continue
        if min(s['width'],s['height'],s.get('font_size',0))<=0:continue
        goal=(preferred_fonts or {}).get(e['id'],float(s['font_size'])*e['width']/s['width'])
        desired=min(float(goal),float(e['font_size'])*1.25)
        if desired<=e['font_size']*1.01:continue
        target=deepcopy(e);_set_font_size(target,desired)
        count=max(1,_text_info(s)[2])
        if _text_info(target)[2]<=count:continue
        align=e.get('text_align','left');fraction=.5 if align=='center' else 1. if align=='right' else 0.
        anchor=e['x']+fraction*e['width'];old_width=e['width']
        best=None
        for increase in (.03,.06,.09,.12,.15,.18):
            p=deepcopy(target);p['width']=old_width*(1+increase);p['x']=anchor-fraction*p['width']
            if p['x']<0 or p['x']+p['width']>width:continue
            box=(p['x'],p['y'],p['x']+p['width'],p['y']+p['height'])
            allowed=True
            for other in result:
                if other['id']==e['id']:continue
                old=src[other['id']]
                if old['width']<=0 or old['height']<=0:continue
                sb=(s['x'],s['y'],s['x']+s['width'],s['y']+s['height'])
                ob=(old['x'],old['y'],old['x']+old['width'],old['y']+old['height'])
                # A source-enclosing backplate bounds the new box. Broad
                # photos and ambient surfaces are not inferred child owners.
                enclosed=(old['type']!='text' and old.get('z_index',0)<=s.get('z_index',0)
                          and _overlap(sb,ob)>=s['width']*s['height']*.99)
                if enclosed and old['width']<source['canvas']['width']*.9:
                    if not (box[0]>=other['x'] and box[2]<=other['x']+other['width'] and box[1]>=other['y'] and box[3]<=other['y']+other['height']):allowed=False;break
                if _overlap(sb,ob)>0:continue
                target_ob=(other['x'],other['y'],other['x']+other['width'],other['y']+other['height'])
                if _overlap(box,target_ob)>1e-6:allowed=False;break
            if not allowed or _text_info(p)[2]>count:continue
            ink=_render_text_ink(p)
            if not ink or ink[0]<-1 or ink[1]<-1 or ink[2]>round(p['width']) or ink[3]>round(p['height']):continue
            best=p;break
        if best:
            notes.append(dict(id=e['id'],font_before=e['font_size'],font_after=best['font_size'],width_ratio=best['width']/old_width,source_lines=count))
            e.update(best)
    return result,notes


def topology(source,design,strict_ownership=False):
    """Compare literal proposed topology to a more conservative owned graph.

    60% bbox overlap is not proof of a backing, and a source row may correctly
    become a column. Literal and source-owned interpretations are kept separate
    to measure false vetoes, not presented as universal design invariants.
    """
    from engine.resize import _overlap
    from engine.macro_layout import composites
    from engine.nested_structure import unpainted_ids
    src={e['id']:e for e in source['elements']};out={e['id']:e for e in design['elements']};inactive=unpainted_ids(source)
    sw,sh=source['canvas']['width'],source['canvas']['height'];edges=[];broken=[];orders=[]
    def box(e):return (e['x'],e['y'],e['x']+e['width'],e['y']+e['height'])
    for child in source['elements']:
        if child['id'] in inactive:continue
        possible=[]
        for parent in source['elements']:
            if parent['id']==child['id'] or parent['type']=='text' or parent['id'] in inactive:continue
            area=child['width']*child['height']
            if area<=0 or _overlap(box(parent),box(child))/area<.60:continue
            if strict_ownership and not (parent.get('group_id') is not None and parent.get('group_id')==child.get('group_id')
                    and parent.get('z_index',0)<=child.get('z_index',0) and parent['width']*parent['height']<.8*sw*sh):continue
            possible.append(parent)
        if strict_ownership and possible:possible=[min(possible,key=lambda p:p['width']*p['height'])]
        for p in possible:
            edges.append((p['id'],child['id']));a,b=out.get(p['id']),out.get(child['id'])
            share=_overlap(box(a),box(b))/max(1e-9,b['width']*b['height']) if a and b else 0.
            if share<.60-1e-6:broken.append(dict(parent=p['id'],child=child['id'],share=share))
    for block in composites(source,tight=True)['blocks']:
        members=[e for e in block if e['id'] not in inactive]
        if len(members)<2:continue
        # Test the suggested exact source-axis permutation, retaining ties.
        xs=[e['x']+e['width']/2 for e in members];ys=[e['y']+e['height']/2 for e in members]
        axis=0 if (max(xs)-min(xs))/sw>(max(ys)-min(ys))/sh else 1
        for i,a in enumerate(members):
            for b in members[:i]:
                k='x' if axis==0 else 'y';dim='width' if axis==0 else 'height'
                delta=a[k]+a[dim]/2-b[k]-b[dim]/2
                if abs(delta)<=4:continue
                aa,bb=out.get(a['id']),out.get(b['id'])
                if aa is None or bb is None:continue
                new=aa[k]+aa[dim]/2-bb[k]-bb[dim]/2
                if delta*new<-.01:orders.append([a['id'],b['id']])
    return dict(containment_edges=len(edges),broken_containment=broken,order_inversions=orders,
                passes=not broken and not orders,strict_ownership=strict_ownership)


def hierarchy_band(elements,source):
    """Test the proposed .75–1.25 scale band inside existing tight blocks.

    Not a hard rule: independent legal/caption scale may be intentional. Its
    effects are measured separately from the horizontal-slack experiment.
    """
    from engine.macro_layout import composites
    from engine.resize import _set_font_size
    from engine.nested_structure import unpainted_ids
    result=deepcopy(elements);out={e['id']:e for e in result};inactive=unpainted_ids(source);notes=[]
    for block in composites(source,tight=True)['blocks']:
        texts=[e for e in block if e['type']=='text' and e['id'] not in inactive and e.get('font_size',0)>0 and e['id'] in out]
        if len(texts)<2:continue
        head=max(texts,key=lambda e:e['font_size']);scale=out[head['id']]['font_size']/head['font_size']
        for s in texts:
            if s is head:continue
            e=out[s['id']];desired=max(.75*scale*s['font_size'],min(1.25*scale*s['font_size'],e['font_size']))
            if abs(desired-e['font_size'])<.01:continue
            notes.append(dict(id=e['id'],anchor=head['id'],before=e['font_size'],after=desired))
            _set_font_size(e,desired)
    return result,notes


def gated_critic(global_scores,tiled_scores,size):
    """Offline fixed top-two/0.15-margin policy; scores not commensurate yet.

    Untouched other candidates retain global scores. This deliberately exposes
    the calibration problem when mixing two score systems in one shortlist.
    """
    w,h=size;result=list(global_scores)
    if max(w/h,h/w)<2.2 or len(result)<2:return result,[]
    ids=sorted(range(len(result)),key=lambda i:result[i],reverse=True)[:2]
    if result[ids[0]]-result[ids[1]]>.15:return result,[]
    for i in ids:result[i]=tiled_scores[i]
    return result,ids
