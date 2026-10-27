"""Source-backed attachment groups and conservative, independently ranked reflows.

No target reference, template identity or content strings participate. Painted
copy axes, enclosing panels and edge contact survive as group relationships.
"""
from copy import deepcopy
from functools import lru_cache
import json
from math import hypot
from statistics import median


def _box(elements):
    from engine.resize import _rotated_bounds
    boxes=[_rotated_bounds(e) for e in elements]
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))


def groups(source,units=None):
    return _groups_cached(json.dumps(source,sort_keys=True,separators=(',',':')))


@lru_cache(maxsize=256)
def _groups_cached(payload):
    from engine.resize import _analyze,_text_visible_bounds,_overlap,_rotated_bounds
    source=json.loads(payload);sw,sh=source['canvas']['width'],source['canvas']['height'];short=min(sw,sh)
    backgrounds,units=_analyze(source);excluded={e['id'] for e in backgrounds}
    src={e['id']:e for e in source['elements'] if e['id'] not in excluded};parent={i:i for i in src}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):parent[find(b)]=find(a)
    texts=[e for e in src.values() if e['type']=='text' and e.get('content','').strip()]
    panels={};backings=[]
    for t in texts:
        tb=_text_visible_bounds(t);area=max(1.,(tb[2]-tb[0])*(tb[3]-tb[1]));choices=[]
        for p in src.values():
            if p['type']=='text' or p.get('z_index',0)>=t.get('z_index',0):continue
            pa=p['width']*p['height']
            if pa>.70*sw*sh or pa<area:continue
            if p.get('shape_type')=='ellipse' and len(t.get('content',''))>40:continue
            if _overlap(_rotated_bounds(p),tb)/area>=.92:choices.append(p)
        if choices:
            p=min(choices,key=lambda e:e['width']*e['height']);panels[t['id']]=p['id'];join(p['id'],t['id']);backings.append((p['id'],t['id']))
    # Schema groups and existing compact painted lockups are stronger evidence
    # than a new semantic guess based only on position in a poster.
    explicit={}
    for e in src.values():
        if e.get('group_id') is not None:
            if e['group_id'] in explicit:join(e['id'],explicit[e['group_id']])
            else:explicit[e['group_id']]=e['id']
    for u in units:
        es=[e for e in u['items'] if e['id'] in src]
        if len(es)<2:continue
        b=_box(es)
        if ((b[2]-b[0])*(b[3]-b[1])<.18*sw*sh or all(e['type']=='image' for e in es)):
            for e in es[1:]:join(es[0]['id'],e['id'])
    # Join nearby copy stacks only when they share a painted edge/center and
    # do not already belong to different enclosing panels.
    axes=[]
    for i,a in enumerate(texts):
        ab=_text_visible_bounds(a)
        for b in texts[i+1:]:
            bb=_text_visible_bounds(b)
            if panels.get(a['id']) and panels.get(b['id']) and panels[a['id']]!=panels[b['id']]:continue
            gap=max(0.,ab[1]-bb[3],bb[1]-ab[3])
            offsets=[abs(ab[0]-bb[0]),abs((ab[0]+ab[2]-bb[0]-bb[2])/2),abs(ab[2]-bb[2])]
            axis=min(range(3),key=offsets.__getitem__)
            frame_axes=[abs(a['x']-b['x']),abs(a['x']+a['width']/2-b['x']-b['width']/2),abs(a['x']+a['width']-b['x']-b['width'])]
            shared=min(range(3),key=frame_axes.__getitem__)
            if frame_axes[shared]<=max(4.,.025*sw):axes.append((a['id'],b['id'],shared))
            if offsets[axis]<=max(3.,short*.02) and gap<=short*.10:
                join(a['id'],b['id'])
            # A price suffix is tied to its number, not independently placed.
            small,large=sorted((a,b),key=lambda e:e.get('font_size',0));sb=_text_visible_bounds(small);lb=_text_visible_bounds(large)
            if (len(small.get('content','').strip())<=3 and small.get('font_size',1)<.4*large.get('font_size',1)
                    and abs(sb[1]-lb[1])<=large.get('font_size',1)*.7
                    and max(0.,sb[0]-lb[2],lb[0]-sb[2])<=max(2*small.get('font_size',1),.16*(lb[2]-lb[0]))):join(small['id'],large['id'])
    # Thin rules and compact marks attach to their nearest source text. A
    # two-edge corner bracket remains an edge decoration, never a text leader.
    ornaments=[]
    for art in src.values():
        if art['type']=='text' or art['id'] in {p for p,_ in backings}:continue
        ab=_rotated_bounds(art);aw,ah=ab[2]-ab[0],ab[3]-ab[1]
        if not (min(aw,ah)<=short*.025 or aw*ah<=sw*sh*.0025):continue
        nearx=min(abs(ab[0]),abs(sw-ab[2]))<.08*sw;neary=min(abs(ab[1]),abs(sh-ab[3]))<.08*sh
        if nearx and neary:continue
        options=[]
        for t in texts:
            tb=_text_visible_bounds(t);dx=max(0.,tb[0]-ab[2],ab[0]-tb[2]);dy=max(0.,tb[1]-ab[3],ab[1]-tb[3])
            distance=hypot(dx,dy)
            if distance<=short*.07:options.append((distance,t))
        if options:
            _,t=min(options,key=lambda p:p[0]);join(art['id'],t['id']);ornaments.append((art['id'],t['id']))
    # Compact illustrative companions (for example a bird above a callout)
    # belong to a nearby copy group, even when they are not thin ornaments.
    for art in src.values():
        if art['type']=='text' or sum(find(i)==find(art['id']) for i in src)>1:continue
        ab=_rotated_bounds(art)
        if art['width']*art['height']>.08*sw*sh:continue
        options=[]
        for t in texts:
            tb=_text_visible_bounds(t);dx=max(0.,tb[0]-ab[2],ab[0]-tb[2]);dy=max(0.,tb[1]-ab[3],ab[1]-tb[3])
            horizontal=min(ab[2],tb[2])-max(ab[0],tb[0])
            if horizontal>0 and hypot(dx,dy)<=short*.04:options.append((hypot(dx,dy),t))
        if options:join(art['id'],min(options,key=lambda p:p[0])[1]['id'])
    # A nested control backed by a larger panel is still part of that panel.
    for pid,_ in backings:
        p=src[pid];pb=_rotated_bounds(p);area=max(1.,p['width']*p['height'])
        for other in {i for i,_ in backings}:
            q=src[other]
            if other!=pid and q['width']*q['height']>area*1.5 and _overlap(pb,_rotated_bounds(q))/area>=.65:join(pid,other)
    # Compact, source-centered emblems couple the text, paired illustrations
    # and rules as one lockup, even when no single panel encloses them all.
    from engine.qwen_refine import _emblem_groups
    for emblem in _emblem_groups({i:e for i,e in src.items() if e['type']=='text' or float(e.get('opacity',1.))>=.9 or i in {p for p,_ in backings}},sw,sh):
        ids=sorted(emblem)
        for eid in ids[1:]:join(ids[0],eid)
    result={}
    for eid in src:result.setdefault(find(eid),[]).append(eid)
    return dict(members=[tuple(sorted(v)) for v in result.values()],backings=backings,ornaments=ornaments,axes=axes,
                semantics={e['id']:u.get('semantic') for u in units for e in u['items']})


def _transform(members,scale,left,top,source_box):
    from engine.resize import _set_font_size
    result=deepcopy(members)
    for e in result:
        e.update(x=left+(e['x']-source_box[0])*scale,y=top+(e['y']-source_box[1])*scale,width=e['width']*scale,height=e['height']*scale)
        if e['type']=='text':_set_font_size(e,e['font_size']*scale)
    return result


def seed(source,units,width,height,flow=False):
    """Preserve source groups; move containers, not the leaves of each card."""
    spec=groups(source);src={e['id']:e for e in source['elements']};sw,sh=source['canvas']['width'],source['canvas']['height'];scale=min(width/sw,height/sh)
    margin=.035*min(width,height);blocks=[]
    for ids in spec['members']:
        es=[src[i] for i in ids];raw_box=_box(es)
        # Hidden raster bleed must not push the visible panel and its copy
        # inward. The parent canvas clips that hidden part in the source.
        b=(max(0.,raw_box[0]),max(0.,raw_box[1]),min(sw,raw_box[2]),min(sh,raw_box[3]))
        bw,bh=b[2]-b[0],b[3]-b[1]
        if bw<=0 or bh<=0:return None
        edges=(raw_box[0]<=1.,raw_box[1]<=1.,raw_box[2]>=sw-1.,raw_box[3]>=sh-1.)
        s=scale
        # Full-height/width edge ornaments retain the same parent edge band.
        if all(e['type']!='text' for e in es):
            if edges[1] and edges[3] and bw<.45*sw:s=min(height/max(1,bh),width*.30/max(1,bw))
            elif edges[0] and edges[2] and bh<.45*sh:s=min(width/max(1,bw),height*.30/max(1,bh))
        s=min(s,(width-2*margin)/max(1,bw),(height-2*margin)/max(1,bh))
        x=(width-sw*scale)/2+b[0]*scale;y=(height-sh*scale)/2+b[1]*scale
        if edges[0] and not edges[2]:x=0.
        if edges[2] and not edges[0]:x=width-bw*s
        if edges[1] and not edges[3]:y=0.
        if edges[3] and not edges[1]:y=height-bh*s
        x=min(width-bw*s,max(0.,x));y=min(height-bh*s,max(0.,y))
        blocks.append(dict(ids=ids,source_box=b,scale=s,x=x,y=y,edges=edges,items=es))
    if flow and max((width/height)/(sw/sh),(sw/sh)/(width/height))>=1.8:
        # A few containers across a wide source become rows, or vice versa.
        # The largest multi-element panel owns the main band; side callouts
        # stay separate, with their source reading order and internal geometry.
        central=[p for p in blocks if any(e['type']=='text' for e in p['items'])]
        if not 2<=len(central)<=5:return None
        ordered=sorted(central,key=lambda p:(p['source_box'][1]+p['source_box'][3])/2 if sw<sh else (p['source_box'][0]+p['source_box'][2])/2)
        areas=[(p['source_box'][2]-p['source_box'][0])*(p['source_box'][3]-p['source_box'][1]) for p in ordered]
        weights=[max(.1,(a/max(areas))**.5) for a in areas];gap=.035*min(width,height)
        available=height-2*margin-gap*(len(ordered)-1);cursor=margin
        primary=ordered[areas.index(max(areas))];pb=primary['source_box']
        primary_room=available*max(weights)/sum(weights)
        common_scale=min((width-2*margin)/max(1,pb[2]-pb[0]),primary_room/max(1,pb[3]-pb[1]))
        for p,weight in zip(ordered,weights):
            b=p['source_box'];bw,bh=b[2]-b[0],b[3]-b[1];room=available*weight/sum(weights)
            # Small footer/control groups must not grow their type several
            # times larger than the main panel just to consume a wide row.
            s=min((width-2*margin)/max(1,bw),room/max(1,bh),common_scale*1.25)
            p.update(scale=s,x=(width-bw*s)/2,y=cursor+(room-bh*s)/2)
            if p['edges'][1] and not p['edges'][3] and p is ordered[0]:p['y']=0.
            cursor+=room+gap
    if len(blocks)==1:
        p=blocks[0];b=p['source_box'];bw,bh=b[2]-b[0],b[3]-b[1]
        if (sum(e['type']=='text' for e in p['items'])>=2 and any(e['type']!='text' for e in p['items'])
                and bw*bh<.55*sw*sh and abs((b[0]+b[2])/2-sw/2)<.08*sw):
            s=min(width*.84/max(1,bw),height*.84/max(1,bh))
            p.update(scale=max(p['scale'],s),x=(width-bw*max(p['scale'],s))/2,y=(height-bh*max(p['scale'],s))/2)
    output=[]
    for p in blocks:output+=_transform(p['items'],p['scale'],p['x'],p['y'],p['source_box'])
    # Legalize only authored clipped image frames. Text and interior artwork
    # retain the shared transform; no asset pixels are changed or replaced.
    for e in output:
        old=src[e['id']]
        if e['type']=='text' or not (old['x']<0 or old['y']<0 or old['x']+old['width']>sw or old['y']+old['height']>sh):continue
        factor=min(1.,width/e['width'],height/e['height']);e['width']*=factor;e['height']*=factor
        e['x']=min(width-e['width'],max(0.,e['x']));e['y']=min(height-e['height'],max(0.,e['y']))
        # If revealing the hidden part would paint over previously separate
        # copy, reduce that ornament to its authored visible extent instead.
        # Asset bytes are unchanged. Owned labels are centered/fitted below.
        from engine.resize import _text_visual_coverage
        new_collision=any(_text_visual_coverage(t,e)>.02 and _text_visual_coverage(src[t['id']],old)<=.05 for t in output if t['type']=='text')
        if new_collision:
            fx=max(0.,min(sw,old['x']+old['width'])-max(0.,old['x']))/old['width']
            fy=max(0.,min(sh,old['y']+old['height'])-max(0.,old['y']))/old['height']
            shrink=max(.1,min(fx,fy));cx=e['x']+e['width']/2;cy=e['y']+e['height']/2
            e['width']*=shrink;e['height']*=shrink
            e['x']=0. if old['x']<0 else width-e['width'] if old['x']+old['width']>sw else cx-e['width']/2
            e['y']=0. if old['y']<0 else height-e['height'] if old['y']+old['height']>sh else cy-e['height']/2
    return output


def fit_groups_to_canvas(elements,source,width,height):
    """Final font/rotation calibration cannot leave a group outside canvas."""
    from engine.resize import _set_font_size
    out={e['id']:e for e in elements}
    for ids in groups(source)['members']:
        es=[out[i] for i in ids if i in out]
        if not es:continue
        b=_box(es);factor=min(1.,width/max(1.,b[2]-b[0]),height/max(1.,b[3]-b[1]))
        if factor<1.:
            for e in es:
                e.update(x=b[0]+(e['x']-b[0])*factor,y=b[1]+(e['y']-b[1])*factor,width=e['width']*factor,height=e['height']*factor)
                if e['type']=='text':_set_font_size(e,e['font_size']*factor)
            b=_box(es)
        dx=max(0.,-b[0])-max(0.,b[2]-width);dy=max(0.,-b[1])-max(0.,b[3]-height)
        for e in es:e['x']+=dx;e['y']+=dy


def relationship_cost(elements,source,width,height):
    """Dimensionless error in actual source attachments, not template priors."""
    from engine.resize import _text_visible_bounds,_rotated_bounds,_overlap
    spec=groups(source);src={e['id']:e for e in source['elements']};out={e['id']:e for e in elements};errors=[]
    for panel,tid in spec['backings']:
        if panel not in out or tid not in out:continue
        old=src[panel];p=out[panel];a=src[tid];t=out[tid]
        s=min(p['width']/old['width'],p['height']/old['height']);ink=_text_visible_bounds(t);box=_rotated_bounds(p)
        oldink=_text_visible_bounds(a);oldbox=_rotated_bounds(old)
        oldgaps=[oldink[0]-oldbox[0],oldink[1]-oldbox[1],oldbox[2]-oldink[2],oldbox[3]-oldink[3]]
        gaps=[ink[0]-box[0],ink[1]-box[1],box[2]-ink[2],box[3]-ink[3]]
        errors.append(sum(max(0.,-v-max(0.,-before*s)-1.) for v,before in zip(gaps,oldgaps))/max(1.,min(p['width'],p['height'])))
    for ids in spec['members']:
        if len(ids)<2 or not all(i in out for i in ids):continue
        es=[src[i] for i in ids];major=[e for e in es if e['type']=='text' or min(e['width'],e['height'])>3]
        if not major:continue
        scales=[min(out[e['id']]['width']/e['width'],out[e['id']]['height']/e['height']) for e in major];s=median(scales)
        tx=median(out[e['id']]['x']-e['x']*s for e in major);ty=median(out[e['id']]['y']-e['y']*s for e in major)
        b=_box(es);den=max(1.,max(b[2]-b[0],b[3]-b[1])*s)
        # Soft, bounded lockup error: fitted text is allowed small changes.
        errors.append(sum(min(1.,hypot(out[i]['x']-(src[i]['x']*s+tx),out[i]['y']-(src[i]['y']*s+ty))/den) for i in ids)/len(ids))
    # Source-aligned groups share an axis even if a large deliberate gap kept
    # them out of one rigid container. Modest resizes do not need to scramble
    # a title/subtitle into unrelated columns.
    sw,sh=source['canvas']['width'],source['canvas']['height'];reference=min(width/sw,height/sh)
    shift=max((width/height)/(sw/sh),(sw/sh)/(width/height))
    for a,b,axis in spec['axes']:
        if a not in out or b not in out:continue
        f=(0.,.5,1.)[axis]
        before=(src[a]['x']+src[a]['width']*f)-(src[b]['x']+src[b]['width']*f)
        after=(out[a]['x']+out[a]['width']*f)-(out[b]['x']+out[b]['width']*f)
        if shift<1.8 or abs(after)<=.08*width:errors.append(abs(after-before*reference)/width)
    # Edge contact is a parent attachment, on all four sides.
    for original in src.values():
        if original['id'] not in out or original['type']=='text':continue
        sw,sh=source['canvas']['width'],source['canvas']['height']
        if original['width']*original['height']>.7*sw*sh:continue
        e=out[original['id']];old=_rotated_bounds(original);new=_rotated_bounds(e)
        for before,after,extent in [(old[0],new[0],width),(old[1],new[1],height),(sw-old[2],width-new[2],width),(sh-old[3],height-new[3],height)]:
            if before<=1.:errors.append(max(0.,after)/extent)
    return sum(errors)/max(1,len(errors))


def containment_issues(elements,source):
    """A source-supported parent owns its text even without schema group_id."""
    from engine.resize import _text_visible_bounds,_rotated_bounds
    spec=groups(source);src={e['id']:e for e in source['elements']};out={e['id']:e for e in elements};result=[]
    for pid,tid in spec['backings']:
        if pid not in out or tid not in out:continue
        p,t=out[pid],out[tid];old,old_t=src[pid],src[tid]
        b=_rotated_bounds(p);ink=_text_visible_bounds(t);ob=_rotated_bounds(old);oi=_text_visible_bounds(old_t)
        s=min(p['width']/old['width'],p['height']/old['height'])
        old_gaps=(ob[0]-oi[0],ob[1]-oi[1],oi[2]-ob[2],oi[3]-ob[3])
        gaps=(b[0]-ink[0],b[1]-ink[1],ink[2]-b[2],ink[3]-b[3])
        amount=max(a-max(0.,before)*s-1. for a,before in zip(gaps,old_gaps))
        if amount>1.:result.append(dict(kind='source_parent_containment',ids=[pid,tid],amount=amount))
    return result


def coherent_copy_axes(elements,source,units,width,height):
    """Use a reflowed column's shared axis for copy and its action control.

    Inline action columns remain independent. A centered brand/hero stack is
    evidence for centered copy; a left-aligned editorial copy column supplies
    its painted left edge to the button directly below it.
    """
    from engine.resize import _text_visible_bounds,_bounds,_overlap
    from engine.layout_contracts import controls,control_body
    out={e['id']:e for e in elements};by_unit={e['id']:u for u in units for e in u['items']}
    originals={e['id']:e for e in source['elements']};labels={tid for _,tid in controls(source)}
    copy=[e for e in elements if e['type']=='text' and e['id'] not in labels and by_unit.get(e['id'],{}).get('semantic') in ('headline','subheading','body')]
    anchors=[]
    for u in units:
        if u.get('semantic') not in ('logo','hero'):continue
        es=[out[e['id']] for e in u['items'] if e['id'] in out]
        if es:anchors.append(_bounds(es))
    centered=(width/height<1.35 and len(anchors)>=2
              and all(abs((b[0]+b[2])/2-width/2)<width*.035 for b in anchors))
    if centered:
        for t in copy:
            if t['width']<width*.55:continue
            t['text_align']='center';t['x']=(width-t['width'])/2
    for pid,tid in controls(source):
        if pid not in out or tid not in out:continue
        if by_unit.get(tid,{}).get('semantic')!='cta' or by_unit.get(tid,{}).get('role')!='action':continue
        panel,label=out[pid],out[tid]
        if abs(panel.get('rotation',0))>.5:continue
        pb=control_body(panel);candidates=[]
        for t in copy:
            tb=_text_visible_bounds(t)
            if tb[3]>pb[1]+max(2.,t.get('font_size',1)*.3):continue
            if pb[1]-tb[3]>.22*height:continue
            if min(tb[2],pb[2])<=max(tb[0],pb[0]):continue
            # Only a formerly attached action follows the copy column. A
            # separate source badge/corner callout does not acquire this role.
            a,b=originals[pid],originals[t['id']]
            if a['y']<b['y'] or a['y']-(b['y']+b['height'])>.20*source['canvas']['height']:continue
            candidates.append((pb[1]-tb[3],t,tb))
        if not candidates:continue
        _,t,tb=min(candidates,key=lambda p:p[0]);align=t.get('text_align','left')
        delta=((tb[0]+tb[2]-pb[0]-pb[2])/2 if align=='center' else
               tb[2]-pb[2] if align=='right' else tb[0]-pb[0])
        members=[out[i] for i in (pid,tid)]
        if not all(0<=e['x']+delta and e['x']+delta+e['width']<=width for e in members):continue
        moved=(pb[0]+delta,pb[1],pb[2]+delta,pb[3])
        # Being below a headline does not make an action part of its column
        # when another paragraph occupies that row. Preserve an inline slot.
        if any(_overlap(moved,_text_visible_bounds(other))>_overlap(pb,_text_visible_bounds(other))+1.
               for other in copy):continue
        for e in members:e['x']+=delta
