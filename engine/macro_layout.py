"""Source-derived composite blocks and independently gated macro layouts.

No template identifiers, target reference coordinates or text strings drive
partitioning. Existing candidates stay intact; new layouts must earn selection.
"""
from copy import deepcopy
from functools import lru_cache
from itertools import combinations
from statistics import median
from math import sqrt,ceil
import json,os


def edge_strips(source):
    w,h=source['canvas']['width'],source['canvas']['height'];found={}
    for e in source['elements']:
        if e['type']!='shape' or e.get('shape_type','rectangle') not in ('rectangle','rounded_rectangle','line') or abs(e.get('rotation',0))>.1:continue
        if e['width']<=.06*w and e['height']>=.85*h:
            if e['x']<=1:found[e['id']]='left'
            elif e['x']+e['width']>=w-1:found[e['id']]='right'
        elif e['height']<=.06*h and e['width']>=.85*w:
            if e['y']<=1:found[e['id']]='top'
            elif e['y']+e['height']>=h-1:found[e['id']]='bottom'
    return found


def clamp_edge_strips(elements,source,width,height):
    originals={e['id']:e for e in source['elements']};sw,sh=source['canvas']['width'],source['canvas']['height']
    for e in elements:
        edge=edge_strips(source).get(e['id'])
        if edge is None:continue
        old=originals[e['id']]
        if edge in ('left','right'):
            e.update(width=old['width']/sw*width,height=height,y=0.)
            e['x']=0. if edge=='left' else width-e['width']
        else:
            e.update(height=old['height']/sh*height,width=width,x=0.)
            e['y']=0. if edge=='top' else height-e['height']


def bounds(items):
    from engine.resize import _rotated_bounds
    b=[_rotated_bounds(e) for e in items]
    return min(p[0] for p in b),min(p[1] for p in b),max(p[2] for p in b),max(p[3] for p in b)


def composites(source,tight=None):
    return deepcopy(_composites(json.dumps(source,sort_keys=True,separators=(',',':')),tight))


@lru_cache(maxsize=256)
def _composites(payload,tight=None):
    from engine.resize import _analyze,_overlap,_text_visible_bounds
    source=json.loads(payload);w,h=source['canvas']['width'],source['canvas']['height']
    from engine.geometry_policy import ambient_shape
    from engine.nested_structure import regional_surfaces, repeated_text_rows, unpainted_ids
    surfaces=regional_surfaces(source)
    bg,_=_analyze(source);fixed={e['id'] for e in bg}|set(edge_strips(source))
    # Backgrounds and edge strips have their own realization paths. A layer
    # may also be translucent, but must only be emitted once.
    # A translucent hairline is still a divider. Alpha alone cannot detach
    # row leaders from the same source neighbors as opaque rules.
    rules={e['id'] for e in source['elements'] if e['type']=='shape'
           and e.get('shape_type','rectangle') in ('rectangle','line')
           and 0<min(e['width'],e['height'])<=max(6.,.008*min(w,h))}
    ambient=(({e['id'] for e in source['elements'] if ambient_shape(e,source)}-rules)|set(surfaces)|unpainted_ids(source))-fixed
    excluded=fixed|ambient
    src={e['id']:e for e in source['elements'] if e['id'] not in excluded};parent={i:i for i in src};links=[]
    # A previously established compact emblem is already a complete functional
    # unit. Whitespace between its leaves is not permission to split it.
    from engine.source_structure import groups as source_groups
    shared=source_groups(source)['members']
    if not surfaces and tight is not False and len(shared)==1 and set(shared[0])==set(src):
        box=bounds(list(src.values()))
        if (box[2]-box[0])*(box[3]-box[1])<=.65*w*h:
            return dict(blocks=[list(src.values())],links=[],edge_ids=list(edge_strips(source)),ambient_ids=sorted(ambient))

    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b,kind):
        parent[root(b)]=root(a);links.append((a,b,kind))
    explicit={}
    for e in src.values():
        if e.get('group_id'):
            g=e['group_id']
            if g in explicit:join(explicit[g],e['id'],'explicit')
            else:explicit[g]=e['id']
    from engine.nested_structure import control_captions
    for p,t,c in control_captions(source):
        if p in src and c in src:join(p,c,'control-caption')
    for a,b in repeated_text_rows(list(src.values()),source['canvas']):
        join(a,b,'repeated-row')
        left,right=src[a],src[b]
        for rule in src.values():
            if rule['id'] not in rules or rule['width']<.6*(right['x']+right['width']-left['x']):continue
            if (abs(rule['x']-left['x'])<=4 and rule['x']+rule['width']<=right['x']+right['width']+4
                    and max(left['y'],right['y'])<rule['y']<=max(left['y']+left['height'],right['y']+right['height'])+4):
                join(a,rule['id'],'row-divider')
    for p in src.values():
        if p['type']=='text' or p['width']*p['height']>.70*w*h:continue
        pb=bounds([p]);area=p['width']*p['height'];children=[]
        for t in src.values():
            if p is t or t.get('z_index',0)<=p.get('z_index',0):continue
            tb=_text_visible_bounds(t) if t['type']=='text' else bounds([t]);ta=max(1.,(tb[2]-tb[0])*(tb[3]-tb[1]))
            if ta>area*.95:continue
            if _overlap(pb,tb)/ta>=.92:children.append(t)
        # Photos own overlay badges; shapes own all of their card children,
        # including another grouped CTA. Avoid a small loose ornament bridge.
        if p['type']=='shape' or p['type']=='image' and area>.025*w*h:
            for t in children:join(p['id'],t['id'],'parent')
    photos=[p for p in src.values() if p['type']=='image' and p['width']*p['height']>.025*w*h]
    for photo in photos:
        for p in src.values():
            if p is photo or p['type']=='image' or p['width']<=0:continue
            # A tall regional panel following a photo is not its caption.
            if p['height'] > .25*photo['height']:continue
            if abs(p['width']/photo['width']-1)>.08:continue
            if abs(p['x']-photo['x'])>.04*photo['width']:continue
            if abs(p['y']-(photo['y']+photo['height']))<=max(2.,.02*h):join(photo['id'],p['id'],'caption')
    # Divider affiliation is source-local and axis-aware, never an independent
    # semantic visual slot. Edge-spanning bands were removed above.
    for line in src.values():
        if line['type']=='text' or min(line['width'],line['height'])>max(6.,.008*min(w,h)):continue
        if any(root(i)==root(line['id']) and i!=line['id'] for i in src):continue
        horizontal=line['width']>=line['height'];axis=1 if horizontal else 0
        b=bounds([line]);options=[]
        for p in src.values():
            if p is line or p['type']!='text' and p['width']*p['height']<.02*w*h:continue
            q=bounds([p]);
            if horizontal and b[2]-b[0]>(q[2]-q[0])*1.5:continue
            gap=max(0.,b[axis]-q[axis+2],q[axis]-b[axis+2]);cross=1-axis
            overlap=min(b[cross+2],q[cross+2])-max(b[cross],q[cross])
            if overlap>0 and gap<.15*(h if horizontal else w):options.append((gap,p))
        if options:join(line['id'],min(options,key=lambda p:p[0])[1]['id'],'divider')
    # Tiny marks share the nearest source cluster instead of becoming grid cells.
    for art in src.values():
        if art['type']=='text' or art['width']*art['height']>.003*w*h:continue
        if any(root(i)==root(art['id']) and i!=art['id'] for i in src):continue
        b=bounds([art]);choices=[]
        for t in src.values():
            if t['type']!='text':continue
            q=bounds([t]);dx=max(0.,b[0]-q[2],q[0]-b[2]);dy=max(0.,b[1]-q[3],q[1]-b[3]);choices.append((dx*dx+dy*dy,t))
        if choices:join(art['id'],min(choices,key=lambda p:p[0])[1]['id'],'mark')
    if tight is True:
        # Existing hierarchy supplies compact functional lockups, rather than
        # matching every isolated text line to a separate human template zone.
        # Background paint cannot bridge groups; native ownership above wins.
        for members in shared:
            ids=[i for i in members if i in src]
            if len(ids)<2:continue
            b=bounds([src[i] for i in ids])
            if (b[2]-b[0])*(b[3]-b[1])<=.65*w*h:
                for i in ids[1:]:join(ids[0],i,'hierarchy')
    if tight is False:
        # Relax only the inferred whole-poster union. Existing compact atomic
        # units (e.g. a number and its suffix) still own their children.
        for u in _analyze(source)[1]:
            es=[e for e in u['items'] if e['id'] in src]
            if len(es)<2:continue
            b=bounds(es)
            if (b[2]-b[0])*(b[3]-b[1])<.18*w*h:
                for e in es[1:]:join(es[0]['id'],e['id'],'unit')
    blocks={}
    for i,e in src.items():blocks.setdefault(root(i),[]).append(e)
    return dict(blocks=list(blocks.values()),links=links,edge_ids=list(edge_strips(source)),ambient_ids=sorted(ambient))


def _ambient_layers(source,width,height,ids):
    """Carry translucent paint outside normal flow, preserving artwork aspect."""
    from engine.nested_structure import unpainted_ids
    unpainted=unpainted_ids(source)
    sw,sh=source['canvas']['width'],source['canvas']['height'];result=[]
    for old in source['elements']:
        if old['id'] not in ids:continue
        e=deepcopy(old);s=min(width/sw,height/sh)
        w,h=old['width']*s,old['height']*s
        cx=(old['x']+old['width']/2)/sw*width;cy=(old['y']+old['height']/2)/sh*height
        e.update(width=w,height=h,x=max(0.,min(width-w,cx-w/2)),y=max(0.,min(height-h,cy-h/2)))
        if old['id'] in unpainted:
            # Style fields are immutable. Retain opacity exactly; a source
            # layer with no viewport footprint gets a minimal bounded box.
            e.update(width=1.,height=1.)
            if e['type']=='text':
                from engine.resize import _set_font_size
                _set_font_size(e,.1)
            e['x']=max(0.,min(width-e['width'],e['x']));e['y']=max(0.,min(height-e['height'],e['y']))
        result.append(e)
    return result


def _fit_regional_surfaces(items,source,width,height):
    """Wrap scope paint around its children without welding their layout."""
    from engine.nested_structure import regional_surfaces
    surfaces=regional_surfaces(source);out={e['id']:e for e in items}
    scale=min(width/source['canvas']['width'],height/source['canvas']['height'])
    originals={e['id']:e for e in source['elements']}
    for pid,ids in surfaces.items():
        children=[out[i] for i in ids if i in out and i not in surfaces]
        if pid not in out or not children:continue
        old=originals[pid];sb=bounds([originals[i] for i in ids if i in out and i not in surfaces]);b=bounds(children)
        pads=[max(0.,sb[0]-old['x']),max(0.,sb[1]-old['y']),
              max(0.,old['x']+old['width']-sb[2]),max(0.,old['y']+old['height']-sb[3])]
        pads=[min(p*scale,.05*min(width,height)) for p in pads]
        x=max(0.,b[0]-pads[0]);y=max(0.,b[1]-pads[1])
        right=min(width,b[2]+pads[2]);bottom=min(height,b[3]+pads[3])
        out[pid].update(x=x,y=y,width=max(1.,right-x),height=max(1.,bottom-y))


def _cap_control_growth(items,source,width,height):
    """An isolated flow slot cannot reverse the authored type hierarchy."""
    from engine.layout_contracts import controls,control_font_floor
    from engine.source_structure import _transform
    from engine.nested_structure import photo_caption_panels
    captions=photo_caption_panels(source)
    old={e['id']:e for e in source['elements']};out={e['id']:e for e in items}
    texts=[e for e in source['elements'] if e['type']=='text' and e['id'] in out]
    if not texts:return
    title=max(texts,key=lambda e:e['font_size']);target=out[title['id']]
    for pid,tid in controls(source):
        if pid not in out or tid not in out or tid==title['id'] or pid in captions:continue
        label=out[tid];limit=max(control_font_floor(source,tid,width,height),
            target['font_size']*old[tid]['font_size']/max(1.,title['font_size'])*1.35)
        if label['font_size']<=limit:continue
        pair=[out[pid],label];b=bounds(pair);s=limit/label['font_size']
        adjusted=_transform(pair,s,(b[0]+b[2]-(b[2]-b[0])*s)/2,
            (b[1]+b[3]-(b[3]-b[1])*s)/2,b)
        for e in adjusted:out[e['id']].update(e)


def _fit_flow(blocks,slot,source,axis,align='left'):
    """One intrinsic-size flow of atomic containers, without leaf deformation."""
    from engine.source_structure import _transform
    x,y,w,h=slot
    if not blocks or min(w,h)<=0:return []
    sw,sh=source['canvas']['width'],source['canvas']['height']
    source_axis=0 if sw/sh>=1.6 else 1
    ordered=sorted(blocks,key=lambda es:((bounds(es)[source_axis]+bounds(es)[source_axis+2])/2,
                                         bounds(es)[1-source_axis]))
    boxes=[bounds(es) for es in ordered];gap=.035*min(w,h)
    extents=[b[axis+2]-b[axis] for b in boxes];cross=[b[3-axis]-b[1-axis] for b in boxes]
    main=w if axis==0 else h;other=h if axis==0 else w
    scale=min((main-gap*(len(boxes)-1))/max(1.,sum(extents)),other/max(1.,max(cross)))
    if scale<=0:return []
    used=sum(extents)*scale+gap*(len(boxes)-1);cursor=(x if axis==0 else y)+(main-used)/2;output=[]
    for es,b,extent,c in zip(ordered,boxes,extents,cross):
        offset=(other-c*scale)/2 if align=='center' else 0.
        left,top=(cursor,y+offset) if axis==0 else (x+offset,cursor)
        output+=_transform(es,scale,left,top,b)
        cursor+=extent*scale+gap
    return output


def _prepared(source):
    """Tighten only unused text-frame space, preserving the painted anchor."""
    from engine.resize import _render_text_ink
    result=deepcopy(source)
    for e in result['elements']:
        if e['type']!='text':continue
        ink=_render_text_ink(e)
        if ink:
            e['height']=max(1.,ceil(ink[3])+2.)
            e['width']=max(e['width'],ceil(ink[2])+2.)
    return result


def _partition(blocks,axis,n):
    """Whitespace cuts cannot sever an atomic card, photo badge or caption."""
    ordered=sorted(blocks,key=lambda es:(bounds(es)[axis]+bounds(es)[axis+2])/2)
    if len(ordered)<n:return []
    legal=[]
    for i in range(1,len(ordered)):
        left=max(bounds(es)[axis+2] for es in ordered[:i]);right=min(bounds(es)[axis] for es in ordered[i:])
        if right>=left-2.:legal.append((i,right-left))
    result=[]
    for cuts in combinations(legal,n-1):
        indices=[0]+[v[0] for v in cuts]+[len(ordered)]
        bands=[ordered[indices[i]:indices[i+1]] for i in range(n)]
        # Prefer substantial cuts while avoiding a lone footer dominating a cell.
        sizes=[sum((bounds(es)[2]-bounds(es)[0])*(bounds(es)[3]-bounds(es)[1]) for es in band) for band in bands]
        score=sum(v[1] for v in cuts)/max(1.,sum(sizes)**.5)-max(sizes)/max(1.,min(sizes))*.02
        result.append((score,bands))
    return [bands for _,bands in sorted(result,key=lambda p:-p[0])[:3]]


def _fit_band(blocks,slot,source,align='left'):
    """Auto-layout between locked atoms; child geometry stays source-relative."""
    from engine.source_structure import _transform
    from engine.resize import _set_font_size,_fit_text
    x,y,w,h=slot;sw,sh=source['canvas']['width'],source['canvas']['height']
    # A source row stays a row inside a band when its atom centers align.
    ordered=sorted(blocks,key=lambda es:(bounds(es)[1],bounds(es)[0]));rows=[]
    for es in ordered:
        b=bounds(es)
        if rows:
            prev=rows[-1];q=bounds([e for block in prev for e in block])
            if min(b[3],q[3])-max(b[1],q[1])>.3*min(b[3]-b[1],q[3]-q[1]):prev.append(es);continue
        rows.append([es])
    row_items=[[e for block in row for e in block] for row in rows];boxes=[bounds(es) for es in row_items]
    gap=min(w,h)*.045
    scale=min(w/max(1.,max(b[2]-b[0] for b in boxes)),
              (h-gap*(len(boxes)-1))/max(1.,sum(b[3]-b[1] for b in boxes)))
    if scale<=0:return []
    # Cap a tiny isolated footer/control at a modest multiple of area scale.
    scale=min(scale,sqrt(w*h/(max(1.,sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes))))*1.35)
    used=sum((b[3]-b[1])*scale for b in boxes)+gap*(len(boxes)-1)
    cursor=y+(h-used)/2;result=[]
    for es,b in zip(row_items,boxes):
        left=x+(w-(b[2]-b[0])*scale)/2 if align=='center' else x
        local=_transform(es,scale,left,cursor,b)
        for e in local:
            if e['type']=='text':
                # No canvas-relative font inflation inside an intact card.
                _set_font_size(e,_fit_text(e,e['width'],e['height'],e['font_size']))
        result+=local;cursor+=(b[3]-b[1])*scale+gap
    return result



def measured_candidates(source, blocks, box, width, height, proposal=None):
    """Coarse model bands and source-order measured packing share one realizer.

    Predicted leaf dimensions/font guesses never enter realization. Substantial
    source leaves vote on their group's proposed center; thin rules do not.
    Raw Qwen candidates remain available independently in the main selector.
    """
    from engine.container_flow import measured_columns
    # Complete explicit siblings may change row count together. Their backing
    # panels remain separate atoms; regional paint never becomes the parent.
    from collections import defaultdict
    siblings=defaultdict(list)
    for es in blocks:
        ids={e.get('group_id') for e in es}
        if len(ids)!=1 or None in ids or len(es)<2:continue
        signature=tuple(sorted(e['type'] for e in es))
        parents={e.get('parent_group_id') for e in es}
        key=('parent',next(iter(parents))) if len(parents)==1 and None not in parents else ('signature',signature)
        siblings[key].append(es)
    for family in siblings.values():
        if len(family)<3:continue
        bs=[bounds(es) for es in family]
        areas=[(b[2]-b[0])*(b[3]-b[1]) for b in bs]
        if max(areas)>3*max(1.,min(areas)):continue
        xs=[(b[0]+b[2])/2 for b in bs];ys=[(b[1]+b[3])/2 for b in bs]
        if min((max(xs)-min(xs))/source['canvas']['width'],(max(ys)-min(ys))/source['canvas']['height'])>.04:continue
        members={e['id'] for es in family for e in es}
        blocks=[es for es in blocks if not any(e['id'] in members for e in es)]+[sum(family,[])]
    # Use existing source peer evidence so a repeated row gets one allocation.
    # Joining is legal only when every peer unit is already a complete atom.
    from engine.resize import _analyze
    by_ids={frozenset(e['id'] for e in es):es for es in blocks};peer_rows={}
    for unit in _analyze(source)[1]:
        if unit.get('peer_ids'):
            ids=frozenset(e['id'] for e in unit['items'])
            if ids in by_ids:peer_rows.setdefault(tuple(unit['peer_ids']),[]).append(ids)
    for key,row in peer_rows.items():
        if len(row)!=len(key) or len(row)<2:continue
        members=[e for ids in row for e in by_ids[ids]]
        blocks=[es for es in blocks if frozenset(e['id'] for e in es) not in row]+[members]
    if not 4 <= len(blocks) <= 40:
        return []
    area=source['canvas']['width']*source['canvas']['height']
    # Small independent artwork does not become a copy column. Place it in
    # proven free space after the primary flow; never fall back onto content.
    floating=[block for block in blocks if not any(e['type']=='text' or
        e['type']=='image' and e['width']*e['height']>.025*area for e in block)]
    blocks=[b for b in blocks if b not in floating]
    if len(blocks)<4:return []
    sw, sh = source['canvas']['width'], source['canvas']['height']
    reference = sqrt(width * height / (sw * sh))
    source_axis = 0 if sw / sh >= 1.6 else 1
    ordered = sorted(blocks, key=lambda es: (bounds(es)[source_axis], bounds(es)[1-source_axis]))
    predictions = {e['id']: e for e in proposal or []}
    centers = []
    for block in blocks:
        votes = []
        for e in block:
            q = predictions.get(e['id'])
            if q is None or not all(k in q for k in ('x', 'y', 'width', 'height')):
                continue
            if e['type'] != 'text' and min(e['width'], e['height']) < max(4., .008 * min(sw, sh)):
                continue
            # Source mass, never a noisy predicted giant rectangle, sets weight.
            weight = max(1., e['width'] * e['height'])
            votes.append((q['x'] + q['width']/2, q['y'] + q['height']/2, weight))
        if not votes:
            centers = []
            break
        def weighted_median(axis):
            values = sorted((v[axis], v[2]) for v in votes);half = sum(v[1] for v in values)/2;total = 0.
            for value, weight in values:
                total += weight
                if total >= half:
                    return value
        centers.append((weighted_median(0), weighted_median(1), block))
    result = []
    for columns in (range(2, min(6, len(blocks)) + 1) if width/height>=1.35 else range(1,min(2,len(blocks))+1)):
        orders = [('macro-measured-columns', ordered)]
        if centers and width/height>=1.35:
            sorted_centers = sorted(centers, key=lambda p: (p[0], p[1]))
            cuts = sorted(sorted(range(1, len(centers)),
                key=lambda i: sorted_centers[i][0]-sorted_centers[i-1][0], reverse=True)[:columns-1])
            bands = [0] + cuts + [len(centers)]
            model_order = [p[2] for a,b in zip(bands,bands[1:]) for p in sorted(sorted_centers[a:b], key=lambda p:p[1])]
            # Keep the model's band membership as well as its coarse order.
            orders.append(('qwen-macro-band-flow', model_order))
        for family, order in orders:
            for multiplier in (.8, 1., 1.2):
                if family.startswith('qwen-'):
                    items = []
                    x,y,w,h = box;gap=max(3., .045*min(w,h));cw=(w-gap*(columns-1))/columns
                    for i,(a,b) in enumerate(zip(bands,bands[1:])):
                        group = [p[2] for p in sorted(sorted_centers[a:b],key=lambda p:p[1])]
                        fitted = measured_columns(group,(x+i*(cw+gap),y,cw,h),source,reference*multiplier,(width,height),1)
                        if fitted is None:
                            items = None
                            break
                        items += fitted
                else:
                    items = measured_columns(order,box,source,reference*multiplier,(width,height),columns)
                if items:
                    result.append(dict(name=f'{family}-{columns}-{multiplier}', elements=items,
                        energy=0., macro_locked=True, source_attached=True,
                        learned_macro=family.startswith('qwen-'), macro_family=family,
                        macro_realization=dict(columns=columns,font_scale=reference*multiplier,
                            order=[[e['id'] for e in es] for es in order])))
    if centers and width/height<1.35:
        # The same group law also applies to tall/square outputs: model Y
        # clusters allocate bands; measured X packing realizes their children.
        sorted_centers=sorted(centers,key=lambda p:(p[1],p[0]))
        x,y,w,h=box;gap=max(3., .045*min(w,h))
        for rows in range(2,min(4,len(blocks))+1):
            cuts=sorted(sorted(range(1,len(centers)),key=lambda i:sorted_centers[i][1]-sorted_centers[i-1][1],reverse=True)[:rows-1])
            bands=[0]+cuts+[len(centers)];bh=(h-gap*(rows-1))/rows
            for multiplier in (.8,1.,1.2):
                items=[];order=[]
                for i,(a,b) in enumerate(zip(bands,bands[1:])):
                    group=[p[2] for p in sorted(sorted_centers[a:b],key=lambda p:p[0])];order+=group
                    fitted=None
                    for columns in range(1,min(3,len(group))+1):
                        fitted=measured_columns(group,(x,y+i*(bh+gap),w,bh),source,reference*multiplier,(width,height),columns)
                        if fitted:break
                    if not fitted:
                        items=None;break
                    items+=fitted
                if items:
                    result.append(dict(name=f'qwen-macro-band-flow-rows-{rows}-{multiplier}',elements=items,
                        energy=0.,macro_locked=True,source_attached=True,learned_macro=True,
                        macro_family='qwen-macro-band-flow',macro_realization=dict(rows=rows,
                            font_scale=reference*multiplier,order=[[e['id'] for e in es] for es in order])))
    if floating:
        from engine.free_space import free_rectangles,place
        from engine.source_structure import _transform
        complete=[]
        for c in result:
            items=c['elements'];ok=True
            for block in floating:
                b=bounds(block);swid=max(1.,b[2]-b[0]);shei=max(1.,b[3]-b[1])
                anchor=((b[0]+b[2])/2/sw*width,(b[1]+b[3])/2/sh*height)
                slots=free_rectangles(width,height,[bounds([e]) for e in items],margin=.025*min(width,height),padding=2.)
                fitted=place((swid,shei),slots,anchor,(width,height),scale_cap=reference)
                if not fitted or fitted['scale']<.15*reference:ok=False;break
                items+=_transform(block,fitted['scale'],fitted['x'],fitted['y'],b)
            if ok:complete.append(c)
        result=complete
    return result

def candidates(source,width,height,proposal=None,*,tight=None,extras=True):
    from engine.resize import _analyze,_energy
    prepared=_prepared(source);spec=composites(source,tight=tight)
    ready={e['id']:e for e in prepared['elements']}
    blocks=[[ready[e['id']] for e in block] for block in spec['blocks']]
    if len(blocks)<2:
        sw,sh=source['canvas']['width'],source['canvas']['height']
        shift=max((width/height)/(sw/sh),(sw/sh)/(width/height))
        return candidates(source,width,height,proposal,tight=False,extras=extras) if tight is not False and shift>=1.8 and len(composites(prepared,tight=False)['blocks'])>1 else []
    sw,sh=source['canvas']['width'],source['canvas']['height'];axis=0 if sw/sh>1.6 else 1
    margin=.045*min(width,height);gap=.04*min(width,height)
    edges=edge_strips(source);src={e['id']:e for e in source['elements']}
    left=max([margin]+[src[i]['width']/sw*width+gap for i,side in edges.items() if side=='left'])
    right=max([margin]+[src[i]['width']/sw*width+gap for i,side in edges.items() if side=='right'])
    top=max([margin]+[src[i]['height']/sh*height+gap for i,side in edges.items() if side=='top'])
    bottom=max([margin]+[src[i]['height']/sh*height+gap for i,side in edges.items() if side=='bottom'])
    w=width-left-right;h=height-top-bottom;result=[]
    ambient=_ambient_layers(source,width,height,spec.get('ambient_ids',[]))
    # Additional intrinsic flows retain the existing band candidates. All
    # source-backed cards, rotated labels and media attachments stay atomic.
    for flow_axis in (0,1):
        for align in ('left','center'):
            items=_fit_flow(blocks,(left,top,w,h),source,flow_axis,align)
            if not items:continue
            items+=deepcopy(ambient)+[deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
            clamp_edge_strips(items,source,width,height)
            from engine.layout_contracts import center_control_labels
            center_control_labels(items,source,width,height)
            result.append(dict(name=f'macro-intrinsic-flow-{flow_axis}-{align}',elements=items,
                               energy=0.,macro_locked=True,source_attached=True,macro_links=spec['links']))
    arrangements=[(n,1) for n in (2,3,4)] if width/height>=1.35 else [(1,n) for n in (2,3,4)]+([(2,2)] if width/height>.65 else [])
    seen=set()
    for cols,rows in arrangements:
        for k,bands in enumerate(_partition(blocks,axis,cols*rows)):
            for align in ('left','center'):
                items=[];cw=(w-gap*(cols-1))/cols;ch=(h-gap*(rows-1))/rows
                for i,band in enumerate(bands):items+=_fit_band(band,(left+(i%cols)*(cw+gap),top+(i//cols)*(ch+gap),cw,ch),source,align)
                items += deepcopy(ambient)+[deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
                clamp_edge_strips(items,source,width,height)
                from engine.layout_contracts import _grow_controls,center_control_labels
                _grow_controls(items,source,width,height)
                center_control_labels(items,source,width,height)
                signature=tuple((e['id'],round(e['x'],1),round(e['y'],1),round(e['font_size'],1) if e['type']=='text' else 0) for e in items)
                if signature in seen:continue
                seen.add(signature)
                result.append(dict(name=f'macro-band-reflow-{cols}x{rows}-{k}-{align}',elements=items,
                                   energy=0.,macro_locked=True,source_attached=True,macro_links=spec['links']))
    # An overlapping source hero prevents a horizontal whitespace cut through
    # its height. Offer a separate media column without severing its overlays.
    media=[es for es in blocks if any(e['type']=='image' and e['width']*e['height']>.08*sw*sh for e in es)]
    if len(media)==1 and width/height>=1.35:
        hero=media[0];rest=[es for es in blocks if es is not hero]
        for share in (.18,.28):
            for n in (2,3):
                for k,bands in enumerate(_partition(rest,1,n)[:2]):
                    items=_fit_band([hero],(left,top,w*share,h),source,'center')
                    space=w*(1-share)-gap;cw=(space-gap*(n-1))/n
                    for i,band in enumerate(bands):items+=_fit_band(band,(left+w*share+gap+i*(cw+gap),top,cw,h),source,'left')
                    items += deepcopy(ambient)+[deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
                    clamp_edge_strips(items,source,width,height)
                    from engine.layout_contracts import _grow_controls,center_control_labels
                    _grow_controls(items,source,width,height);center_control_labels(items,source,width,height)
                    result.append(dict(name=f'macro-media-split-{share}-{n}-{k}',elements=items,energy=0.,macro_locked=True,source_attached=True))
    for c in measured_candidates(source, blocks, (left,top,w,h), width, height, proposal):
        c['elements'] += deepcopy(ambient)+[deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
        clamp_edge_strips(c['elements'],source,width,height)
        from engine.layout_contracts import center_control_labels
        center_control_labels(c['elements'],source,width,height)
        c['macro_links']=spec['links']
        result.append(c)
    additions=[]
    if extras and os.environ.get('MAGICANVAS_EXPERIMENTAL_MACRO_LIBRARY')=='1':
        sw,sh=source['canvas']['width'],source['canvas']['height']
        shift=max((width/height)/(sw/sh),(sw/sh)/(width/height))
        signature=lambda spec:{frozenset(e['id'] for e in es) for es in spec['blocks']}
        locked=composites(prepared,tight=True)
        if shift>=1.8:
            for mode in (True,False):
                alternative=composites(prepared,tight=mode)
                if signature(alternative)==signature(spec):continue
                for c in candidates(source,width,height,proposal,tight=mode,extras=False)[:1]:
                    c['name']=('macro-locked-' if mode else 'band-split-')+c['name']
                    c['macro_family']='dual-granularity';c['grouping']='tight' if mode else 'band-split';additions.append(c)
        from engine.retrieve import candidates as retrieved
        for c in retrieved(prepared,locked['blocks'],width,height):
            c['elements']+=deepcopy(ambient)+[deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
            clamp_edge_strips(c['elements'],source,width,height)
            from engine.layout_contracts import center_control_labels
            center_control_labels(c['elements'],source,width,height)
            additions.append(c)
    # Geometry quality preselects a bounded diverse render budget, not an oracle.
    _,units=_analyze(source)
    from engine.resize import _assemble
    backgrounds=_analyze(source)[0]
    for c in result:
        _cap_control_growth(c['elements'],source,width,height)
        from engine.nested_structure import regional_surfaces
        if regional_surfaces(source):
            from engine.layout_contracts import _primary_axis
            _primary_axis(c['elements'],source,units,width,height,apply=True)
        _fit_regional_surfaces(c['elements'],source,width,height)
        c['precheck']=checks(source,_assemble(c['elements'],backgrounds,source,width,height),units,{},c['name'])
    ranked=sorted(result,key=lambda c:(c['precheck']['flagged'],len(c['precheck']['defect_flags']),quality(c['elements'],source,width,height)['cost']));chosen=[];families=set()
    # Reserve measured and learned macro alternatives without starving the
    # existing families or allowing an unbounded render sweep.
    for family in ('qwen-macro-band-flow','macro-measured-columns'):
        match=next((c for c in ranked if c.get('macro_family')==family),None)
        if match:chosen.append(match)
    for c in ranked:
        family=c.get('macro_family') or c['name'].split('-')[3]
        if c not in chosen and family not in families:
            chosen.append(c);families.add(family)
        if len(chosen)>=8:break
    for c in ranked:
        if len(chosen)>=8:break
        if c not in chosen:chosen.append(c)
    # Supplemental families do not evict the existing eight macro slots.
    # Render at most two retrieved and two alternate-granularity proposals.
    for family,limit in (('human-retrieved',2),('dual-granularity',2)):
        pool=[c for c in additions if c.get('macro_family')==family]
        chosen+=sorted(pool,key=lambda c:quality(c['elements'],source,width,height)['cost'])[:limit]
    return chosen


def _union_area(boxes):
    """Rectangle union area, so overlapping backplates cannot inflate fill."""
    xs=sorted({b[0] for b in boxes}|{b[2] for b in boxes});area=0.
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in boxes if b[0]<right and b[2]>left)
        bottom=None;covered=0.
        for top,end in intervals:
            if bottom is None or top>bottom:covered+=max(0.,end-top);bottom=end
            elif end>bottom:covered+=end-bottom;bottom=end
        area+=(right-left)*covered
    return area


def quality(elements,source,width,height):
    """Geometry/ink-visible quality; excluded edge strips cannot inflate occupancy."""
    from engine.resize import _analyze,_text_visible_bounds,_overlap
    from engine.geometry_policy import ambient_shape
    from engine.nested_structure import regional_surfaces,unpainted_ids
    bg,units=_analyze(source);excluded={e['id'] for e in bg}|set(edge_strips(source))|set(regional_surfaces(source))|unpainted_ids(source)|{e['id'] for e in source['elements'] if ambient_shape(e,source)};original={e['id']:e for e in source['elements']};out={e['id']:e for e in elements}
    content=[e for e in elements if e['id'] not in excluded];texts=[e for e in content if e['type']=='text' and e.get('content','').strip()]
    if not content:return dict(cost=1000.,width_use=0.,height_use=0.)
    bs=[_text_visible_bounds(e) if e['type']=='text' else bounds([e]) for e in content]
    box=(min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs));wx=(box[2]-box[0])/width;hy=(box[3]-box[1])/height
    area_scale=sqrt(width*height/(source['canvas']['width']*source['canvas']['height']))
    ratios=[e['font_size']/max(1.,original[e['id']]['font_size']*area_scale) for e in texts]
    minfont=min((e['font_size'] for e in texts),default=0.);medfont=median([e['font_size'] for e in texts]) if texts else 0.
    # Supporting/legal annotations contribute less than major copy, but cannot
    # make a microscopic headline acceptable merely by spreading corner marks.
    major=sorted(texts,key=lambda e:-original[e['id']]['font_size'])[:max(1,len(texts)//2)]
    headline=max(texts,key=lambda e:original[e['id']]['font_size']) if texts else None
    photos=[e for e in source['elements']
            if e['id'] not in excluded and e['type']=='image']
    hero=max(photos,key=lambda e:e['width']*e['height']) if photos else None
    target_hero=out.get(hero['id']) if hero else None
    hero_share=(target_hero['width']*target_hero['height']/(width*height)) if target_hero else 0.
    floor=min(16.,max(5.,min(width,height)*.024));read=sum(max(0.,1.-e['font_size']/floor) for e in major)/max(1,len(major))
    links=composites(source)['links'];orphans=0;drift=0.
    for a,b,kind in links:
        if a not in out or b not in out:orphans+=1;continue
        if kind=='explicit':continue
        p,q=out[a],out[b];oa,ob=original[a],original[b]
        if min(oa['width'],oa['height'])<=0:continue
        s=min(p['width']/oa['width'],p['height']/oa['height'])
        distance=((q['x']-p['x']-(ob['x']-oa['x'])*s)**2+(q['y']-p['y']-(ob['y']-oa['y'])*s)**2)**.5
        normal=max(8.,min(width,height)*.04);drift+=min(1.,distance/normal)
        if distance>normal:orphans+=1
    balance=(abs(box[0]-(width-box[2]))/width+abs(box[1]-(height-box[3]))/height)/2
    cost=35.*(max(0.,.72-wx)+max(0.,.72-hy))+35.*read+20.*max(0.,.65-median(ratios))/.65 if ratios else 100.
    local_axes=subpanels(elements,source);source_gap=clearance(elements,source)
    center_error=0.
    source_items=[e for e in source['elements'] if e['id'] not in excluded]
    if center_evidence(source,excluded) and source_items:
        sb=bounds(source_items)
        if abs((sb[0]+sb[2])/2-source['canvas']['width']/2)<.08*source['canvas']['width'] and wx<.6 and abs((box[0]+box[2])/2-width/2)<.15*width:
            center_error=abs((box[0]+box[2])/2-width/2)/width
    cost+=18.*drift/max(1,len(links))+8.*orphans/max(1,len(links))+35.*local_axes+10.*source_gap+50.*center_error
    from engine.layout_contracts import controls
    from engine.nested_structure import photo_caption_panels
    captions=photo_caption_panels(source)
    prominence=0.
    if headline and headline['font_size']>0:
        for p,t in controls(source):
            if t not in out or p in captions:continue
            expected=original[t]['font_size']/original[headline['id']]['font_size']
            actual=out[t]['font_size']/headline['font_size']
            prominence+=max(0.,actual/max(.01,expected)-1.5)
    cost+=35.*prominence
    return dict(cost=cost,width_use=wx,height_use=hy,content_area_share=sum(max(0,b[2]-b[0])*max(0,b[3]-b[1]) for b in bs)/(width*height),
                content_union_share=_union_area(bs)/(width*height),
                hero_area_share=hero_share,headline_font=headline['font_size'] if headline else 0.,
                control_prominence_error=prominence,subpanel_axis_error=local_axes,source_gap_deficit=source_gap,near_center_error=center_error,median_font=medfont,minimum_font=minfont,relative_font_median=median(ratios) if ratios else 0.,orphaned_attachments=orphans,attachment_drift=drift,margin_imbalance=balance)


def upgrade(incumbent,visual,backgrounds,units,source,width,height,desired,render_rank,proposal=None):
    """Only a measured, materially better sibling can replace the incumbent."""
    from engine.resize import _assemble
    old_doc=_assemble(incumbent['elements'],incumbent.get('backgrounds',backgrounds),source,width,height)
    old_checks=checks(source,old_doc,units,visual,incumbent['name'])
    before=quality(incumbent['elements'],source,width,height)
    sw,sh=source['canvas']['width'],source['canvas']['height'];shift=max((width/height)/(sw/sh),(sw/sh)/(width/height))
    # Do not regenerate familiar clean layouts by default. Complex, poor
    # reflows must demonstrate a quality shortfall or an actual safety defect.
    local=local_variants(incumbent['elements'],source,width,height)
    if incumbent['name'].startswith('qwen-'):
        for c in local:c['name']='qwen-'+c['name']
    flow_opportunity=(shift>=1.6 and min(before['width_use'],before['height_use'])<.60
                      and before.get('relative_font_median',1.)<.85)
    eligible=bool(local) or (len(source['elements'])>=14 and (old_checks['flagged'] or flow_opportunity or shift>=1.6 and (before['cost']>20. or before.get('orphaned_attachments',0)>=2 or min(before['width_use'],before['height_use'])<.45)))
    diagnostics=dict(considered=eligible,incumbent=incumbent['name'],before=before,old_flags=old_checks['defect_flags'],candidates=[])
    if not eligible:return incumbent,visual,diagnostics
    alternatives=local+(candidates(source,width,height,proposal) if len(source['elements'])>=14 else [])
    if not alternatives:return incumbent,visual,diagnostics
    for c in alternatives:
        winner,report=render_rank([c],backgrounds,units,source,width,height,desired)
        if winner['name']!=c['name'] or not report.get('used'):continue
        doc=_assemble(winner['elements'],backgrounds,source,width,height);gate=checks(source,doc,units,report,winner['name'])
        score=quality(winner['elements'],source,width,height)
        item=dict(name=c['name'],flags=gate['defect_flags'],quality=score)
        diagnostics['candidates'].append(item)
        if c.get('macro_family')=='human-retrieved':
            item['retrieval']=c['retrieval']
            # A library proposal must not win by erasing primary prominence.
            if any(before.get(key,0.)>0. and score.get(key,0.)<.70*before[key]
                   for key in ('headline_font','hero_area_share')):
                item['rejection_reason']='Retrieved layout loses more than 30% primary prominence'
                continue
        # A previously clean layout can only be replaced by another clean
        # layout with at least 20% quality improvement and a two-point margin.
        critical=set(gate['defect_flags'])-{'control_readability'}
        if critical:continue
        from engine.layout_contracts import issues
        from engine.layout_contracts import control_font_floor
        def readability_deficit(es):
            return sum(d.get('amount',0.)/max(1.,control_font_floor(source,d['ids'][1],width,height))
                       for d in issues(es,source,width,height,units) if d['kind']=='control_readability')
        deficit=readability_deficit(winner['elements'])
        item['readability_deficit']=deficit
        # A bounded readable-but-imperfect fallback may replace an already
        # defective result. Bounds, ink overlap/clipping, attachment collisions
        # and contrast remain mandatory. Never replace a clean result this way.
        tier=1 if gate['flagged'] else 0
        if tier and (not old_checks['flagged'] or score['cost']+25.*deficit >= before['cost']+25.*readability_deficit(incumbent['elements'])):continue
        if not old_checks['flagged'] and score['cost']>=before['cost']-max(2.,before['cost']*.15):continue
        # Median type size overweights tiny annotations. The measured quality
        # cost already checks readability; use occupied union area here so a
        # larger hero/heading can beat a sparse layout with larger footnotes.
        if (not old_checks['flagged'] and
                score.get('content_union_share',0.)<.90*before.get('content_union_share',0.)):
            item['rejection_reason']='Clean incumbent retains larger occupied content area'
            continue
        # Reflow can trade headline size for a much larger hero. Reject joint
        # shrinkage, not every individual trade-off in a fixed-size canvas.
        hero_lost=(before.get('hero_area_share',0.)==0. or
                   score.get('hero_area_share',0.)<.90*before['hero_area_share'])
        if (not old_checks['flagged'] and hero_lost and
                score.get('headline_font',0.)<.90*before.get('headline_font',0.)):
            item['rejection_reason']='Clean incumbent retains stronger combined primary prominence'
            continue
        selection=(tier,score['cost']+25.*deficit)
        if tuple(diagnostics.get('selected_key',(float('inf'),float('inf'))))<=selection:continue
        diagnostics.update(selected=c['name'],selected_cost=selection[1],selected_key=list(selection),remaining_flags=gate['defect_flags']);best=(winner,report)
    if 'selected' in diagnostics:
        winner,report=best;report['prior_selection']=dict(winner=incumbent['name'],flags=old_checks['defect_flags'],quality=before)
        return winner,report,diagnostics
    return incumbent,visual,diagnostics


def checks(source,doc,units,visual,name):
    from engine.main import self_check
    from engine.resize import _objective_layout_cost,_source_occlusion_defects,_unintended_text_overlaps,_unintended_control_overlaps
    from engine.layout_contracts import issues
    w,h=doc['canvas']['width'],doc['canvas']['height'];es=doc['elements']
    flags=[d['kind'] for d in issues(es,source,w,h,units)]
    if self_check(source,doc):flags.append('structural_check')
    from engine.raster_metrics import glyph_overflow
    if glyph_overflow(source,es,w,h):flags.append('glyph_overflow')
    if _unintended_text_overlaps(es,units):flags.append('text_on_text')
    if _unintended_control_overlaps(es,source):flags.append('control_collision')
    if _source_occlusion_defects(es,source):flags.append('text_hero_occlusion')
    if _objective_layout_cost(es,units,source,w,h,1.)>1. and not flags:flags.append('glyph_geometry')
    row=next((r for r in visual.get('candidates',[]) if r['name']==name),{})
    if row.get('invisible_text'):flags.append('invisible_text')
    if row.get('contrast_regressions'):flags.append('contrast_below_source_floor')
    return dict(flagged=bool(flags),defect_flags=sorted(set(flags)))


@lru_cache(maxsize=128)
def image_cells(path):
    """Find stroked rectangular subpanels in a sparse transparent frame asset."""
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            im=im.convert('RGBA');im.thumbnail((900,900));a=np.asarray(im)[:,:,3]>128
        h,w=a.shape
        if a.mean()>.2 or min(w,h)<30:return ()
        def centers(mask):
            runs=[];start=None
            for i,v in enumerate(list(mask)+[False]):
                if v and start is None:start=i
                if not v and start is not None:runs.append((start+i-1)/2);start=None
            return runs
        xs=centers(a.mean(axis=0)>.85)
        if len(xs)<2:return ()
        result=[]
        for left,right in zip(xs,xs[1:]):
            if right-left<.08*w:continue
            ys=centers(a[:,round(left):round(right)+1].mean(axis=1)>.85)
            for top,bottom in zip(ys,ys[1:]):
                if bottom-top>.10*h:result.append((left/w,top/h,right/w,bottom/h))
        return tuple(result) if 2<=len(result)<=8 else ()
    except (ImportError,OSError,ValueError):return ()


def subpanels(elements,source,apply=False):
    from engine.resize import _asset_root,_text_visible_bounds,_overlap
    original={e['id']:e for e in source['elements']};out={e['id']:e for e in elements};errors=[]
    for p in source['elements']:
        if p['type']!='image' or p['id'] not in out or abs(p.get('rotation',0))>.1 or not p.get('asset_id'):continue
        cells=image_cells(str(_asset_root(p)/p['asset_id']))
        if not cells:continue
        target=out[p['id']]
        for t in source['elements']:
            if t['type']!='text' or t['id'] not in out or t.get('z_index',0)<=p.get('z_index',0):continue
            old=_text_visible_bounds(t);area=max(1.,(old[2]-old[0])*(old[3]-old[1]));choices=[]
            for c in cells:
                b=(p['x']+c[0]*p['width'],p['y']+c[1]*p['height'],p['x']+c[2]*p['width'],p['y']+c[3]*p['height'])
                if _overlap(b,old)/area>=.85:choices.append((c,b))
            if len(choices)!=1:continue
            cell,b=choices[0];tb=(target['x']+cell[0]*target['width'],target['y']+cell[1]*target['height'],target['x']+cell[2]*target['width'],target['y']+cell[3]*target['height'])
            q=out[t['id']];ink=_text_visible_bounds(q)
            cx=tb[0]+((old[0]+old[2])/2-b[0])/max(1.,b[2]-b[0])*(tb[2]-tb[0]);cy=tb[1]+((old[1]+old[3])/2-b[1])/max(1.,b[3]-b[1])*(tb[3]-tb[1])
            dx=cx-(ink[0]+ink[2])/2;dy=cy-(ink[1]+ink[3])/2
            errors.append(abs(dx)/max(1.,tb[2]-tb[0])+abs(dy)/max(1.,tb[3]-tb[1]))
            if apply:
                # Source-relative painted axes retain the panel's own grid,
                # even when a global source-axis repair moved its leaves.
                q['x']+=dx;q['y']+=dy
    return sum(errors)


def clearance(elements,source,apply=False):
    from engine.resize import _analyze,_text_visible_bounds
    bg,_=_analyze(source);excluded={e['id'] for e in bg};out={e['id']:e for e in elements};errors=[]
    w,h=source['canvas']['width'],source['canvas']['height'];src={e['id']:e for e in source['elements']}
    for art in source['elements']:
        if art['id'] in excluded or art['type']!='image' or art['id'] not in out:continue
        a=bounds([art]);q=out[art['id']];b=bounds([q])
        required=[]
        for t in source['elements']:
            if t['type']!='text' or t['id'] not in out:continue
            old=_text_visible_bounds(t);new=_text_visible_bounds(out[t['id']]);gap=old[0]-a[2]
            if not 1.<gap<.1*min(w,h) or min(old[3],a[3])<=max(old[1],a[1]):continue
            if min(new[3],b[3])<=max(new[1],b[1]):continue
            wanted=max(3.,gap*min(q['width']/art['width'],q['height']/art['height']))
            missing=max(0.,wanted-(new[0]-b[2]))
            if missing>0:required.append(missing);errors.append(missing/max(3.,wanted))
        if apply and required:
            move=min(max(required),q['x'])
            # This sibling moves only an independently authored visual atom.
            if not any(art['id'] in {e['id'] for e in es} and len(es)>1 for es in composites(source)['blocks']):q['x']-=move
    return sum(errors)


def local_variants(elements,source,width,height):
    variants=[]
    for name,fn in [('subpanel-axes',subpanels),('source-copy-clearance',clearance)]:
        result=deepcopy(elements);before=fn(result,source)
        if before>.015:
            fn(result,source,True)
            if result!=elements:variants.append(dict(name='source-local-'+name,elements=result,energy=0.,macro_locked=True,source_attached=True))
    from engine.resize import _analyze
    bg,_=_analyze(source);exclude={e['id'] for e in bg}|set(edge_strips(source));old=[e for e in source['elements'] if e['id'] not in exclude];es=[e for e in elements if e['id'] not in exclude]
    if es and center_evidence(source,exclude):
        sb=bounds(old);b=bounds(es);sw=source['canvas']['width'];cx=(b[0]+b[2])/2
        if abs((sb[0]+sb[2])/2-sw/2)<.08*sw and abs(cx-width/2)<.15*width and b[2]-b[0]<.6*width and abs(cx-width/2)>.015*width:
            result=deepcopy(elements)
            for e in result:
                if e['id'] not in exclude:e['x']+=width/2-cx
            variants.append(dict(name='source-local-near-center',elements=result,energy=0.,macro_locked=True,source_attached=True))
    return variants


def center_evidence(source,excluded):
    if len(composites(source)['blocks'])<=3:return True
    es=[e for e in source['elements'] if e['id'] not in excluded]
    w=source['canvas']['width']
    return bool(es and len(es)<=10 and all(abs(e['x']+e['width']/2-w/2)<.10*w for e in es))
