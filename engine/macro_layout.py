"""Source-derived composite blocks and independently gated macro layouts.

No template identifiers, target reference coordinates or text strings drive
partitioning. Existing candidates stay intact; new layouts must earn selection.
"""
from copy import deepcopy
from functools import lru_cache
from itertools import combinations
from statistics import median
from math import sqrt,ceil
import json


def edge_strips(source):
    w,h=source['canvas']['width'],source['canvas']['height'];found={}
    for e in source['elements']:
        if e['type']!='shape' or abs(e.get('rotation',0))>.1:continue
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


def composites(source):
    return deepcopy(_composites(json.dumps(source,sort_keys=True,separators=(',',':'))))


@lru_cache(maxsize=256)
def _composites(payload):
    from engine.resize import _analyze,_overlap,_text_visible_bounds
    source=json.loads(payload);w,h=source['canvas']['width'],source['canvas']['height']
    bg,_=_analyze(source);excluded={e['id'] for e in bg}|set(edge_strips(source))
    src={e['id']:e for e in source['elements'] if e['id'] not in excluded};parent={i:i for i in src};links=[]
    # A previously established compact emblem is already a complete functional
    # unit. Whitespace between its leaves is not permission to split it.
    from engine.source_structure import groups as source_groups
    shared=source_groups(source)['members']
    if len(shared)==1 and set(shared[0])==set(src):
        box=bounds(list(src.values()))
        if (box[2]-box[0])*(box[3]-box[1])<=.65*w*h:
            return dict(blocks=[list(src.values())],links=[],edge_ids=list(edge_strips(source)))

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
    blocks={}
    for i,e in src.items():blocks.setdefault(root(i),[]).append(e)
    return dict(blocks=list(blocks.values()),links=links,edge_ids=list(edge_strips(source)))


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


def candidates(source,width,height):
    from engine.resize import _analyze,_energy
    prepared=_prepared(source);spec=composites(prepared);blocks=spec['blocks']
    if len(blocks)<2:return []
    sw,sh=source['canvas']['width'],source['canvas']['height'];axis=0 if sw/sh>1.6 else 1
    margin=.045*min(width,height);gap=.04*min(width,height)
    edges=edge_strips(source);src={e['id']:e for e in source['elements']}
    left=max([margin]+[src[i]['width']/sw*width+gap for i,side in edges.items() if side=='left'])
    right=max([margin]+[src[i]['width']/sw*width+gap for i,side in edges.items() if side=='right'])
    top=max([margin]+[src[i]['height']/sh*height+gap for i,side in edges.items() if side=='top'])
    bottom=max([margin]+[src[i]['height']/sh*height+gap for i,side in edges.items() if side=='bottom'])
    w=width-left-right;h=height-top-bottom;result=[]
    arrangements=[(n,1) for n in (2,3,4)] if width/height>=1.35 else [(1,n) for n in (2,3,4)]+([(2,2)] if width/height>.65 else [])
    seen=set()
    for cols,rows in arrangements:
        for k,bands in enumerate(_partition(blocks,axis,cols*rows)):
            for align in ('left','center'):
                items=[];cw=(w-gap*(cols-1))/cols;ch=(h-gap*(rows-1))/rows
                for i,band in enumerate(bands):items+=_fit_band(band,(left+(i%cols)*(cw+gap),top+(i//cols)*(ch+gap),cw,ch),source,align)
                items += [deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
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
                    items += [deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
                    clamp_edge_strips(items,source,width,height)
                    from engine.layout_contracts import _grow_controls,center_control_labels
                    _grow_controls(items,source,width,height);center_control_labels(items,source,width,height)
                    result.append(dict(name=f'macro-media-split-{share}-{n}-{k}',elements=items,energy=0.,macro_locked=True,source_attached=True))
    # Geometry quality preselects a bounded diverse render budget, not an oracle.
    _,units=_analyze(source)
    from engine.resize import _assemble
    backgrounds=_analyze(source)[0]
    for c in result:c['precheck']=checks(source,_assemble(c['elements'],backgrounds,source,width,height),units,{},c['name'])
    ranked=sorted(result,key=lambda c:(c['precheck']['flagged'],len(c['precheck']['defect_flags']),quality(c['elements'],source,width,height)['cost']));chosen=[];families=set()
    for c in ranked:
        family=c['name'].split('-')[3]
        if family not in families:chosen.append(c);families.add(family)
    for c in ranked:
        if c not in chosen:chosen.append(c)
        if len(chosen)>=8:break
    return chosen


def quality(elements,source,width,height):
    """Geometry/ink-visible quality; excluded edge strips cannot inflate occupancy."""
    from engine.resize import _analyze,_text_visible_bounds,_overlap
    bg,units=_analyze(source);excluded={e['id'] for e in bg}|set(edge_strips(source));original={e['id']:e for e in source['elements']};out={e['id']:e for e in elements}
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
    return dict(cost=cost,width_use=wx,height_use=hy,content_area_share=sum(max(0,b[2]-b[0])*max(0,b[3]-b[1]) for b in bs)/(width*height),
                subpanel_axis_error=local_axes,source_gap_deficit=source_gap,near_center_error=center_error,median_font=medfont,minimum_font=minfont,relative_font_median=median(ratios) if ratios else 0.,orphaned_attachments=orphans,attachment_drift=drift,margin_imbalance=balance)


def upgrade(incumbent,visual,backgrounds,units,source,width,height,desired,render_rank):
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
    eligible=bool(local) or (len(source['elements'])>=14 and (old_checks['flagged'] or shift>=1.6 and (before['cost']>20. or before.get('orphaned_attachments',0)>=2 or min(before['width_use'],before['height_use'])<.45)))
    diagnostics=dict(considered=eligible,incumbent=incumbent['name'],before=before,old_flags=old_checks['defect_flags'],candidates=[])
    if not eligible:return incumbent,visual,diagnostics
    alternatives=local+(candidates(source,width,height) if len(source['elements'])>=14 else [])
    if not alternatives:return incumbent,visual,diagnostics
    for c in alternatives:
        winner,report=render_rank([c],backgrounds,units,source,width,height,desired)
        if winner['name']!=c['name'] or not report.get('used'):continue
        doc=_assemble(winner['elements'],backgrounds,source,width,height);gate=checks(source,doc,units,report,winner['name'])
        score=quality(winner['elements'],source,width,height)
        item=dict(name=c['name'],flags=gate['defect_flags'],quality=score)
        diagnostics['candidates'].append(item)
        # A previously clean layout can only be replaced by another clean
        # layout with at least 20% quality improvement and a two-point margin.
        if gate['flagged']:continue
        if not old_checks['flagged'] and score['cost']>=before['cost']-max(2.,before['cost']*.15):continue
        if diagnostics.get('selected_cost',float('inf'))<=score['cost']:continue
        diagnostics.update(selected=c['name'],selected_cost=score['cost']);best=(winner,report)
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
