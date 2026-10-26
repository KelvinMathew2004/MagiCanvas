"""Measured semantic bands for source-format-independent brand reflow.

This is an independent fallback family; it does not replace a clean model
proposal. Whole source units, rather than independent leaves, own each band.
"""
from copy import deepcopy
from math import sqrt


def stack_seed(source,units,width,height):
    from engine.resize import _unit,_row,_copy,_footer_band,_fit_raw_block,_raw_bounds,_shift,_decorations,_bounds
    if width/height>=1.35:return None
    if not any(u.get('semantic')=='logo' for u in units) or not any(u.get('semantic')=='hero' for u in units):return None
    roles={r:[u for u in units if u['role']==r and not (u.get('corner_callout') and u['role']=='decoration')] for r in ['header','visual','copy','tag','action','footer','decoration']}
    margin=.045*min(width,height);gap=.018*min(width,height);cw=width-2*margin
    scale=sqrt(width*height/(source['canvas']['width']*source['canvas']['height']))
    blocks=[]
    for role,share in [('header',.075),('visual',.26),('tag',.10),('copy',.28),('action',.08),('footer',.08)]:
        members=deepcopy(roles[role]);room=height*share
        if not members:continue
        if role=='copy':items=_copy(members,cw,room,scale*1.5,gap,align='center')
        elif role=='footer':items,_=_footer_band(members,cw,room,scale,gap)
        elif role in ('header','visual'):
            items=[];cursor=0.;per=room/len(members)
            for u in members:
                local=_unit(u,cw,per,scale*2,fill=True);local=_fit_raw_block(local,cw,per);b=_raw_bounds(local)
                items+=_shift(local,(cw-b[2]+b[0])/2-b[0],cursor-b[1]);cursor+=b[3]-b[1]+gap
        else:
            for u in members:u['canvas_scale']=scale
            if role=='tag' and width/height<=.4:
                items=[];cursor=0.;per=(room-gap*(len(members)-1))/len(members)
                for u in members:
                    local=_fit_raw_block(_unit(u,cw,per,scale),cw,per);b=_raw_bounds(local)
                    items+=_shift(local,(cw-b[2]+b[0])/2-b[0],cursor-b[1]);cursor+=b[3]-b[1]+gap
            else:items,_=_row(members,cw,room,scale,gap)
        if items:blocks.append(items)
    result=[]
    for block in blocks:result+=block
    result+=_decorations([u for u in units if u['role']=='decoration'],width,height,source['canvas']['width'],source['canvas']['height'],result)
    return resolve_stack(result,source,units,width,height)


def resolve_stack(elements,source,units,width,height):
    """Pack measured atomic rows after glyph/control growth; preserve each row."""
    from engine.resize import _raw_bounds,_bounds,_shift,_set_font_size
    out={e['id']:e for e in elements};bands=[];seen=set();corner=[]
    order={'header':0,'visual':1,'tag':2,'copy':3,'action':4,'footer':5,'decoration':6}
    for u in sorted(units,key=lambda u:(order.get(u['role'],6),u['bounds'][1])):
        members=[out[e['id']] for e in u['items'] if e['id'] in out and e['id'] not in seen]
        if not members:continue
        seen.update(e['id'] for e in members)
        if u.get('corner_callout') and u['role']=='decoration':corner+=members;continue
        if u['role']=='decoration' or u.get('backing_panel'):continue
        bands.append((u,members))
    # Peer badges retain a shared row except on true skyscrapers.
    if width/height>.4:
        tags=[(u,b) for u,b in bands if u['role']=='tag']
        if len(tags)>1:
            boxes=[_raw_bounds(b) for _,b in tags];gap=.018*min(width,height)
            total=sum(b[2]-b[0] for b in boxes)+gap*(len(tags)-1)
            if total<=width*.91:
                merged=[];cursor=0.;maxh=max(b[3]-b[1] for b in boxes)
                for (_,block),box in zip(tags,boxes):
                    merged+=_shift(block,cursor-box[0],(maxh-box[3]+box[1])/2-box[1]);cursor+=box[2]-box[0]+gap
                bands=[p for p in bands if p[0]['role']!='tag'];bands.append((dict(role='tag'),merged));bands.sort(key=lambda p:order[p[0]['role']])
    margin=.045*min(width,height);gap=.018*min(width,height)
    top=margin
    if corner:top=max(top,_raw_bounds(corner)[3]+gap)
    bottom=height-margin;heights=[_raw_bounds(b)[3]-_raw_bounds(b)[1] for _,b in bands]
    usable=bottom-top-gap*max(0,len(bands)-1)
    factor=min(1.,usable/max(1.,sum(heights)))
    for _,block in bands:
        if factor<1:
            l,t,_,_=_raw_bounds(block)
            for e in block:
                e.update(x=(e['x']-l)*factor,y=(e['y']-t)*factor,width=e['width']*factor,height=e['height']*factor)
                if e['type']=='text':_set_font_size(e,e['font_size']*factor)
    used=sum(_raw_bounds(b)[3]-_raw_bounds(b)[1] for _,b in bands)
    spacing=max(gap,(bottom-top-used)/max(1,len(bands)-1));cursor=top
    result=[]
    for u,block in bands:
        l,t,r,b=_raw_bounds(block);dx=(width-(r-l))/2-l
        if u['role']=='footer' and u.get('footer_anchor')=='right':dx=width-margin-r
        result+=_shift(block,dx,cursor-t);cursor+=b-t+spacing
    result+=corner
    # Followers/decorative layers remain present; no output ID is dropped.
    result_by={e['id']:e for e in result}
    return [result_by.get(e['id'],e) for e in elements]
