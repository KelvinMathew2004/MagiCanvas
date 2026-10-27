"""Bounded, auditable completion of omitted IDs; never credit raw validity."""
from copy import deepcopy
from math import sqrt
from statistics import median


def free_rectangles(width,height,occupied):
    """Deterministic maximal empty rectangles; prune contained fragments."""
    rects=[(0.,0.,float(width),float(height))]
    for x,y,r,b in occupied:
        next_rects=[]
        for a,t,c,d in rects:
            if min(c,r)<=max(a,x) or min(d,b)<=max(t,y):next_rects.append((a,t,c,d));continue
            next_rects += [(a,t,min(c,x),d),(max(a,r),t,c,d),(a,t,c,min(d,y)),(a,max(t,b),c,d)]
        valid=sorted(set(q for q in next_rects if q[2]-q[0]>1 and q[3]-q[1]>1))
        rects=[q for i,q in enumerate(valid) if not any(i!=j and p[0]<=q[0] and p[1]<=q[1] and p[2]>=q[2] and p[3]>=q[3] for j,p in enumerate(valid))]
        rects=sorted(rects,key=lambda q:(-(q[2]-q[0])*(q[3]-q[1]),q))[:256]
    return rects


def complete_missing(source,proposals,base_elements,target):
    """Keep >=85% raw IDs, max three omissions; no recovered anchor chains."""
    src={e['id']:e for e in source['elements']};missing=set(src)-set(proposals)
    if not proposals or len(missing)>3 or len(proposals)/max(1,len(src))<.85:return None,{}
    result=deepcopy(proposals);notes={};sw,sh=source['canvas']['width'],source['canvas']['height'];w,h=target['width'],target['height']
    from engine.geometry_policy import ambient_shape
    from engine.source_structure import groups
    spec=groups(source);backings={t:p for p,t in spec['backings']}
    background={eid for eid,e in src.items() if e['type']!='text' and e['width']*e['height']>=.90*sw*sh}
    available={i:p for i,p in proposals.items() if i not in background and not ambient_shape(src[i],source)}
    scales=[sqrt(p['width']*p['height']/max(1.,src[i]['width']*src[i]['height'])) for i,p in available.items()]
    scale=median(scales) if scales else min(w/sw,h/sh)
    occupied=[(p['x'],p['y'],p['x']+p['width'],p['y']+p['height']) for p in available.values()]
    for eid in sorted(missing):
        old=src[eid];anchor=None;method=None
        owner=backings.get(eid)
        if owner in available:anchor=owner;method='source-parent'
        if anchor is None:
            peers=[i for i in available if old.get('group_id') is not None and old.get('group_id')==src[i].get('group_id')]
            if not peers:
                peers=[i for members in spec['members'] if eid in members for i in members if i in available]
            if peers:
                anchor=min(peers,key=lambda i:((old['x']-src[i]['x'])/sw)**2+((old['y']-src[i]['y'])/sh)**2);method='source-group'
        if anchor is not None:
            a=src[anchor];p=available[anchor];sx=p['width']/max(1.,a['width']);sy=p['height']/max(1.,a['height']);s=min(sx,sy)
            value=dict(x=p['x']+(old['x']-a['x'])*sx,y=p['y']+(old['y']-a['y'])*sy,width=old['width']*s,height=old['height']*s)
        elif eid in background:
            value=dict(x=0.,y=0.,width=float(w),height=float(h));s=min(w/sw,h/sh);method='background'
        else:
            rects=free_rectangles(w,h,occupied)
            if not rects:return None,notes
            def fit(q):return min(scale,(q[2]-q[0])/old['width'],(q[3]-q[1])/old['height'])
            q=max(rects,key=lambda q:(fit(q),-(q[0]/w-old['x']/sw)**2-(q[1]/h-old['y']/sh)**2))
            s=fit(q);value=dict(x=q[0],y=q[1],width=old['width']*s,height=old['height']*s);method='free-rectangle'
        if old['type']=='text':value['font_size']=old['font_size']*s
        value['width']=min(float(w),max(1.,value['width']));value['height']=min(float(h),max(1.,value['height']))
        value['x']=max(0.,min(w-value['width'],value['x']));value['y']=max(0.,min(h-value['height'],value['y']))
        result[eid]=value;notes[eid]=dict(method=method,anchor=anchor)
        if method=='free-rectangle':occupied.append((value['x'],value['y'],value['x']+value['width'],value['y']+value['height']))
    return result,notes
