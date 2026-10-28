"""Training-only human zone retrieval; no borrowed pixels, text or template IDs.

Assignments use a deterministic rectangular Hungarian solver. SciPy is not a
runtime dependency. Realization and rendered safety gates are separate costs.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from math import log,sqrt,isfinite
from collections import Counter
import json,os,time
LIBRARY=Path(__file__).parent/'data/layout_library.json'


def assign(cost):
    """Minimum sum assignment, every row exactly once, distinct columns (n<=m)."""
    n=len(cost);m=len(cost[0]) if n else 0
    if not n:return []
    if n>m or any(len(row)!=m or any(not isfinite(float(v)) for v in row) for row in cost):raise ValueError('Finite rectangular costs require rows <= columns')
    u=[0.]*(n+1);v=[0.]*(m+1);owner=[0]*(m+1);prior=[0]*(m+1)
    for i in range(1,n+1):
        owner[0]=i;column=0;distance=[float('inf')]*(m+1);seen=[False]*(m+1)
        while True:
            seen[column]=True;row=owner[column];delta=float('inf');next_column=0
            for j in range(1,m+1):
                if seen[j]:continue
                reduced=float(cost[row-1][j-1])-u[row]-v[j]
                if reduced<distance[j]:distance[j]=reduced;prior[j]=column
                if distance[j]<delta:delta=distance[j];next_column=j
            for j in range(m+1):
                if seen[j]:u[owner[j]]+=delta;v[j]-=delta
                else:distance[j]-=delta
            column=next_column
            if owner[column]==0:break
        while column:
            old=prior[column];owner[column]=owner[old];column=old
    return sorted((owner[j]-1,j-1) for j in range(1,m+1) if owner[j])


@lru_cache(maxsize=4)
def _library(path,mtime,size):
    try:
        doc=json.loads(Path(path).read_text());entries=doc.get('layouts',[])
        if doc.get('version')!=1:return []
        valid=[]
        for e in entries:
            try:
                if not .05<float(e['aspect'])<20 or not 2<=len(e['zones'])<=40:continue
                if not isinstance(e['roles'],dict) or not isinstance(e['key'],str):continue
                boxes=[z['box'] for z in e['zones']]
                if any(z.get('role') not in ('header','visual','copy','tag','action','footer','art') or z.get('kind') not in ('media','copy','control','card') for z in e['zones']):continue
                if any(len(b)!=4 or not all(isfinite(float(v)) for v in b) or min(b[2:])<=0 or min(b[:2])<0 or b[0]+b[2]>1.001 or b[1]+b[3]>1.001 for b in boxes):continue
                valid.append(e)
            except (ValueError,TypeError,KeyError):continue
        return valid
    except (OSError,ValueError,TypeError,KeyError):return []


def library():
    path=Path(os.environ.get('MAGICANVAS_LAYOUT_LIBRARY',LIBRARY))
    try:stat=path.stat()
    except OSError:return []
    return _library(str(path.resolve()),stat.st_mtime_ns,stat.st_size)


def plans(source,blocks,width,height,limit=5):
    from engine.layout_descriptors import describe
    if min(width,height)<=0 or not 2<=len(blocks)<=40:return []
    desc=describe(source,blocks);want=Counter(d['role'] for d in desc);target=width/height;started=time.perf_counter()
    near=[e for e in library() if abs(log(e['aspect']/target))<=.20]
    near.sort(key=lambda e:(sum(abs(want.get(k,0)-e['roles'].get(k,0)) for k in set(want)|set(e['roles'])),abs(log(e['aspect']/target)),e['key']))
    pool=near[:30]
    positions=sorted({round(i*(len(pool)-1)/max(1,min(limit,len(pool))-1)) for i in range(min(limit,len(pool)))}) if pool else []
    results=[]
    for rank in positions:
        template=pool[rank];zones=template['zones'];matrix=[]
        for d in desc:
            row=[]
            for z in zones:
                x,y,w,h=z['box'];area=max(1e-9,w*h);aspect=w*width/max(1.,h*height)
                row.append(1.5*(d['role']!=z['role'])+.75*(d['kind']!=z['kind'])+.5*abs(log(d['area']/area))+.5*abs(log(d['aspect']/aspect)))
            matrix.append(row+[3.5]*len(desc))
        matching=assign(matrix);placements=[];unassigned=[]
        for i,j in matching:
            if j>=len(zones):unassigned.append(i)
            else:placements.append((i,j))
        if any(desc[i]['role'] not in ('tag','footer','art') for i in unassigned):continue
        results.append(dict(key=template['key'],rank=rank,placements=placements,unassigned=unassigned,zones=zones,
            descriptors=desc,matching_cost=sum(matrix[i][j] for i,j in matching),lookup_seconds=time.perf_counter()-started))
    return results


def candidates(source,blocks,width,height):
    from engine.container_flow import measured_block
    from engine.macro_layout import bounds
    from engine.free_space import free_rectangles,place
    from engine.source_structure import _transform
    reference=sqrt(width*height/(source['canvas']['width']*source['canvas']['height']));result=[]
    for plan in plans(source,blocks,width,height):
        items=[];occupied=[];failed=False
        for i,j in plan['placements']:
            x,y,w,h=plan['zones'][j]['box'];x*=width;y*=height;w*=width;h*=height
            # Measure before accepting a zone; no subsequent leaf stretching.
            lo,hi=reference*.20,reference*1.4;fitted=None
            for _ in range(9):
                mid=(lo+hi)/2;local=measured_block(blocks[i],w,h,source,mid,(width,height))
                if local is not None:lo=mid;fitted=local
                else:hi=mid
            if fitted is None:failed=True;break
            b=bounds(fitted);dx=x+(w-(b[2]-b[0]))/2-b[0];dy=y+(h-(b[3]-b[1]))/2-b[1]
            for e in fitted:e['x']+=dx;e['y']+=dy
            occupied.append(bounds(fitted));items+=fitted
        if failed:continue
        for i in plan['unassigned']:
            block=blocks[i];b=bounds(block);sw,sh=(source['canvas'][k] for k in ('width','height'))
            rects=free_rectangles(width,height,occupied,margin=.045*min(width,height),padding=max(2.,.015*min(width,height)))
            placement=place((b[2]-b[0],b[3]-b[1]),rects,((b[0]+b[2])/2/sw*width,(b[1]+b[3])/2/sh*height),(width,height),scale_cap=reference)
            if placement is None:failed=True;break
            local=_transform(block,placement['scale'],placement['x'],placement['y'],b);items+=local;occupied.append(bounds(local))
        if not failed:
            result.append(dict(name=f'human-retrieved-{plan["rank"]}-{plan["key"][:8]}',elements=items,energy=0.,macro_locked=True,source_attached=True,
                macro_family='human-retrieved',retrieval={k:plan[k] for k in ('key','rank','matching_cost','unassigned','lookup_seconds')}))
    return result
