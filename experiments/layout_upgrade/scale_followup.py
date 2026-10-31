"""Three isolated follow-ups, with no model training or reference-coordinate input."""
from copy import deepcopy
from math import sqrt
import numpy as np


def partition_statistics(background,mask,color,opacity=1.):
    """Same legacy statistics, selecting only the three required order ranks."""
    from engine.raster_metrics import luminance
    pixels=np.asarray(background,dtype=np.uint8).reshape(-1,3)
    active=pixels[np.asarray(mask).reshape(-1)>12]
    if not len(active):return None
    alpha=color[3]*opacity
    fg=np.array(color[:3],dtype=np.float64)*alpha+active.astype(np.float64)*(1-alpha)
    a,b=luminance(fg),luminance(active)
    ratios=(np.maximum(a,b)+.05)/(np.minimum(a,b)+.05)
    ranks=sorted({0,len(ratios)//20,len(ratios)//5})
    contrasts=np.partition(ratios,ranks)
    indices=np.arange(1,len(b));rough=float(np.abs(np.diff(b))[indices%8!=0].sum()/56.)
    return dict(minimum=float(contrasts[0]),lower_quintile=float(contrasts[len(contrasts)//5]),
        p05=float(contrasts[len(contrasts)//20]),ink_pixels=len(active),roughness=rough,
        luma_std=float(b.std()),background_rgb_jump=float(np.ptp(pixels,axis=0).max()))


def quota_funnel(candidates,source,units,width,height,desired,budget=12):
    """Reserve independent paths, then use true pre-raster defect cost only.

    This is intentionally an experiment: pixels may distinguish equal geometry,
    so passing quotas does not guarantee the original winner survives pruning.
    """
    from engine.resize import _objective_layout_cost
    if len(candidates)<=budget:return list(candidates),dict(input=len(candidates),kept=len(candidates),pruned=0)
    ranked=sorted(range(len(candidates)),key=lambda i:(
        _objective_layout_cost(candidates[i]['elements'],units,source,width,height,desired),i))
    keep=set()
    predicates=[lambda c:c['name'].startswith('qwen-') and not c.get('macro_locked'),
        lambda c:c['name'].startswith('macro-band-reflow'),
        lambda c:c['name'].startswith('source-attached-proportional'),
        lambda c:'semantic-stack' in c['name'],lambda c:'editorial' in c['name'],
        lambda c:c['name']=='proportional',lambda c:c.get('measured_stack')]
    for name in ('qwen-original','qwen-font-fitted','qwen-pre-container-font-fitted','qwen-raw-constraint-repaired'):
        predicates.append(lambda c,name=name:c['name']==name)
    for predicate in predicates:
        index=next((i for i in ranked if predicate(candidates[i])),None)
        if index is not None:keep.add(index)
    for i in ranked:
        if len(keep)>=budget:break
        keep.add(i)
    retained=[c for i,c in enumerate(candidates) if i in keep]
    return retained,dict(input=len(candidates),kept=len(retained),pruned=len(candidates)-len(retained),
        names=[c['name'] for c in retained])


def container_growth(elements,source,width,height,scale_cap=1.25,translation_cap=.05,minimum_scale_gain=.02):
    """Maximize a SHARED uniform scale with bounded whole-block translations.

    Two LPs: maximize scale, then minimize absolute motion at that scale.
    Source-owned children never become independent variables. Existing pair
    separation directions and confirmed shared source/target axes stay fixed.
    This cannot solve reflow topology; all results still need native verification.
    """
    from scipy.optimize import linprog
    from engine.macro_layout import composites,bounds
    from engine.source_structure import _transform
    by={e['id']:e for e in elements};original={e['id']:e for e in source['elements']}
    spec=composites(source,tight=True)
    blocks=[[by[e['id']] for e in group if e['id'] in by] for group in spec['blocks']]
    blocks=[b for b in blocks if b]
    if not blocks or len(blocks)>32:return None,dict(reason='unsupported block count')
    boxes=np.asarray([bounds(b) for b in blocks],dtype=float)
    sizes=boxes[:,2:]-boxes[:,:2]
    if np.any(sizes<=0):return None,dict(reason='degenerate block')
    dims=np.asarray([width,height],dtype=float);b=boxes/np.tile(dims,2)
    half=(b[:,2:]-b[:,:2])/2;centers=(b[:,:2]+b[:,2:])/2
    old=np.asarray([bounds([original[e['id']] for e in group]) for group in blocks])
    n=len(blocks);variables=1+4*n;ub=[];rhs=[];eq=[];erhs=[]
    def row(terms):
        v=np.zeros(variables)
        for k,value in terms.items():v[k]+=value
        return v
    def move(i,axis):return 1+2*i+axis
    def absolute(i,axis):return 1+2*n+2*i+axis
    def lower(i,axis,edge):
        ub.append(row({0:half[i,axis],move(i,axis):-1}));rhs.append(centers[i,axis]-edge)
    def upper(i,axis,edge):
        ub.append(row({0:half[i,axis],move(i,axis):1}));rhs.append(edge-centers[i,axis])
    sw,sh=source['canvas']['width'],source['canvas']['height']
    for i in range(n):
        for axis,dim in enumerate(dims):
            # Do not impose a new inset larger than the existing one.
            margin=min(.025*min(width,height)/dim,max(0.,b[i,axis]),max(0.,1-b[i,axis+2]))
            lower(i,axis,margin);upper(i,axis,1-margin)
            for sign in (-1.,1.):
                ub.append(row({move(i,axis):sign,absolute(i,axis):-1}));rhs.append(0.)
            source_dim=(sw,sh)[axis]
            if old[i,axis]<=2 and b[i,axis]<=.02:
                eq.append(row({move(i,axis):1,0:-half[i,axis]}));erhs.append(-centers[i,axis])
            elif old[i,axis+2]>=source_dim-2 and b[i,axis+2]>=.98:
                eq.append(row({move(i,axis):1,0:half[i,axis]}));erhs.append(1-centers[i,axis])
    for i in range(n):
        for j in range(i):
            separations=[]
            for axis in (0,1):
                delta=centers[i,axis]-centers[j,axis]
                gap=abs(delta)-half[i,axis]-half[j,axis]
                if gap>=-1e-9:separations.append((gap,axis,1. if delta>=0 else -1.))
            if separations:
                gap,axis,direction=max(separations)
                clearance=min(max(0.,gap),max(1.,.008*min(width,height))/dims[axis])
                ub.append(row({move(j,axis):direction,move(i,axis):-direction,
                    0:half[i,axis]+half[j,axis]}))
                rhs.append(direction*(centers[i,axis]-centers[j,axis])-clearance)
            else:
                # Existing overlapping blocks keep their relative arrangement.
                for axis in (0,1):
                    delta=centers[i,axis]-centers[j,axis]
                    eq.append(row({move(i,axis):1,move(j,axis):-1,0:-delta}));erhs.append(-delta)
            for axis in (0,1):
                for fraction in (0.,.5,1.):
                    oa=old[i,axis]+fraction*(old[i,axis+2]-old[i,axis])
                    ob=old[j,axis]+fraction*(old[j,axis+2]-old[j,axis])
                    a=centers[i,axis]+(2*fraction-1)*half[i,axis]
                    c=centers[j,axis]+(2*fraction-1)*half[j,axis]
                    if abs(oa-ob)<=4 and abs(a-c)<=.003:
                        eq.append(row({move(i,axis):1,move(j,axis):-1,
                            0:(2*fraction-1)*(half[i,axis]-half[j,axis])}))
                        erhs.append(centers[j,axis]-centers[i,axis])
    bounds_lp=[(1.,scale_cap)]+[(-translation_cap,translation_cap)]*(2*n)+[(0,None)]*(2*n)
    objective=np.zeros(variables);objective[0]=-1
    kwargs=dict(A_ub=np.asarray(ub),b_ub=np.asarray(rhs),A_eq=np.asarray(eq) if eq else None,
        b_eq=np.asarray(erhs) if eq else None,bounds=bounds_lp,method='highs',options={'time_limit':.2})
    fit=linprog(objective,**kwargs)
    if not fit.success:return None,dict(reason='scale solve failed',status=int(fit.status))
    scale=float(fit.x[0])
    if scale<1+minimum_scale_gain:return None,dict(reason='no material scale headroom',scale=scale)
    bounds_lp[0]=(scale,scale);objective[0]=0;objective[1+2*n:]=1
    stable=linprog(objective,**kwargs)
    if not stable.success:return None,dict(reason='motion solve failed',status=int(stable.status))
    output=deepcopy(elements);mapped={e['id']:e for e in output}
    for i,group in enumerate(blocks):
        center=(centers[i]+stable.x[1+2*i:1+2*i+2])*dims
        left,top=center-sizes[i]*scale/2
        for e in _transform(group,scale,left,top,boxes[i]):mapped[e['id']].update(e)
    return output,dict(scale=scale,blocks=n,axis_constraints=len(eq),
        max_normalized_translation=float(np.abs(stable.x[1:1+2*n]).max()),
        source_ownership='tight composites; one shared scale, rigid child transforms',
        reason='proposal requires normal full raster/geometry verification')
