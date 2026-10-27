"""Repeated siblings share an allocated box; each card remains an atomic unit."""
from collections import defaultdict
from math import ceil
from copy import deepcopy


def arrange(blocks,box,source):
    from engine.macro_layout import bounds
    from engine.source_structure import _transform
    x,y,w,h=box;n=len(blocks)
    if n<2 or min(w,h)<=0:return None
    ordered=sorted(blocks,key=lambda es:(bounds(es)[1],bounds(es)[0]))
    # Preserve a source horizontal row's reading order; otherwise row-major.
    bs=[bounds(es) for es in ordered]
    if max((b[1]+b[3])/2 for b in bs)-min((b[1]+b[3])/2 for b in bs)<=.04*source['canvas']['height']:
        ordered.sort(key=lambda es:bounds(es)[0]);bs=[bounds(es) for es in ordered]
    widths=[b[2]-b[0] for b in bs];heights=[b[3]-b[1] for b in bs]
    # Source gaps are normalized by sibling size. Empty target space does not
    # turn a compact source row into widely dispersed, unrelated items.
    sb=sorted(bs);gaps=[max(0.,b[0]-a[2]) for a,b in zip(sb,sb[1:])]
    ratio=min(.5,max(.08,(sum(gaps)/max(1,len(gaps)))/max(1.,sum(heights)/n)))
    gap=ratio*min(w/max(1,n),h/max(1,n));options=[]
    for cols in sorted({1,n,ceil(n/2)}):
        rows=ceil(n/cols);s=min((w-gap*(cols-1))/(cols*max(widths)),(h-gap*(rows-1))/(rows*max(heights)))
        if s<=0:continue
        # Scale is the primary objective; prefer fewer unused cells on ties.
        options.append((s,-(cols*rows-n),cols,rows))
    if not options:return None
    scale,_,cols,rows=max(options);row_h=max(heights)*scale
    total_h=row_h*rows+gap*(rows-1);items=[]
    for row in range(rows):
        indices=list(range(row*cols,min(n,(row+1)*cols)));used=sum(widths[i]*scale for i in indices)+gap*(len(indices)-1)
        cursor=x+(w-used)/2
        for i in indices:
            top=y+(h-total_h)/2+row*(row_h+gap)+(row_h-heights[i]*scale)/2
            items+=_transform(ordered[i],scale,cursor,top,bs[i]);cursor+=widths[i]*scale+gap
    fonts=[e['font_size'] for e in items if e['type']=='text']
    return items,dict(rows=rows,columns=cols,scale=scale,minimum_font=min(fonts) if fonts else None,clear_gap=gap)


def variants(elements,source):
    """Offer only source-evidenced repeated structures, not arbitrary blocks."""
    from engine.macro_layout import composites,bounds
    out={e['id']:e for e in elements};families=defaultdict(list)
    for block in composites(source)['blocks']:
        if not all(e['id'] in out for e in block):continue
        signature=tuple(sorted(e['type'] for e in block))
        if 'text' not in signature:continue
        families[signature].append(block)
    result=[]
    for blocks in families.values():
        if not 3<=len(blocks)<=12:continue
        font_sizes=[max(e.get('font_size',0) for e in block if e['type']=='text') for block in blocks]
        if max(font_sizes)/max(1.,min(font_sizes))>1.15:continue
        boxes=[bounds(b) for b in blocks];areas=[(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        if max(areas)/max(1.,min(areas))>4:continue
        sx=[(b[0]+b[2])/2 for b in boxes];sy=[(b[1]+b[3])/2 for b in boxes]
        sw,sh=source['canvas']['width'],source['canvas']['height']
        if min((max(sx)-min(sx))/sw,(max(sy)-min(sy))/sh)>.04:continue
        members=[[out[e['id']] for e in b] for b in blocks];b=bounds([e for es in members for e in es])
        fitted=arrange(blocks,(b[0],b[1],b[2]-b[0],b[3]-b[1]),source)
        if not fitted:continue
        items,report=fitted;ids={e['id'] for e in items}
        result.append(([deepcopy(e) for e in elements if e['id'] not in ids]+items,report))
        if len(result)==2:break
    return result
