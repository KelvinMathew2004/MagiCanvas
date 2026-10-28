"""Repeated siblings share an allocated box; each card remains an atomic unit."""
from collections import defaultdict
from math import ceil
from copy import deepcopy


def ordered_peer_rows(widths, available, gap, rows):
    """Exact contiguous row breaks for measured unequal-width siblings.

    Keep reading order and intrinsic widths; slack is never converted into
    stretched controls. O(rows * peers^2), with deterministic tie breaking.
    """
    n=len(widths)
    if not 1<=rows<=n or any(w<=0 or w>available for w in widths):return None
    prefix=[0.]
    for w in widths:prefix.append(prefix[-1]+w)
    states={(0,0):(0.,[])}
    for r in range(1,rows+1):
        for end in range(r,n+1):
            choices=[]
            for start in range(r-1,end):
                prior=states.get((r-1,start));used=prefix[end]-prefix[start]+gap*(end-start-1)
                if prior is not None and used<=available+1e-7:
                    choices.append((prior[0]+(available-used)**2,prior[1]+[(start,end)]))
            if choices:states[r,end]=min(choices)
    return states.get((rows,n),(None,None))[1]


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
    for cols in sorted({1,2,3,n,ceil(n/2),ceil(n/3)} & set(range(1,n+1))):
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


def measured_block(block, width, height, source, font_scale, canvas):
    """Measure source atoms before packing; supported copy can rewrap.

    Native text backings may wrap; photos keep their aspect and overlays. A
    source-adjacent caption can move beside its photo, but cannot become a new
    overlay. Unsupported complex/rotated atoms retain their source transform.
    Returned coordinates are local and include all original IDs exactly once.
    """
    from engine.macro_layout import bounds
    from engine.source_structure import _transform
    from engine.resize import _set_font_size, _text_info, _render_text_ink
    from engine.layout_contracts import controls, control_font_floor
    if min(width, height, font_scale) <= 0 or not block:
        return None
    by = {e['id']: e for e in block}
    pairs = [(by[p], by[t]) for p, t in controls(source) if p in by and t in by]

    def text(old, available, size, shrink=True):
        if available<=1:return None
        e = deepcopy(old)
        _set_font_size(e, size)
        e.update(x=0., y=0., width=available, height=height)
        measured, natural, lines, *_ = _text_info(e, available)
        if natural>available+.5 and shrink:
            # Wrapping cannot split an unbreakable word/URL. Fit that token
            # only after using the entire allocation; preserve tracking ratio.
            _set_font_size(e,max(.1,size*available/natural*.98))
            measured,natural,lines,*_=_text_info(e,available)
        if natural > available + .5 or e.get('max_lines') and lines > e['max_lines']:
            return None
        ink = _render_text_ink(e)
        if ink is None:
            return None  # This candidate requires the real local font renderer.
        for _ in range(6):
            if ink[2]<=round(available):break
            if not shrink:return None
            _set_font_size(e,e['font_size']*min(.98,max(1.,round(available))/ink[2]*.98))
            measured,natural,lines,*_=_text_info(e,available)
            ink=_render_text_ink(e)
            if ink is None:return None
        if ink[2]>round(available):return None
        e['height'] = max(measured, ink[3]) + 2.
        return e

    def control(panel, label, available):
        if panel['type'] != 'shape' or abs(panel.get('rotation', 0)) > .1:
            return None
        size = max(label['font_size'] * font_scale,
                   control_font_floor(source, label['id'], *canvas))
        pad = max(3., min(size * .55, max(0., label['x'] - panel['x']) * size / label['font_size']))
        probe=deepcopy(label);_set_font_size(probe,size)
        natural=_text_info(probe,available)[1]
        # An allocation is a maximum, not an instruction to stretch a short
        # action to an entire copy column. Keep source padding and hierarchy.
        used=min(available,max(panel['width']*size/label['font_size'],natural+2*pad))
        t = text(label, used - 2 * pad, size,shrink=False) if used > 2 * pad else None
        if t is None:
            return None
        p = deepcopy(panel)
        p.update(x=0., y=0., width=used,
                 height=max(panel['height'] * size / label['font_size'], t['height'] + 2 * pad))
        t.update(x=pad, y=(p['height'] - t['height']) / 2)
        # Center the painted ink, not the nominal line box.
        ink = _render_text_ink(t)
        t['y'] = p['height'] / 2 - (ink[1] + ink[3]) / 2
        return [p, t]

    # A caption below a button is not a second label inside its backplate.
    # Keep the evidenced attachment while measuring it independently.
    from engine.nested_structure import control_captions
    attachments=control_captions(source)
    if len(block)==3 and len(pairs)==1:
        panel,label=pairs[0]
        caption=next((by[c] for p,t,c in attachments if p==panel['id'] and c in by),None)
        if caption is not None:
            items=control(panel,label,width)
            note=text(caption,width,caption['font_size']*font_scale)
            if items is None or note is None:return None
            gap=max(3.,min(8.,(caption['y']-panel['y']-panel['height'])*font_scale))
            note.update(y=items[0]['height']+gap)
            return items+[note] if note['y']+note['height']<=height else None

    # Two actual contained labels retain inline versus stacked ownership.
    # Authored intersecting type stays in the keep-arrangement path below.
    native_panels=[e for e in block if e['type']=='shape' and abs(e.get('rotation',0))<=.1]
    native_labels=[e for e in block if e['type']=='text' and abs(e.get('rotation',0))<=.1]
    if len(block)==2 and len(native_panels)==len(native_labels)==1 and not pairs:
        p,t=native_panels[0],native_labels[0]
        if (len(t.get('content',''))>80 and p.get('shape_type','rectangle') in ('rectangle','rounded_rectangle')
                and t['x']>=p['x'] and t['y']>=p['y']
                and t['x']+t['width']<=p['x']+p['width']+1
                and t['y']+t['height']<=p['y']+p['height']+1):
            pad=max(3.,min(width*.08,(t['x']-p['x'])*font_scale))
            copy=text(t,width-2*pad,t['font_size']*font_scale)
            if copy is None:return None
            panel=deepcopy(p);panel.update(x=0.,y=0.,width=width,height=copy['height']+2*pad)
            copy.update(x=pad,y=pad)
            return [panel,copy] if panel['height']<=height else None
    if len(block)==3 and len(native_panels)==1 and len(native_labels)==2:
        p=native_panels[0];a,b=sorted(native_labels,key=lambda e:(e['y'],e['x']))
        contained=all(t['x']>=p['x']-1 and t['y']>=p['y']-1 and
                      t['x']+t['width']<=p['x']+p['width']+1 and
                      t['y']+t['height']<=p['y']+p['height']+1 for t in (a,b))
        inline=abs(a['y']+a['height']/2-b['y']-b['height']/2)<=4 and (
            a['x']+a['width']<=b['x']+1 or b['x']+b['width']<=a['x']+1)
        stacked=a['y']+a['height']<=b['y']+1
        if contained and (inline or stacked):
            labels=sorted((a,b),key=lambda e:e['x'] if inline else e['y'])
            pad=max(3.,min(width*.08,max(t['font_size'] for t in labels)*font_scale*.55));gap=max(3.,pad*.6)
            sizes=[max(t['font_size']*font_scale,control_font_floor(source,t['id'],*canvas)) for t in labels]
            probes=[]
            for t,size in zip(labels,sizes):
                probe=deepcopy(t);_set_font_size(probe,size);probes.append(_text_info(probe,width)[1]+2.)
            allocations=probes if inline else [width-2*pad]*2
            if inline and sum(allocations)+gap>width-2*pad:return None
            painted=[text(t,aw,size,shrink=False) for t,aw,size in zip(labels,allocations,sizes)]
            if any(t is None for t in painted):return None
            inner_h=max(t['height'] for t in painted) if inline else sum(t['height'] for t in painted)+gap
            panel=deepcopy(p);panel.update(x=0.,y=0.,width=min(width,sum(allocations)+gap+2*pad) if inline else width,
                height=max(p['height']*font_scale,inner_h+2*pad))
            if panel['height']>height:return None
            cursor=pad
            for t in painted:
                t['x']=cursor if inline else pad
                t['y']=(panel['height']-t['height'])/2 if inline else cursor
                cursor+=(t['width'] if inline else t['height'])+gap
            return [panel]+painted

    # Repeated controls are a source-evidenced sibling group, not independent
    # leaves. Rewrap the row into a small grid without separating labels.
    if len(pairs) >= 2 and len(block) == 2 * len(pairs) and all(
            p['type'] == 'shape' and abs(p.get('rotation',0)) <= .1 for p,t in pairs):
        centers=[p['y']+p['height']/2 for p,t in pairs]
        horizontal=max(centers)-min(centers)<=max(4.,.04*source['canvas']['height'])
        pairs.sort(key=lambda pair:(pair[0]['x'],pair[0]['y']) if horizontal else (pair[0]['y'],pair[0]['x']))
        if len(pairs)>=4:
            gap=max(3.,min(6.,.045*min(width,height)));cards=[control(p,t,width) for p,t in pairs]
            if any(es is None for es in cards):return None
            widths=[es[0]['width'] for es in cards];rh=max(bounds(es)[3] for es in cards);options=[]
            for rows in range(1,len(cards)+1):
                if rows*rh+(rows-1)*gap>height:continue
                breaks=ordered_peer_rows(widths,width,gap,rows)
                if breaks is None:continue
                items=[]
                for row,(start,end) in enumerate(breaks):
                    cursor=0.
                    for es in deepcopy(cards[start:end]):
                        for e in es:e['x']+=cursor;e['y']+=row*(rh+gap)
                        cursor+=es[0]['width']+gap;items+=es
                options.append((rows,items))
            return min(options,key=lambda p:p[0])[1] if options else None
        gap = max(3., .045*min(width,height));options=[]
        for cols in sorted({1,2,3,len(pairs),ceil(len(pairs)/2),ceil(len(pairs)/3)} & set(range(1,len(pairs)+1))):
            cw=(width-gap*(cols-1))/cols
            cards=[control(p,t,cw) for p,t in pairs]
            if any(es is None for es in cards):continue
            row_heights=[max(bounds(es)[3] for es in cards[i:i+cols]) for i in range(0,len(cards),cols)]
            used=sum(row_heights)+gap*(len(row_heights)-1)
            if used > height:continue
            items=[];top=0.
            for row,rh in enumerate(row_heights):
                for col,es in enumerate(cards[row*cols:(row+1)*cols]):
                    for e in es:e['x']+=col*(cw+gap);e['y']+=top
                    items+=es
                top+=rh+gap
            options.append((used,-cols,items))
        return min(options,key=lambda p:p[:2])[2] if options else None

    # A real photo/caption attachment can change orientation. Everything else
    # on the photo (badge, price, marker) keeps its exact relative transform.
    photos = [e for e in block if e['type'] == 'image']
    panels=[e for e in block if e['type']=='shape' and e.get('shape_type','rectangle')
            in ('rectangle','rounded_rectangle') and abs(e.get('rotation',0))<=.1]
    labels=[e for e in block if e['type']=='text']
    if len(photos)==1 and len(panels)==1 and labels and len(block)==len(labels)+2:
        photo=photos[0];panel=panels[0]
        # A photo followed by copy inside one card can wrap its copy; an
        # authored overlay/badge keeps the original rigid realization below.
        if (photo['y']>=panel['y']-1 and photo['y']+photo['height']<=panel['y']+panel['height']+1
                and all(t['y']>=photo['y']+photo['height']-2 for t in labels)) and all(
                e['x']>=panel['x']-1 and e['x']+e['width']<=panel['x']+panel['width']+1
                for e in [photo]+labels):
            padding=max(2.,min(.05*width,max(0.,photo['x']-panel['x'])*font_scale))
            pw=width-2*padding
            pic=deepcopy(photo);pic.update(x=padding,y=padding,width=pw,height=pw*photo['height']/max(1.,photo['width']))
            items=[deepcopy(panel),pic];cursor=pic['y']+pic['height']
            previous=photo
            for old in sorted(labels,key=lambda e:(e['y'],e['x'])):
                cursor+=max(2.,min(6.,max(0.,old['y']-previous['y']-previous['height'])*font_scale))
                t=text(old,pw,old['font_size']*font_scale)
                if t is None:return None
                t.update(x=padding,y=cursor);items.append(t);cursor+=t['height'];previous=old
            cursor+=padding
            if cursor<=height:
                items[0].update(x=0.,y=0.,width=width,height=cursor)
                return items
            return None

    # A collection of complete explicit sibling cards uses the same measured
    # packer. Source row/column evidence is required by the caller's grouping.
    groups=defaultdict(list)
    for e in block:
        if e.get('group_id') is not None:groups[e['group_id']].append(e)
    if len(groups)>=3 and sum(map(len,groups.values()))==len(block):
        ordered=sorted(groups.values(),key=lambda es:(bounds(es)[1],bounds(es)[0]))
        gap=max(3.,.04*min(width,height));options=[];n=len(ordered)
        for cols in sorted({1,2,3,n,ceil(n/2),ceil(n/3)} & set(range(1,n+1))):
            cw=(width-gap*(cols-1))/cols
            children=[measured_block(es,cw,height,source,font_scale,canvas) for es in ordered]
            if any(es is None for es in children):continue
            rows=[children[i:i+cols] for i in range(0,n,cols)]
            used=sum(max(bounds(es)[3] for es in row) for row in rows)+gap*(len(rows)-1)
            if used>height:continue
            items=[];cursor=0.
            for row in rows:
                rh=max(bounds(es)[3] for es in row)
                for col,es in enumerate(row):
                    for e in es:e['x']+=col*(cw+gap);e['y']+=cursor
                    items+=es
                cursor+=rh+gap
            # Maximize retained type scale and media prominence, rather than
            # minimizing used height (which chooses six thumbnail columns).
            ratios=[e['font_size']/by[e['id']]['font_size'] for e in items if e['type']=='text']
            media=sum(e['width']*e['height'] for e in items if e['type']=='image')
            options.append((min(ratios,default=font_scale),media,-used,items))
        if options:return max(options,key=lambda p:p[:3])[3]
        return None
    media_options = []
    if len(photos) == 1:
        photo = photos[0]
        for panel, label in pairs:
            gap = panel['y'] - photo['y'] - photo['height']
            if not (-1 <= gap <= max(2., .02 * source['canvas']['height'])
                    and abs(panel['x'] - photo['x']) <= .04 * photo['width']
                    and abs(panel['width'] / photo['width'] - 1) <= .08):
                continue
            media = [e for e in block if e['id'] not in (panel['id'], label['id'])]
            mb = bounds(media)
            for share in (1., .5, .65):
                pw = width * share
                caption_width = width if share == 1 else width - pw - 3.
                caption = control(panel, label, caption_width)
                if caption is None:
                    continue
                scale = min(pw / max(1., mb[2] - mb[0]), height / max(1., mb[3] - mb[1]))
                items = _transform(media, scale, 0., 0., mb)
                b = bounds(items)
                # Bottom/side contacts are explicit; no new text/photo overlap.
                dx, dy = (0., b[3]) if share == 1 else (b[2] + 3., 0.)
                for e in caption:
                    e['x'] += dx
                    e['y'] += dy
                items += caption
                b = bounds(items)
                if b[2] <= width + .01 and b[3] <= height + .01:
                    media_options.append(items)
        if media_options:
            return max(media_options, key=lambda es: next(e['width'] * e['height'] for e in es if e['id'] == photo['id']))

    if len(block) == 2 and len(pairs) == 1 and pairs[0][0]['type'] == 'shape' and abs(pairs[0][0].get('rotation', 0)) <= .1:
        items = control(*pairs[0], width)
        if items and bounds(items)[3] <= height:
            return items
        return None

    # A repeated unbacked two-field row is semi-elastic: the trailing field
    # keeps its measured width while the leading field wraps in the remainder.
    from engine.nested_structure import repeated_text_rows
    row_pairs=repeated_text_rows(source['elements'],source['canvas'])
    pair=next(((by[a],by[b]) for a,b in row_pairs if a in by and b in by),None)
    rules=[e for e in block if e['type']=='shape' and e['height']<=6
           and e.get('shape_type','rectangle') in ('line','rectangle')]
    if pair and len(block)==2+len(rules):
        left,right=pair;gap=max(3.,min(.06*width,(right['x']-left['x']-left['width'])*font_scale))
        probe=deepcopy(right);_set_font_size(probe,right['font_size']*font_scale)
        rw=min(width*.4,_text_info(probe,width)[1]+2.)
        a=text(left,width-rw-gap,left['font_size']*font_scale)
        b=text(right,rw,right['font_size']*font_scale)
        if a is None or b is None:return None
        b.update(x=width-rw,text_align='right')
        row_height=max(a['height'],b['height']);items=[a,b]
        for old in rules:
            e=deepcopy(old);e.update(x=0.,y=row_height+2.,width=width,height=max(.5,old['height']*font_scale));items.append(e)
        return items if bounds(items)[3]<=height else None

    # Copy and its native horizontal rules reflow as one measured stack.
    texts = [e for e in block if e['type'] == 'text']
    rules = [e for e in block if e['type'] == 'shape' and e.get('shape_type', 'rectangle') in ('line', 'rectangle')
             and e['height'] <= max(6., source['canvas']['height'] * .008)]
    if texts and len(texts) + len(rules) == len(block) and all(abs(e.get('rotation', 0)) <= .1 for e in block):
        ordered = sorted(block, key=lambda e: (e['y'], e['x']))
        # Do not reinterpret authored overlapping typography as a paragraph.
        if any(a['y'] + a['height'] > b['y'] + 2 for a, b in zip(ordered, ordered[1:])):
            ordered = []
        items = []
        cursor = 0.
        previous = None
        for old in ordered:
            if previous:
                cursor += max(2., min(8., max(0., old['y'] - previous['y'] - previous['height']) * font_scale))
            if old['type'] == 'text':
                e = text(old, width, old['font_size'] * font_scale)
                if e is None:
                    return None
            else:
                e = deepcopy(old)
                e.update(x=0., width=width, height=max(.5, old['height'] * font_scale))
            e['y'] = cursor
            cursor += e['height']
            items.append(e)
            previous = old
        if items:
            return items if cursor <= height else None

    b = bounds(block)
    scale = min(font_scale, width / max(1., b[2] - b[0]), height / max(1., b[3] - b[1]))
    result=_transform(block, scale, 0., 0., b)
    # An unresolved atomic lockup is a keep-arrangement atom. Fractional font
    # scaling/tracking must not introduce new lines inside that arrangement.
    from engine.resize import _fit_text
    originals={e['id']:e for e in block}
    for e in result:
        if e['type']!='text':continue
        old=originals[e['id']];limit=_text_info(old)[2]
        probe=deepcopy(e);probe['max_lines']=min(limit,e.get('max_lines') or limit)
        _set_font_size(e,_fit_text(probe,e['width'],e['height'],e['font_size']))
        if _text_info(e)[2]>limit:return None
    return result


def measured_columns(blocks, box, source, font_scale, canvas, columns):
    """Bounded contiguous partition of measured atoms into equal-width columns.

    This uses actual wrapped heights before assigning column breaks, rather
    than four arbitrary source tiers sharing the same short slot. No ID or
    content matching determines the partition. The caller owns reading order.
    """
    from engine.macro_layout import bounds
    x, y, w, h = box
    if not 1 <= columns <= min(6, len(blocks)):
        return None
    gap = max(3., .045 * min(w, h))
    cw = (w - gap * (columns - 1)) / columns
    measured = [measured_block(b, cw, h, source, font_scale, canvas) for b in blocks]
    if any(es is None for es in measured):
        return None
    heights = [bounds(es)[3] - bounds(es)[1] for es in measured]
    # DP minimizes unused column height; every atom belongs to exactly one
    # band. Bounded O(columns * atoms^2), with no random restarts.
    states = {(0, 0): (0., [])}
    n = len(blocks)
    for c in range(1, columns + 1):
        for end in range(c, n + 1):
            options = []
            for start in range(c - 1, end):
                prior = states.get((c - 1, start))
                if prior is None:
                    continue
                used = sum(heights[start:end]) + gap * (end - start - 1)
                if used <= h:
                    options.append((prior[0] + (h - used) ** 2, prior[1] + [(start, end, used)]))
            if options:
                states[c, end] = min(options, key=lambda p: p[0])
    solved = states.get((columns, n))
    if solved is None:
        return None
    result = []
    for column, (start, end, used) in enumerate(solved[1]):
        cursor = y + (h - used) / 2
        for i in range(start, end):
            local = deepcopy(measured[i]);b = bounds(local)
            for e in local:
                e['x'] += x + column * (cw + gap) - b[0]
                e['y'] += cursor - b[1]
            result += local
            cursor += heights[i] + gap
    return result
