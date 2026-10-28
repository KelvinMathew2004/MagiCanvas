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

    def text(old, available, size):
        e = deepcopy(old)
        _set_font_size(e, size)
        e.update(x=0., y=0., width=available, height=height)
        measured, natural, lines, *_ = _text_info(e, available)
        if natural > available + .5 or e.get('max_lines') and lines > e['max_lines']:
            return None
        ink = _render_text_ink(e)
        if ink is None:
            return None  # This candidate requires the real local font renderer.
        e['height'] = max(measured, ink[3]) + 2.
        return e

    def control(panel, label, available):
        if panel['type'] != 'shape' or abs(panel.get('rotation', 0)) > .1:
            return None
        size = max(label['font_size'] * font_scale,
                   control_font_floor(source, label['id'], *canvas))
        pad = max(3., min(size * .55, max(0., label['x'] - panel['x']) * size / label['font_size']))
        t = text(label, available - 2 * pad, size) if available > 2 * pad else None
        if t is None:
            return None
        p = deepcopy(panel)
        p.update(x=0., y=0., width=available,
                 height=max(panel['height'] * size / label['font_size'], t['height'] + 2 * pad))
        t.update(x=pad, y=(p['height'] - t['height']) / 2)
        # Center the painted ink, not the nominal line box.
        ink = _render_text_ink(t)
        t['y'] = p['height'] / 2 - (ink[1] + ink[3]) / 2
        return [p, t]

    # Repeated controls are a source-evidenced sibling group, not independent
    # leaves. Rewrap the row into a small grid without separating labels.
    if len(pairs) >= 2 and len(block) == 2 * len(pairs) and all(
            p['type'] == 'shape' and abs(p.get('rotation',0)) <= .1 for p,t in pairs):
        pairs.sort(key=lambda pair:pair[0]['x'])
        gap = max(3., .045*min(width,height));options=[]
        for cols in sorted({1,len(pairs),ceil(len(pairs)/2)}):
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
