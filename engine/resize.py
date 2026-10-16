"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.

Implementation: offline v0.9 reflow adapted from Project Chameleon's
analysis -> candidate generation -> ranking pipeline. No editor/model runtime.
Explicit groups travel together. Simple text/backplate groups can widen as a
unit; artwork groups receive one proportional transform. Inferred roles stay
internal and do not become output schema fields.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import cos, isfinite, log, radians, sin, sqrt
from pathlib import Path
import re
from typing import Any


def _bounds(items):
    boxes = []
    for e in items:
        w, h, a = e['width'], e['height'], radians(e.get('rotation', 0))
        rw, rh = abs(w*cos(a))+abs(h*sin(a)), abs(w*sin(a))+abs(h*cos(a))
        cx, cy = e['x']+w/2, e['y']+h/2
        boxes.append((cx-rw/2, cy-rh/2, cx+rw/2, cy+rh/2))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else (0, 0, 0, 0)


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def _area(e):
    return max(1e-9, e['width']*e['height'])


def _contains(a, b, tolerance=1):
    return (b['x'] >= a['x']-tolerance and b['y'] >= a['y']-tolerance
            and b['x']+b['width'] <= a['x']+a['width']+tolerance
            and b['y']+b['height'] <= a['y']+a['height']+tolerance)


@lru_cache(maxsize=128)
def _font(family, bold, italic):
    """Load actual supplied fonts locally; unknown fonts use an estimate."""
    try:
        from PIL import ImageFont
        key = re.sub('[^a-z0-9]', '', family.lower())
        directory = Path(__file__).resolve().parents[1]/'brand_kit'/'fonts'
        matches = []
        for p in sorted(directory.glob('*')):
            name, _, suffix = p.stem.partition('-')
            if p.suffix.lower() in ('.ttf', '.otf') and re.sub('[^a-z0-9]', '', name.lower()) == key:
                penalty = int(('bold' in suffix.lower()) != bold)+int(('italic' in suffix.lower()) != italic)
                matches.append((penalty, str(p)))
        if matches:
            return ImageFont.truetype(min(matches)[1], 1000)
    except (ImportError, OSError, ValueError):
        pass
    return None


@lru_cache(maxsize=4096)
def _advances(content, family, bold, italic, runs):
    font, result = _font(family, bold, italic), []
    for i, c in enumerate(content):
        selected = font
        for start, end, rb, ri in runs:
            if start <= i < end:
                selected = _font(family, bold if rb is None else rb, italic if ri is None else ri)
        if c == '\n':
            result.append(0)
        elif selected is not None:
            result.append(selected.getlength(c)/1000)
        else:
            result.append(.30 if c.isspace() or c in "ilI.,'!:;|" else .90 if c in 'MW@#%&' else .65 if c.isupper() else .56)
    return tuple(result)


def _text(e, width=None, size=None):
    """Mirror converter/wrap.py: replace spaces, never split a long word.

    Conservatively account for leading, letter spacing and styled characters.
    Font metrics use the brand-kit font files rather than editor approximations.
    """
    width = e['width'] if width is None else width
    size = e['font_size'] if size is None else size
    content = e.get('content', '')
    runs = tuple((r['start'], r['end'], None if 'font_weight' not in r else r['font_weight']=='bold',
                  None if 'font_style' not in r else r['font_style']=='italic') for r in (e.get('style_runs') or []))
    advances = _advances(content, e.get('font_family', ''), e.get('font_weight')=='bold', e.get('font_style')=='italic', runs)
    widths = [v*size for v in advances]
    chars, line_width, space = list(content), 0, None
    for i, c in enumerate(chars):
        if c == '\n':
            line_width, space = 0, None
            continue
        if line_width+widths[i] > width and space is not None:
            chars[space] = '\n'
            line_width, space = sum(widths[space+1:i+1]), None
        else:
            line_width += widths[i]
        if c == ' ':
            space = i
    lines, line_width, count = [], 0, 0
    for c, advance in zip(chars, widths):
        if c == '\n':
            lines.append(line_width+max(0, count-1)*e.get('letter_spacing', 0))
            line_width, count = 0, 0
        else:
            line_width, count = line_width+advance, count+1
    lines.append(line_width+max(0, count-1)*e.get('letter_spacing', 0))
    return len(lines)*size*e.get('line_height', 1.2)+size*.12, max(lines, default=0), len(lines)


def _fit_text(e, width, height, desired):
    lo, hi = 1.0, max(1.0, desired)
    for _ in range(18):
        mid = (lo+hi)/2
        h, w, lines = _text(e, width, mid)
        if h <= height and w <= width and (not e.get('max_lines') or lines <= e['max_lines']):
            lo = mid
        else:
            hi = mid
    return lo


def _scale(items, factor):
    left, top, _, _ = _bounds(items)
    result = deepcopy(items)
    for e in result:
        e['x'], e['y'] = (e['x']-left)*factor, (e['y']-top)*factor
        e['width'], e['height'] = e['width']*factor, e['height']*factor
        for key in ('font_size', 'letter_spacing', 'corner_radius', 'stroke_width'):
            if key in e:
                e[key] *= factor
        if e['type'] == 'text':
            e['font_size'] = max(1, e['font_size'])
    return result


def _move(items, x, y):
    left, top, _, _ = _bounds(items)
    for e in items:
        e['x'], e['y'] = e['x']+x-left, e['y']+y-top
    return items


def _shift(items, x, y):
    """Translate a slot's contents without discarding its internal padding."""
    for e in items:
        e['x'] += x
        e['y'] += y
    return items


def _contain(items, width, height):
    l, t, r, b = _bounds(items)
    return _scale(items, min(1, width/max(1e-9, r-l), height/max(1e-9, b-t)))


def _analyze(source):
    elements = source['elements']
    sw, sh = source['canvas']['width'], source['canvas']['height']
    max_font = max((e.get('font_size', 0) for e in elements), default=24)
    backgrounds, rest = [], []
    for e in elements:
        full = e['x'] <= sw*.02 and e['y'] <= sh*.02 and e['x']+e['width'] >= sw*.98 and e['y']+e['height'] >= sh*.98
        behind = all(e.get('z_index', 0) <= f.get('z_index', 0) for f in elements if f['type']=='text')
        (backgrounds if full and behind and not e.get('group_id') and e['type']!='text' else rest).append(e)
    parent = list(range(len(rest)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[root(j)] = root(i)
    for i, a in enumerate(rest):
        for j, b in enumerate(rest[:i]):
            if a.get('group_id') and a.get('group_id') == b.get('group_id'):
                join(i, j)
    # Infer only tight text/backplate containment; never group a whole page
    # merely because its elements are close or aligned.
    for i, text in enumerate(rest):
        if text['type'] != 'text' or text.get('group_id'):
            continue
        backs = [(j, s) for j, s in enumerate(rest) if s['type']=='shape' and s.get('shape_type')=='rectangle'
                 and not s.get('group_id') and _contains(s, text) and _area(s) < sw*sh*.16
                 and s['height'] < text['font_size']*6 and s.get('z_index', 0) <= text.get('z_index', 0)
                 and abs(s.get('rotation', 0)-text.get('rotation', 0)) < .1]
        if backs:
            join(i, min(backs, key=lambda p: _area(p[1]))[0])
    grouped = {}
    for i, e in enumerate(rest):
        grouped.setdefault(root(i), []).append(e)
    units = []
    for items in grouped.values():
        x, y, r, b = _bounds(items)
        texts = [e for e in items if e['type']=='text']
        shapes = [e for e in items if e['type']=='shape']
        images = [e for e in items if e['type']=='image']
        label = (len(shapes)==1 and len(texts)==len(items)-1 and bool(texts)
                 and all(_contains(shapes[0], t, 3) for t in texts))
        if label:
            role = 'action' if y/sh > .6 else 'tag'
        elif len(texts)==len(items):
            role = 'footer' if y/sh > .90 and max(t['font_size'] for t in texts) < max_font*.5 else 'copy'
        elif images and not texts and (r-x)/max(1, b-y)>2 and y/sh<.22 and (r-x)*(b-y)<sw*sh*.15:
            role = 'header'
        elif images and not texts and y/sh>.88 and (r-x)*(b-y)<sw*sh*.05:
            role = 'footer'
        elif len(items)==1 and shapes and (shapes[0].get('opacity', 1)<.4 or shapes[0].get('shape_type') in ('ellipse', 'line')):
            role = 'decoration'
        else:
            role = 'visual'
        units.append(dict(items=items, bounds=(x,y,r,b), role=role, label=label))
    return backgrounds, units


def _unit(u, width, height, scale, fill=False):
    width, height = max(.0001, width), max(.0001, height)
    items, (x,y,r,b) = u['items'], u['bounds']
    if len(items)==1 and items[0]['type']=='text':
        e = deepcopy(items[0])
        e['width'] = width
        e['font_size'] = _fit_text(e, width, height, max(1, e['font_size']*scale))
        e['height'] = min(height, _text(e)[0])
        e['x'], e['y'] = 0, 0
        return _contain([e], width, height)
    if u['label']:
        shape = next(e for e in items if e['type']=='shape')
        texts = sorted((e for e in items if e['type']=='text'), key=lambda e:(e['y'],e['x']))
        label_w = min(width, max((r-x)*scale, max(_text(e, 1e9, max(1,e['font_size']*scale))[1]+e['font_size']*scale*1.6 for e in texts)))
        result = deepcopy(items)
        back = next(e for e in result if e['type']=='shape')
        labels = sorted((e for e in result if e['type']=='text'), key=lambda e:(e['y'],e['x']))
        pad = min(label_w*.10, max(e['font_size'] for e in texts)*scale*.8)
        total_h = 0
        for old, e in zip(texts, labels):
            e['width'] = max(.01, label_w-2*pad)
            e['font_size'] = _fit_text(old, e['width'], height/len(texts)*.75, max(1,old['font_size']*scale))
            e['height'] = _text(e)[0]
            total_h += e['height']
        label_h = min(height, max(shape['height']*scale, total_h*1.5))
        back.update(x=0, y=0, width=label_w, height=label_h)
        if 'corner_radius' in back:
            back['corner_radius'] = min(label_h/2, shape['corner_radius']*scale)
        if 'stroke_width' in back:
            back['stroke_width'] *= scale
        cursor = (label_h-total_h)/2
        for e in labels:
            e['x'], e['y'] = pad, cursor
            cursor += e['height']
        return _contain(result, width, height)
    factor = min(width/max(1e-9,r-x), height/max(1e-9,b-y))
    if not fill:
        factor = min(factor, scale)
    result = _scale(items, factor)
    for e in result:
        if e['type']=='text':
            e['font_size'] = _fit_text(e, e['width'], e['height'], e['font_size'])
    return _contain(result, width, height)


def _row(units, width, height, scale, gap):
    rows, current, used = [], [], 0
    for u in units:
        items = _unit(u, width, height, scale)
        _,_,w,h = _bounds(items)
        if current and used+gap+w > width:
            rows.append(current)
            current, used = [], 0
        current.append((items,w,h))
        used += w+(gap if len(current)>1 else 0)
    if current:
        rows.append(current)
    output, cursor = [], 0
    for row in rows:
        rh = max(h for _,_,h in row)
        x = (width-sum(w for _,w,_ in row)-gap*(len(row)-1))/2
        for items,w,h in row:
            output.extend(_move(items,x,cursor+(rh-h)/2))
            x += w+gap
        cursor += rh+gap
    return output, max(0,cursor-gap)


def _copy(units, width, height, scale, gap):
    if not units:
        return []
    def build(factor):
        blocks = [_unit(u,width,height,factor) for u in units]
        sizes = [_bounds(b)[3] for b in blocks]
        return blocks,sizes,sum(sizes)+gap*max(0,len(blocks)-1)
    lo,hi = .0001,scale
    for _ in range(16):
        mid = (lo+hi)/2
        if build(mid)[2] <= height:
            lo = mid
        else:
            hi = mid
    blocks,sizes,occupied = build(lo)
    # Impossible low-resolution cases still retain every element. Compress
    # gaps first; a final proportional fit is the last structural fallback.
    spacing = min(gap, max(0,(height-sum(sizes))/max(1,len(blocks)-1)))
    if occupied <= height and len(blocks)>1:
        spacing += min((height-occupied)/(len(blocks)+1),gap*2.5)
    occupied = sum(sizes)+spacing*max(0,len(blocks)-1)
    cursor,output = max(0,(height-occupied)/2),[]
    for block,h in zip(blocks,sizes):
        w = _bounds(block)[2]
        output.extend(_move(block,(width-w)/2,cursor))
        cursor += h+spacing
    return _contain(output,width,height)


def _visuals(units,width,height,scale,gap):
    best,best_use = [],-1
    for cols in range(1,len(units)+1):
        rows = (len(units)+cols-1)//cols
        cw,ch = (width-gap*(cols-1))/cols,(height-gap*(rows-1))/rows
        if min(cw,ch)<=0:
            continue
        items,occupied = [],0
        for i,u in enumerate(units):
            block = _unit(u,cw,ch,scale,fill=True)
            _,_,w,h = _bounds(block)
            items.extend(_move(block,(i%cols)*(cw+gap)+(cw-w)/2,(i//cols)*(ch+gap)+(ch-h)/2))
            occupied += w*h
        if occupied>best_use:
            best,best_use = items,occupied
    return best


def _decorations(units,width,height,sw,sh,occupied):
    result=[]
    for u in units:
        x,y,r,b=u['bounds']
        block=_unit(u,width*.32,height*.32,min(width/sw,height/sh))
        _,_,w,h=_bounds(block)
        preferred=(max(0,min(width-w,(x+r)/2/sw*width-w/2)),max(0,min(height-h,(y+b)/2/sh*height-h/2)))
        positions=[preferred,(0,0),(width-w,0),(0,height-h),(width-w,height-h)]
        def cost(p):
            box=(p[0],p[1],p[0]+w,p[1]+h)
            collisions=sum(_overlap(box,_bounds([e]))/max(1,w*h) for e in occupied if e['type']=='text')
            return collisions*10+((p[0]-preferred[0])/width)**2+((p[1]-preferred[1])/height)**2
        result.extend(_move(block,*min(positions,key=cost)))
    return result


def _reflow(units,width,height,sw,sh,mode,share,scale):
    margin,gap=min(width,height)*.05,min(width,height)*.035
    cw,ch=width-2*margin,height-2*margin
    roles={r:sorted((u for u in units if u['role']==r),key=lambda u:(u['bounds'][1],u['bounds'][0]))
           for r in ('copy','header','footer','visual','tag','action','decoration')}
    side_header=mode=='split' and width/height>3 and bool(roles['header'])
    header,hh=_row([] if side_header else roles['header'],cw,ch*.13,scale,gap)
    footer,fh=_row(roles['footer'],cw,ch*.10,scale,gap*.6)
    labels,lh=_row(roles['tag']+roles['action'],cw,ch*.17,scale,gap)
    reserved=hh+fh+lh+gap*sum(bool(v) for v in (hh,fh,lh))
    if reserved>ch*.53:
        shrink=ch*.53/reserved
        header,hh=_row([] if side_header else roles['header'],cw,ch*.13,scale*shrink,gap*shrink)
        footer,fh=_row(roles['footer'],cw,ch*.10,scale*shrink,gap*shrink*.6)
        labels,lh=_row(roles['tag']+roles['action'],cw,ch*.17,scale*shrink,gap*shrink)
    top,bottom=margin,height-margin
    output=[]
    if hh:
        output.extend(_shift(header,margin,top)); top+=hh+gap
    if fh:
        bottom-=fh; output.extend(_shift(footer,margin,bottom)); bottom-=gap
    if lh:
        bottom-=lh; output.extend(_shift(labels,margin,bottom)); bottom-=gap
    available=max(.001,bottom-top)
    gap=min(gap,available*.08)
    cx,cwidth=margin,cw
    if side_header:
        hw=cw*.15
        block,used=_row(roles['header'],hw,available,scale,gap)
        output.extend(_shift(block,margin,top+(available-used)/2))
        cx+=hw+gap; cwidth-=hw+gap
    copy,visuals=roles['copy'],roles['visual']
    if not copy or not visuals:
        block=_copy(copy,cwidth,available,scale,gap) if copy else _visuals(visuals,cwidth,available,scale,gap)
        if block:
            output.extend(_move(block,cx,top+max(0,(available-_bounds(block)[3])/2)))
    elif mode=='split':
        vw=(cwidth-gap)*share; tw=cwidth-gap-vw
        text=_copy(copy,tw,available,scale,gap)
        art=_visuals(visuals,vw,available,scale,gap)
        art_left=sum((u['bounds'][0]+u['bounds'][2])/2 for u in visuals)/len(visuals)<sw*.45
        tx,vx=(cx+vw+gap,cx) if art_left else (cx,cx+tw+gap)
        output.extend(_move(text,tx,top+(available-_bounds(text)[3])/2))
        output.extend(_move(art,vx,top+(available-_bounds(art)[3])/2))
    else:
        vh=(available-gap)*share; th=available-gap-vh
        text=_copy(copy,cwidth,th,scale,gap)
        art=_visuals(visuals,cwidth,vh,scale,gap)
        ty,vy=(top,top+th+gap) if mode=='copy-first' else (top+vh+gap,top)
        output.extend(_move(text,cx,ty+(th-_bounds(text)[3])/2))
        output.extend(_move(art,cx,vy+(vh-_bounds(art)[3])/2))
    output.extend(_decorations(roles['decoration'],width,height,sw,sh,output))
    return output


def _proportional(units,width,height,sw,sh):
    factor=min(width/sw,height/sh)
    result=[]
    for u in units:
        block=_contain(_scale(u['items'],factor),width,height)
        for e in block:
            if e['type']=='text':
                e['font_size']=_fit_text(e,e['width'],e['height'],e['font_size'])
        x,y,_,_=u['bounds']; _,_,w,h=_bounds(block)
        px=(width-sw*factor)/2+x*factor; py=(height-sh*factor)/2+y*factor
        result.extend(_move(block,max(0,min(width-w,px)),max(0,min(height-h,py))))
    return result


def _score(elements,units,source,width,height,desired):
    """Internal heuristic, not the official challenge evaluator."""
    by_id={e['id']:e for e in elements}
    penalty,rects=0.0,[]
    max_font=max((e.get('font_size',0) for e in source['elements']),default=24)
    for u in units:
        block=[by_id[e['id']] for e in u['items']]
        bounds=_bounds(block)
        if u['role']!='decoration':
            rects.append((u,bounds))
        for old,e in zip(u['items'],block):
            if not all(isfinite(e[k]) for k in ('x','y','width','height')) or min(e['width'], e['height']) < 0:
                return -1e12
            box=_bounds([e])
            penalty+=(1-_overlap(box,(0,0,width,height))/max(1e-9,(box[2]-box[0])*(box[3]-box[1])))*2000
            if e['type']=='text':
                th,tw,lines=_text(e)
                penalty+=(max(0,th/max(.001,e['height'])-1)+max(0,tw/max(.001,e['width'])-1))*400
                if e.get('max_lines'):
                    penalty+=max(0,lines-e['max_lines'])*100
                importance=2 if old['font_size']>=max_font*.75 else 1
                penalty+=max(0,log(max(1,old['font_size']*desired)/e['font_size']))*16*importance
                penalty+=max(0,min(8,min(width,height)*.065)-e['font_size'])*3
    for i,(a,ar) in enumerate(rects):
        for b,br in rects[:i]:
            intentional=a['role']==b['role']=='visual' and _overlap(a['bounds'],b['bounds'])>0
            if not intentional:
                penalty+=_overlap(ar,br)/max(1,min((ar[2]-ar[0])*(ar[3]-ar[1]),(br[2]-br[0])*(br[3]-br[1])))*700
    meaningful=[b for u,b in rects if u['role'] not in ('header','footer')]
    if meaningful:
        x0,y0=min(b[0] for b in meaningful),min(b[1] for b in meaningful)
        x1,y1=max(b[2] for b in meaningful),max(b[3] for b in meaningful)
        penalty+=(max(0,.78-(x1-x0)/width)+max(0,.72-(y1-y0)/height))*50
        penalty+=(abs((x0+x1)/width/2-.5)+abs((y0+y1)/height/2-.5))*12
    change=abs(log((width/height)/(source['canvas']['width']/source['canvas']['height'])))
    if change<.32:
        for u,b in rects:
            a=u['bounds']
            penalty+=(abs((b[0]+b[2])/width-(a[0]+a[2])/source['canvas']['width'])
                      +abs((b[1]+b[3])/height-(a[1]+a[3])/source['canvas']['height']))*(1-change/.32)*20
    return -penalty


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    This is where your actual approach goes: constraint solving, a learned
    layout model, heuristics, whatever you choose. Latency is not a scoring
    concern (see RULES.md) — spend your effort on quality, not speed.

    Current implementation notes:
        The original starter contract above is retained verbatim.
        The uniform-stretch baseline was replaced as the starter requested;
        selected deterministic ideas from Project Chameleon were translated
        into the helpers in this file. See IMPLEMENTATION.md for provenance
        and a complete historical copy of the starter, including its comments.

        There is no editor-specific 240px minimum or 5:1 restriction: extreme
        banners are official inputs. Every element survives even when the
        supplied resolution cannot make all of its content legible. Invalid
        dimensions and duplicate source IDs fail clearly. Input is not mutated.
    """
    sw,sh=source['canvas']['width'],source['canvas']['height']
    width,height=target_canvas['width'],target_canvas['height']
    if any(not isinstance(n,(int,float)) or isinstance(n,bool) or not isfinite(n) or n<=0 for n in (sw,sh,width,height)):
        raise ValueError('Canvas dimensions must be positive finite numbers.')
    if int(width)!=width or int(height)!=height:
        raise ValueError('Target dimensions must be integer pixels.')
    if len({e['id'] for e in source['elements']})!=len(source['elements']):
        raise ValueError('Source element IDs must be unique.')
    backgrounds,units=_analyze(source)
    desired=sqrt(width*height/(sw*sh))
    candidates=[_proportional(units,width,height,sw,sh)]
    for mode in ('split','copy-first','visual-first'):
        for share in (.25,.38,.50,.62):
            for multiplier in (.80,1.0,1.25):
                candidates.append(_reflow(units,width,height,sw,sh,mode,share,desired*multiplier))
    winner=max(candidates,key=lambda c:_score(c,units,source,width,height,desired))
    by_id={e['id']:e for e in winner}
    for old in backgrounds:
        e=deepcopy(old); e.update(x=0,y=0,width=width,height=height)
        if e['type']=='image':
            e['fit']='cover'
        by_id[e['id']]=e
    return {'canvas':{'width':int(width),'height':int(height)},
            'elements':[by_id[e['id']] for e in source['elements']]}
