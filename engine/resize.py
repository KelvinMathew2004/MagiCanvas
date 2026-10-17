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
from math import cos, exp, isfinite, log, radians, sin, sqrt
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter
from pathlib import Path
import re
from typing import Any


@lru_cache(maxsize=256)
def _png_alpha_rect(asset_id):
    """Normalized visible-pixel bounds for a PNG asset, or None if opaque."""
    if not isinstance(asset_id,str) or Path(asset_id).suffix.lower() != '.png':
        return None
    path=Path(__file__).resolve().parents[1]/'brand_kit'/'assets'/asset_id
    try:
        from PIL import Image
        with Image.open(path) as image:
            rgba=image.convert('RGBA')
            alpha=rgba.getchannel('A')
            if alpha.getextrema()==(255,255):
                return None
            # Ignore tiny alpha noise from antialiasing/export fringes; those
            # pixels should not reserve layout space for an otherwise clear PNG.
            box=alpha.point(lambda value:255 if value>8 else 0).getbbox()
            if not box:
                return None
            left,top,right,bottom=box
            width,height=rgba.size
            if box==(0,0,width,height):
                return None
            return left/width,top/height,right/width,bottom/height
    except (OSError,ValueError):
        return None


def _image_content_rect(e):
    """Visible image rectangle inside its layout box, honoring the fit mode."""
    rect=_png_alpha_rect(e.get('asset_id'))
    x,y,w,h=e['x'],e['y'],e['width'],e['height']
    if rect is None or w<=0 or h<=0:
        return x,y,x+w,y+h
    left,top,right,bottom=rect
    path=Path(__file__).resolve().parents[1]/'brand_kit'/'assets'/e['asset_id']
    try:
        from PIL import Image
        with Image.open(path) as image:
            iw,ih=image.size
    except (OSError,ValueError):
        return x,y,x+w,y+h
    fit=e.get('fit','contain')
    if fit=='contain':
        factor=min(w/iw,h/ih)
        draw_w,draw_h=iw*factor,ih*factor
        ox,oy=(w-draw_w)/2,(h-draw_h)/2
        l,r=ox+left*draw_w,ox+right*draw_w
        t,b=oy+top*draw_h,oy+bottom*draw_h
    elif fit=='cover':
        factor=max(w/iw,h/ih)
        draw_w,draw_h=iw*factor,ih*factor
        ox,oy=(w-draw_w)/2,(h-draw_h)/2
        l,r=ox+left*draw_w,ox+right*draw_w
        t,b=oy+top*draw_h,oy+bottom*draw_h
        l,r=max(0,min(w,l)),max(0,min(w,r))
        t,b=max(0,min(h,t)),max(0,min(h,b))
        if r<l:l=r=(l+r)/2
        if b<t:t=b=(t+b)/2
    else:
        l,r=left*w,right*w
        t,b=top*h,bottom*h
    return x+l,y+t,x+r,y+b


def _trim_transparent_frame(items):
    """Crop the dominant transparent PNG gutter without scaling visible pixels."""
    if len(items)!=1 or items[0].get('type')!='image':
        return items
    e=items[0]
    rect=_png_alpha_rect(e.get('asset_id'))
    if rect is None or e.get('fit')=='fill':
        return items
    path=Path(__file__).resolve().parents[1]/'brand_kit'/'assets'/e['asset_id']
    try:
        from PIL import Image
        with Image.open(path) as image:
            iw,ih=image.size
    except (OSError,ValueError):
        return items
    left,top,right,bottom=rect
    # The cover window is centered, so its minimum safe extent is the distance
    # from the image center to the farther visible edge on each axis.
    crop_w=max((right-left)*iw,2*max(.5-left,right-.5)*iw)
    crop_h=max((bottom-top)*ih,2*max(.5-top,bottom-.5)*ih)
    # Trim one axis only. This preserves the raster scale exactly and keeps
    # asymmetric but visible artwork inside a centered cover viewport.
    trim_x=1-crop_w/iw
    trim_y=1-crop_h/ih
    if max(trim_x,trim_y)<.12:
        return items
    x,y,w,h=e['x'],e['y'],e['width'],e['height']
    fit=e.get('fit','contain')
    scale=(max(w/iw,h/ih) if fit=='cover' else min(w/iw,h/ih))
    draw_w,draw_h=iw*scale,ih*scale
    ox,oy=(w-draw_w)/2,(h-draw_h)/2
    if trim_x>=trim_y:
        crop_left=(iw-crop_w)/2
        e['x']=x+ox+crop_left*scale
        e['y']=y+oy
        e['width']=crop_w*scale
        e['height']=ih*scale
    else:
        crop_top=(ih-crop_h)/2
        e['x']=x+ox
        e['y']=y+oy+crop_top*scale
        e['width']=iw*scale
        e['height']=crop_h*scale
    e['fit']='cover'
    return items


def _raw_bounds(items):
    boxes = []
    for e in items:
        w, h, a = e['width'], e['height'], radians(e.get('rotation', 0))
        rw, rh = abs(w*cos(a))+abs(h*sin(a)), abs(w*sin(a))+abs(h*cos(a))
        cx, cy = e['x']+w/2, e['y']+h/2
        # Both visual rotation bounds and the runner's raw geometry must fit.
        boxes.append((min(e['x'],cx-rw/2), min(e['y'],cy-rh/2),
                      max(e['x']+w,cx+rw/2), max(e['y']+h,cy+rh/2)))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else (0, 0, 0, 0)


def _visual_bounds(items):
    """Visible-pixel bounds used for layout scoring and alignment."""
    boxes=[]
    for e in items:
        if e['type']=='image':
            left,top,right,bottom=_image_content_rect(e)
        else:
            left,top,right,bottom=e['x'],e['y'],e['x']+e['width'],e['y']+e['height']
        angle=radians(e.get('rotation',0))
        cx,cy=e['x']+e['width']/2,e['y']+e['height']/2
        corners=[(left,top),(right,top),(right,bottom),(left,bottom)]
        rotated=[(cx+(px-cx)*cos(angle)-(py-cy)*sin(angle),
                  cy+(px-cx)*sin(angle)+(py-cy)*cos(angle)) for px,py in corners]
        boxes.append((min(p[0] for p in rotated),min(p[1] for p in rotated),
                      max(p[0] for p in rotated),max(p[1] for p in rotated)))
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes)) if boxes else (0,0,0,0)


def _bounds(items):
    """Full element-frame bounds used for transforms and schema safety."""
    return _raw_bounds(items)


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


def _text_info(e, width=None, size=None):
    """Cached actual-font word wrapping; preserve explicit newlines/style runs."""
    width = e['width'] if width is None else width
    size = e['font_size'] if size is None else size
    runs = tuple((r['start'], r['end'], None if 'font_weight' not in r else r['font_weight']=='bold',
                  None if 'font_style' not in r else r['font_style']=='italic') for r in (e.get('style_runs') or []))
    return _text_cached(e.get('content',''),e.get('font_family',''),e.get('font_weight')=='bold',
                        e.get('font_style')=='italic',e.get('line_height',1.2),
                        e.get('letter_spacing',0),runs,width,size)


def _text(e, width=None, size=None):
    # Keep the public measurement helper's three-value contract.
    return _text_info(e,width,size)[:3]


@lru_cache(maxsize=65536)
def _text_cached(content,family,bold,italic,line_height,spacing,runs,width,size):
    # Match the supplied converter's space-replacement wrap rule. No hard
    # splitting of long words or text rewriting is hidden in the measurement.
    advances = _advances(content,family,bold,italic,runs)
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
            lines.append(line_width+max(0, count-1)*spacing)
            line_width, count = 0, 0
        else:
            line_width, count = line_width+advance, count+1
    lines.append(line_width+max(0, count-1)*spacing)
    words=tuple(len(line.split()) for line in ''.join(chars).split('\n'))
    return len(lines)*size*line_height+size*.12, max(lines, default=0), len(lines), tuple(lines), words


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


def _rag(e, width=None, size=None):
    # Author-authored breaks may be deliberate; never rewrite their content.
    if '\n' in e.get('content',''):
        return 0.
    _,maximum,count,widths,words=_text_info(e,width,size)
    if count<2 or maximum<=0:
        return 0.
    # A moderately short last line is normal; penalize severe imbalance/runts.
    rag=sum((1-v/maximum)**2 for v in widths)/count
    return rag+(0.5 if words[-1]==1 and widths[-1]<maximum*.55 else 0.)


def _target_font(e, u, scale):
    median=u.get('font_median',e['font_size'])
    gamma=u.get('font_gamma',1.)
    return max(1,median*scale*(e['font_size']/max(1,median))**gamma)


def _balanced_text(e, width, height, desired):
    size=_fit_text(e,width,height,desired)
    best=( _rag(e,width,size), width,size)
    if '\n' not in e.get('content','') and _text(e,width,size)[2]>1:
        for wf in (1.,.96,.90,.84):
            for sf in (1.,.97,.94,.92):
                w=width*wf; f=max(1,size*sf)
                measured_h,measured_w,lines=_text(e,w,f)
                if measured_h>height or measured_w>w or e.get('max_lines') and lines>e['max_lines']:
                    continue
                # Prefer readable type over a cosmetic improvement to its rag.
                cost=_rag(e,w,f)+1.5*(1-f/max(1,size))+.08*(1-wf)
                if cost<best[0]-1e-6:
                    best=(cost,w,f)
    return best[1],best[2]


def _tighten_text(e):
    """Remove unused left-aligned box width without changing measured wrapping."""
    if e.get('text_align','left')!='left':
        return
    old_h,old_w,old_lines=_text(e)
    width=min(e['width'],max(.001,old_w+max(.01,e['font_size']*.001)))
    new_h,new_w,new_lines=_text(e,width)
    if new_lines==old_lines and new_h<=old_h+1e-6 and new_w<=width+1e-6:
        e['width']=width


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


def _move_visual(items, x, y):
    """Place visible pixels at x/y while preserving each image's transparent frame."""
    left, top, _, _ = _visual_bounds(items)
    return _shift(items, x-left, y-top)


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
    # Infer only one-to-one text/backplate containment; never group a whole page
    # merely because its elements are close or aligned.
    for i, text in enumerate(rest):
        if text['type'] != 'text' or text.get('group_id'):
            continue
        backs = [(j, s) for j, s in enumerate(rest) if s['type']=='shape' and s.get('shape_type')=='rectangle'
                 and not s.get('group_id') and _contains(s, text) and _area(s) < sw*sh*.16
                 and s['height'] < text['font_size']*6 and s.get('z_index', 0) <= text.get('z_index', 0)
                 and abs(s.get('rotation', 0)-text.get('rotation', 0)) < .1
                 and not any(other is not text and other is not s and _contains(s, other)
                             for other in rest)]
        if backs:
            join(i, min(backs, key=lambda p: _area(p[1]))[0])
    grouped = {}
    for i, e in enumerate(rest):
        grouped.setdefault(root(i), []).append(e)
    units = []
    for items in grouped.values():
        image_only=all(e['type']=='image' for e in items)
        x, y, r, b = _visual_bounds(items) if image_only else _bounds(items)
        texts = [e for e in items if e['type']=='text']
        shapes = [e for e in items if e['type']=='shape']
        images = [e for e in items if e['type']=='image']
        label = (len(shapes)==1 and len(texts)==1 and len(items)==2
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
    units=_enrich_units(source, units)
    fonts=sorted(e['font_size'] for e in elements if e['type']=='text')
    median=fonts[len(fonts)//2] if fonts else 24
    for u in units:
        u['font_median']=median
    return backgrounds, units


def _unit(u, width, height, scale, fill=False):
    width, height = max(.0001, width), max(.0001, height)
    items, (x,y,r,b) = u['items'], u['bounds']
    # A transparent PNG's frame can contain substantial empty space. Fit its
    # visible pixels to the slot, then position by those pixels while retaining
    # the original PNG and element frame. This is generic alpha geometry, not
    # a logo-specific scale multiplier.
    if len(items)==1 and items[0]['type']=='image' and _png_alpha_rect(items[0].get('asset_id')):
        visible=_visual_bounds(items)
        visual_w=max(1e-9,visible[2]-visible[0]);visual_h=max(1e-9,visible[3]-visible[1])
        factor=min(width/visual_w,height/visual_h)
        if not fill:
            factor=min(factor,scale)
        raw=_raw_bounds(items)
        raw_w=max(1e-9,raw[2]-raw[0]);raw_h=max(1e-9,raw[3]-raw[1])
        # Visible pixels may fit a narrow slot while the transparent frame
        # still exceeds the real canvas. Cap by both the local vertical slot
        # and the actual target canvas before the projection phase.
        factor=min(factor,height/raw_h)
        target_w,target_h=u.get('target_canvas',(0,0))
        if target_w>0 and target_h>0:
            factor=min(factor,target_w/raw_w,target_h/raw_h)
        result=_scale(items,factor)
        _trim_transparent_frame(result)
        visible=_visual_bounds(result)
        return _move_visual(result,(width-(visible[2]-visible[0]))/2,
                            (height-(visible[3]-visible[1]))/2)
    if len(items)==1 and items[0]['type']=='text':
        e = deepcopy(items[0])
        desired=_target_font(e,u,scale)
        if u.get('semantic') in ('headline','subheading'):
            e['width'],e['font_size']=_balanced_text(e,width,height,desired)
        else:
            e['width']=width
            e['font_size']=_fit_text(e,width,height,desired)
        e['height'] = min(height, _text(e)[0])
        e['x'], e['y'] = 0, 0
        return _contain([e], width, height)
    if u['label'] and not u.get('corner_callout'):
        shape = next(e for e in items if e['type']=='shape')
        texts = sorted((e for e in items if e['type']=='text'), key=lambda e:(e['y'],e['x']))
        label_w = min(width, max((r-x)*scale, max(_text(e, 1e9, _target_font(e,u,scale))[1]+_target_font(e,u,scale)*1.6 for e in texts)))
        result = deepcopy(items)
        back = next(e for e in result if e['type']=='shape')
        labels = sorted((e for e in result if e['type']=='text'), key=lambda e:(e['y'],e['x']))
        pad = min(label_w*.10, max(_target_font(e,u,scale) for e in texts)*.8)
        total_h = 0
        for old, e in zip(texts, labels):
            e['width'] = max(.01, label_w-2*pad)
            desired_font=_target_font(old,u,scale)
            fit_height=height/len(texts)*.75
            if '\n' not in old.get('content',''):
                fit_height=min(fit_height,desired_font*(old.get('line_height',1.2)+.12)+.001)
            e['font_size'] = _fit_text(old,e['width'],fit_height,desired_font)
            e['height'] = _text(e)[0]
            total_h += e['height']
        if u.get('semantic') == 'cta' and shape['width'] > 0:
            # The text can force a narrower CTA slot than the source button.
            # Derive its height from the fitted width so the pill keeps its
            # source proportions instead of collapsing into a thin strip.
            fitted_shape_h = label_w * shape['height'] / shape['width']
        else:
            fitted_shape_h = shape['height']*scale
        label_h = min(height, max(fitted_shape_h, total_h*1.5))
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
    fit_w,fit_h=u.get('fit_size',(r-x,b-y))
    factor = min(width/max(1e-9,fit_w), height/max(1e-9,fit_h))
    if not fill:
        factor = min(factor, scale)
    result = _scale(items, factor)
    for e in result:
        if e['type']=='text':
            e['font_size'] = _fit_text(e, e['width'], e['height'], e['font_size'])
    return _contain(result, width, height)


def _row(units, width, height, scale, gap, align='center'):
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
        row_width=sum(w for _,w,_ in row)+gap*(len(row)-1)
        x = 0 if align=='left' else width-row_width if align=='right' else (width-row_width)/2
        for items,w,h in row:
            output.extend(_move(items,x,cursor+(rh-h)/2))
            x += w+gap
        cursor += rh+gap
    return output, max(0,cursor-gap)


def _badge_row(units,width,height,scale,gap,align='center'):
    # Keep verified peers on one row by sharing a fitting scale when possible.
    if len(units)>1 and units[0].get('peer_ids') and all(u.get('peer_ids')==units[0]['peer_ids'] for u in units):
        widths=[_bounds(_unit(u,width,height,scale))[2] for u in units]
        needed=sum(widths)
        if needed+gap*(len(units)-1)>width:
            gap=min(gap,width/max(1,len(units)*8))
            scale*=max(.001,(width-gap*(len(units)-1))/max(.001,needed))*.995
    return _row(units,width,height,scale,gap,align)


def _action_band(tags,actions,width,height,scale,gap,stacked,align='center'):
    if not (stacked or len(tags)>=2) or not tags or not actions:
        return _badge_row(tags,width,height,scale,gap,align) if not actions else _row(tags+actions,width,height,scale,gap,align)
    badges,bh=_badge_row(tags,width,height*.40,scale*.82,gap,align)
    buttons,ah=_row(actions,width,max(.001,height-bh-gap),scale,gap,align)
    if align=='left':
        _move(badges,0,0);_move(buttons,0,0)
    output=badges+_shift(buttons,0,bh+gap)
    return output,bh+gap+ah


def _footer_band(units,width,height,scale,gap,one_line_text=False):
    """Keep small source-corner images subordinate to centered footer copy."""
    corners=[u for u in units if u.get('footer_anchor') in ('left','right')]
    central=[u for u in units if u not in corners]
    if (one_line_text and len(central)==1 and len(central[0]['items'])==1
            and central[0]['items'][0]['type']=='text'
            and '\n' not in central[0]['items'][0].get('content','')):
        # Reserve the footer's horizontal span for the full legal sentence.
        # Source footnote frames (often only a small fraction of the canvas)
        # should not force it to wrap after a wide-banner conversion.
        unit=central[0]
        e=deepcopy(unit['items'][0])
        desired=_target_font(unit['items'][0],unit,scale)
        e['width']=width
        e['font_size']=_fit_text(e,width,height,desired)
        text_h,text_w,_=_text(e)
        # Keep a generous text frame: the layout's measurement font and the
        # final renderer can differ slightly, and a tightly measured frame
        # makes a visually single-line disclaimer wrap in the exported PNG.
        e['width']=max(width*.72,text_w+e['font_size']*2)
        e['height']=min(height,text_h)
        e['text_align']='center'
        e['x']=(width-e['width'])/2
        e['y']=(height-e['height'])/2
        output=[e]
        for side in ('left','right'):
            items=[u for u in corners if u.get('footer_anchor')==side]
            if not items:
                continue
            block,bh=_row(items,width*.14,height,scale,gap)
            left,_,right,_=_visual_bounds(block)
            bw=right-left
            x=0 if side=='left' else width-bw
            if side=='left' and x+bw>=e['x']-gap:
                continue
            if side=='right' and x<=e['x']+e['width']+gap:
                continue
            output.extend(_move_visual(block,x,(height-bh)/2))
        return output,e['height']
    if not corners:
        return _row(central,width,height,scale,gap)
    text,th=_row(central,width,height*.40 if central else height,scale,gap)
    budget=max(.001,height-th-(gap if th else 0))
    output=[];corner_h=0
    for side in ('left','right'):
        items=[u for u in corners if u['footer_anchor']==side]
        if not items:continue
        block,bh=_row(items,width*.40,budget,scale,gap)
        left,_,right,_=_bounds(block);bw=right-left
        output.extend(_move(block,0 if side=='left' else width-bw,0))
        corner_h=max(corner_h,bh)
    if text:output.extend(_shift(text,0,corner_h+gap))
    return output,corner_h+(gap+th if text else 0)


def _axis_active(u,source,width,height):
    state=_aspect_state(source,width,height)
    vertical=state['bucket'] in ('story','portrait','square')
    return vertical and u.get('source_centered',False)


def _badge_flow(records,width,height):
    bands={}
    actions=[_bounds(items) for u,items in records if u['role']=='action']
    for u,items in records:
        if u['role']=='tag' and u.get('peer_ids'):
            bands.setdefault(u['peer_ids'],[]).append((u,_bounds(items)))
    cost=0.
    for members in bands.values():
        if len(members)<2:continue
        members.sort(key=lambda row:row[0]['bounds'][0])
        boxes=[b for _,b in members]
        center_y=sum((b[1]+b[3])/2 for b in boxes)/len(boxes)
        cost+=sum(((b[1]+b[3])/2-center_y)**2/height**2 for b in boxes)*4
        gaps=[b[0]-a[2] for a,b in zip(boxes,boxes[1:])]
        if len(gaps)>1:
            mean=sum(gaps)/len(gaps)
            cost+=sum(((g-mean)/max(min(width,height)*.025,abs(mean)))**2 for g in gaps)*.03
        for a,b in zip(boxes,boxes[1:]):
            cost+=max(0,(a[0]-b[0])/width)
        if width/height<1.3:
            bottom=max(b[3] for b in boxes)
            # A centered strip's independent members should retain its shared axis.
            if all(not u.get('source_upper',False) for u,_ in members) and actions:
                action=min(actions,key=lambda b:abs(b[1]-bottom))
                cost+=max(0,(bottom-action[1])/height)*2
                cost+=max(0,(action[1]-bottom)/height-.12)
                band_center=(min(b[0] for b in boxes)+max(b[2] for b in boxes))/2
                cost+=((band_center-(action[0]+action[2])/2)/width)**2*2
    return cost


def _copy(units, width, height, scale, gap, align=None):
    if not units:
        return []
    def build(factor):
        blocks = [_unit(u,width,height,factor) for u in units]
        if align is not None:
            for u,block in zip(units,blocks):
                if len(u['items'])==1 and u['items'][0]['type']=='text' and not u['items'][0].get('group_id'):
                    for e in block:
                        e['text_align']=align
                        # Tight line extents permit a common visible left axis.
                        if align=='left':
                            _tighten_text(e)
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
        output.extend(_move(block,0 if align=='left' else (width-w)/2,cursor))
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
        block=_callout_block(u,width,height,sw,sh) if u.get('corner_callout') else _unit(u,width*.32,height*.32,min(width/sw,height/sh))
        _,_,w,h=_bounds(block)
        preferred=(max(0,min(width-w,(x+r)/2/sw*width-w/2)),max(0,min(height-h,(y+b)/2/sh*height-h/2)))
        positions=[preferred,(0,0),(width-w,0),(0,height-h),(width-w,height-h)]
        if u.get('corner_callout'):
            edge=min(width,height)*.025
            preferred=(width-edge-w if (x+r)/2>sw/2 else edge,edge)
            positions=[preferred]
        def cost(p):
            box=(p[0],p[1],p[0]+w,p[1]+h)
            collisions=sum(_overlap(box,_bounds([e]))/max(1,w*h) for e in occupied if e['type']=='text')
            return collisions*10+((p[0]-preferred[0])/width)**2+((p[1]-preferred[1])/height)**2
        result.extend(_move(block,*min(positions,key=cost)))
    return result


def _logo_row(units,width,height,scale,gap,source_size=None,fill=False):
    """Fit logos into a slot, optionally filling it with a bounded scale."""
    if not units:return [],0
    slot=max(.001,(width-gap*(len(units)-1))/len(units))
    blocks=[]
    for u in units:
        if fill:
            if len(u['items'])==1 and u['items'][0]['type']=='image' and _png_alpha_rect(u['items'][0].get('asset_id')):
                cap=max(scale*3.4,min(width/source_size[0],height/source_size[1])*3.4) if source_size else scale*3.4
                block=_unit(u,slot,height,cap,fill=True)
                blocks.append(block)
                continue
            fit_w,fit_h=u.get('fit_size',(u['bounds'][2]-u['bounds'][0],u['bounds'][3]-u['bounds'][1]))
            cap=max(scale*3.4,min(width/source_size[0],height/source_size[1])*3.4) if source_size else scale*3.4
            factor=min(slot/max(1e-9,fit_w),height/max(1e-9,fit_h),cap)
            block=_contain(_scale(u['items'],factor),slot,height)
        else:
            block=_unit(u,slot,height,scale*1.8)
        blocks.append(block)
    sizes=[_visual_bounds(b) for b in blocks]
    used=sum(b[2]-b[0] for b in sizes)+gap*(len(blocks)-1)
    row_h=max(b[3]-b[1] for b in sizes)
    output=[];x=max(0,(width-used)/2)
    for block,b in zip(blocks,sizes):
        w,h=b[2]-b[0],b[3]-b[1]
        output.extend(_move_visual(block,x,(row_h-h)/2));x+=w+gap
    return output,row_h


def _callout_block(u,width,height,sw,sh):
    desired=sqrt(width*height/(sw*sh))
    return _unit(u,width*.32,height*.38,max(min(width/sw,height/sh),desired*.85))


def _editorial_copy(roles,width,height,scale,gap):
    """Kicker, copy and CTA occupy one left-aligned column."""
    upper=[u for u in roles['tag'] if u.get('source_upper')]
    lower=[u for u in roles['tag'] if not u.get('source_upper')]
    logo,lh=_logo_row(roles['header'],width,height*.18,scale,gap)
    kicker,kh=_badge_row(upper,width,height*.13,scale*.82,gap,align='left')
    actions,ah=_action_band(lower,roles['action'],width,height*.34,scale,gap,True,align='left')
    reserved=sum(v for v in (lh,kh,ah))
    count=sum(bool(v) for v in (lh,kh,ah))+bool(roles['copy'])
    copy=_copy(roles['copy'],width,max(.001,height-reserved-gap*max(0,count-1)),scale,gap,align='left')
    mutable={u['items'][0]['id'] for u in roles['copy']+roles['action']+roles['tag']
             if len(u['items'])==1 and u['items'][0]['type']=='text' and not u['items'][0].get('group_id')}
    output=[];cursor=0
    for block in (logo,kicker,copy,actions):
        if not block:continue
        for e in block:
            if e['id'] in mutable:
                e['text_align']='left';_tighten_text(e)
        _,_,_,bottom=_bounds(_move(block,0,0))
        output.extend(_move(block,0,cursor));cursor+=bottom+gap
    return _contain(output,width,height)


def _banner_actions(units,width,height,scale,gap,banner_width=None):
    """Keep a plain text CTA on one line when the banner has room for it."""
    block,used=_row(units,width,height,scale,gap)
    if len(units)==1 and len(units[0]['items'])==1 and units[0]['items'][0]['type']=='text' and not units[0]['label']:
        e=block[0]
        desired=_target_font(units[0]['items'][0],units[0],scale)
        max_text_width=min(width,(banner_width or width)*.32)
        natural=_text(e,1e9,desired)[1]
        chosen=desired
        if natural>max_text_width:
            lo,hi=1.,desired
            for _ in range(18):
                mid=(lo+hi)/2
                if _text(e,1e9,mid)[1]<=max_text_width:lo=mid
                else:hi=mid
            if lo>=max(12.,desired*.55):chosen=lo
        natural=_text(e,1e9,chosen)[1]
        if natural<=max_text_width:
            e['font_size']=_fit_text(e,natural+chosen*.2,height,chosen)
            natural=_text(e,1e9,e['font_size'])[1]
            # Leave room for renderer/font-metric differences after the
            # cached measurement used by the layout engine.
            e['width']=min(max_text_width,natural*1.18+e['font_size']*.25)
            e['height']=min(height,_text(e)[0])
    return _contain(block,width,height),used


def _editorial_split_seed(units,width,height,sw,sh,scale,share=.42,reverse=False):
    if width/height>=3.2:
        return _banner_seed(units,width,height,scale,share,reverse,(sw,sh))
    margin,gap=min(width,height)*.05,min(width,height)*.035
    cw,ch=width-2*margin,height-2*margin
    roles={r:sorted((u for u in units if u['role']==r),key=_semantic_order)
           for r in ('header','footer','copy','action','tag','visual','decoration')}
    footer,fh=_footer_band(roles['footer'],cw,ch*.10,scale,gap*.6)
    top=margin;available=max(.001,ch-fh-(gap if fh else 0))
    side_header=width/height>=2.4 and bool(roles['header'])
    left=margin;main_width=cw;output=[]
    if side_header:
        logo_w=max(cw*.12,min(cw*.18,available*.70))
        brand,bh=_logo_row(roles['header'],logo_w,available*.75,scale,gap,(sw,sh),fill=True)
        output.extend(_shift(brand,left,top+(available-bh)/2))
        left+=logo_w+gap;main_width-=logo_w+gap
    art_left=bool(roles['visual']) and sum((u['bounds'][0]+u['bounds'][2])/2 for u in roles['visual'])/len(roles['visual'])<sw*.45
    if reverse:art_left=not art_left
    vw=(main_width-gap)*share if roles['visual'] else 0
    tw=max(.001,main_width-vw-(gap if vw else 0))
    tx=left+vw+gap if art_left else left
    vx=left if art_left else left+tw+gap
    # Reserve actual corner-callout clearance in the affected column.
    copy_clear=art_clear=0
    for u in roles['decoration']:
        if not u.get('corner_callout'):continue
        b=_bounds(_callout_block(u,width,height,sw,sh));clear=b[3]-b[1]+gap
        on_left=(u['bounds'][0]+u['bounds'][2])/2<sw/2
        if roles['visual'] and on_left==art_left:art_clear=max(art_clear,clear)
        elif not side_header or not on_left:copy_clear=max(copy_clear,clear)
    local={**roles,'header':[] if side_header else roles['header']}
    copy=_editorial_copy(local,tw,max(.001,available-copy_clear),scale,gap)
    b=_bounds(copy);bh=b[3]-b[1]
    output.extend(_move(copy,tx,top+copy_clear+(available-copy_clear-bh)/2))
    if roles['visual']:
        art=_visuals(roles['visual'],max(.001,vw),max(.001,available-art_clear),scale,gap)
        b=_bounds(art);bw,bh=b[2]-b[0],b[3]-b[1]
        output.extend(_move(art,vx+(vw-bw)/2,top+art_clear+(available-art_clear-bh)/2))
    if footer:output.extend(_shift(footer,margin,height-margin-fh))
    decoration_layout=_decorations(roles['decoration'],width,height,sw,sh,output)
    output.extend(decoration_layout)
    # If a source logo nearly touches a large decorative ellipse, preserve that
    # visual relationship after reflow: center the mark over its nearby accent.
    # The logo remains its own movable unit; this only chooses its seed position.
    for logo_unit in (u for u in roles['header'] if u['semantic']=='logo'):
        lx,ly,lr,lb=logo_unit['bounds']
        nearby=[]
        for deco_unit in roles['decoration']:
            if not any(e.get('shape_type')=='ellipse' for e in deco_unit['items'] if e['type']=='shape'):
                continue
            dx,dy,dr,db=deco_unit['bounds']
            vertical_gap=max(0,dy-lb,ly-db);horizontal_gap=max(0,dx-lr,lx-dr)
            if vertical_gap<=sh*.12 and horizontal_gap<=sw*.08:
                nearby.append((vertical_gap/max(1,sh)+horizontal_gap/max(1,sw),deco_unit))
        if nearby:
            deco_unit=min(nearby,key=lambda pair:pair[0])[1]
            logo_ids={e['id'] for e in logo_unit['items']}
            deco_ids={e['id'] for e in deco_unit['items']}
            logo_items=[e for e in output if e['id'] in logo_ids]
            deco_items=[e for e in output if e['id'] in deco_ids]
            if logo_items and deco_items:
                logo_box=_visual_bounds(logo_items);deco_box=_visual_bounds(deco_items)
                delta=(deco_box[0]+deco_box[2]-logo_box[0]-logo_box[2])/2
                _shift(logo_items,delta,0)
    return output


def _reflow(units,width,height,sw,sh,mode,share,scale):
    if width/height>=2.4 or (mode=='split' and width/height>=1.35):
        return _editorial_split_seed(units,width,height,sw,sh,scale,share,reverse=mode=='visual-first')
    margin,gap=min(width,height)*.05,min(width,height)*.035
    cw,ch=width-2*margin,height-2*margin
    roles={r:sorted((u for u in units if u['role']==r),key=_semantic_order)
           for r in ('copy','header','footer','visual','tag','action','decoration')}
    upper=[u for u in roles['tag'] if u['bounds'][1]<sh*.45 and width/height<1.35]
    vertical_flow=width/height<1.35
    upper_row=upper if vertical_flow else []
    if not vertical_flow:
        roles['header']+=upper
    roles['tag']=[u for u in roles['tag'] if u not in upper]
    side_header=width/height>=2.4 and bool(roles['header'])
    compact_bands=.75<=width/height<=1.3
    header,hh=_row([] if side_header else roles['header'],cw,ch*(.09 if compact_bands else .13),scale,gap)
    kicker,kh=_badge_row(upper_row,cw,ch*.09,scale*.82,gap,align='center')
    footer,fh=_footer_band(roles['footer'],cw,ch*(.06 if compact_bands else .10),scale,gap*.6)
    labels,lh=_action_band(roles['tag'],roles['action'],cw,ch*(.14 if compact_bands else .20),scale,gap,width/height<1.3)
    reserved=hh+kh+fh+lh+gap*sum(bool(v) for v in (hh,kh,fh,lh))
    reserved_limit=ch*(.34 if compact_bands else .53)
    if reserved>reserved_limit:
        shrink=reserved_limit/reserved
        header,hh=_row([] if side_header else roles['header'],cw,ch*(.09 if compact_bands else .13),scale*shrink,gap*shrink)
        kicker,kh=_badge_row(upper_row,cw,ch*.09,scale*.82*shrink,gap*shrink,align='center')
        footer,fh=_footer_band(roles['footer'],cw,ch*(.06 if compact_bands else .10),scale*shrink,gap*shrink*.6)
        labels,lh=_action_band(roles['tag'],roles['action'],cw,ch*(.14 if compact_bands else .20),scale*shrink,gap*shrink,width/height<1.3)
    top,bottom=margin,height-margin
    output=[]
    if hh:
        output.extend(_shift(header,margin,top)); top+=hh+gap
    if kh:
        output.extend(_shift(kicker,margin,top)); top+=kh+gap
    if fh:
        bottom-=fh; output.extend(_shift(footer,margin,bottom)); bottom-=gap
    if lh:
        bottom-=lh; output.extend(_shift(labels,margin,bottom)); bottom-=gap
    available=max(.001,bottom-top)
    gap=min(gap,available*.08)
    cx,cwidth=margin,cw
    if side_header:
        hw=cw*.15
        block,used=_logo_row(roles['header'],hw,available*.75,scale,gap)
        output.extend(_shift(block,margin,top+(available-used)/2))
        cx+=hw+gap; cwidth-=hw+gap
    copy,visuals=roles['copy'],roles['visual']
    if not copy or not visuals:
        block=_copy(copy,cwidth,available,scale,gap) if copy else _visuals(visuals,cwidth,available,scale,gap)
        if block:
            output.extend(_move(block,cx+(cwidth-(_bounds(block)[2]-_bounds(block)[0]))/2,top+max(0,(available-(_bounds(block)[3]-_bounds(block)[1]))/2)))
    elif mode=='split':
        vw=(cwidth-gap)*share; tw=cwidth-gap-vw
        text=_copy(copy,tw,available,scale,gap,align='left')
        art=_visuals(visuals,vw,available,scale,gap)
        art_left=sum((u['bounds'][0]+u['bounds'][2])/2 for u in visuals)/len(visuals)<sw*.45
        tx,vx=(cx+vw+gap,cx) if art_left else (cx,cx+tw+gap)
        output.extend(_move(text,tx,top+(available-_bounds(text)[3])/2))
        output.extend(_move(art,vx+(vw-(_bounds(art)[2]-_bounds(art)[0]))/2,top+(available-(_bounds(art)[3]-_bounds(art)[1]))/2))
    else:
        vh=(available-gap)*share; th=available-gap-vh
        text=_copy(copy,cwidth,th,scale,gap)
        art=_visuals(visuals,cwidth,vh,scale,gap)
        ty,vy=(top,top+th+gap) if mode=='copy-first' else (top+vh+gap,top)
        output.extend(_move(text,cx+(cwidth-(_bounds(text)[2]-_bounds(text)[0]))/2,ty+(th-(_bounds(text)[3]-_bounds(text)[1]))/2))
        output.extend(_move(art,cx+(cwidth-(_bounds(art)[2]-_bounds(art)[0]))/2,vy+(vh-(_bounds(art)[3]-_bounds(art)[1]))/2))
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


def _enrich_units(source, units):
    """Feature-based role hypotheses, with uncertainty kept internal.

    No ID/filename lookup. Being at the top alone does not make every image a
    logo, and colored rectangles alone are not automatically calls to action.
    """
    sw, sh = source['canvas']['width'], source['canvas']['height']
    image_units = [u for u in units if any(e['type'] == 'image' for e in u['items'])]
    image_areas=[(lambda box:(box[2]-box[0])*(box[3]-box[1]))(_visual_bounds([e]))
                 for u in image_units for e in u['items'] if e['type']=='image']
    largest_image=max(image_areas,default=1)
    text_units = [u for u in units if u['role'] == 'copy']
    headline = max(text_units, key=lambda u: (max(e.get('font_size', 0) for e in u['items']),
                   -u['bounds'][1], u['bounds'][2]-u['bounds'][0]), default=None)
    head_font = max((e.get('font_size', 0) for e in headline['items']), default=24) if headline else 24
    labels = [u for u in units if u['label']]
    for u in units:
        x, y, r, b = u['bounds']
        # Repeated compact pills on the same source row are badges, rather
        # than multiple CTAs. A taller, distinct button remains an action.
        angle=sum(e.get('rotation',0) for e in u['items'])/len(u['items'])
        u['corner_callout']=bool(u['label'] and abs(angle)>2 and y<sh*.25 and (x<sw*.2 or r>sw*.8))
        if u['corner_callout']:
            u.update(role='decoration',semantic='callout',confidence=.78)
            continue
        size=max((e.get('font_size',0) for e in u['items']),default=0)
        peers = [v for v in labels if v is not u and abs(v['bounds'][1]-y) < sh*.025
                 and .85 < (v['bounds'][3]-v['bounds'][1])/max(1,b-y) < 1.15
                 and abs(sum(e.get('rotation',0) for e in v['items'])/len(v['items'])-angle)<=2
                 and .85 < max((e.get('font_size',0) for e in v['items']),default=0)/max(1,size) < 1.15]
        if u['label'] and peers:
            u['peer_ids']=tuple(sorted(v['items'][0]['id'] for v in [u]+peers))
        if u['label'] and peers and b-y < sh*.09:
            u['role'] = 'tag'
        images = [e for e in u['items'] if e['type'] == 'image']
        small_image = images and not any(e['type'] == 'text' for e in u['items']) and (r-x)*(b-y) < sw*sh*.09
        near_top = (y+b)/2 < sh*.22
        near_corner = min(abs(x), abs(sw-r)) < sw*.12 and y < sh*.28
        relative_small = ((r-x)*(b-y)<sw*sh*.12 if len(image_units)==1 else max((_area(e) for e in images), default=0) < largest_image*.55)
        if u['role'] == 'header' or small_image and (near_top or near_corner) and (relative_small or b-y < sh*.15):
            u['role'], semantic, confidence = 'header', 'logo', .72
        elif u is headline:
            semantic, confidence = 'headline', .88
        elif u['role'] == 'copy':
            size = max(e.get('font_size', 0) for e in u['items'])
            words = sum(len(e.get('content', '').split()) for e in u['items'])
            if y > sh*.52 and words <= 8 and size < head_font*.65 and any(e.get('font_weight') == 'bold' for e in u['items']) and (y > sh*.72 or headline and any(e.get('color') != headline['items'][0].get('color') for e in u['items'])):
                u['role'], semantic, confidence = 'action', 'cta', .56
            elif headline and b <= headline['bounds'][1] and size < head_font*.55:
                semantic, confidence = 'kicker', .62
            else:
                semantic, confidence = ('subheading' if size >= head_font*.48 else 'body'), .62
        elif u['role'] == 'action':
            semantic, confidence = 'cta', .80 if u['label'] else .50
        elif u['role'] == 'tag':
            semantic, confidence = 'badge', .72
        elif u['role'] == 'footer':
            semantic, confidence = 'footer', .70
        elif u['role'] == 'decoration':
            semantic, confidence = 'decoration', .65
        else:
            semantic, confidence = 'art', .55
        u.update(semantic=semantic, confidence=confidence)
    visuals = [u for u in units if u['role'] == 'visual']
    if visuals:
        hero = max(visuals, key=lambda u: (u['bounds'][2]-u['bounds'][0])*(u['bounds'][3]-u['bounds'][1]))
        hero.update(semantic='hero', confidence=.80)
    for u in units:
        x,y,r,b=u['bounds']
        u['source_upper']=y<sh*.45
        u['source_centered']=u['semantic'] in ('logo','hero','headline','subheading','body','cta') and abs((x+r)/2/sw-.5)<.06
        if u['role']=='footer' and any(e['type']=='image' for e in u['items']) and (r-x)*(b-y)<sw*sh*.05:
            center=(x+r)/2/sw
            u['footer_anchor']='right' if center>.75 else 'left' if center<.25 else 'center'
    return units


def _aspect_state(source, width, height):
    ratio = width/height
    delta = ratio/(source['canvas']['width']/source['canvas']['height'])
    # Bucket only by the selected target format. Keep `flow=preserve` as a
    # separate signal for near-identical source/target ratios; format-specific
    # rules still come from the bucket and therefore remain predictable.
    if ratio >= 3.2:
        bucket='banner'
    elif ratio <= .68:
        bucket='story'
    elif .82 <= ratio <= 1.2:
        bucket='square'
    elif ratio > 1.2:
        bucket='wide'
    else:
        bucket='portrait'
    if abs(log(delta)) < .18:
        flow = 'preserve'
    else:
        flow = bucket
    return dict(flow=flow, bucket=bucket, delta=delta, ratio=ratio)


def _semantic_order(u):
    return ({'logo': 0, 'kicker': 1, 'headline': 2, 'subheading': 3, 'body': 4,
             'badge': 5, 'cta': 6, 'hero': 7, 'art': 8, 'footer': 9,
             'decoration': 10}.get(u['semantic'], 8), u['bounds'][1], u['bounds'][0])


def _banner_seed(units, width, height, scale, hero_share=.18, reverse=False, source_size=None):
    """True inline banner: logo | copy | action | hero, plus secondary rows."""
    margin, gap = max(min(width, height)*.05,width*.02), min(width, height)*.06
    cw, ch = width-2*margin, height-2*margin
    roles = {r: sorted((u for u in units if u['role'] == r), key=_semantic_order)
             for r in ('header','copy','action','tag','visual','footer','decoration')}
    footer, fh = _footer_band(roles['footer'], cw, ch*.12, scale, gap,
                              one_line_text=width/max(1,height)>=3)
    available = ch-fh-(gap if fh else 0)
    categories = [('header', .16), ('copy', .50), ('action', .34)]
    if roles['visual']:
        protected_right=any(u.get('corner_callout') and (u['bounds'][0]+u['bounds'][2])/2>(source_size or (width,height))[0]/2 for u in roles['decoration']) or any(u.get('footer_anchor')=='right' for u in roles['footer'])
        if protected_right:
            categories = [('header', .06), ('copy', .47), ('action', .16), ('visual', max(hero_share,.31))]
        else:
            categories = [('header', .12), ('copy', .36), ('action', .24), ('visual', max(hero_share,.28))]
    if reverse:
        categories = ([('header', .06), ('visual', max(hero_share,.31)), ('copy', .47), ('action', .16)]
                      if roles['visual'] and protected_right else
                      [('header', .12), ('visual', max(hero_share,.28)), ('copy', .36), ('action', .24)]
                      if roles['visual'] else categories)
    active = [(key, weight) for key, weight in categories if roles[key]]
    total = sum(weight for _, weight in active) or 1
    usable = cw-gap*max(0, len(active)-1)
    slot_widths={key:usable*weight/total for key,weight in active}
    if any(u['semantic']=='logo' for u in roles['header']) and len(active)>1:
        logo_min=max(cw*.12,min(cw*.18,available*.70))
        logo_width=min(usable,max(slot_widths.get('header',0),logo_min))
        other_weight=sum(weight for key,weight in active if key!='header') or 1
        slot_widths['header']=logo_width
        for key,weight in active:
            if key!='header':
                slot_widths[key]=(usable-logo_width)*weight/other_weight
    output, blocks = [], []
    for key, weight in active:
        w = slot_widths[key]
        if key == 'copy':
            local_roles={**roles,'header':[],'action':[],'visual':[],'footer':[],'decoration':[]}
            block=_editorial_copy(local_roles,w,available,scale,gap)
        elif key == 'header':
            # A banner logo should support the headline, not consume the full
            # short-side height. Alpha-aware fitting still uses only its mark.
            block,used=_logo_row(roles[key],w,available*.50,scale,gap,source_size,fill=True)
        elif key == 'visual':
            block = _visuals(roles[key], w, available, scale, gap)
        else:
            block, used = _banner_actions(roles[key], w, available*.85, scale, gap,cw)
        if block:
            box=_visual_bounds(block);bw,bh=box[2]-box[0],box[3]-box[1]
            blocks.append((key,block,bw,bh,box))
    right_clear=width-margin
    for u in roles['decoration']:
        if not u.get('corner_callout'):
            continue
        if source_size and (u['bounds'][0]+u['bounds'][2])/2<=source_size[0]/2:
            continue
        callout=_callout_block(u,width,height,*(source_size or (width,height)))
        cb=_visual_bounds(callout);edge=min(width,height)*.025
        right_clear=min(right_clear,width-edge-(cb[2]-cb[0])-gap)
    right_footers=[u for u in roles['footer'] if u.get('footer_anchor')=='right']
    if right_footers:
        footer_ids={e['id'] for u in right_footers for e in u['items']}
        fb=_visual_bounds([e for e in footer if e['id'] in footer_ids]) if any(e['id'] in footer_ids for e in footer) else None
        if fb:
            right_clear=min(right_clear,fb[0]-gap)
    # Use actual transparent-frame insets to establish a shared visible margin.
    # This keeps a padded PNG mark from appearing detached while text/actions
    # sit much closer to the opposite canvas edge.
    edge_insets=[]
    for _,block,_,_,visual_box in (blocks[:1]+blocks[-1:] if blocks else []):
        raw_box=_raw_bounds(block)
        edge_insets.extend((max(0.,visual_box[0]-raw_box[0]),
                            max(0.,raw_box[2]-visual_box[2])))
    content_margin=max([margin]+edge_insets)
    right_clear=min(right_clear,width-content_margin)
    # Pack measured artwork/text extents. A tight left-aligned copy column
    # must not reserve half the banner as invisible whitespace.
    sum_bw=sum(row[2] for row in blocks)
    pack_width=min(max(0,width-2*content_margin),max(0,right_clear-content_margin))
    if len(blocks)>=3:
        gap_x=max(gap,(pack_width-sum_bw)/max(1,len(blocks)-1))
        cursor=content_margin
    else:
        gap_x=min(pack_width*.08,max(gap,(pack_width-sum_bw)/max(1,len(blocks)+1)))
        occupied=sum_bw+gap_x*max(0,len(blocks)-1)
        cursor=content_margin+max(0,(pack_width-occupied)/2)
    for index,(key,block,bw,bh,box) in enumerate(blocks):
        if key=='visual' and index==len(blocks)-1 and right_clear<width-margin:
            available_slot=max(.001,right_clear-(cursor+gap))
            if available_slot<bw:
                block=_visuals(roles['visual'],available_slot,available,scale,gap)
                box=_visual_bounds(block);bw,bh=box[2]-box[0],box[3]-box[1]
                gap_x=min(gap_x,max(0,available_slot-bw))
                occupied=sum(row[2] for row in blocks if row[0]!='visual')+bw+gap_x*max(0,len(blocks)-1)
                if len(blocks)>=3:
                    cursor=content_margin+sum(row[2]+gap_x for row in blocks[:index])
                else:
                    cursor=content_margin+max(0,(pack_width-occupied)/2)+sum(row[2]+gap_x for row in blocks[:index])
        px=cursor-box[0]
        py=margin+(available-bh)/2-box[1]
        output.extend(_move_visual(block,px,py))
        cursor+=bw+gap_x
    if roles['tag'] and not roles['copy']:
        tags, _ = _row(roles['tag'], cw, available*.2, scale*.8, gap,align='left')
        output.extend(_shift(tags, margin, margin))
    if footer:
        output.extend(_shift(footer, margin, height-margin*.45-fh))
    output.extend(_decorations(roles['decoration'],width,height,
                  (source_size or (width,height))[0],(source_size or (width,height))[1],output))
    return output


def _stack_seed(units, width, height, scale, visual_share, hero_first, source_size):
    """Story/square flow with separate top badges and bottom CTA/footer bands."""
    working = deepcopy(units)
    # Use the existing text-aware packing mechanics, but keep top badges out
    # of the CTA row and preserve headline/body order independently of grouping.
    sw, sh = source_size
    # `_reflow` gives upper tags their own centered kicker row. Keep their
    # semantic role intact so they cannot shift a source-centered logo.
    return _reflow(working,width,height,sw,sh,'visual-first' if hero_first else 'copy-first',visual_share,scale)


def _centered_content_seed(units, width, height, scale, source_size):
    """Keep simple logo/copy/action compositions as one balanced, centered stack."""
    margin=min(width,height)*.06
    gap=min(width,height)*.055
    usable_w=width-2*margin
    roles={r:sorted((u for u in units if u['role']==r),key=_semantic_order)
           for r in ('header','copy','tag','action')}
    blocks=[]
    if roles['header']:
        block,_=_logo_row(roles['header'],usable_w,min(height*.16,usable_w*.28),scale,gap,fill=True)
        if block:blocks.append(block)
    if roles['copy']:
        block=_copy(roles['copy'],usable_w,height*.34,scale,gap,align='center')
        if block:blocks.append(block)
    if roles['tag'] or roles['action']:
        block,_=_action_band(roles['tag'],roles['action'],usable_w,height*.18,scale,gap,False,align='center')
        if block:blocks.append(block)
    if not blocks:
        return []
    sizes=[]
    for block in blocks:
        box=_visual_bounds(block)
        sizes.append((box[2]-box[0],box[3]-box[1]))
    # Use an even flex-like gap, but preserve each semantic block internally.
    total_h=sum(h for _,h in sizes)
    actual_gap=min(gap,max(0,(height*.66-total_h)/max(1,len(blocks)-1)))
    occupied=total_h+actual_gap*max(0,len(blocks)-1)
    # If content cannot fit the preferred cluster, scale each atomic block
    # together so text and related artwork keep their own proportions.
    if occupied>height*.72:
        factor=height*.72/occupied
        blocks=[_scale(block,factor) for block in blocks]
        sizes=[(w*factor,h*factor) for w,h in sizes]
        total_h=sum(h for _,h in sizes)
        actual_gap=min(gap*factor,max(0,(height*.66-total_h)/max(1,len(blocks)-1)))
        occupied=total_h+actual_gap*max(0,len(blocks)-1)
    cursor=(height-occupied)/2
    output=[]
    for block,(bw,bh) in zip(blocks,sizes):
        box=_visual_bounds(block)
        output.extend(_move_visual(block,(width-bw)/2,cursor))
        cursor+=bh+actual_gap
    output.extend(_decorations([u for u in units if u['role']=='decoration'],width,height,
                               source_size[0],source_size[1],output))
    return output


def _safe_grid(units, width, height, scale):
    """Last-resort disjoint slots, retaining every element and atomic group."""
    if not units:
        return []
    margin = min(width,height)*.025
    count = len(units)
    cols = max(1, min(count, round(sqrt(count*width/height))))
    rows = (count+cols-1)//cols
    cw, ch = (width-2*margin)/cols, (height-2*margin)/rows
    output = []
    for i, u in enumerate(sorted(units,key=_semantic_order)):
        block = _unit(u, cw*.92, ch*.92, scale, fill=u['role']=='visual')
        _,_,w,h = _bounds(block)
        output.extend(_move(block,margin+(i%cols)*cw+(cw-w)/2,margin+(i//cols)*ch+(ch-h)/2))
    return output


def _layered(a, b):
    """Only source-evidenced artwork layering; text-bearing units stay clear."""
    if a['role'] == 'decoration' or b['role'] == 'decoration':
        # A text-bearing corner callout gets separation from main content.
        if (a.get('corner_callout') and b['role']!='decoration') or (b.get('corner_callout') and a['role']!='decoration'):
            return False
        return True
    if any(e['type']=='text' for e in a['items']+b['items']):
        return False
    ar, br = a['bounds'], b['bounds']
    fraction = _overlap(ar,br)/max(1e-9,min((ar[2]-ar[0])*(ar[3]-ar[1]),(br[2]-br[0])*(br[3]-br[1])))
    return fraction > .30


def _records(elements, units):
    by_id = {e['id']: e for e in elements}
    return [(u, [by_id[e['id']] for e in u['items']]) for u in units]


def _unit_margin(box, width, height):
    # A strict 5% margin can be infeasible for a source-wide unit. Prefer it,
    # but never make an otherwise valid canvas impossible solely for padding.
    return max(0, min(min(width,height)*.045,
               (width-box[2]+box[0])/2, (height-box[3]+box[1])/2))


def _pair_gap(a, b, width, height):
    """Keep a banner CTA visibly separate from product artwork."""
    roles={a.get('semantic'),b.get('semantic')}
    if width/max(1,height)>=3 and 'cta' in roles and roles & {'hero','art'}:
        return max(min(width,height)*.008,height*.10)
    return min(width,height)*.008


def _feasible(elements, units, width, height):
    records = [(u,_visual_bounds(items)) for u,items in _records(elements,units)]
    raw_records=[(u,_raw_bounds(items)) for u,items in _records(elements,units)]
    for i,((u,b),(_,raw)) in enumerate(zip(records,raw_records)):
        if not all(isfinite(v) for v in b):
            return False
        margin = 0 if u['role']=='decoration' else _unit_margin(b,width,height)
        if b[0]<margin-1e-5 or b[1]<margin-1e-5 or b[2]>width-margin+1e-5 or b[3]>height-margin+1e-5:
            return False
        # Transparent pixels are ignored for layout margins, but the original
        # element frame must remain on-canvas for renderer/schema validity.
        if raw[0]<-1e-5 or raw[1]<-1e-5 or raw[2]>width+1e-5 or raw[3]>height+1e-5:
            return False
        for v,c in records[:i]:
            if _layered(u,v):
                continue
            gap=_pair_gap(u,v,width,height)
            if not (b[2]+gap<=c[0]+1e-5 or c[2]+gap<=b[0]+1e-5 or b[3]+gap<=c[1]+1e-5 or c[3]+gap<=b[1]+1e-5):
                return False
    return True


def _constrain(elements, units, width, height, topology='nearest'):
    """Cassowary placement with required bounds/separation and soft preferences.

    Non-overlap is disjunctive, not one linear equation. Each candidate chooses
    its separating axes first; Kiwi then solves that particular topology. An
    infeasible topology is rejected rather than silently weakening constraints.
    """
    try:
        import kiwisolver as kiwi
    except ImportError:
        return None
    records = _records(elements, units)
    boxes = [_visual_bounds(items) for _,items in records]
    solver = kiwi.Solver()
    xs = [kiwi.Variable(f'x{i}') for i in range(len(records))]
    ys = [kiwi.Variable(f'y{i}') for i in range(len(records))]
    try:
        for i, ((u,_),box) in enumerate(zip(records,boxes)):
            x,y,r,b = box
            w,h = r-x,b-y
            margin = 0 if u['role']=='decoration' else _unit_margin(box,width,height)
            solver.addConstraint(xs[i]>=margin)
            solver.addConstraint(ys[i]>=margin)
            solver.addConstraint(xs[i]+w<=width-margin)
            solver.addConstraint(ys[i]+h<=height-margin)
            raw=_raw_bounds(records[i][1])
            solver.addConstraint(xs[i]+raw[0]-x>=0)
            solver.addConstraint(ys[i]+raw[1]-y>=0)
            solver.addConstraint(xs[i]+raw[2]-x<=width)
            solver.addConstraint(ys[i]+raw[3]-y<=height)
            if u.get('corner_callout'):
                sw,sh=u.get('source_canvas',(width,height))
                sx,sy,sr,_=u['bounds']
                edge=min(width,height)*.025
                anchor_x=width-edge-w if (sx+sr)/2>sw/2 else edge
                solver.addConstraint(xs[i]==anchor_x)
                solver.addConstraint(ys[i]==edge)
            else:
                solver.addConstraint((xs[i]==x) | kiwi.strength.strong)
                solver.addConstraint((ys[i]==y) | kiwi.strength.strong)
        for i,(u,_) in enumerate(records):
            a=boxes[i]; aw,ah=a[2]-a[0],a[3]-a[1]
            for j,(v,_) in enumerate(records[:i]):
                if _layered(u,v):
                    continue
                gap=_pair_gap(u,v,width,height)
                b=boxes[j]; bw,bh=b[2]-b[0],b[3]-b[1]
                # Prefer the closest separating direction, in normalized units.
                choices=[((a[2]+gap-b[0])/width, xs[i]+aw+gap<=xs[j]),
                         ((b[2]+gap-a[0])/width, xs[j]+bw+gap<=xs[i]),
                         ((a[3]+gap-b[1])/height, ys[i]+ah+gap<=ys[j]),
                         ((b[3]+gap-a[1])/height, ys[j]+bh+gap<=ys[i])]
                if topology != 'nearest':
                    # Centers give consistent direction on each chosen axis.
                    dx=(a[0]+a[2]-b[0]-b[2])/width
                    dy=(a[1]+a[3]-b[1]-b[3])/height
                    horizontal=abs(dx)>=abs(dy)
                    if topology=='alternate' and _overlap(a,b)>0:
                        horizontal=not horizontal
                    choice=(1 if dx>=0 else 0) if horizontal else (3 if dy>=0 else 2)
                    solver.addConstraint(choices[choice][1])
                else:
                    solver.addConstraint(min(choices,key=lambda p:p[0])[1])
                if abs(a[0]-b[0])<width*.015:
                    solver.addConstraint((xs[i]==xs[j]) | kiwi.strength.weak)
                elif abs((a[0]+a[2])-(b[0]+b[2]))<width*.025:
                    solver.addConstraint((xs[i]+aw/2==xs[j]+bw/2) | kiwi.strength.weak)
        solver.updateVariables()
    except kiwi.UnsatisfiableConstraint:
        return None
    output=[]
    for i,(_,items) in enumerate(records):
        output.extend(_move_visual(deepcopy(items),xs[i].value(),ys[i].value()))
    return output if _feasible(output,units,width,height) else None


def _perceptual_context(source, backgrounds, units, width, height):
    """Conservative geometry affiliations and an image-busyness proxy, not semantics."""
    sw,sh=source['canvas']['width'],source['canvas']['height']
    icons=[u for u in units if len(u['items'])==1 and u['items'][0]['type'] in ('image','shape')
           and not u['items'][0].get('group_id') and u['role'] not in ('decoration','footer')
           and (u['bounds'][2]-u['bounds'][0])*(u['bounds'][3]-u['bounds'][1])<sw*sh*.065]
    captions=[u for u in units if u['role']=='copy' and u['semantic'] not in ('headline','kicker')
              and len(u['items'])==1 and not u['items'][0].get('group_id')
              and len(u['items'][0].get('content','').split())<=14]
    proposals=[]
    for icon in icons:
        a=icon['bounds'];iw,ih=a[2]-a[0],a[3]-a[1]
        options=[]
        for caption in captions:
            b=caption['bounds'];tw,th=b[2]-b[0],b[3]-b[1]
            below=b[1]>=a[3]-1 and b[1]-a[3]<min(sw,sh)*.065 and abs((a[0]+a[2]-b[0]-b[2])/2)<max(iw,tw)*.25
            beside=b[0]>=a[2]-1 and b[0]-a[2]<min(sw,sh)*.065 and abs((a[1]+a[3]-b[1]-b[3])/2)<max(ih,th)*.3
            if below or beside:
                distance=(max(0,b[1]-a[3])/sh if below else max(0,b[0]-a[2])/sw)
                options.append((distance,caption,'below' if below else 'beside'))
        if options:
            distance,caption,orientation=min(options,key=lambda p:p[0])
            proposals.append((distance,icon,caption,orientation))
    selected=[];used=set()
    for distance,icon,caption,orientation in sorted(proposals,key=lambda p:p[0]):
        ids=(icon['items'][0]['id'],caption['items'][0]['id'])
        if not used.intersection(ids):
            selected.append(dict(icon=ids[0],caption=ids[1],orientation=orientation,
                                 width=icon['bounds'][2]-icon['bounds'][0],height=icon['bounds'][3]-icon['bounds'][1]))
            used.update(ids)
    # Require at least two similarly sized repeated pairs before inferring intent.
    affiliations=[]
    for pair in selected:
        peers=[v for v in selected if v is not pair and v['orientation']==pair['orientation']
               and .5<v['width']/max(.001,pair['width'])<2 and .5<v['height']/max(.001,pair['height'])<2]
        if peers:
            affiliations.append({k:v for k,v in pair.items() if k not in ('width','height')})
    context=dict(affinities=affiliations,background_map=[],background_note='No image background to analyze')
    if not any(e['type']=='image' for e in backgrounds) or width*height>16000000:
        return context
    try:
        from renderer.converter.convert import render_design_to_png
        from PIL import Image, ImageFilter, ImageStat
        items=deepcopy(backgrounds)
        for e in items:
            e.update(x=0,y=0,width=width,height=height)
            if e['type']=='image':e['fit']='cover'
        with TemporaryDirectory(prefix='chameleon-background-') as directory:
            path=Path(directory)/'background.png'
            warnings=render_design_to_png(dict(canvas=dict(width=width,height=height),elements=items),
                         Path(__file__).resolve().parents[1]/'brand_kit'/'assets',path,max_size=96)
            # Missing imagery must not turn placeholder edges into a subject guess.
            if warnings:
                context['background_note']='Background analysis skipped: renderer conversion warnings'
                return context
            with Image.open(path) as image:
                pixels=image.convert('RGB').resize((96,96))
            cells=[]
            for row in range(3):
                for col in range(3):
                    tile=pixels.crop((col*32,row*32,(col+1)*32,(row+1)*32))
                    edges=tile.convert('L').filter(ImageFilter.FIND_EDGES).crop((1,1,31,31))
                    edge=ImageStat.Stat(edges).mean[0]/255
                    variance=sum(ImageStat.Stat(tile).var)/(3*255**2)
                    cells.append(min(1,edge*3+variance*4))
            if max(cells)-min(cells)>.015:
                context['background_map']=cells
            context['background_note']='3x3 edge/color-variance busyness proxy; no subject recognition'
    except Exception as exc:
        context['background_note']=f'Optional background analysis unavailable: {exc}'
    return context


def _affiliated_seed(units, context, width, height, sw, sh, scale):
    """Candidate-only composites; the solver still sees independent source units."""
    by_id={e['id']:u for u in units for e in u['items']}
    paired=set();working=[]
    for pair in context['affinities']:
        a,b=by_id[pair['icon']],by_id[pair['caption']]
        items=deepcopy(a['items']+b['items'])
        caption=items[-1]
        if pair['orientation']=='below' and '\n' not in caption.get('content',''):
            # Candidate-only caption widening, while its icon keeps proportion.
            caption['width']=max(caption['width'],min(sw*.4,_text(caption,1e9)[1]+caption['font_size']*.15))
            icon_box=_bounds([items[0]])
            caption['x']=(icon_box[0]+icon_box[2]-caption['width'])/2
            caption['height']=_text(caption)[0]
        working.append(dict(items=items,bounds=_bounds(items),role='visual',semantic='art',label=False))
        paired.update((pair['icon'],pair['caption']))
    if working:
        # Repeated feature cells share one fitting envelope, so caption length
        # alone does not make identical source icons scale differently.
        cell_w=max(u['bounds'][2]-u['bounds'][0] for u in working)
        cell_h=max(u['bounds'][3]-u['bounds'][1] for u in working)
        for u in working:
            u['fit_size']=(cell_w,cell_h)
    working.extend(u for u in units if not any(e['id'] in paired for e in u['items']))
    return _reflow(working,width,height,sw,sh,'copy-first',.60,scale)


def _quiet_seed(elements, units, width, height, cell, scale=1.):
    """Repack quiet copy/action and artwork in complementary regions.

    A cell is a preference within a region, not a tiny mandatory text box.
    Header/footer get their own bands; projection remains the final hard gate.
    """
    roles={r:sorted((u for u in units if u['role']==r),key=_semantic_order)
           for r in ('header','footer','copy','action','tag','visual','decoration')}
    upper=[u for u in roles['tag'] if u.get('source_upper')]
    roles['header']+=upper
    roles['tag']=[u for u in roles['tag'] if u not in upper]
    margin,gap=min(width,height)*.05,min(width,height)*.035
    cw,ch=width-2*margin,height-2*margin
    header,hh=_row(roles['header'],cw,ch*.12,scale,gap)
    footer,fh=_footer_band(roles['footer'],cw,ch*.10,scale,gap)
    top=margin+hh+(gap if hh else 0)
    bottom=height-margin-fh-(gap if fh else 0)
    available=max(.001,bottom-top)
    col,row=cell%3,cell//3
    horizontal=(col!=1 and width/height>=.85) or row==1
    copy_rect=(margin,top,cw,available)
    art_rect=None
    if roles['visual'] and not (roles['copy'] or roles['action'] or roles['tag']):
        art_rect=(margin,top,cw,available)
    elif roles['visual']:
        if horizontal:
            tw=max(.001,(cw-gap)*.58);vw=max(.001,cw-gap-tw)
            left=col!=2
            copy_rect=(margin if left else margin+vw+gap,top,tw,available)
            art_rect=(margin+tw+gap if left else margin,top,vw,available)
        else:
            th=max(.001,(available-gap)*.56);vh=max(.001,available-gap-th)
            above=row!=2
            copy_rect=(margin,top if above else top+vh+gap,cw,th)
            art_rect=(margin,top+th+gap if above else top,cw,vh)
    x,y,w,h=copy_rect
    actions,ah=_action_band(roles['tag'],roles['action'],w,h*.32,scale,gap,width/height<1.3)
    copy=_copy(roles['copy'],w,max(.001,h-ah-(gap if ah else 0)),scale,gap,
               align='left' if horizontal else None)
    content=[]
    copy_h=_bounds(copy)[3]-_bounds(copy)[1]
    if copy:content.extend(_move(copy,0,0))
    if actions:
        action_w=_bounds(actions)[2]-_bounds(actions)[0]
        content.extend(_move(actions,0 if horizontal else (w-action_w)/2,copy_h+(gap if copy else 0)))
    # Fit the assembled compact cluster together if extreme budgets demand it.
    content=_contain(content,w,h)
    _,_,bw,bh=_bounds(content)
    cx=(col+.5)*width/3;cy=(row+.5)*height/3
    px=max(x,min(x+w-bw,cx-bw/2));py=max(y,min(y+h-bh,cy-bh/2))
    output=_move(content,px,py)
    if art_rect:
        ax,ay,aw,ah=art_rect
        art=_visuals(roles['visual'],aw,ah,scale,gap)
        _,_,bw,bh=_bounds(art)
        output.extend(_move(art,ax+(aw-bw)/2,ay+(ah-bh)/2))
    if header:output.extend(_shift(header,margin,margin))
    if footer:output.extend(_shift(footer,margin,height-margin-fh))
    decorative_ids={e['id'] for u in roles['decoration'] for e in u['items']}
    output.extend(deepcopy(e) for e in elements if e['id'] in decorative_ids)
    return output


def _perceptual_terms(records, boxes, units, width, height):
    terms=dict(affinity=0.,whitespace=0.,background_busyness=0.)
    context=units[0].get('perceptual',{}) if units else {}
    by_id={e['id']:e for _,items in records for e in items}
    for pair in context.get('affinities',[]):
        a,b=_bounds([by_id[pair['icon']]]),_bounds([by_id[pair['caption']]])
        dx=max(0,a[0]-b[2],b[0]-a[2])/width
        dy=max(0,a[1]-b[3],b[1]-a[3])/height
        distance=sqrt(dx*dx+dy*dy)
        terms['affinity']+=max(0,distance-.035)**2*8
        # Soft neighborhood exclusivity: its own icon should remain nearest.
        own=sqrt(((a[0]+a[2]-b[0]-b[2])/2/width)**2+((a[1]+a[3]-b[1]-b[3])/2/height)**2)
        others=[_bounds([by_id[p['icon']]]) for p in context['affinities'] if p is not pair]
        nearest=min((sqrt(((v[0]+v[2]-b[0]-b[2])/2/width)**2+((v[1]+v[3]-b[1]-b[3])/2/height)**2) for v in others),default=own)
        terms['affinity']+=max(0,own-nearest)
    main=[box for (u,_),box in zip(records,boxes) if u['role'] not in ('decoration','footer','header')]
    if len(main)>=2:
        # Only large central voids get a small penalty; don't demand a filled grid.
        for nx,ny in ((.33,.33),(.67,.33),(.33,.67),(.67,.67)):
            distance=min(sqrt((max(0,b[0]/width-nx,nx-b[2]/width))**2+
                              (max(0,b[1]/height-ny,ny-b[3]/height))**2) for b in main)
            terms['whitespace']+=max(0,distance-.22)**2
    cells=context.get('background_map',[])
    if cells:
        for (u,items),b in zip(records,boxes):
            if not any(e['type']=='text' for e in items):continue
            area=max(1,(b[2]-b[0])*(b[3]-b[1]))
            for i,cost in enumerate(cells):
                cell=((i%3)*width/3,(i//3)*height/3,(i%3+1)*width/3,(i//3+1)*height/3)
                terms['background_busyness']+=cost*_overlap(b,cell)/area
    return terms


def _text_right_limit(unit,box,records,width,height):
    """Horizontal room at this anchor, respecting neighboring occupied rows."""
    limit=width-min(width,height)*.045;gap=min(width,height)*.008
    for other,items in records:
        if other is unit or _layered(unit,other):continue
        b=_visual_bounds(items)
        if min(box[3],b[3])>max(box[1],b[1]) and b[0]>=box[2]-1e-6:
            limit=min(limit,b[0]-gap)
    return max(box[2],limit)


def _banner_vertical_neighbor(unit,box,records):
    """Whether a unit shares a copy column with another vertically stacked unit."""
    left,top,right,bottom=box
    for other,items in records:
        if other is unit or other['role'] not in ('header','copy','tag','action','visual'):
            continue
        other_box=_visual_bounds(items)
        overlap=max(0,min(right,other_box[2])-max(left,other_box[0]))
        if overlap>=min(right-left,other_box[2]-other_box[0])*.2:
            return True
    return False


def _banner_centerline(records,width,height):
    margin=min(width,height)*.05
    footer_tops=[_visual_bounds(items)[1] for unit,items in records if unit['role']=='footer']
    bottom=min(footer_tops)-min(width,height)*.035 if footer_tops else height-margin
    return (margin+max(margin,bottom))/2


def _energy_terms(elements, units, source, width, height, desired):
    records = _records(elements,units)
    boxes = [_visual_bounds(items) for _,items in records]
    state=_aspect_state(source,width,height)
    terms = dict(overlap=0.,boundary=0.,text_fit=0.,readability=0.,balance=0.,
                 hierarchy=0.,alignment=0.,spacing=0.,canvas_use=0.,source_fidelity=0.)
    terms.update(rag=0.,affinity=0.,whitespace=0.,background_busyness=0.,axial=0.,badge_flow=0.,footer_anchor=0.,hero_prominence=0.,brand_visibility=0.,column_alignment=0.,avoidable_wrap=0.,banner_action_stack=0.,banner_gap_evenness=0.,banner_centering=0.,centered_flow=0.)
    masses=[]
    for i,((u,items),box) in enumerate(zip(records,boxes)):
        x,y,r,b=box; area=max(1e-9,(r-x)*(b-y))
        if not all(isfinite(v) for v in box) or any(min(e['width'],e['height'])<0 for e in items):
            return {**terms,'boundary':1e9}
        terms['boundary'] += 1-_overlap(box,(0,0,width,height))/area
        for (v,_),other in zip(records[:i],boxes[:i]):
            if not _layered(u,v):
                other_area=max(1e-9,(other[2]-other[0])*(other[3]-other[1]))
                terms['overlap'] += _overlap(box,other)/min(area,other_area)
        if u['role']!='decoration':
            # Perceptual mass: sqrt(area) avoids a large blank textbox dominating.
            mass=sqrt(area)*(.2 if u.get('footer_anchor') in ('left','right') else .4 if u['role']=='footer' else 1)
            masses.append((mass,(x+r)/2/width,(y+b)/2/height))
        for original,e in zip(u['items'],items):
            if e['type']!='text':
                continue
            th,tw,lines=_text(e)
            terms['text_fit'] += max(0,th/max(.001,e['height'])-1)+max(0,tw/max(.001,e['width'])-1)
            if width/height>=1.35 and lines>1 and '\n' not in e.get('content','') and (u['semantic']=='headline' or len(e.get('content','').split())<=14) and u['role'] in ('copy','action'):
                room=_text_right_limit(u,box,records,width,height)-x
                possible=_text(e,max(e['width'],room))[2]
                terms['avoidable_wrap']+=max(0,(lines-possible)/lines)
            if e.get('max_lines'):
                terms['text_fit'] += max(0,lines-e['max_lines'])
            expected=_target_font(original,u,desired)
            if u['semantic'] in ('headline','subheading') or (u['role']=='copy' and len(e.get('content','').split())<=14):
                terms['rag']+=_rag(e)
            importance=2 if u['semantic']=='headline' else 1
            terms['readability'] += importance*max(0,log(expected/max(1,e['font_size'])))**2
            terms['readability'] += max(0,min(8,min(width,height)*.065)-e['font_size'])/4
        if u['semantic']=='hero' and width/height<3.2:
            source_box=u['bounds'];source_share=(source_box[2]-source_box[0])*(source_box[3]-source_box[1])/(source['canvas']['width']*source['canvas']['height'])
            floor=source_share*.45
            terms['hero_prominence']+=max(0,(floor-area/(width*height))/max(1e-9,floor))**2
            if state['bucket'] in ('square','portrait','story','wide'):
                source_height=(source_box[3]-source_box[1])/source['canvas']['height']
                target_height=(b-y)/height
                minimum=max(source_height*.72,{'square':.42,'portrait':.27,'story':.25,'wide':.28}[state['bucket']])
                shortfall=max(0,(minimum-target_height)/max(.01,minimum))
                terms['hero_prominence']+=shortfall if state['bucket']=='square' else shortfall**2
        if u['semantic']=='logo' and width/height>=2.4:
            old=u['bounds'];ratio=(old[2]-old[0])/max(1e-9,old[3]-old[1])
            min_h=min(height*.16,(old[3]-old[1])*desired*.75)
            min_w=min(width*.07,(old[2]-old[0])*desired*.75)
            terms['brand_visibility']+=max(0,(min_h-(b-y))/max(1,min_h))**2
            terms['brand_visibility']+=max(0,(min_w-(r-x))/max(1,min_w))**2
            wanted=min(height*.60,width*.15/max(1e-9,ratio),(old[3]-old[1])*desired*1.8)
            terms['brand_visibility']+=max(0,(wanted-(b-y))/max(1,wanted))**2
        if state['bucket']=='banner' and u['role'] in ('header','copy','tag','action','visual') and not _banner_vertical_neighbor(u,box,records):
            center=(y+b)/2
            terms['banner_centering']+=((center-_banner_centerline(records,width,height))/height)**2
        if _axis_active(u,source,width,height):
            terms['axial']+=((x+r)/2/width-.5)**2*(2 if u['semantic'] in ('logo','hero','headline','cta') else 1)
        if u.get('footer_anchor') in ('left','right'):
            margin=min(width,height)*.05
            terms['footer_anchor']+=((x-margin)/width)**2 if u['footer_anchor']=='left' else ((r-width+margin)/width)**2
    if masses:
        total=sum(m for m,_,_ in masses)
        terms['balance']=((sum(m*x for m,x,_ in masses)/total-.5)**2+(sum(m*y for m,_,y in masses)/total-.5)**2)
    copy=[(u,items,b) for (u,items),b in zip(records,boxes) if u['role']=='copy']
    copy.sort(key=lambda row:_semantic_order(row[0]))
    for (a,ai,ab),(b,bi,bb) in zip(copy,copy[1:]):
        # Heading/body order is vertical within their column, even in banners.
        horizontal_overlap=max(0,min(ab[2],bb[2])-max(ab[0],bb[0]))
        if horizontal_overlap>0:
            terms['hierarchy']+=max(0,(ab[3]-bb[1])/height)
            terms['alignment']+=min(abs(ab[0]-bb[0]),abs(ab[2]-bb[2]),abs((ab[0]+ab[2]-bb[0]-bb[2])/2))/width
            preferred=max(min(width,height)*.022,min(e.get('font_size',12) for e in ai+bi)*.65)
            actual=bb[1]-ab[3]
            terms['spacing']+=(max(0,preferred-actual)/max(1,preferred))**2
        if a['semantic']=='headline' and b['semantic'] in ('subheading','body'):
            af=max(e.get('font_size',1) for e in ai); bf=max(e.get('font_size',1) for e in bi)
            original_ratio=max(e.get('font_size',1) for e in a['items'])/max(e.get('font_size',1) for e in b['items'])
            if original_ratio>1.25:
                terms['hierarchy']+=max(0,min(original_ratio,1.6)-af/max(1,bf))
    if width/height>=1.35:
        left_copy=[b for (u,items),b in zip(records,boxes) if u['role']=='copy' and any(e.get('text_align','left')=='left' for e in items)]
        copy_boxes=[b for (u,_),b in zip(records,boxes) if u['role']=='copy']
        if copy_boxes:
            copy_top=min(b[1] for b in copy_boxes)
            for (u,_),b in zip(records,boxes):
                if u['role']=='tag' and u.get('source_upper'):
                    terms['hierarchy']+=max(0,(b[3]-copy_top)/height)
        if left_copy:
            axis=min(b[0] for b in left_copy)
            terms['column_alignment']+=sum(abs(b[0]-axis)/width for b in left_copy)
            tags=[b for (u,_),b in zip(records,boxes) if u['role']=='tag']
            terms['column_alignment']+=sum(abs(b[0]-axis)/width for b in tags)
            if state['bucket']!='banner':
                actions=[b for (u,_),b in zip(records,boxes) if u['role']=='action']
                terms['column_alignment']+=sum(abs(b[0]-axis)/width for b in actions)
    if state['bucket']=='banner':
        copy_boxes=[b for (u,_),b in zip(records,boxes) if u['role']=='copy']
        action_boxes=[b for (u,_),b in zip(records,boxes) if u['role']=='action']
        tag_count=sum(u['role']=='tag' for u,_ in records)
        if len(copy_boxes)+len(action_boxes)+tag_count>=3:
            for action in action_boxes:
                aw=max(1.,action[2]-action[0])
                for copy in copy_boxes:
                    overlap=max(0.,min(action[2],copy[2])-max(action[0],copy[0]))
                    terms['banner_action_stack']+=overlap/min(aw,max(1.,copy[2]-copy[0]))
        columns=[]
        for roles in ({'logo'},{'headline','subheading','body','kicker'},{'cta'},{'hero','art'}):
            member_boxes=[box for (u,_),box in zip(records,boxes)
                          if u['semantic'] in roles or roles=={'headline','subheading','body','kicker'} and u['role']=='tag']
            if member_boxes:
                columns.append((min(box[0] for box in member_boxes),max(box[2] for box in member_boxes)))
        columns.sort()
        gaps=[b[0]-a[1] for a,b in zip(columns,columns[1:])]
        if len(gaps)>1:
            mean=sum(gaps)/len(gaps)
            normalizer=max(min(width,height)*.04,abs(mean),1.)
            terms['banner_gap_evenness']=sum(((gap-mean)/normalizer)**2 for gap in gaps)/len(gaps)
    # Simple centered ads benefit from a shared axis and a compact, even stack.
    # Score only independent semantic bands; the contents of linked units remain atomic.
    simple_centered=(state['bucket']!='banner') and not any(
        u['role']=='visual' or u['semantic'] in ('hero','art') for u,_ in records)
    if simple_centered:
        bands=[]
        for roles in ({'logo'},{'headline','subheading','body','kicker'},{'cta'}):
            members=[box for (u,_),box in zip(records,boxes) if u['semantic'] in roles]
            if members:
                bands.append((min(b[0] for b in members),min(b[1] for b in members),
                              max(b[2] for b in members),max(b[3] for b in members)))
        if len(bands)>=2:
            centers=[((b[0]+b[2])/2,(b[1]+b[3])/2) for b in bands]
            terms['centered_flow']+=sum(((x-width/2)/width)**2 for x,_ in centers)*5
            cluster_top=min(b[1] for b in bands);cluster_bottom=max(b[3] for b in bands)
            max_cluster=.42 if state['bucket']=='story' else .64
            terms['centered_flow']+=max(0,(cluster_bottom-cluster_top)/height-max_cluster)**2*8
            gaps=[b[1]-a[3] for a,b in zip(bands,bands[1:])]
            if len(gaps)>1:
                mean=sum(gaps)/len(gaps)
                terms['centered_flow']+=sum(((g-mean)/height)**2 for g in gaps)*5
            terms['centered_flow']+=((sum((b[1]+b[3])/2 for b in bands)/len(bands)-height/2)/height)**2*3
    def role_bounds(roles):
        return _visual_bounds([e for (u,items) in records if u['semantic'] in roles for e in items])
    flow_roles=[{'logo'},{'headline','subheading','body','kicker'},{'cta'},{'hero','art'}]
    if state['bucket']=='story':
        flow_roles=[{'logo'},{'hero','art'},{'headline','subheading','body','kicker'},{'cta'}]
    flow=[]
    for roles in flow_roles:
        if any(u['semantic'] in roles for u,_ in records):
            flow.append(role_bounds(roles))
    if state['bucket'] in ('banner','story'):
        axis,extent=(0,width) if state['bucket']=='banner' else (1,height)
        for a,b in zip(flow,flow[1:]):
            ac=(a[axis]+a[axis+2])/2;bc=(b[axis]+b[axis+2])/2
            terms['hierarchy']+=max(0,(ac-bc)/extent)*2
    main=[b for (u,_),b in zip(records,boxes) if u['role'] not in ('decoration','footer')]
    if main:
        left,top=min(b[0] for b in main),min(b[1] for b in main)
        right,bottom=max(b[2] for b in main),max(b[3] for b in main)
        min_height_use=.40 if simple_centered and state['bucket']=='story' else .76
        terms['canvas_use']=(max(0,.80-(right-left)/width)**2+max(0,min_height_use-(bottom-top)/height)**2)
    terms['badge_flow']=_badge_flow(records,width,height)
    terms.update(_perceptual_terms(records,boxes,units,width,height))
    change=abs(log(state['delta']))
    if change<.22:
        for (u,_),b in zip(records,boxes):
            a=u['bounds']
            terms['source_fidelity']+=(abs((b[0]+b[2])/width-(a[0]+a[2])/source['canvas']['width'])+
                                      abs((b[1]+b[3])/height-(a[1]+a[3])/source['canvas']['height']))*(1-change/.22)
    return terms


_ENERGY_WEIGHTS=dict(overlap=1000.,boundary=1500.,text_fit=500.,readability=13.,
                     balance=18.,hierarchy=30.,alignment=16.,spacing=8.,
                     canvas_use=35.,source_fidelity=18.,rag=5.,affinity=18.,whitespace=5.,background_busyness=7.,axial=160.,badge_flow=45.,footer_anchor=100.,hero_prominence=210.,brand_visibility=45.,column_alignment=120.,avoidable_wrap=25.,banner_action_stack=180.,banner_gap_evenness=80.,banner_centering=80.,centered_flow=95.)


def _energy(elements,units,source,width,height,desired):
    return sum(value*_ENERGY_WEIGHTS[name] for name,value in
               _energy_terms(elements,units,source,width,height,desired).items())


def _constraint_violation(elements, units, width, height):
    """Continuous repair cost, including required gaps that IoU alone misses."""
    records=[(u,_visual_bounds(items)) for u,items in _records(elements,units)]
    value=0.
    raw_records=[(u,_raw_bounds(items)) for u,items in _records(elements,units)]
    for i,((u,b),(_,raw)) in enumerate(zip(records,raw_records)):
        margin=0 if u['role']=='decoration' else _unit_margin(b,width,height)
        value+=(max(0,margin-b[0])+max(0,b[2]-width+margin))/width
        value+=(max(0,margin-b[1])+max(0,b[3]-height+margin))/height
        value+=(max(0,-raw[0])+max(0,raw[2]-width))/width
        value+=(max(0,-raw[1])+max(0,raw[3]-height))/height
        for v,c in records[:i]:
            if not _layered(u,v):
                gap=_pair_gap(u,v,width,height)
                value+=max(0,min((b[2]+gap-c[0])/width,(c[2]+gap-b[0])/width,
                                 (b[3]+gap-c[1])/height,(c[3]+gap-b[1])/height))
    return value


def _polish(elements, units, source, width, height, desired, seed=0, iterations=36):
    """Bounded seeded annealing over atomic units, not per-element jitter.

    Wrapping and disjunctive separation are nonsmooth; a gradient-only optimizer
    would not enforce them. Keep the best feasible result, even if an annealing
    step temporarily accepts a slightly worse arrangement. Infeasible starts
    first reduce hard-constraint violations, including missing separation gaps.
    """
    rng=Random(seed)
    state=_aspect_state(source,width,height)
    current=deepcopy(elements)
    current_energy=_energy(current,units,source,width,height,desired)
    current_feasible=_feasible(current,units,width,height)
    violation=_constraint_violation(current,units,width,height)
    best,best_energy=deepcopy(current),current_energy
    best_feasible=current_feasible
    moving=[u for u in units if (u['role']!='decoration' or not current_feasible)
            and not u.get('corner_callout')]
    if not moving:
        return current,dict(before=current_energy,after=current_energy,accepted=0,iterations=0)
    accepted=0
    for step in range(iterations):
        u=moving[step%len(moving)] if step<len(moving)*2 else rng.choice(moving)
        ids={e['id'] for e in u['items']}
        block=[e for e in current if e['id'] in ids]
        bounds=_visual_bounds(block); x,y,r,b=bounds
        cooling=1-step/max(1,iterations)
        motion=rng.randrange(5)
        proposal=deepcopy(block)
        if motion<2:
            axis=motion
            current_records=_records(current,units)
            extent=width if axis==0 else height
            old=x if axis==0 else y
            size=r-x if axis==0 else b-y
            targets=[min(width,height)*.045,(extent-size)/2,extent-min(width,height)*.045-size]
            # Also snap to other units' matching edges/centers.
            for v,other in _records(current,units):
                if v is u:
                    continue
                box=_visual_bounds(other)
                targets.extend([box[axis],box[axis+2]-size,(box[axis]+box[axis+2]-size)/2])
            targets=[v for v in targets if abs(v-old)>extent*.001]
            if axis==0 and u.get('footer_anchor') in ('left','right'):
                margin=min(width,height)*.05
                target=margin if u['footer_anchor']=='left' else width-margin-size
            elif axis==0 and state['bucket']=='banner' and u['semantic']=='logo':
                target=min(width*.05,max(0,width-size))
            elif axis==0 and _axis_active(u,source,width,height) and u['semantic'] in ('logo','hero'):
                target=(width-size)/2
            elif axis==0 and state['bucket']=='banner' and u['role'] in ('copy','tag'):
                is_left_column=u['role']=='tag' or any(e.get('text_align')=='left' for e in block)
                anchors=[_visual_bounds(other)[0] for v,other in current_records if v is not u and v['role'] in ('copy','tag')
                         and (v['role']=='tag' or any(e.get('text_align')=='left' for e in other))]
                target=min(anchors,key=lambda value:abs(value-old)) if is_left_column and anchors else old
            elif axis==1 and state['bucket']=='banner' and not _banner_vertical_neighbor(u,bounds,current_records):
                target=_banner_centerline(current_records,width,height)-size/2
            else:
                target=min(targets,key=lambda v:abs(v-old)) if targets and rng.random()<.55 else old+rng.uniform(-1,1)*extent*(.04*cooling+.002)
            proposal=_move_visual(proposal,target if axis==0 else x,target if axis==1 else y)
        elif motion==2 and len(block)==1 and block[0]['type']=='text':
            e=proposal[0]
            if u['role']=='footer':
                # Footer width is an explicit layout affordance, particularly
                # for legal copy in banners; annealing must not tighten it back
                # to a renderer-dependent text measurement.
                e['width']=block[0]['width']
            elif width/height>=1.35 and '\n' not in e.get('content','') and rng.random()<.5:
                room=_text_right_limit(u,bounds,_records(current,units),width,height)-x
                e['width']=max(1,min(room,_text(e,1e9)[1]+e['font_size']*.02))
            else:
                e['width']=max(1,e['width']*rng.choice((.90,1.10)))
            e['font_size']=_fit_text(e,e['width'],height*.7,max(1,e['font_size']*rng.choice((1.,1.04))))
            e['height']=_text(e)[0]
            _tighten_text(e)
            new_box=_visual_bounds(proposal);new_w=new_box[2]-new_box[0]
            align=e.get('text_align','left')
            left=(x+r-new_w)/2 if align=='center' else r-new_w if align=='right' else x
            proposal=_move_visual(proposal,left,y)
        else:
            factor=rng.choice((.95,1.05))
            if u['semantic']=='logo' and factor<1:
                source_height=u['bounds'][3]-u['bounds'][1]
                minimum=min(height*.28,source_height*desired*.85)
                if b-y<minimum:
                    factor=1.05
            proposal=_scale(proposal,factor)
            left=(x+r-(r-x)*factor)/2
            if u.get('footer_anchor') in ('left','right'):
                margin=min(width,height)*.05
                left=margin if u['footer_anchor']=='left' else width-margin-(r-x)*factor
            proposal=_move_visual(proposal,left,(y+b-(b-y)*factor)/2)
        changed=[e for e in current if e['id'] not in ids]+proposal
        feasible=_feasible(changed,units,width,height)
        if current_feasible and not feasible:
            continue
        value=_energy(changed,units,source,width,height,desired)
        next_violation=_constraint_violation(changed,units,width,height)
        temperature=.35*cooling+.005
        if not current_feasible:
            # Repair hard violations first, not merely soft layout energy.
            accept=feasible or next_violation<violation-1e-12 or (abs(next_violation-violation)<1e-12 and value<current_energy)
        else:
            accept=value<current_energy or rng.random()<exp(min(0,(current_energy-value)/temperature))
        if accept:
            current,current_energy=changed,value
            current_feasible,violation=feasible,next_violation
            accepted+=1
            if (feasible and (not best_feasible or value<best_energy)) or (not best_feasible and not feasible):
                best,best_energy=deepcopy(changed),value
                best_feasible=feasible
    return best,dict(before=_energy(elements,units,source,width,height,desired),after=best_energy,accepted=accepted,iterations=iterations)


def _rgba(value):
    try:
        from PIL import ImageColor
        if value.startswith(('rgb(', 'rgba(')):
            values=[float(v) for v in re.findall(r'[\d.]+',value)]
            return tuple(values[:3])+(values[3] if len(values)>3 else 1.,)
        rgb=ImageColor.getrgb(value)
        return tuple(rgb[:3])+((rgb[3]/255) if len(rgb)>3 else 1.,)
    except (ValueError,AttributeError):
        return None


def _luminance(rgb):
    channels=[v/255 for v in rgb[:3]]
    return sum((v/12.92 if v<=.04045 else ((v+.055)/1.055)**2)*w for v,w in zip(channels,(.2126,.7152,.0722)))


def _visual_relationship_context(units,source,width,height):
    """Compact visual-semantic summary for the image/text judge."""
    roles={u.get('semantic') for u in units}
    grouped=[]
    peer_rows=False
    for u in units:
        items=u.get('items',[])
        group_ids={e.get('group_id') for e in items if e.get('group_id')}
        if group_ids and len(items)>1:
            types={e.get('type') for e in items}
            grouped.append('mixed' if len(types)>1 else next(iter(types),'art'))
        peer_rows=peer_rows or bool(u.get('peer_ids'))
    sw,sh=source['canvas']['width'],source['canvas']['height']
    callout_quadrants=[]
    for u in units:
        if not u.get('corner_callout'):
            continue
        x,y,r,b=u['bounds']
        vertical='top' if (y+b)/2<sh/2 else 'bottom'
        horizontal='left' if (x+r)/2<sw/2 else 'right'
        callout_quadrants.append(f'{vertical}-{horizontal}')
    bucket=_aspect_state(source,width,height)['bucket']
    density='dense' if len(units)>=14 else 'moderate' if len(units)>=7 else 'sparse'
    return dict(roles=sorted(role for role in roles if role),bucket=bucket,density=density,
                grouped_units=len(grouped),grouped_types=sorted(set(grouped)),
                repeated_badge_row=peer_rows,
                callout_quadrants=callout_quadrants,
                has_footer=any(u.get('role')=='footer' for u in units))


def _visual_rerank(candidates, backgrounds, units, source, width, height, desired):
    """Rendered diagnostics plus an optional learned SigLIP perceptual judge.

    Analyze top geometry-safe candidates. A text-free diagnostic rendition lets
    us inspect actual backing pixels without measuring the glyph's own color.
    When the baked SigLIP weights are present, a learned image/text score is
    blended with the deterministic cost as a closer proxy for the PNG evaluator;
    otherwise the deterministic ranking is used unchanged. Either way this stage
    only *reorders* candidates that already passed every hard constraint -- it
    never edits geometry, text, assets, or the element list.
    """
    if not candidates or width*height>16000000:
        return candidates[0],dict(used=False,reason='Canvas exceeds diagnostic raster budget')
    try:
        from renderer.converter.convert import render_design_to_png
        from PIL import Image, ImageChops
    except ImportError as exc:
        return candidates[0],dict(used=False,reason=f'Renderer unavailable: {exc}')
    entries=[]
    sw,sh=source['canvas']['width'],source['canvas']['height']
    source_ratio=sw/sh
    target_ratio=width/height
    ratio_shift=max(target_ratio/source_ratio,source_ratio/target_ratio)
    base_energy=candidates[0]['energy']
    def family(name):
        if name.startswith('proportional'):
            return 'proportional'
        if 'centered-content' in name:
            return 'centered'
        if 'semantic-stack' in name:
            return 'stack'
        if 'inline-banner' in name or 'hero-banner' in name:
            return 'banner'
        if 'editorial-split' in name:
            return 'editorial-split'
        if name.startswith(('split-','copy-first-','visual-first-')):
            return 'reflow'
        if 'affiliated' in name:
            return 'affiliated'
        if 'quiet-region' in name:
            return 'quiet-region'
        return name.split('-',1)[0]
    # A large aspect-ratio change needs a true reflow. Proportional scaling is
    # still generated as a fallback, but on its own it preserves the source
    # hierarchy and leaves the newly available axis unused. Letting SigLIP
    # choose it here repeatedly produced attractive-looking thumbnails with
    # the original stack merely enlarged, so omit that topology from judging
    # when a feasible structural alternative exists.
    judge_pool=candidates
    if ratio_shift>=1.6:
        reflowed=[candidate for candidate in candidates
                  if not candidate['name'].startswith('proportional')]
        if reflowed:
            judge_pool=reflowed
    # Give the rendered judge genuinely different layout topologies. Sorting
    # only by geometry energy often leaves a single candidate in its shortlist.
    shortlist=[];seen_families=set()
    for candidate in judge_pool:
        key=family(candidate['name'])
        if key not in seen_families:
            shortlist.append(candidate);seen_families.add(key)
        if len(shortlist)>=5:
            break
    for candidate in judge_pool:
        if len(shortlist)>=5:break
        if candidate not in shortlist:shortlist.append(candidate)
    with TemporaryDirectory(prefix='chameleon-rank-') as directory:
        # Render the source first so it is the reference available before any
        # adapted candidate is produced or sent to the visual judge.
        reference_path=Path(directory)/'original.png'
        reference_image=None
        try:
            render_design_to_png(source,Path(__file__).resolve().parents[1]/'brand_kit'/'assets',reference_path,max_size=640)
            with Image.open(reference_path) as image:
                reference_image=image.convert('RGB')
        except Exception:
            # Source comparison is an optional cue; candidate judging remains
            # available if an original asset or renderer path is unavailable.
            pass
        for index,candidate in enumerate(shortlist):
            # All entrants have already passed hard geometry checks. A wider,
            # still bounded band lets perceptual ranking compare meaningful
            # structures instead of nearly identical energy winners only.
            if candidate['energy']>base_energy+max(50,base_energy*.85):
                continue
            doc=_assemble(candidate['elements'],backgrounds,source,width,height)
            full_path=Path(directory)/f'{index}.png'
            backing_path=Path(directory)/f'{index}-backing.png'
            try:
                warnings=render_design_to_png(doc,Path(__file__).resolve().parents[1]/'brand_kit'/'assets',full_path,max_size=640)
                backing=dict(canvas=doc['canvas'],elements=[e for e in doc['elements'] if e['type']!='text'])
                render_design_to_png(backing,Path(__file__).resolve().parents[1]/'brand_kit'/'assets',backing_path,max_size=640)
                with Image.open(full_path) as image:
                    full=image.convert('RGB')
                with Image.open(backing_path) as image:
                    under=image.convert('RGB')
                ink=ImageChops.difference(full,under).convert('L')
                cost,checked,weak,busy,invisible=0.,0,[],[],[]
                for e in doc['elements']:
                    if e['type']!='text' or not e.get('content','').strip():
                        continue
                    color=_rgba(e.get('color',''))
                    if not color or abs(e.get('rotation',0))>1:
                        continue
                    left=max(0,round(e['x']/width*under.width));top=max(0,round(e['y']/height*under.height))
                    right=min(under.width,max(left+1,round((e['x']+e['width'])/width*under.width)))
                    bottom=min(under.height,max(top+1,round((e['y']+e['height'])/height*under.height)))
                    if right<=left or bottom<=top:
                        continue
                    patch=under.crop((left,top,right,bottom)).resize((8,8))
                    contrasts,lumas=[],[]
                    for pixel in (patch.get_flattened_data() if hasattr(patch, 'get_flattened_data') else patch.getdata()):
                        alpha=color[3]*e.get('opacity',1)
                        foreground=tuple(color[k]*alpha+pixel[k]*(1-alpha) for k in range(3))
                        a,b=_luminance(foreground),_luminance(pixel)
                        contrasts.append((max(a,b)+.05)/(min(a,b)+.05));lumas.append(b)
                    target=3 if e['font_size']>=24 or e.get('font_weight')=='bold' and e['font_size']>=18 else 4.5
                    low=sorted(contrasts)[6]
                    deficit=max(0,(target-low)/target)
                    cost+=deficit*5
                    if deficit>.03:
                        weak.append(e['id'])
                    roughness=sum(abs(lumas[j]-lumas[j-1]) for j in range(1,len(lumas)) if j%8)/56
                    cost+=max(0,roughness-.04)*8
                    if roughness>.08:
                        busy.append(e['id'])
                    if ink.crop((left,top,right,bottom)).getextrema()[1]<5:
                        cost+=3;invisible.append(e['id'])
                    checked+=1
                cost/=max(1,checked)
                entries.append(dict(candidate=candidate,cost=cost,checked=checked,low_contrast=weak,busy_background=busy,invisible_text=invisible,warnings=warnings,image=full.copy()))
            except Exception as exc:
                # Diagnostics are optional; isolate third-party renderer faults.
                # Do not catch BaseException (interrupts and shutdown still work).
                entries.append(dict(candidate=candidate,cost=0.,error=str(exc)))
    if not entries or all('error' in item for item in entries):
        return candidates[0],dict(used=False,reason='Diagnostic rendering failed',errors=[e.get('error') for e in entries])
    valid=[e for e in entries if 'error' not in e]
    heuristic_cost=lambda e:e['candidate']['energy']+e['cost']
    relationship_context=_visual_relationship_context(units,source,width,height)
    perceptual=_siglip_rerank(valid,heuristic_cost,reference_image,relationship_context)
    index=perceptual.get('selected_index')
    selected=valid[index] if index is not None else min(valid,key=heuristic_cost)
    report=dict(used=True,
                method='rendered contrast/busyness/visible-text checks'+(' + SigLIP perceptual re-rank' if perceptual.get('used') else ''),
                perceptual=perceptual,
                candidates=[{k:v for k,v in e.items() if k not in ('candidate','image')}|{'name':e['candidate']['name']} for e in entries])
    return selected['candidate'],report


def _siglip_rerank(valid, heuristic_cost, reference_image=None, layout_context=None):
    """Optional learned SigLIP re-rank among already-valid candidate renders.

    Returns a diagnostic dict. When used, ``selected_index`` points at the chosen
    entry in ``valid``; otherwise it is ``None`` and the caller keeps the pure
    heuristic winner. SigLIP may only pick among the bounded, hard-valid visual
    shortlist whose heuristic cost is within a generous but finite band, so it
    cannot rescue an illegible or off-canvas layout. Degrades to a no-op
    when torch/transformers or the baked weights are absent.
    """
    if len(valid)<2:
        return dict(used=False,selected_index=None,reason='Fewer than two candidates to compare')
    try:
        from engine.vision import score_layouts
    except Exception as exc:
        return dict(used=False,selected_index=None,reason=f'Vision module unavailable: {exc}')
    result=score_layouts([e['image'] for e in valid],reference_image=reference_image,
                         layout_context=layout_context)
    if result is None:
        return dict(used=False,selected_index=None,reason='SigLIP returned no scores')
    scores,note=result
    if scores is None:
        return dict(used=False,selected_index=None,reason=note)
    costs=[heuristic_cost(e) for e in valid]
    best_cost=min(costs)
    # Near-tie band: candidates within this tolerance of the best heuristic cost
    # are considered interchangeable on the heuristics, so the learned judge
    # decides among them. Anything worse than the band is never chosen here.
    tolerance=max(45.,abs(best_cost)*.85)
    band=[i for i,c in enumerate(costs) if c<=best_cost+tolerance]
    best_cost=min(costs)
    # Let the visual model decide between layouts, while a small capped
    # geometry penalty keeps a barely higher image/text similarity from
    # overruling a clearly more balanced deterministic layout.
    penalties=[min(.18,max(0.,cost-best_cost)*.04) for cost in costs]
    adjusted=[score-penalty for score,penalty in zip(scores,penalties)]
    chosen=max(band,key=lambda i:adjusted[i])
    return dict(used=True,note=note,selected_index=chosen,
                scores=[round(s,4) for s in scores],
                adjusted_scores=[round(s,4) for s in adjusted],
                heuristic_costs=[round(c,4) for c in costs],
                heuristic_penalties=[round(p,4) for p in penalties],
                tie_band=band,tolerance=round(tolerance,4),
                overrode_heuristic=costs[chosen]>best_cost+1e-9)


def _assemble(elements,backgrounds,source,width,height):
    by_id={e['id']:deepcopy(e) for e in elements}
    for old in backgrounds:
        e=deepcopy(old);e.update(x=0,y=0,width=width,height=height)
        if e['type']=='image':
            e['fit']='cover'
        by_id[e['id']]=e
    return dict(canvas=dict(width=int(width),height=int(height)),elements=[by_id[e['id']] for e in source['elements']])


def _adapt_with_diagnostics(source,target_canvas):
    start=perf_counter()
    sw,sh=source['canvas']['width'],source['canvas']['height']
    width,height=target_canvas['width'],target_canvas['height']
    if any(not isinstance(n,(int,float)) or isinstance(n,bool) or not isfinite(n) or n<=0 for n in (sw,sh,width,height)):
        raise ValueError('Canvas dimensions must be positive finite numbers.')
    if int(width)!=width or int(height)!=height:
        raise ValueError('Target dimensions must be integer pixels.')
    if len({e['id'] for e in source['elements']})!=len(source['elements']):
        raise ValueError('Source element IDs must be unique.')
    backgrounds,units=_analyze(source)
    for unit in units:
        unit['target_canvas']=(width,height)
        unit['source_canvas']=(sw,sh)
    desired=sqrt(width*height/(sw*sh))
    state=_aspect_state(source,width,height)
    context=_perceptual_context(source,backgrounds,units,width,height)
    context['typography_gamma']=min(1.,.65+.7*desired)
    for u in units:
        u['perceptual']=context
        u['font_gamma']=min(1.,.65+.7*desired)
    seeds=[dict(name='proportional',elements=_proportional(units,width,height,sw,sh))]
    for mode in ('split','copy-first','visual-first'):
        for share in (.30,.45,.60):
            family=f'editorial-{mode}' if width/height>=2.4 else mode
            seeds.append(dict(name=f'{family}-{share}',elements=_reflow(units,width,height,sw,sh,mode,share,desired*1.15)))
    if width/height>=1.35:
        seeds.append(dict(name='editorial-split',elements=_editorial_split_seed(units,width,height,sw,sh,desired*1.15,.42)))
        seeds.append(dict(name='editorial-split-reversed',elements=_editorial_split_seed(units,width,height,sw,sh,desired*1.15,.42,True)))
    if state['bucket']=='banner':
        for share in (.14,.22,.30):
            seeds.append(dict(name=f'inline-banner-{share}',elements=_banner_seed(units,width,height,desired*1.15,share,source_size=(sw,sh))))
        seeds.append(dict(name='left-hero-banner',elements=_banner_seed(units,width,height,desired*1.15,.24,True,source_size=(sw,sh))))
    if state['bucket'] in ('story','portrait','square'):
        for hero_first in (True,False):
            shares=(.42,.52,.64,.72,.82) if .75<=width/height<=1.3 else (.34,.42,.48,.52)
            for share in shares:
                seeds.append(dict(name=f'semantic-stack-{hero_first}-{share}',elements=_stack_seed(units,width,height,desired*1.15,share,hero_first,(sw,sh))))
    if state['bucket']!='banner' and not any(
        u['role']=='visual' or u['semantic'] in ('hero','art') for u in units):
        seeds.append(dict(name='centered-content-stack',elements=_centered_content_seed(units,width,height,desired*1.15,(sw,sh))))
    if context['affinities']:
        seeds.append(dict(name='affiliated-features',elements=_affiliated_seed(units,context,width,height,sw,sh,desired)))
    if context['background_map']:
        base=seeds[0]['elements']
        for cell in sorted(range(9),key=lambda i:context['background_map'][i])[:3]:
            seeds.append(dict(name=f'quiet-region-{cell}',elements=_quiet_seed(base,units,width,height,cell,desired*1.15)))
    seeds.append(dict(name='safe-grid',elements=_safe_grid(units,width,height,desired)))
    feasible=[]
    rejected=[]
    for seed in seeds:
        projected=None
        for topology in ('nearest','centers','alternate'):
            projected=_constrain(seed['elements'],units,width,height,topology)
            if projected is not None:
                break
        if projected is None and _feasible(seed['elements'],units,width,height):
            projected=seed['elements']
        if projected is None:
            rejected.append(seed)
        else:
            feasible.append(dict(name=seed['name'],elements=projected,energy=_energy(projected,units,source,width,height,desired)))
    # Spend a bounded repair budget on the most promising rejected seeds.
    rejected.sort(key=lambda c:_constraint_violation(c['elements'],units,width,height))
    recovered=0
    for i,seed in enumerate(rejected[:3]):
        repaired,_=_polish(seed['elements'],units,source,width,height,desired,seed=701+i,iterations=60)
        if _feasible(repaired,units,width,height):
            recovered+=1
            feasible.append(dict(name=seed['name']+'-repaired',elements=repaired,energy=_energy(repaired,units,source,width,height,desired)))
    generation_time=perf_counter()-start
    if not feasible:
        # No valid input element is dropped, even in a physically impossible
        # target. Keep bounded geometry and expose fallback in diagnostics.
        fallback=_safe_grid(units,width,height,desired)
        feasible=[dict(name='bounded-grid-fallback',elements=fallback,energy=_energy(fallback,units,source,width,height,desired))]
    feasible.sort(key=lambda c:(c['energy'],c['name']))
    optimized=[]
    for i,candidate in enumerate(feasible[:10]):
        if _feasible(candidate['elements'],units,width,height):
            elements,optimization=_polish(candidate['elements'],units,source,width,height,desired,seed=90210+i,iterations=36)
        else:
            elements=candidate['elements'];optimization=dict(before=candidate['energy'],after=candidate['energy'],accepted=0,iterations=0)
        optimized.append(dict(name=candidate['name'],elements=elements,energy=_energy(elements,units,source,width,height,desired),optimization=optimization))
    optimized.sort(key=lambda c:(c['energy'],c['name']))
    geometry_time=perf_counter()-start
    winner,visual=_visual_rerank(optimized,backgrounds,units,source,width,height,desired)
    # Run up to two bounded refinement rounds. Each round proposes local
    # changes from the current best, then compares parent and child against
    # both the inferred relationship cues and the original rendered design.
    # Stop early when the judge keeps the parent or the geometry has settled.
    refinement=dict(generated=0,accepted=0,used=False,rounds=[])
    for round_index in range(2):
        if not _feasible(winner['elements'],units,width,height):
            break
        parent=winner
        refined,stats=_polish(parent['elements'],units,source,width,height,desired,
                              seed=120319+round_index*3571,iterations=48)
        refined_by_id={e['id']:e for e in refined}
        changed=any(any(abs(refined_by_id[old['id']].get(key,0)-old.get(key,0))>.05
                        for key in ('x','y','width','height','font_size'))
                    for old in parent['elements'] if old['id'] in refined_by_id)
        refinement['generated']+=1
        refinement['accepted']+=stats['accepted']
        round_report=dict(round=round_index+1,accepted=stats['accepted'],used=False)
        if not changed or not _feasible(refined,units,width,height):
            refinement['rounds'].append(round_report)
            break
        variant=dict(name=f"refined-{parent['name']}",elements=refined,
                     energy=_energy(refined,units,source,width,height,desired))
        refined_winner,refined_visual=_visual_rerank(
            [parent,variant],backgrounds,units,source,width,height,desired)
        round_report.update(used=True,winner=refined_winner['name'])
        refinement['rounds'].append(round_report)
        visual=dict(refined_visual,first_pass=visual,refinement_judge=refined_visual)
        if refined_winner['name']==parent['name']:
            break
        refinement['used']=True
        refinement['winner']=refined_winner['name']
        winner=refined_winner
    output=_assemble(winner['elements'],backgrounds,source,width,height)
    report=dict(perceptual=context,aspect=state,roles=[dict(ids=[e['id'] for e in u['items']],role=u['semantic'],confidence=u['confidence']) for u in units],
                generated=len(seeds),feasible=len(feasible),rejected_topologies=len(rejected),recovered=recovered,starts=len(optimized),winner=winner['name'],
                energy_terms=_energy_terms(winner['elements'],units,source,width,height,desired),
                candidates=[{k:v for k,v in c.items() if k!='elements'} for c in optimized],
                refinement=refinement,
                visual=visual,hard_constraints_passed=_feasible(winner['elements'],units,width,height),
                seconds=dict(generation=round(generation_time,4),geometry=round(geometry_time,4),total=round(perf_counter()-start,4)))
    return output,report


def _score(elements,units,source,width,height,desired):
    """Compatibility helper: higher is better, using the new energy terms."""
    return -_energy(elements,units,source,width,height,desired)


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
    return _adapt_with_diagnostics(source, target_canvas)[0]
