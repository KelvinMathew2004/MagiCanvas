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

from renderer.converter.assets import visible_content_bounds


_ASSETS_DIR=Path(__file__).resolve().parents[1]/'brand_kit'/'assets'


def _bounds(items):
    boxes = []
    for e in items:
        if e.get('type')=='image':
            boxes.append(visible_content_bounds(e,_ASSETS_DIR))
            continue
        w, h, a = e['width'], e['height'], radians(e.get('rotation', 0))
        rw, rh = abs(w*cos(a))+abs(h*sin(a)), abs(w*sin(a))+abs(h*cos(a))
        cx, cy = e['x']+w/2, e['y']+h/2
        # Both visual rotation bounds and the runner's raw geometry must fit.
        boxes.append((min(e['x'],cx-rw/2), min(e['y'],cy-rh/2),
                      max(e['x']+w,cx+rw/2), max(e['y']+h,cy+rh/2)))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else (0, 0, 0, 0)


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def _area(e):
    if e.get('type')=='image':
        x,y,r,b=_bounds([e])
        return max(1e-9,(r-x)*(b-y))
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
    desired=max(1,median*scale*(e['font_size']/max(1,median))**gamma)
    # Narrow formats should reflow before labels and copy collapse to pixels.
    # Floors scale with the target canvas and source type scale, not a sample.
    if u.get('target_aspect_ratio',1.)<.55:
        short=u.get('target_short_side',0.)
        if short:
            if u.get('semantic')=='headline': floor=min(short*.07,median*.65)
            elif u.get('semantic') in ('subheading','body'): floor=min(short*.045,median*.45)
            elif u.get('role') in ('tag','action') or u.get('semantic') in ('badge','cta','callout'):
                floor=min(short*.055,median*.55)
            elif u.get('role')=='footer': floor=min(short*.03,median*.30)
            else: floor=0.
            desired=max(desired,floor)
    return desired


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
    # A sizeable translucent card behind foreground copy is a follower panel,
    # not hero artwork. Require strict z-order and near-total containment of
    # text-bearing units so decorative shapes are not mistaken for containers.
    panel_children={}
    groups=list(grouped.values())
    panel_candidates=[]
    for items in groups:
        if len(items)!=1:continue
        panel=items[0]
        if (panel['type']!='shape' or panel.get('shape_type')!='rectangle'
                or not sw*sh*.06<=_area(panel)<=sw*sh*.80):continue
        panel_box=_bounds([panel]);backed=[]
        for child in groups:
            if child is items or not any(e['type']=='text' for e in child):continue
            if not all(panel.get('z_index',0)<e.get('z_index',0) for e in child):continue
            child_box=_bounds(child);child_area=max(1e-9,(child_box[2]-child_box[0])*(child_box[3]-child_box[1]))
            if _overlap(panel_box,child_box)/child_area>=.85:
                backed.extend(e['id'] for e in child)
        text_groups=sum(any(e['type']=='text' for e in child) and
                        any(e['id'] in backed for e in child) for child in groups)
        panel_area=_area(panel)
        if text_groups>=2 or (text_groups==1 and sw*sh*.16<=panel_area<sw*sh*.75):
            panel_candidates.append((panel_area,panel,tuple(sorted(set(backed)))) )
    assigned=set()
    for _,panel,backed in sorted(panel_candidates,key=lambda candidate:candidate[0]):
        children=tuple(eid for eid in backed if eid not in assigned)
        if not children:continue
        panel_children[panel['id']]=children
        assigned.update(children)
    units = []
    for items in grouped.values():
        x, y, r, b = _bounds(items)
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
        elif len(items)==1 and shapes and shapes[0]['id'] in panel_children:
            role = 'decoration'
        elif len(items)==1 and shapes and (shapes[0].get('opacity', 1)<.4 or shapes[0].get('shape_type') in ('ellipse', 'line')):
            role = 'decoration'
        else:
            role = 'visual'
        unit=dict(items=items,bounds=(x,y,r,b),role=role,label=label)
        if len(items)==1 and items[0]['id'] in panel_children:
            panel=items[0]
            unit.update(backing_panel=True,backed_ids=panel_children[panel['id']],
                        backing_source_width=panel['width'],backing_source_height=panel['height'],
                        backing_source_radius=panel.get('corner_radius',0.))
        units.append(unit)
    units=_enrich_units(source, units)
    fonts=sorted(e['font_size'] for e in elements if e['type']=='text')
    median=fonts[len(fonts)//2] if fonts else 24
    for u in units:
        u['font_median']=median
    return backgrounds, units


def _unit(u, width, height, scale, fill=False):
    width, height = max(.0001, width), max(.0001, height)
    items, (x,y,r,b) = u['items'], u['bounds']
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
    fit_w,fit_h=u.get('fit_size',(r-x,b-y))
    factor = min(width/max(1e-9,fit_w), height/max(1e-9,fit_h))
    if not fill:
        # A corner media asset has communicative weight in the source even if
        # its file is small. Preserve a modest fraction of that prominence.
        anchored_footer_image=(u['role']=='footer' and u.get('source_footer_anchor',u.get('footer_anchor')) in ('left','right')
                               and any(e['type']=='image' for e in items))
        scale_cap=min(scale*1.35,1.) if anchored_footer_image else scale
        if u.get('inflow_footer_asset'):
            # Moving a small footer badge into the CTA area should free its
            # corner without turning it into a dominant product image.
            scale_cap=min(scale*1.55,width*.28/max(1.,fit_w))
        factor = min(factor, scale_cap)
    # PNGs keep their original frame scale; visible alpha bounds, rather than
    # transparent padding, determine where their artwork fits.
    result = _scale(items, factor)
    for e in result:
        if e['type']=='text':
            e['font_size'] = _fit_text(e, e['width'], e['height'], e['font_size'])
    return _contain(result, width, height)


def _row(units, width, height, scale, gap, align='center', equalize_control_height=False):
    if equalize_control_height and units:
        controls=[_unit(u,width,height,scale) for u in units]
        tag_heights=[_bounds(items)[3]-_bounds(items)[1]
                     for u,items in zip(units,controls) if u.get('role')=='tag']
        if tag_heights:
            target_h=max(max(tag_heights),max((_text(e)[0]+4 for items in controls for e in items if e['type']=='text'),default=0))
            for items in controls:
                _,top,_,bottom=_bounds(items);center=(top+bottom)/2
                for e in items:
                    if e['type']=='shape':
                        e['height']=target_h;e['y']=center-target_h/2
                        if 'corner_radius' in e:e['corner_radius']=min(target_h/2,e['corner_radius'])
                    elif e['type']=='text':
                        measured_h=_text(e)[0]
                        e['height']=max(measured_h,min(e['height'],target_h))
                        e['y']=center-e['height']/2
            widths=[_bounds(items)[2]-_bounds(items)[0] for items in controls]
            total=sum(widths)+gap*(len(controls)-1)
            if total>width:
                factor=width/max(1e-9,total)
                controls=[_scale(items,factor) for items in controls]
                widths=[w*factor for w in widths]
            elif total>0:
                factor=min(1.12,width*.94/total)
                if factor>1:
                    controls=[_scale(items,factor) for items in controls]
                    widths=[w*factor for w in widths]
            rh=max((_bounds(items)[3]-_bounds(items)[1] for items in controls),default=0)
            total=sum(widths)+gap*(len(controls)-1)
            x=0 if align=='left' else width-total if align=='right' else (width-total)/2
            output=[]
            for items,w in zip(controls,widths):
                output.extend(_move(items,x,0));x+=w+gap
            return output,rh
    rows, current, used = [], [], 0
    for u in units:
        items = _unit(u, width, height, scale)
        _,_,w,h = _bounds(items)
        if current and used+gap+w > width:
            rows.append(current)
            current, used = [], 0
        current.append((u,items,w,h))
        used += w+(gap if len(current)>1 else 0)
    if current:
        rows.append(current)
    output, cursor = [], 0
    for row in rows:
        rh = max(h for _,_,_,h in row)
        row_width=sum(w for _,_,w,_ in row)+gap*(len(row)-1)
        x = 0 if align=='left' else width-row_width if align=='right' else (width-row_width)/2
        for _,items,w,h in row:
            output.extend(_move(items,x,cursor+(rh-h)/2))
            x += w+gap
        cursor += rh+gap
    return output, max(0,cursor-gap)


def _stacked_badge_order(units,width,height,scale):
    """Group wrapped controls by fill, then order each group into a smooth stack."""
    measured=[]
    for unit in units:
        box=_bounds(_unit(unit,width,height,scale))
        shapes=[e for e in unit['items'] if e['type']=='shape']
        fill=(shapes[0].get('fill_color') or shapes[0].get('stroke_color') or '') if shapes else ''
        label=' '.join(e.get('content','') for e in unit['items'] if e['type']=='text')
        font=max((e.get('font_size',0) for e in unit['items'] if e['type']=='text'),default=0)
        measured.append((unit,max(0.,box[2]-box[0]),str(fill).strip().lower(),font,label))
    # Repeated fills stay together. Within a fill family, widest/most prominent
    # labels come first, creating a deliberate taper instead of alternating
    # wide and narrow pills. Different color families are ordered by their
    # average width and type scale; horizontal rows retain their source order.
    groups={}
    for row in measured:groups.setdefault(row[2],[]).append(row)
    ordered_groups=sorted(groups.values(),key=lambda group:(
        -sum(row[1] for row in group)/len(group),
        -sum(row[3] for row in group)/len(group),
        min((row[4] for row in group),default='')))
    return [row[0] for group in ordered_groups
            for row in sorted(group,key=lambda row:(-row[1],-row[3],row[4]))]


def _badge_row(units,width,height,scale,gap,align='center'):
    # Keep verified peers on one row by sharing a fitting scale when possible.
    shared_peers=(len(units)>1 and units[0].get('peer_ids') and
                  all(u.get('peer_ids')==units[0]['peer_ids'] for u in units))
    wrap_without_scaling=False
    if shared_peers:
        widths=[]
        for unit in units:
            box=_bounds(_unit(unit,width,height,scale))
            widths.append(box[2]-box[0])
        needed=sum(widths)
        if needed+gap*(len(units)-1)>width:
            target_ratio=units[0].get('target_aspect_ratio',width/max(.001,height))
            narrow_skyscraper=min(target_ratio,width/max(.001,height))<.55
            # Natural widths already include the narrow-format text floor.
            # Let the row builder wrap when those readable labels need space;
            # never compress the whole peer group just to keep it on one line.
            if not narrow_skyscraper:
                gap=min(gap,width/max(1,len(units)*8))
                scale*=max(.001,(width-gap*(len(units)-1))/max(.001,needed))*.995
            else:
                wrap_without_scaling=True
                units=_stacked_badge_order(units,width,height,scale)
                # Peer controls stay visually grouped when they wrap. Scale
                # the row rhythm with the narrow column instead of reusing
                # the much larger gap intended between content sections.
                gap=max(width*.028,min(gap,width*.034))
    return _row(units,width,height,scale,gap,align,
                equalize_control_height=bool(shared_peers and not wrap_without_scaling))


def _action_band(tags,actions,width,height,scale,gap,stacked,align='center'):
    if not (stacked or len(tags)>=2) or not tags or not actions:
        return _badge_row(tags,width,height,scale,gap,align) if not actions else _row(tags+actions,width,height,scale,gap,align)
    # `stacked` is decided from the target canvas by _reflow. The action band
    # itself is intentionally wide, so deriving the flow from its local aspect
    # ratio incorrectly dropped migrated footer assets on story layouts.
    vertical=stacked
    peer_gap=max(width*.028,min(gap,width*.034)) if vertical else gap
    section_gap=(min(height*.15,max(min(width,height)*.10,gap*2.))
                 if vertical else min(gap*1.6,min(width,height)*.04))
    badges,bh=_badge_row(tags,width,height*.42,scale*.78,peer_gap,align)
    # A migrated corner/footer badge is a separate content group. Keep it
    # between the peer controls and primary action with the same section
    # rhythm used elsewhere, while retaining its source-scale cap in _unit.
    assets=[u for u in actions if u.get('inflow_footer_asset')]
    primary=[u for u in actions if u not in assets]
    asset_items=[];asset_h=0.
    if vertical and assets:
        asset_items,asset_h=_row(assets,width,height*.18,scale,peer_gap,align)
    # A primary action keeps greater visual weight than secondary tags while
    # the available slot allows it; _unit still shrinks it to fit when tight.
    # Measure the action at its natural width-constrained size before laying it
    # out. Giving it all remaining band height made _contain center it in a tall
    # empty slot, creating an artificial gap below the badges.
    action_room=max(.001,height-bh-section_gap-asset_h-section_gap)
    natural=[]
    for unit in primary:
        natural.extend(_unit(unit,width,action_room,scale*1.06))
    natural_box=_bounds(natural)
    natural_h=max(1.,natural_box[3]-natural_box[1])
    # The primary CTA should keep its lead over secondary badges. On a tall
    # layout, use available band height for a restrained size increase instead
    # of pinning it to source scale while unused space remains below. The cap
    # avoids the oversized CTA that can feel unbalanced in wide banners.
    action_scale=scale*1.06
    if vertical and primary and action_room>natural_h*1.05:
        action_scale*=min(1.12,sqrt(action_room/max(1.,natural_h)))
    buttons,ah=_row(primary,width,min(action_room,natural_h*(action_scale/(scale*1.06))),action_scale,peer_gap if vertical else gap,align)
    if align=='left':
        _move(badges,0,0);_move(asset_items,0,0);_move(buttons,0,0)
    if vertical and assets:
        output=badges+_shift(asset_items,0,bh+section_gap)+_shift(buttons,0,bh+section_gap+asset_h+section_gap)
        return output,bh+section_gap+asset_h+section_gap+ah
    output=badges+_shift(buttons,0,bh+section_gap)
    return output,bh+section_gap+ah


def _footer_band(units,width,height,scale,gap,center_corners=False):
    """Use the footer band efficiently while preserving corner image visibility."""
    if center_corners:
        # In a vertical composition, a corner badge and legal line read better
        # as one centered footer group than as isolated left/right fragments.
        return _row(units,width,height,scale,gap)
    corners=[u for u in units if u.get('footer_anchor') in ('left','right')]
    central=[u for u in units if u not in corners]
    if not corners:
        return _row(central,width,height,scale,gap)
    output=[];corner_h=0;left_w=right_w=0
    for side in ('left','right'):
        items=[u for u in corners if u['footer_anchor']==side]
        if not items:continue
        # Give corner assets the whole band height. Previously their height was
        # only the leftover after reserving space below for legal copy, which
        # made small badges nearly disappear in short banners.
        block,bh=_row(items,width*.30,height,scale,gap)
        left,_,right,_=_bounds(block);bw=right-left
        output.extend(_move(block,0 if side=='left' else width-bw,0))
        corner_h=max(corner_h,bh)
        if side=='left':left_w=bw
        else:right_w=bw
    # Reserve equal space on both sides so a one-sided corner asset does not
    # pull otherwise centered footer copy off the canvas axis.
    side_reserve=max(left_w,right_w)+gap
    text_width=max(.001,width-2*side_reserve)
    text,th=_row(central,text_width,height*.65 if corners else height,scale,gap)
    row_h=max(corner_h,th)
    # Center the legal copy in the free middle span, and vertically align it
    # with corner assets rather than stacking it beneath them.
    if text:
        output.extend(_shift(text,(width-text_width)/2,(row_h-th)/2))
    for e in output:
        # Corner elements are generated at y=0; center them in the shared row.
        if e not in text:
            e['y']+=(row_h-corner_h)/2
    return output,row_h


def _centered_copy_layout(elements,units,width):
    by_id={e['id']:e for e in elements}
    copy_text=[by_id[item['id']] for unit in units if unit['role']=='copy'
               for item in unit['items'] if item['type']=='text' and item['id'] in by_id]
    return (bool(copy_text) and all(e.get('text_align','left')=='center' and
            abs((e['x']+e['width']/2)/max(1.,width)-.5)<.12 for e in copy_text))


def _axis_active(u,source,width,height,elements=None,units=None):
    state=_aspect_state(source,width,height)
    # Preserve a source-centered axis only while the target remains a
    # single-column composition. Landscape reflow intentionally assigns
    # separate columns to copy and artwork, so centering each source-centered
    # unit would fight those split candidates in both scoring and polishing.
    centered_stack=(state['flow']=='landscape' and elements is not None and units is not None
                    and _centered_copy_layout(elements,units,width))
    return (state['flow'] in ('preserve','story','portrait','square') or centered_stack) and u.get('source_centered',False)


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
        if height/width>1.35:
            # Polishing can collapse the deliberate row gap from a wrapped
            # peer group because non-overlap alone considers a near-touching
            # stack acceptable. Preserve a modest, canvas-scaled minimum gap
            # between rows while keeping controls within each row aligned.
            rows=_peer_row_groups(list(range(len(members))),boxes)
            row_boxes=[(min(boxes[i][1] for i in row),max(boxes[i][3] for i in row))
                       for row in rows]
            preferred=min(width,height)*.034
            for (_,bottom),(top,_) in zip(row_boxes,row_boxes[1:]):
                actual=top-bottom
                cost+=(max(0.,preferred-actual)/max(1.,preferred))**2*.8
        gaps=[b[0]-a[2] for a,b in zip(boxes,boxes[1:])]
        if len(gaps)>1:
            mean=sum(gaps)/len(gaps)
            cost+=sum(((g-mean)/max(min(width,height)*.025,abs(mean)))**2 for g in gaps)*.03
        for a,b in zip(boxes,boxes[1:]):
            cost+=max(0,(a[0]-b[0])/width)
        # Preserve source order. When one peer has a contrasting fill and the
        # others share a style, prefer it between the matching peers.
        def fill_color(unit):
            shape=next((e for e in unit['items'] if e['type']=='shape'),None)
            return _rgba(shape.get('fill_color','')) if shape else None
        colors=[fill_color(u) for u,_ in members]
        if len(members)>=3 and all(c is not None for c in colors):
            distances=[sum((colors[i][k]-colors[j][k])**2 for k in range(3))**.5
                       for i in range(len(colors)) for j in range(i+1,len(colors))]
            if distances and max(distances)>90:
                for outlier,color in enumerate(colors):
                    similar=[j for j,other in enumerate(colors) if j!=outlier and
                             sum((color[k]-other[k])**2 for k in range(3))**.5<55]
                    if len(similar)==len(members)-1:
                        order=sorted(range(len(members)),key=lambda i:boxes[i][0])
                        if order.index(outlier) in (0,len(order)-1):
                            cost+=.35
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
        # Compress supporting copy faster than headlines. This keeps a short
        # headline from collapsing just because one paragraph is long, while
        # retaining the original relative scale when the slot has room.
        ratio=max(.0001,factor/max(.0001,scale))
        compact_banner=width/max(1.,height)>=6 and height<=120
        blocks=[]
        headline_width=None
        for u in units:
            semantic=u.get('semantic')
            exponent=(1. if compact_banner else
                      {'headline':.58,'kicker':.72,'subheading':.88,'body':1.18}.get(semantic,1.))
            local_scale=scale*(ratio**exponent)
            block=_unit(u,width,height,local_scale)
            if semantic=='headline':
                box=_bounds(block);headline_width=box[2]-box[0]
            elif (semantic in ('subheading','body') and headline_width
                  and width>=320 and height>=120 and width>headline_width*1.15):
                capped_width=min(width,max(width*.52,headline_width*.88))
                if capped_width<width*.98:
                    narrower=_unit(u,capped_width,height,local_scale)
                    original_sizes=[e['font_size'] for e in block if e['type']=='text']
                    narrower_sizes=[e['font_size'] for e in narrower if e['type']=='text']
                    if not original_sizes or min(narrower_sizes,default=0)>=min(original_sizes)*.94:
                        block=narrower
            blocks.append(block)
        if align is not None:
            for u,block in zip(units,blocks):
                if len(u['items'])==1 and u['items'][0]['type']=='text' and not u['items'][0].get('group_id'):
                    for e in block:
                        e['text_align']=align
                        # Tight line extents permit a common visible left axis.
                        if align=='left':
                            _tighten_text(e)
        sizes = [_bounds(b)[3] for b in blocks]
        gaps = []
        for left, right in zip(units, units[1:]):
            hierarchy_pair = (left.get('semantic') == 'headline' and
                              right.get('semantic') == 'subheading')
            gaps.append(gap * (.58 if hierarchy_pair else 1.0))
        return blocks,sizes,gaps,sum(sizes)+sum(gaps)
    lo,hi = .0001,scale
    for _ in range(16):
        mid = (lo+hi)/2
        if build(mid)[3] <= height:
            lo = mid
        else:
            hi = mid
    blocks,sizes,gaps,occupied = build(lo)
    # Impossible low-resolution cases still retain every element. Compress
    # gaps first; a final proportional fit is the last structural fallback.
    if len(blocks)>1:
        available=max(0,height-sum(sizes))
        requested=sum(gaps)
        if requested>available:
            gaps=[g*available/max(.001,requested) for g in gaps]
        elif occupied<=height:
            # Use some available space, while keeping the headline and its
            # supporting line visually connected as one copy block.
            extra=min((height-occupied)/(len(blocks)+1),gap*1.5)
            gaps=[g+extra*(.55 if units[i].get('semantic')=='headline' and
                            units[i+1].get('semantic')=='subheading' else 1.0)
                  for i,g in enumerate(gaps)]
    occupied=sum(sizes)+sum(gaps)
    cursor,output = max(0,(height-occupied)/2),[]
    for i,(block,h) in enumerate(zip(blocks,sizes)):
        w = _bounds(block)[2]
        output.extend(_move(block,0 if align=='left' else (width-w)/2,cursor))
        cursor += h+(gaps[i] if i<len(gaps) else 0)
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
            anchored_media=(u['role']=='footer' and u.get('footer_anchor') in ('left','right')
                            and any(e['type']=='image' for e in u['items']))
            # Hero art fits the space left by the layout, but cannot grow
            # beyond its source-relative target scale just to fill a slot.
            vertical_hero=(u['semantic']=='hero' and u.get('target_aspect_ratio',1.)<.68)
            if vertical_hero:
                block=_unit(u,cw,ch,scale)
                full_slot=_unit(u,cw,ch,scale,fill=True)
                current=_bounds(block);maximum=_bounds(full_slot)
                current_w,current_h=max(1.,current[2]-current[0]),max(1.,current[3]-current[1])
                maximum_w,maximum_h=max(1.,maximum[2]-maximum[0]),max(1.,maximum[3]-maximum[1])
                growth=min(1.30,maximum_w/current_w,maximum_h/current_h)
                if growth>1.001:block=_scale(block,growth)
            else:
                block = _unit(u,cw,ch,scale,fill=not (u['semantic']=='hero' or anchored_media))
            _,_,w,h = _bounds(block)
            items.extend(_move(block,(i%cols)*(cw+gap)+(cw-w)/2,(i//cols)*(ch+gap)+(ch-h)/2))
            occupied += w*h
        if occupied>best_use:
            best,best_use = items,occupied
    return best


def _decoration_obstacle(element):
    """Use visible glyph bounds, not a wide text container, for open-space search."""
    if element.get('type')!='text':
        return _bounds([element])
    text_h,_,_,line_widths,_=_text_info(element)
    ink_w=max(line_widths,default=0.)
    align=element.get('text_align','left')
    if align=='center':left=element['x']+(element['width']-ink_w)/2
    elif align=='right':left=element['x']+element['width']-ink_w
    else:left=element['x']
    return left,element['y'],left+ink_w,element['y']+min(element['height'],text_h)


def _sync_backing_panels(elements,units,width,height):
    """Resize inferred card backgrounds to follow their adapted children."""
    by_id={e['id']:e for e in elements}
    short=min(width,height)
    for unit in units:
        if not unit.get('backing_panel'):
            continue
        children=[by_id[eid] for eid in unit.get('backed_ids',()) if eid in by_id]
        if not children:
            continue
        left,top,right,bottom=_bounds(children)
        pad_x,pad_y=short*.028,short*.022
        x=max(0.,left-pad_x);y=max(0.,top-pad_y)
        r=min(float(width),right+pad_x);b=min(float(height),bottom+pad_y)
        panel=by_id.get(unit['items'][0]['id'])
        if panel is None:
            panel=deepcopy(unit['items'][0]);elements.append(panel);by_id[panel['id']]=panel
        source_w=max(1.,unit.get('backing_source_width',panel['width']))
        source_h=max(1.,unit.get('backing_source_height',panel['height']))
        panel.update(x=x,y=y,width=max(1.,r-x),height=max(1.,b-y))
        if 'corner_radius' in panel:
            panel['corner_radius']=min(panel['width']/2,panel['height']/2,
                                       unit.get('backing_source_radius',panel['corner_radius'])
                                       *min(width/source_w,height/source_h))
    return elements


def _decorations(units,width,height,sw,sh,occupied):
    result=[]
    for u in units:
        if u.get('backing_panel'):
            backed=set(u.get('backed_ids',()))
            children=[e for e in occupied if e['id'] in backed]
            if children:
                panel=deepcopy(u['items'][0])
                _sync_backing_panels(children+[panel],[u],width,height)
                result.append(panel)
            continue
        x,y,r,b=u['bounds']
        # Match the source's scale to the target's usable short side, then
        # search for modest growth when the surrounding layout leaves room.
        # A geometric-mean scale would enlarge accents just because a banner
        # gets wider, even when its height (and the available visual space)
        # shrinks.
        decoration_scale=min(width/sw,height/sh)
        if u.get('corner_callout'):
            block=_callout_block(u,width,height,sw,sh)
            _,_,w,h=_bounds(block)
            max_x,max_y=max(0,width-w),max(0,height-h)
            edge=min(width,height)*.045
            preferred=(width-edge-w if (x+r)/2>sw/2 else edge,edge)
            preferred=(min(max(0,preferred[0]),max_x),min(max(0,preferred[1]),max_y))
            result.extend(_move(block,*preferred))
            continue

        source_x=(x+r)/2/max(1.,sw);source_y=(y+b)/2/max(1.,sh)
        prior_boxes=[_bounds([e]) for e in result]
        short=min(width,height)
        base=_unit(u,width*.58,height*.58,decoration_scale)
        base_box=_bounds(base);base_w=max(1.,base_box[2]-base_box[0]);base_h=max(1.,base_box[3]-base_box[1])
        base_max=max(base_w,base_h)
        opacity=sum(e.get('opacity',1.) for e in u['items'])/max(1,len(u['items']))
        # Let large background accents grow into genuinely open areas, while
        # keeping tiny sparkles and other small ornaments close to source size.
        growth_cap=1.40 if base_max>=short*.12 else 1.10
        growth_cap=min(growth_cap,short*.62/base_max)
        scales=sorted({min(growth_cap,factor) for factor in (1.,1.12,1.26,1.40) if factor<=growth_cap+1e-6})
        if not scales:scales=[max(.01,growth_cap)]
        best=None
        for factor in scales:
            # _unit fits raw shapes to its slot and intentionally ignores the
            # text-scale parameter for those units. Apply the bounded growth to
            # the fitted atomic artwork itself so circles and arbitrary shapes
            # can actually expand into available negative space.
            block=_scale(base,factor)
            _,_,w,h=_bounds(block)
            max_x,max_y=max(0,width-w),max(0,height-h)
            # Include the mapped source position, edge positions, and several
            # side/center stations. This works for circles, polygons, and
            # irregular transparent artwork without assuming a specific ID.
            preferred=(min(max(0,source_x*width-w/2),max_x),
                       min(max(0,source_y*height-h/2),max_y))
            xs=sorted({0.,max_x,preferred[0],max(0.,min(max_x,(width-w)/2)),
                       max(0.,min(max_x,width*.16)),max(0.,min(max_x,width*.84-w))})
            ys=sorted({0.,max_y,preferred[1],max(0.,min(max_y,(height-h)/2)),
                       max(0.,min(max_y,height*.20)),max(0.,min(max_y,height*.80-h))})
            for px in xs:
                for py in ys:
                    box=(px,py,px+w,py+h);area=max(1.,w*h)
                    collision=0.
                    for element in occupied:
                        other=_decoration_obstacle(element)
                        overlap=_overlap(box,other)/max(1.,min(area,(other[2]-other[0])*(other[3]-other[1])))
                        # Keep decoration away from copy and controls first;
                        # light layering over imagery remains acceptable.
                        if element['type']=='text':
                            # Translucent backdrop shapes are meant to sit
                            # behind copy, so their clearance cost should scale
                            # with their actual visual strength.
                            collision+=overlap*(11.*max(.025,opacity**2))
                        else:
                            collision+=overlap*(.35+1.85*opacity)
                    collision+=sum(_overlap(box,other)/area*2.5 for other in prior_boxes)
                    tx=(px+w/2)/width;ty=(py+h/2)/height
                    ex=.12 if source_x<.30 else .88 if source_x>.70 else source_x
                    ey=.12 if source_y<.30 else .88 if source_y>.70 else source_y
                    source_drift=((tx-ex)**2+(ty-ey)**2)*1.4
                    # Side placement makes use of lateral negative space; the
                    # small reward is bounded so it cannot outweigh collisions.
                    side_reward=min(.20,abs(tx-.5)*.32)
                    spread=0.
                    for other in prior_boxes:
                        ox=(other[0]+other[2])/2/width;oy=(other[1]+other[3])/2/height
                        distance=sqrt((tx-ox)**2+(ty-oy)**2)
                        spread+=max(0.,.34-distance)**2*5.
                    # A gentle growth reward makes the accent fill open space;
                    # the collision and short-side caps stop it from taking over.
                    growth_reward=.65*(max(w,h)/max(1.,short*.62))**2
                    score=collision+source_drift+spread-side_reward-growth_reward
                    candidate=(score,block,px,py)
                    if best is None or candidate[0]<best[0]:best=candidate
        if best is not None:
            _,block,px,py=best
            placed=_move(block,px,py)
            result.extend(placed)
            occupied.extend(placed)
    return result


def _logo_row(units,width,height,scale,gap,source_size=None,fill=False):
    """Fit logos into a slot, optionally filling it with a bounded scale."""
    if not units:return [],0
    slot=max(.001,(width-gap*(len(units)-1))/len(units))
    blocks=[]
    for u in units:
        if fill:
            fit_w,fit_h=u.get('fit_size',(u['bounds'][2]-u['bounds'][0],u['bounds'][3]-u['bounds'][1]))
            cap=max(scale*1.6,min(width/source_size[0],height/source_size[1])*1.35) if source_size else scale*1.6
            factor=min(slot/max(1e-9,fit_w),height/max(1e-9,fit_h),cap)
            block=_contain(_scale(u['items'],factor),slot,height)
        else:
            block=_unit(u,slot,height,scale*1.8)
        blocks.append(block)
    sizes=[_bounds(b) for b in blocks]
    used=sum(b[2]-b[0] for b in sizes)+gap*(len(blocks)-1)
    row_h=max(b[3]-b[1] for b in sizes)
    output=[];x=max(0,(width-used)/2)
    for block,b in zip(blocks,sizes):
        w,h=b[2]-b[0],b[3]-b[1]
        output.extend(_move(block,x,(row_h-h)/2));x+=w+gap
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
    actions,ah=_action_band(lower,roles['action'],width,height*.38,scale*1.06,gap,True,align='left')
    reserved=sum(v for v in (lh,kh,ah))
    count=sum(bool(v) for v in (lh,kh,ah))+bool(roles['copy'])
    copy_height=max(.001,height-reserved-gap*max(0,count-1))
    copy_units=roles['copy']
    headline=[u for u in copy_units if u.get('semantic')=='headline']
    secondary=[u for u in copy_units if u.get('semantic') in ('subheading','body')]
    secondary_words=[sum(len(e.get('content','').split()) for e in u['items'] if e['type']=='text')
                     for u in secondary]
    similar_length=(secondary_words and min(secondary_words)>0 and
                    max(secondary_words)<=min(secondary_words)*2.2)
    grid_flow=(width/max(1.,copy_height)>=1.5 and width>=320 and copy_height>=120
               and len(headline)==1 and len(secondary)>=3 and similar_length
               and all(len(u['items'])==1 and u['items'][0]['type']=='text' and
                       not u['items'][0].get('group_id') for u in secondary))
    if grid_flow:
        # Keep the headline spanning the copy area; flow comparable supporting
        # units into two columns only when their measured text can use the
        # wider banner slot without changing their source reading order.
        head_height=copy_height*.38
        head_block=_copy(headline,width,head_height,scale*1.06,gap*.65,align='left')
        hb=_bounds(head_block);actual_head_h=hb[3]-hb[1]
        grid_top=hb[3]+gap
        grid_height=max(1.,copy_height-grid_top)
        columns=2
        rows=(len(secondary)+columns-1)//columns
        cell_gap=min(gap*.8,width*.035)
        cell_w=(width-cell_gap*(columns-1))/columns
        cell_h=(grid_height-cell_gap*(rows-1))/rows
        grid=[]
        for index,unit in enumerate(secondary):
            block=_unit(unit,cell_w,cell_h,scale*.98)
            box=_bounds(block);bw,bh=box[2]-box[0],box[3]-box[1]
            x=(index%columns)*(cell_w+cell_gap)+(cell_w-bw)/2
            y=grid_top+(index//columns)*(cell_h+cell_gap)+(cell_h-bh)/2
            grid.extend(_move(block,x,y))
        copy=head_block+grid
    else:
        copy=_copy(copy_units,width,copy_height,scale*1.06,gap,align='left')
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


def _dense_leaderboard_seed(roles,width,height,scale,gap,source_size):
    """Keep brand, message, controls, and product in distinct banner zones."""
    margin=min(width,height)*.06
    cw=width-2*margin
    anchored_media=[u for u in roles['footer'] if u.get('footer_anchor') in ('left','right')]
    footer_units=[u for u in roles['footer'] if u not in anchored_media]
    media=roles['visual']+anchored_media
    footer,fh=_footer_band(footer_units,cw,height*.12,scale,gap*.6)
    top=margin
    bottom=height-margin-fh-(gap if fh else 0)
    available=max(1.,bottom-top)
    logo_w=min(cw*.16,max(width*.13,available*1.3)) if roles['header'] else 0
    logo,logo_h=_logo_row(roles['header'],logo_w,available*.82,scale*1.25,gap,source_size,fill=True) if logo_w else ([],0)
    copy_x=margin+(logo_w+gap if logo_w else 0)
    media_w=min(cw*.21,max(width*.17,available*2.2)) if media else 0
    media_left=width-margin-media_w
    copy_w=max(1.,media_left-gap-copy_x)
    media_top=top
    for unit in roles['decoration']:
        if unit.get('corner_callout') and (unit['bounds'][0]+unit['bounds'][2])/2>source_size[0]/2:
            callout_box=_bounds(_callout_block(unit,width,height,*source_size))
            media_top=max(media_top,callout_box[3]+gap*.5)
    media_available=max(1.,bottom-media_top)
    media_items=_visuals(media,media_w,media_available,scale,gap*.35) if media_w else []
    actions=roles['tag']+roles['action']
    action_h=min(available*.42,max(available*.32,min(width,height)*.31)) if actions else 0
    action_row,action_used=_row(actions,copy_w,action_h,scale*1.25,gap*.9,align='left',equalize_control_height=True) if actions else ([],0)
    text_h=max(1.,available-action_used-(gap if actions else 0))
    copy=_copy(roles['copy'],copy_w,text_h,scale*1.3,gap*.45,align='left')
    output=[]
    if logo:
        output.extend(_move(logo,margin,top+(available-logo_h)/2))
    if copy:
        output.extend(_move(copy,copy_x,top))
    if action_row:
        action_y=top+text_h+gap
        output.extend(_move(action_row,copy_x,action_y))
    if media_items:
        media_h=_bounds(media_items)[3]-_bounds(media_items)[1]
        media_w_actual=_bounds(media_items)[2]-_bounds(media_items)[0]
        output.extend(_move(media_items,width-margin-media_w_actual,media_top+(media_available-media_h)/2))
    if footer:
        output.extend(_shift(footer,margin,height-margin-fh))
    output.extend(_decorations(roles['decoration'],width,height,*source_size,output))
    return output


def _dense_short_banner(roles,width,height):
    """Detect compact, information-rich banners from roles and target geometry."""
    return (width/max(1.,height)>=6 and height<=120 and bool(roles['copy'])
            and bool(roles['action']) and len(roles['tag'])>=2
            and bool(roles['visual']) and bool(roles['header']))


def _equalize_control_geometry(elements,units):
    """Restore a shared control baseline after collision constraint projection."""
    by_id={e['id']:e for e in elements}
    peers=[u for u in units if u['role']=='tag']
    if not peers:return
    peer_boxes=[_bounds([by_id[e['id']] for e in u['items'] if e['id'] in by_id]) for u in peers]
    row_center=sum((b[1]+b[3])/2 for b in peer_boxes)/len(peer_boxes)
    target_h=max(max(b[3]-b[1] for b in peer_boxes),
                 max((_text(by_id[e['id']])[0]+4 for u in units if u['role'] in ('tag','action')
                      for e in u['items'] if e['id'] in by_id and e['type']=='text'),default=0))
    for u in units:
        if u['role'] not in ('tag','action'):continue
        for original in u['items']:
            e=by_id.get(original['id'])
            if not e:continue
            if e['type']=='shape':
                e['height']=target_h;e['y']=row_center-target_h/2
                if 'corner_radius' in e:e['corner_radius']=min(target_h/2,e['corner_radius'])
            elif e['type']=='text':
                e['height']=max(_text(e)[0],min(e['height'],target_h))
                e['y']=row_center-e['height']/2


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
        logo_w=max(cw*.09,min(cw*.16,available*.55))
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
    output.extend(_decorations(roles['decoration'],width,height,sw,sh,output))
    return output


def _reflow(units,width,height,sw,sh,mode,share,scale):
    if width/height>=2.4 or (mode=='split' and width/height>=1.35):
        return _editorial_split_seed(units,width,height,sw,sh,scale,share,reverse=mode=='visual-first')
    margin,gap=min(width,height)*.05,min(width,height)*.035
    vertical_flow=width/height<1.35
    # Related items use a compact rhythm; transitions between semantic groups
    # use a larger canvas-relative rhythm so the hierarchy remains legible as
    # the output dimensions change.
    section_gap=max(margin*1.35,gap*1.6) if vertical_flow else gap
    cw,ch=width-2*margin,height-2*margin
    roles={r:sorted((u for u in units if u['role']==r),key=_semantic_order)
           for r in ('copy','header','footer','visual','tag','action','decoration')}
    upper=[u for u in roles['tag'] if u['bounds'][1]<sh*.45 and width/height<1.35]
    upper_row=upper if vertical_flow else []
    if not vertical_flow:
        roles['header']+=upper
    roles['tag']=[u for u in roles['tag'] if u not in upper]
    side_header=width/height>=2.4 and bool(roles['header'])
    compact_bands=.75<=width/height<=1.3
    header,hh=_row([] if side_header else roles['header'],cw,ch*(.09 if compact_bands else .13),scale,gap)
    kicker,kh=_badge_row(upper_row,cw,ch*.09,scale*.82,gap,align='center')
    footer,fh=_footer_band(roles['footer'],cw,ch*(.06 if compact_bands else .10),scale,gap*.6,
                           center_corners=width/height<1.35)
    labels,lh=_action_band(roles['tag'],roles['action'],cw,ch*(.11 if compact_bands else .20),scale,gap,width/height<1.3)
    deferred_action_flow=bool(mode!='split' and width/height<1.35 and lh and roles['copy'] and roles['visual'])
    # Legal copy remains footer content even when its companion badge has
    # migrated into the main flow to clear a crowded corner.
    deferred_footer_flow=False
    reserved=hh+kh+fh+lh+gap*sum(bool(v) for v in (hh,kh,fh,lh))
    reserved_limit=ch*(.34 if compact_bands else .53)
    if reserved>reserved_limit:
        shrink=reserved_limit/reserved
        header,hh=_row([] if side_header else roles['header'],cw,ch*(.09 if compact_bands else .13),scale*shrink,gap*shrink)
        kicker,kh=_badge_row(upper_row,cw,ch*.09,scale*.82*shrink,gap*shrink,align='center')
        footer,fh=_footer_band(roles['footer'],cw,ch*(.06 if compact_bands else .10),scale*shrink,gap*shrink*.6,
                               center_corners=width/height<1.35)
        labels,lh=_action_band(roles['tag'],roles['action'],cw,ch*(.11 if compact_bands else .20),scale*shrink,gap*shrink,width/height<1.3)
    top,bottom=margin,height-margin
    output=[]
    if hh:
        output.extend(_shift(header,margin,top)); top+=hh+(section_gap if vertical_flow and kh else gap)
    if kh:
        output.extend(_shift(kicker,margin,top)); top+=kh+gap
    if fh and not deferred_footer_flow:
        bottom-=fh; output.extend(_shift(footer,margin,bottom)); bottom-=gap
    if lh and not deferred_action_flow:
        bottom-=lh; output.extend(_shift(labels,margin,bottom)); bottom-=gap
    available=max(.001,bottom-top)
    gap=min(gap,available*.08)
    copy_gap=max(gap*.58,margin*.45) if width/height<.68 else gap
    cx,cwidth=margin,cw
    if side_header:
        hw=cw*.15
        block,used=_logo_row(roles['header'],hw,available*.75,scale,gap)
        output.extend(_shift(block,margin,top+(available-used)/2))
        cx+=hw+gap; cwidth-=hw+gap
    copy,visuals=roles['copy'],roles['visual']
    if not copy or not visuals:
        block=_copy(copy,cwidth,available,scale,copy_gap) if copy else _visuals(visuals,cwidth,available,scale,gap)
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
        flow_height=available
        if deferred_footer_flow:flow_height-=fh+gap*1.5
        flow_height=max(.001,flow_height)
        vh=(flow_height-gap)*share; th=flow_height-gap-vh
        # Use the canvas inset as the vertical copy rhythm. This keeps the
        # headline and supporting copy from feeling cramped on long canvases,
        # while _copy retains a tighter headline-to-subheading relationship.
        text=_copy(copy,cwidth,th,scale,copy_gap)
        art=_visuals(visuals,cwidth,vh,scale,gap)
        text_box,art_box=_bounds(text),_bounds(art)
        text_h=max(0.,text_box[3]-text_box[1]);art_h=max(0.,art_box[3]-art_box[1])
        if deferred_action_flow:
            # Distribute available vertical space across measured visible
            # groups. Slot centering used to hide the intended gaps and leave
            # a large unassigned tail below the CTA. Four explicit intervals
            # keep top, hero/copy, copy/actions, and actions/footer breathing
            # room in balance. Tight relationships inside copy and button
            # groups are handled separately by _copy and _action_band.
            free=max(0.,available-text_h-art_h-lh)
            section_space=free/4
            x_text=cx+(cwidth-(text_box[2]-text_box[0]))/2
            x_art=cx+(cwidth-(art_box[2]-art_box[0]))/2
            if mode=='copy-first':
                ty=top+section_space
                vy=top+section_space+text_h+section_space
            else:
                vy=top+section_space
                ty=top+section_space+art_h+section_space
            output.extend(_move(text,x_text,ty))
            output.extend(_move(art,x_art,vy))
            content_bottom=max(ty+text_h,vy+art_h)
            output.extend(_shift(labels,margin,content_bottom+section_space))
        else:
            slack=max(0.,available-text_h-art_h-(fh if deferred_footer_flow else 0.))
            gap_slots=2+bool(deferred_footer_flow)
            inner=max(0.,slack/max(1,gap_slots))
            outer=inner
            if mode=='copy-first':
                ty=top+outer
                vy=top+outer+text_h+inner
            else:
                vy=top+outer
                ty=top+outer+art_h+inner
            output.extend(_move(text,cx+(cwidth-(text_box[2]-text_box[0]))/2,ty))
            output.extend(_move(art,cx+(cwidth-(art_box[2]-art_box[0]))/2,vy))
    output.extend(_decorations(roles['decoration'],width,height,sw,sh,output))
    return output


def _proportional(units,width,height,sw,sh):
    factor=min(width/sw,height/sh)
    desired=sqrt(width*height/(sw*sh))*1.15
    result=[]
    for u in units:
        block=_contain(_scale(u['items'],factor),width,height)
        originals={e['id']:e for e in u['items']}
        for e in block:
            if e['type']=='text':
                original=originals[e['id']]
                target=_target_font(original,u,desired)
                e['font_size']=_fit_text(e,e['width'],e['height'],max(e['font_size'],target))
        x,y,_,_=u['bounds']; _,_,w,h=_bounds(block)
        px=(width-sw*factor)/2+x*factor; py=(height-sh*factor)/2+y*factor
        result.extend(_move(block,max(0,min(width-w,px)),max(0,min(height-h,py))))
    return result


def _repair_copy_hierarchy(elements, units, width, height):
    """Keep a headline and its supporting line together after geometry repair."""
    copy=sorted((u for u in units if u['role']=='copy'),key=_semantic_order)
    by_id={e['id']:e for e in elements}
    for headline,subhead in zip(copy,copy[1:]):
        if headline.get('semantic')!='headline' or subhead.get('semantic')!='subheading':
            continue
        head=[by_id[e['id']] for e in headline['items'] if e['id'] in by_id]
        support=[by_id[e['id']] for e in subhead['items'] if e['id'] in by_id]
        if not head or not support:
            continue
        hb,sb=_bounds(head),_bounds(support)
        if min(hb[2],sb[2])-max(hb[0],sb[0])<=0:
            continue
        head_align={e.get('text_align','left') for e in head if e['type']=='text'}
        support_align={e.get('text_align','left') for e in support if e['type']=='text'}
        if len(head_align)==len(support_align)==1 and head_align==support_align:
            align=next(iter(head_align))
            if align=='left':
                target=max(hb[0],sb[0]);head_dx=target-hb[0];support_dx=target-sb[0]
            elif align=='right':
                target=min(hb[2],sb[2]);head_dx=target-hb[2];support_dx=target-sb[2]
            elif align=='center':
                target=(hb[0]+hb[2]+sb[0]+sb[2])/4
                head_dx=target-(hb[0]+hb[2])/2
                support_dx=target-(sb[0]+sb[2])/2
            else:
                head_dx=support_dx=0.
            if abs(head_dx)>1e-4 or abs(support_dx)>1e-4:
                old_x=[e['x'] for e in head+support]
                for e in head:e['x']+=head_dx
                for e in support:e['x']+=support_dx
                if not _feasible(elements,units,width,height):
                    for e,x in zip(head+support,old_x):
                        e['x']=x
        actual=sb[1]-hb[3]
        font=min((e.get('font_size',12) for e in head+support if e['type']=='text'),default=12)
        preferred=max(min(width,height)*.022,font*.65)
        target=preferred*1.45
        if actual>target:
            dy=actual-target
            old_y=[e['y'] for e in support]
            for e in support:
                e['y']-=dy
            if not _feasible(elements,units,width,height):
                for e,y in zip(support,old_y):
                    e['y']=y


def _repair_primary_action_shape(elements, units, width, height):
    """Keep a primary CTA close to its source shape when its slot permits."""
    by_id={e['id']:e for e in elements}
    for unit in units:
        if unit.get('role')!='action' or unit.get('semantic')!='cta':
            continue
        source_shape=next((e for e in unit['items'] if e['type']=='shape'),None)
        if not source_shape:
            continue
        shape=by_id.get(source_shape['id'])
        if not shape:
            continue
        source_ratio=source_shape['width']/max(1.,source_shape['height'])
        if source_ratio<=0:
            continue
        current_h=shape['height']
        # Reconstruct the height implied by the source button's own proportions
        # at its adapted width. The cap is canvas-relative and only limits cases
        # where that shape would occupy an unreasonable share of a short canvas.
        target_h=min(shape['width']/source_ratio,min(height*.10,width*.32))
        if target_h<=current_h*1.05:
            continue
        old=shape.copy()
        shape['height']=target_h
        # Keep the top edge fixed so the added height uses open space below it
        # instead of squeezing the intentionally separated asset row.
        shape['y']=max(0.,min(height-target_h,shape['y']))
        label_shift=(target_h-current_h)/2
        labels=[by_id[item['id']] for item in unit['items']
                if item['type']=='text' and item['id'] in by_id]
        for label in labels:label['y']+=label_shift
        if 'corner_radius' in shape:
            shape['corner_radius']=min(target_h/2,shape['corner_radius']*target_h/max(1.,current_h))
        if not _feasible(elements,units,width,height):
            shape.update(old)
            for label in labels:label['y']-=label_shift


def _repair_footer_bottom_anchor(elements, units, width, height):
    """Place standalone legal copy at the lower safe margin when space allows."""
    by_id={e['id']:e for e in elements}
    margin=min(width,height)*.045
    for unit in units:
        if unit['role']!='footer' or unit.get('inflow_footer_asset') or any(
                e['type']=='image' for e in unit['items']):
            continue
        items=[by_id[e['id']] for e in unit['items']
               if e['id'] in by_id and by_id[e['id']]['type']=='text']
        if not items:
            continue
        box=_bounds(items)
        target_bottom=height-margin
        dy=max(0.,target_bottom-box[3])
        if dy<=1e-4:
            continue
        old=[e['y'] for e in items]
        for e in items:
            e['y']+=dy
        if not _feasible(elements,units,width,height):
            for e,y in zip(items,old):
                e['y']=y


def _repair_vertical_hero_spacing(elements, units, width, height):
    """Balance a centered hero's clearance above and below the product."""
    by_id={e['id']:e for e in elements}
    records=[]
    for unit in units:
        items=[by_id[e['id']] for e in unit['items'] if e['id'] in by_id]
        if items:records.append((unit,_bounds(items),items))
    brands=[b for u,b,_ in records if u['semantic']=='logo' or
            u['semantic']=='badge' and u.get('source_upper')]
    copies=[b for u,b,_ in records if u['semantic'] in ('headline','subheading','body')]
    if not brands or not copies:return
    brand_bottom=max(b[3] for b in brands)
    copy_top=min(b[1] for b in copies)
    brand_center=sum((b[0]+b[2])/2 for b in brands)/len(brands)
    copy_center=sum((b[0]+b[2])/2 for b in copies)/len(copies)
    for unit,box,items in records:
        if unit['semantic'] not in ('hero','art') or box[1]<brand_bottom or box[3]>copy_top:
            continue
        center=(box[0]+box[2])/2
        if min(abs(center-brand_center),abs(center-copy_center))>width*.18:
            continue
        top_gap=box[1]-brand_bottom
        bottom_gap=copy_top-box[3]
        if abs(top_gap-bottom_gap)<=min(width,height)*.015:
            continue
        dy=(bottom_gap-top_gap)/2
        old=[e['y'] for e in items]
        for e in items:e['y']+=dy
        if not _feasible(elements,units,width,height):
            for e,y in zip(items,old):e['y']=y
            lo,hi=0.,1.
            for _ in range(16):
                mid=(lo+hi)/2
                for e,y in zip(items,old):e['y']=y+dy*mid
                if _feasible(elements,units,width,height):lo=mid
                else:hi=mid
            for e,y in zip(items,old):e['y']=y+dy*lo


def _repair_vertical_brand_clearance(elements, units, width, height):
    """Use spare lower-canvas space to open the transition below the brand."""
    if width/max(1.,height)>=1.35:
        return
    by_id={e['id']:e for e in elements}
    records=[]
    for unit in units:
        items=[by_id[e['id']] for e in unit['items'] if e['id'] in by_id]
        if items:
            records.append((unit,_bounds(items),items))
    brands=[b for u,b,_ in records
            if u['semantic']=='logo' or u.get('source_upper') and u['role'] in ('tag','header')
            or u.get('corner_callout')]
    if not brands:
        return
    brand_bottom=max(b[3] for b in brands)
    flow=[(u,b,items) for u,b,items in records
          if (u['role'] not in ('header','decoration','footer')
              or u['role']=='footer' and u.get('source_footer_anchor') in ('left','right')
              and any(e['type']=='image' for e in u['items']))
          and not (u.get('source_upper') and u['semantic'] in ('badge','kicker'))
          and b[1]>=brand_bottom]
    if not flow:
        return
    first_top=min(b[1] for _,b,_ in flow)
    preferred=min(width,height)*.055
    needed=max(0.,preferred-(first_top-brand_bottom))
    if needed<=1e-4:
        return
    flow_bottom=max(b[3] for _,b,_ in flow)
    footer_tops=[b[1] for u,b,_ in records if u['role']=='footer'
                 and not (u.get('source_footer_anchor') in ('left','right')
                          and any(e['type']=='image' for e in u['items']))]
    lower_limit=min(footer_tops) if footer_tops else height-preferred
    # Keep at least one comfortable section gap before footer content. Moving
    # every later primary group together preserves their internal hierarchy.
    footer_reserve=min(width,height)*.032
    available=max(0.,lower_limit-flow_bottom-footer_reserve)
    shift=min(needed,available)
    if shift<=1e-4:
        return
    moved=[(e,e['y']) for _,_,items in flow for e in items]
    for e,y in moved:
        e['y']=y+shift
    if not _feasible(elements,units,width,height):
        for e,y in moved:
            e['y']=y


def _repair_banner_logo_column_balance(elements, units, width, height):
    """Center visible logo artwork between the canvas inset and copy column."""
    if width/max(1.,height)<1.35:
        return
    by_id={e['id']:e for e in elements}
    logos=[(u,[by_id[e['id']] for e in u['items'] if e['id'] in by_id])
           for u in units if u['semantic']=='logo']
    copies=[_bounds([by_id[e['id']] for e in u['items'] if e['id'] in by_id])
            for u in units if u['role']=='copy'
            and any(e['id'] in by_id for e in u['items'])]
    if not logos or not copies:
        return
    copy_left=min(box[0] for box in copies)
    margin=min(width,height)*.05
    gap=min(width,height)*.025
    moved=[]
    for _,items in logos:
        if not items:
            continue
        box=_bounds(items)
        # Only balance a logo that occupies the leading brand column; logos
        # centered over the message or placed to its right retain their anchor.
        if box[2]+gap>copy_left or (box[0]+box[2])/2>copy_left:
            continue
        target_center=(margin+copy_left-gap)/2
        dx=target_center-(box[0]+box[2])/2
        dx=min(dx,copy_left-gap-box[2],width-margin-box[2])
        dx=max(dx,margin-box[0])
        if abs(dx)<1e-4:
            continue
        moved.extend((e,e['x']) for e in items)
        for e in items:
            e['x']+=dx
    if moved and not _feasible(elements,units,width,height):
        for e,x in moved:
            e['x']=x


def _repair_banner_cta_copy_proximity(elements, units, width, height):
    """Keep banner CTAs visibly distinct while attached to their copy group."""
    if width/max(1.,height)<1.35:
        return
    by_id={e['id']:e for e in elements}
    copies=[e for u in units if u['role']=='copy'
            for e in u['items'] if e['id'] in by_id]
    actions=[u for u in units if u['semantic']=='cta']
    if not copies or not actions:
        return
    copy_box=_bounds([by_id[e['id']] for e in copies])
    copy_font=max((e.get('font_size',0) for e in copies if e.get('type')=='text'),default=0)
    canvas_gap=min(width,height)*.12
    heroes=[_bounds([by_id[e['id']] for e in u['items'] if e['id'] in by_id])
            for u in units if u['semantic'] in ('hero','art')]
    for unit in actions:
        items=[by_id[e['id']] for e in unit['items'] if e['id'] in by_id]
        if not items:
            continue
        box=_bounds(items)
        # The CTA already belongs to the copy column when it sits under or
        # inside that column. For inline CTAs, enforce a restrained minimum gap
        # so the action reads clearly without splitting the copy group apart.
        if box[0]<copy_box[2]:
            continue
        width_cta=box[2]-box[0]
        action_font=max((e.get('font_size',0) for e in items if e.get('type')=='text'),default=0)
        gap=max(canvas_gap,(copy_font+action_font)*.25)
        target=copy_box[2]+gap
        right_limit=width-min(width,height)*.045
        for hero in heroes:
            vertical_overlap=min(box[3],hero[3])-max(box[1],hero[1])>0
            if vertical_overlap and hero[0]>=box[0]:
                right_limit=min(right_limit,hero[0]-gap)
        if target+width_cta>right_limit+1e-5:
            continue
        dx=target-box[0]
        if abs(dx)<1e-4:
            continue
        old_x=[e['x'] for e in items]
        for e in items:
            e['x']+=dx
        if not _feasible(elements,units,width,height):
            for e,x in zip(items,old_x):
                e['x']=x


def _enrich_units(source, units):
    """Feature-based role hypotheses, with uncertainty kept internal.

    No ID/filename lookup. Being at the top alone does not make every image a
    logo, and colored rectangles alone are not automatically calls to action.
    """
    sw, sh = source['canvas']['width'], source['canvas']['height']
    image_units = [u for u in units if any(e['type'] == 'image' for e in u['items'])]
    largest_image = max((_area(e) for u in image_units for e in u['items'] if e['type'] == 'image'), default=1)
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
        size=max((e.get('font_size',0) for e in u['items']),default=0)
        peers = [v for v in labels if v is not u and abs(v['bounds'][1]-y) < sh*.025
                 and .85 < (v['bounds'][3]-v['bounds'][1])/max(1,b-y) < 1.15
                 and abs(sum(e.get('rotation',0) for e in v['items'])/len(v['items'])-angle)<=2
                 and .85 < max((e.get('font_size',0) for e in v['items']),default=0)/max(1,size) < 1.15]
        # A single label placed in a source corner is an accent even when it
        # is not rotated. Keep it anchored to that corner when the canvas
        # changes shape; a repeated row of similar labels remains a badge set.
        source_corner_label=bool(u['label'] and y<sh*.25 and (x<sw*.2 or r>sw*.8))
        u['corner_callout']=bool(source_corner_label and (abs(angle)>2 or not peers))
        if u['corner_callout']:
            # It is still a content badge: preserve its corner preference by
            # default, but let reflowed layouts place it as an upper kicker.
            # Treating it as decoration hid it from the copy/header grouping
            # logic, which could strand it beside unrelated lower content.
            # Tilted corner ribbons retain their anchored/decorative role in
            # landscape formats, where the banner seed uses their corner to
            # reserve clearance from the hero. Vertical migration explicitly
            # promotes them to the upper content stack.
            u.update(role='decoration' if abs(angle)>2 else 'tag',
                     semantic='callout' if abs(angle)>2 else 'badge',confidence=.78)
            continue
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
                # Supporting copy often uses a deliberately smaller size than
                # the headline. A hard .48 cutoff mislabels real subheads as
                # body copy and drops their tight heading relationship.
                semantic, confidence = ('subheading' if size >= head_font*.38 else 'body'), .62
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
    if abs(log(delta)) < .18:
        flow = 'preserve'
    elif ratio >= 3.2:
        flow = 'banner'
    elif ratio <= .68:
        flow = 'story'
    elif .82 <= ratio <= 1.2:
        flow = 'square'
    else:
        flow = 'landscape' if ratio > 1 else 'portrait'
    return dict(flow=flow, delta=delta, ratio=ratio)


def _semantic_order(u):
    return ({'logo': 0, 'kicker': 1, 'headline': 2, 'subheading': 3, 'body': 4,
             'badge': 5, 'cta': 6, 'hero': 7, 'art': 8, 'footer': 9,
             'decoration': 10}.get(u['semantic'], 8), u['bounds'][1], u['bounds'][0])


def _banner_seed(units, width, height, scale, hero_share=.18, reverse=False, source_size=None):
    """True inline banner: logo | copy | action | hero, plus secondary rows."""
    margin, gap = min(width, height)*.05, min(width, height)*.035
    cw, ch = width-2*margin, height-2*margin
    roles = {r: sorted((u for u in units if u['role'] == r), key=_semantic_order)
             for r in ('header','copy','action','tag','visual','footer','decoration')}
    if _dense_short_banner(roles,width,height):
        return _dense_leaderboard_seed(roles,width,height,scale,gap,source_size or (width,height))
    footer, fh = _footer_band(roles['footer'], cw, ch*.12, scale, gap)
    available = ch-fh-(gap if fh else 0)
    categories = [('header', .16), ('copy', .50), ('action', .34)]
    if roles['visual']:
        protected_right=any(u.get('corner_callout') and (u['bounds'][0]+u['bounds'][2])/2>(source_size or (width,height))[0]/2 for u in roles['decoration']) or any(u.get('footer_anchor')=='right' for u in roles['footer'])
        if protected_right:
            categories = [('header', .08), ('copy', .45), ('action', .16), ('visual', max(hero_share,.31))]
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
        logo_min=max(cw*.09,min(cw*.16,available*.55))
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
            block,used=_logo_row(roles[key],w,max(available*.80,available*.70),scale,gap,source_size,fill=True)
        elif key == 'visual':
            block = _visuals(roles[key], w, available, scale, gap)
        else:
            block, used = _banner_actions(roles[key], w, available*.85, scale, gap,cw)
        if block:
            box=_bounds(block);bw,bh=box[2]-box[0],box[3]-box[1]
            blocks.append((key,block,bw,bh,box))
    right_clear=width-margin
    for u in roles['decoration']:
        if not u.get('corner_callout'):
            continue
        if source_size and (u['bounds'][0]+u['bounds'][2])/2<=source_size[0]/2:
            continue
        callout=_callout_block(u,width,height,*(source_size or (width,height)))
        cb=_bounds(callout);edge=min(width,height)*.045
        right_clear=min(right_clear,width-edge-(cb[2]-cb[0])-gap)
    right_footers=[u for u in roles['footer'] if u.get('footer_anchor')=='right']
    if right_footers:
        footer_ids={e['id'] for u in right_footers for e in u['items']}
        fb=_bounds([e for e in footer if e['id'] in footer_ids]) if any(e['id'] in footer_ids for e in footer) else None
        if fb:
            right_clear=min(right_clear,fb[0]-gap)
    # Pack measured artwork/text extents. A tight left-aligned copy column
    # must not reserve half the banner as invisible whitespace.
    sum_bw=sum(row[2] for row in blocks)
    pack_width=min(cw,max(0,right_clear-margin))
    if len(blocks)>=3:
        gap_x=max(gap,(pack_width-sum_bw)/max(1,len(blocks)-1))
        cursor=margin
    else:
        gap_x=min(pack_width*.08,max(gap,(pack_width-sum_bw)/max(1,len(blocks)+1)))
        occupied=sum_bw+gap_x*max(0,len(blocks)-1)
        cursor=margin+max(0,(pack_width-occupied)/2)
    for index,(key,block,bw,bh,box) in enumerate(blocks):
        if key=='visual' and index==len(blocks)-1 and right_clear<width-margin:
            available_slot=max(.001,right_clear-(cursor+gap))
            if available_slot<bw:
                block=_visuals(roles['visual'],available_slot,available,scale,gap)
                box=_bounds(block);bw,bh=box[2]-box[0],box[3]-box[1]
                gap_x=min(gap_x,max(0,available_slot-bw))
                occupied=sum(row[2] for row in blocks if row[0]!='visual')+bw+gap_x*max(0,len(blocks)-1)
                if len(blocks)>=3:
                    cursor=margin+sum(row[2]+gap_x for row in blocks[:index])
                else:
                    cursor=margin+max(0,(pack_width-occupied)/2)+sum(row[2]+gap_x for row in blocks[:index])
        px=cursor-box[0]
        py=margin+(available-bh)/2-box[1]
        output.extend(_move(block,px,py))
        cursor+=bw+gap_x
    if roles['tag'] and not roles['copy']:
        tags, _ = _row(roles['tag'], cw, available*.2, scale*.8, gap,align='left')
        output.extend(_shift(tags, margin, margin))
    if footer:
        output.extend(_shift(footer, margin, height-margin-fh))
    output.extend(_decorations(roles['decoration'],width,height,
                  (source_size or (width,height))[0],(source_size or (width,height))[1],output))
    return output


def _stack_seed(units, width, height, scale, visual_share, hero_first, source_size,
                move_corner_callouts=False):
    """Story/square flow with separate top badges and bottom CTA/footer bands."""
    working = deepcopy(units)
    if move_corner_callouts:
        for unit in working:
            if unit.get('corner_callout'):
                # Preserve the source's top-corner intent: when moved inward,
                # an accent remains an upper kicker instead of joining the CTA.
                unit.update(role='tag',semantic='badge',corner_callout=False,
                            source_upper=True,inflow_callout=True,confidence=.78)
                for item in unit['items']:
                    item['rotation']=0
                unit['bounds']=_bounds(unit['items'])
            elif (unit['role']=='footer' and unit.get('source_footer_anchor',unit.get('footer_anchor')) in ('left','right')
                  and any(item['type']=='image' for item in unit['items'])):
                # Free the legal footer's corner by putting its small image
                # beside the primary action in the main content flow.
                unit.update(role='action',semantic='badge',inflow_footer_asset=True,confidence=.78)
                unit.pop('footer_anchor',None)
                unit.pop('source_footer_anchor',None)
    # Use the existing text-aware packing mechanics, but keep top badges out
    # of the CTA row and preserve headline/body order independently of grouping.
    sw, sh = source_size
    # `_reflow` gives upper tags their own centered kicker row. Keep their
    # semantic role intact so they cannot shift a source-centered logo.
    return _reflow(working,width,height,sw,sh,'visual-first' if hero_first else 'copy-first',visual_share,scale)


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


def _can_bleed(u):
    """Raw output frames must remain on-canvas, including decoration."""
    return False


def _records(elements, units):
    by_id = {e['id']: e for e in elements}
    return [(u, [by_id[e['id']] for e in u['items']]) for u in units]


def _unit_margin(box, width, height, unit=None):
    # Give the primary reading flow a little breathing room when the canvas
    # permits it. Corner/footer content keeps the tighter legacy safe area.
    primary = unit is not None and unit['role'] in ('header', 'copy', 'tag', 'action', 'visual')
    preferred = min(width, height) * (.055 if primary else .045)
    # Never make a source-wide unit infeasible solely to enforce padding.
    return max(0, min(preferred,
               (width-box[2]+box[0])/2, (height-box[3]+box[1])/2))


def _feasible(elements, units, width, height):
    if any(not all(isfinite(v) for v in (e.get('x',0),e.get('y',0),e.get('width',0),e.get('height',0)))
           or (e.get('type')!='image' and (e['x'] < -1e-5 or e['y'] < -1e-5
           or e['x']+e['width'] > width+1e-5 or e['y']+e['height'] > height+1e-5))
           for e in elements):
        return False
    records = [(u,_bounds(items)) for u,items in _records(elements,units)]
    gap = min(width,height)*.008
    for i,(u,b) in enumerate(records):
        if not all(isfinite(v) for v in b):
            return False
        margin = -min(width,height)*.20 if _can_bleed(u) else 0 if u['role']=='decoration' else _unit_margin(b,width,height,u)
        if b[0]<margin-1e-5 or b[1]<margin-1e-5 or b[2]>width-margin+1e-5 or b[3]>height-margin+1e-5:
            return False
        for v,c in records[:i]:
            if _layered(u,v):
                continue
            if not (b[2]+gap<=c[0]+1e-5 or c[2]+gap<=b[0]+1e-5 or b[3]+gap<=c[1]+1e-5 or c[3]+gap<=b[1]+1e-5):
                return False
    return True


def _repair_sparse_vertical_stack(elements, units, width, height):
    """Give tall-layout content groups a consistent, readable vertical rhythm.

    Atomic units keep their internal geometry, source order is retained, and
    legal/footer units remain anchored. Text roles are grouped together so
    related headline and supporting copy retain their close relationship.
    """
    if height/max(1.,width)<=1.35:
        return False
    by_id={e['id']:e for e in elements}
    specs=(('brand',lambda u:u['semantic']=='logo' or
            u['semantic']=='badge' and u.get('source_upper')),
           ('hero',lambda u:u['semantic'] in ('hero','art')),
           ('copy',lambda u:u['semantic'] in ('headline','subheading','body','kicker')),
           ('tags',lambda u:u['role']=='tag' and not u.get('source_upper')),
           ('action',lambda u:u['semantic']=='cta'))
    groups=[]
    for name,predicate in specs:
        members=[u for u in units if predicate(u)]
        if not members:
            continue
        box=_bounds([by_id[e['id']] for u in members for e in u['items'] if e['id'] in by_id])
        groups.append(dict(name=name,members=members,box=box))
    groups.sort(key=lambda g:g['box'][1])
    if len(groups)<3:
        return False
    short=min(width,height)
    # Use the same base rhythm between content sections. Product art and badge
    # bands receive a little more room; related copy remains one atomic group.
    targets=[]
    for a,b in zip(groups,groups[1:]):
        factor=.14
        if 'hero' in (a['name'],b['name']):
            factor=.18
        elif (a['name'],b['name']) in (('copy','tags'),('tags','action')):
            factor=.12
        targets.append(short*factor)
    first_top=groups[0]['box'][1]
    total_height=sum(g['box'][3]-g['box'][1] for g in groups)+sum(targets)
    footer=[u for u in units if u['role']=='footer' and
            not u.get('inflow_footer_asset',False)]
    bottom_limit=min((_bounds([by_id[e['id']] for e in u['items'] if e['id'] in by_id])[1]
                      for u in footer),default=height-short*.06)
    top_limit=short*.06
    available=max(0.,bottom_limit-top_limit)
    if total_height>available:
        return False
    # Center the content sequence in the safe area above footer/legal content.
    cursor=top_limit+(available-total_height)*.5
    original_y={e['id']:e['y'] for e in elements}
    for i,group in enumerate(groups):
        dy=cursor-group['box'][1]
        for unit in group['members']:
            for item in unit['items']:
                if item['id'] in by_id:
                    by_id[item['id']]['y']+=dy
        cursor+=group['box'][3]-group['box'][1]
        if i<len(targets):
            cursor+=targets[i]
    if not _feasible(elements,units,width,height):
        for e in elements:
            e['y']=original_y[e['id']]
        return False
    return True


def _peer_row_groups(indexes,boxes):
    """Cluster same-peer controls by observed row, preserving deliberate stacks."""
    ordered=sorted(indexes,key=lambda i:(boxes[i][1]+boxes[i][3])/2)
    groups=[]
    for index in ordered:
        box=boxes[index];center=(box[1]+box[3])/2;height=max(1e-6,box[3]-box[1])
        if groups:
            previous=groups[-1][-1];other=boxes[previous]
            other_center=(other[1]+other[3])/2
            tolerance=max(2.,min(height,max(1e-6,other[3]-other[1]))*.45)
            if center-other_center<=tolerance:
                groups[-1].append(index)
                continue
        groups.append([index])
    return groups


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
    boxes = [_bounds(items) for _,items in records]
    solver = kiwi.Solver()
    xs = [kiwi.Variable(f'x{i}') for i in range(len(records))]
    ys = [kiwi.Variable(f'y{i}') for i in range(len(records))]
    gap = min(width,height)*.008
    try:
        # Preserve repeated icon/caption relationships through Cassowary
        # projection. These are soft directional preferences: hard bounds and
        # non-overlap remain authoritative when the target cannot fit the pair.
        id_to_unit={item['id']:i for i,(_,items) in enumerate(records) for item in items}
        perceptual=units[0].get('perceptual',{}) if units else {}
        for pair in perceptual.get('affinities',[]):
            icon=id_to_unit.get(pair.get('icon'));caption=id_to_unit.get(pair.get('caption'))
            if icon is None or caption is None or icon==caption:continue
            ix,iy,ir,ib=boxes[icon];cx,cy,cr,cb=boxes[caption]
            iw,ih=ir-ix,ib-iy;cw,ch=cr-cx,cb-cy
            preferred_gap=min(width,height)*.018
            if pair.get('orientation')=='below':
                solver.addConstraint((xs[caption]+cw/2==xs[icon]+iw/2) | kiwi.strength.medium)
                solver.addConstraint((ys[caption]>=ys[icon]+ih) | kiwi.strength.weak)
                solver.addConstraint((ys[caption]<=ys[icon]+ih+preferred_gap*2) | kiwi.strength.weak)
            elif pair.get('orientation')=='beside':
                solver.addConstraint((ys[caption]+ch/2==ys[icon]+ih/2) | kiwi.strength.medium)
                solver.addConstraint((xs[caption]>=xs[icon]+iw) | kiwi.strength.weak)
                solver.addConstraint((xs[caption]<=xs[icon]+iw+preferred_gap*2) | kiwi.strength.weak)
        peer_bands={}
        for i,(u,_) in enumerate(records):
            if u['role']=='tag' and u.get('peer_ids'):
                peer_bands.setdefault(u['peer_ids'],[]).append(i)
        for indexes in peer_bands.values():
            if len(indexes)<2:continue
            for row in _peer_row_groups(indexes,boxes):
                if len(row)<2:continue
                first=row[0];first_h=boxes[first][3]-boxes[first][1]
                for other in row[1:]:
                    other_h=boxes[other][3]-boxes[other][1]
                    solver.addConstraint(ys[first]+first_h/2==ys[other]+other_h/2)
        for i, ((u,_),box) in enumerate(zip(records,boxes)):
            x,y,r,b = box
            w,h = r-x,b-y
            margin = -min(width,height)*.20 if _can_bleed(u) else 0 if u['role']=='decoration' else _unit_margin(box,width,height,u)
            solver.addConstraint(xs[i]>=margin)
            solver.addConstraint(ys[i]>=margin)
            solver.addConstraint(xs[i]+w<=width-margin)
            solver.addConstraint(ys[i]+h<=height-margin)
            for item in records[i][1]:
                if item.get('type')=='image':
                    continue
                dx=item['x']-x;dy=item['y']-y
                solver.addConstraint(xs[i]+dx>=0)
                solver.addConstraint(ys[i]+dy>=0)
                solver.addConstraint(xs[i]+dx+item['width']<=width)
                solver.addConstraint(ys[i]+dy+item['height']<=height)
            solver.addConstraint((xs[i]==x) | kiwi.strength.strong)
            solver.addConstraint((ys[i]==y) | kiwi.strength.strong)
        for i,(u,_) in enumerate(records):
            a=boxes[i]; aw,ah=a[2]-a[0],a[3]-a[1]
            for j,(v,_) in enumerate(records[:i]):
                if _layered(u,v):
                    continue
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
        output.extend(_move(deepcopy(items),xs[i].value(),ys[i].value()))
    _sync_backing_panels(output,units,width,height)
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
    context=dict(affinities=affiliations,background_map=[],background_luma=[],
                 background_note='No stationary background to analyze')
    if not backgrounds or width*height>16000000:
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
            cells=[];lumas=[]
            for row in range(3):
                for col in range(3):
                    tile=pixels.crop((col*32,row*32,(col+1)*32,(row+1)*32))
                    edges=tile.convert('L').filter(ImageFilter.FIND_EDGES).crop((1,1,31,31))
                    edge=ImageStat.Stat(edges).mean[0]/255
                    stats=ImageStat.Stat(tile)
                    variance=sum(stats.var)/(3*255**2)
                    cells.append(min(1,edge*3+variance*4))
                    lumas.append(_luminance(stats.mean))
            if max(cells)-min(cells)>.015:
                context['background_map']=cells
            if max(lumas)-min(lumas)>.08:
                context['background_luma']=lumas
            context['background_note']='3x3 busyness and luminance proxy on stationary backgrounds'
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
    lumas=context.get('background_luma',[])
    backed_ids={eid for unit in units if unit.get('backing_panel')
                for eid in unit.get('backed_ids',())}
    if lumas:
        for (unit,items),_ in zip(records,boxes):
            if unit.get('label') or any(e['type']=='shape' for e in items):
                continue
            for element in items:
                if element['type']!='text' or element['id'] in backed_ids:
                    continue
                rgba=_rgba(element.get('color',''))
                if not rgba:continue
                alpha=rgba[3]*element.get('opacity',1.)
                text_luma=_luminance(rgba[:3])
                target=3.0 if element.get('font_size',0)>=24 or (
                    element.get('font_weight')=='bold' and element.get('font_size',0)>=18) else 4.5
                area=max(1.,element['width']*element['height'])
                for i,bg_luma in enumerate(lumas):
                    cell=((i%3)*width/3,(i//3)*height/3,
                          (i%3+1)*width/3,(i//3+1)*height/3)
                    overlap=_overlap(_bounds([element]),cell)
                    if overlap<=0:continue
                    weight=overlap/area
                    foreground=text_luma*alpha+bg_luma*(1-alpha)
                    contrast=(max(foreground,bg_luma)+.05)/(min(foreground,bg_luma)+.05)
                    terms['background_busyness']+=weight*max(0.,(target-contrast)/target)**2*2.5
    return terms


def _stack_balance(records, boxes, width, height):
    """Balance open space around related vertical groups, not each element."""
    def group_box(predicate):
        members=[b for (u,_),b in zip(records,boxes) if predicate(u)]
        return (min(b[1] for b in members),max(b[3] for b in members)) if members else None
    # Treat related copy and peer controls as single blocks, but keep a migrated
    # corner badge, primary CTA, and footer distinct so their gaps are balanced
    # too. This catches cramped internal control bands without spacing every
    # line of text as if it were an unrelated section.
    predicates=(
        lambda u: u['semantic']=='logo' or (u['semantic']=='badge' and u.get('source_upper')),
        lambda u: u['semantic'] in ('hero','art'),
        lambda u: u['semantic'] in ('headline','subheading','body','kicker'),
        lambda u: u['role']=='tag' and not u.get('source_upper'),
        lambda u: u.get('inflow_footer_asset',False) or
                  u['role']=='footer' and any(e['type']=='image' for e in u['items']),
        lambda u: u['semantic']=='cta',
        lambda u: u['role']=='footer' and not u.get('inflow_footer_asset',False) and
                  not any(e['type']=='image' for e in u['items']),
    )
    groups=[box for predicate in predicates if (box:=group_box(predicate)) is not None]
    # Actual vertical order allows the product to sit above or below the copy.
    groups.sort()
    if len(groups)<2:
        return 0.
    inset=min(width,height)*.055
    top_inset=inset*.88
    bottom_inset=inset*1.12
    gaps=[max(0.,groups[0][0]-top_inset)]
    gaps.extend(max(0.,b[0]-a[1]) for a,b in zip(groups,groups[1:]))
    gaps.append(max(0.,height-bottom_inset-groups[-1][1]))
    mean=sum(gaps)/len(gaps)
    # Keep this a gentle preference: the final repair centers a clearly
    # isolated hero directly in its available interval, while this term only
    # nudges broad section spacing and must not reward enlarging the artwork.
    balance=sum(((gap-mean)/height)**2 for gap in gaps)/len(gaps)
    # On tall canvases, equal but enormous gaps can still make a small set of
    # related content feel disconnected. Keep the core stack compact and let
    # the remaining whitespace collect around the group. This does not include
    # footer/legal copy or corner footer assets, whose anchoring is deliberate.
    if height/max(1.,width)>1.35:
        core_predicates=predicates[:4]+(predicates[5],)
        core=[box for predicate in core_predicates
              if (box:=group_box(predicate)) is not None]
        core.sort()
        if len(core)>=3:
            inner=[max(0.,b[0]-a[1]) for a,b in zip(core,core[1:])]
            mean_inner=sum(inner)/len(inner)
            target_gap=max(min(width,height)*.045,height*.04)
            overflow=max(0.,(mean_inner-target_gap)/height)
            # Give excessive gaps enough weight to change the candidate choice
            # on sparse, tall formats. Equal 150px gaps can be numerically
            # balanced yet still leave a short message and CTA disconnected.
            balance+=overflow**2*4.0
    return balance


def _text_right_limit(unit,box,records,width,height):
    """Horizontal room at this anchor, respecting neighboring occupied rows."""
    limit=width-min(width,height)*.045;gap=min(width,height)*.008
    for other,items in records:
        if other is unit or _layered(unit,other):continue
        b=_bounds(items)
        if min(box[3],b[3])>max(box[1],b[1]) and b[0]>=box[2]-1e-6:
            limit=min(limit,b[0]-gap)
    return max(box[2],limit)


def _banner_vertical_neighbor(unit,box,records):
    """Whether a unit shares a copy column with another vertically stacked unit."""
    left,top,right,bottom=box
    for other,items in records:
        if other is unit or other['role'] not in ('header','copy','tag','action','visual'):
            continue
        other_box=_bounds(items)
        overlap=max(0,min(right,other_box[2])-max(left,other_box[0]))
        if overlap>=min(right-left,other_box[2]-other_box[0])*.2:
            return True
    return False


def _banner_centerline(records,width,height):
    margin=min(width,height)*.05
    footer_tops=[_bounds(items)[1] for unit,items in records if unit['role']=='footer']
    bottom=min(footer_tops)-min(width,height)*.035 if footer_tops else height-margin
    return (margin+max(margin,bottom))/2


def _energy_terms(elements, units, source, width, height, desired):
    records = _records(elements,units)
    boxes = [_bounds(items) for _,items in records]
    state=_aspect_state(source,width,height)
    terms = dict(overlap=0.,boundary=0.,text_fit=0.,readability=0.,balance=0.,
                 hierarchy=0.,alignment=0.,spacing=0.,canvas_use=0.,source_fidelity=0.)
    terms.update(rag=0.,affinity=0.,whitespace=0.,background_busyness=0.,axial=0.,badge_flow=0.,footer_anchor=0.,hero_prominence=0.,cta_prominence=0.,brand_visibility=0.,column_alignment=0.,avoidable_wrap=0.,banner_action_stack=0.,banner_gap_evenness=0.,banner_centering=0.,stack_balance=0.,badge_alignment=0.,brand_media_spacing=0.,footer_visibility=0.,primary_clearance=0.,callout_migration=0.,source_order=0.,decoration_composition=0.)
    # Keep decorative elements in the broad source regions that gave them
    # compositional purpose. When source accents occupy opposite corners,
    # discourage the optimizer from clustering them into the same corner.
    source_w=max(1.,source['canvas']['width']);source_h=max(1.,source['canvas']['height'])
    decor=[]
    for (unit,_),box in zip(records,boxes):
        if unit['role']!='decoration' or unit.get('corner_callout'):
            continue
        sx=(unit['bounds'][0]+unit['bounds'][2])/2/source_w
        sy=(unit['bounds'][1]+unit['bounds'][3])/2/source_h
        tx=(box[0]+box[2])/2/width;ty=(box[1]+box[3])/2/height
        # Preserve clear edge/corner intent while allowing free placement for
        # shapes that began near the center of the source canvas.
        ex=.12 if sx<.30 else .88 if sx>.70 else sx
        ey=.12 if sy<.30 else .88 if sy>.70 else sy
        terms['decoration_composition']+=(tx-ex)**2+(ty-ey)**2
        decor.append((sx,sy,tx,ty))
    for i,(sx,sy,tx,ty) in enumerate(decor):
        for osx,osy,otx,oty in decor[:i]:
            if abs(sx-osx)>.40 and abs(sy-osy)>.40:
                # Preserve diagonal opposition only when the source already
                # establishes it; arbitrary shapes are never forced into a
                # mirrored arrangement.
                terms['decoration_composition']+=max(0.,.42-abs(tx-otx))**2
                terms['decoration_composition']+=max(0.,.42-abs(ty-oty))**2
    if state['flow'] in ('story','portrait','square'):
        terms['stack_balance']=_stack_balance(records,boxes,width,height)
    masses=[]
    for i,((u,items),box) in enumerate(zip(records,boxes)):
        x,y,r,b=box; area=max(1e-9,(r-x)*(b-y))
        if not all(isfinite(v) for v in box) or any(min(e['width'],e['height'])<0 for e in items):
            return {**terms,'boundary':1e9}
        if not _can_bleed(u):
            terms['boundary'] += 1-_overlap(box,(0,0,width,height))/area
        if u.get('inflow_callout'):
            terms['callout_migration']+=1.
        for (v,_),other in zip(records[:i],boxes[:i]):
            if not _layered(u,v):
                other_area=max(1e-9,(other[2]-other[0])*(other[3]-other[1]))
                terms['overlap'] += _overlap(box,other)/min(area,other_area)
        if u['role']!='decoration':
            # Perceptual mass: sqrt(area) avoids a large blank textbox dominating.
            mass=sqrt(area)*(.2 if u.get('footer_anchor') in ('left','right') else .4 if u['role']=='footer' else 1)
            masses.append((mass,(x+r)/2/width,(y+b)/2/height))
        if u['role']=='footer' and u.get('footer_anchor') in ('left','right') and any(e['type']=='image' for e in items):
            source_h=max(1.,u['bounds'][3]-u['bounds'][1])
            # Corner imagery still carries information at small sizes. Keep a
            # source-relative visibility floor, capped by the target canvas,
            # so compact footer assets do not collapse to near-invisible marks.
            minimum=min(height*.12,source_h*min(desired,1.)*.90)
            terms['footer_visibility']+=max(0.,(minimum-(b-y))/max(1.,minimum))**2
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
            if state['flow'] in ('square','portrait','story','landscape'):
                source_height=(source_box[3]-source_box[1])/source['canvas']['height']
                target_height=(b-y)/height
                minimum=max(source_height*.72,{'square':.32,'portrait':.27,'story':.25,'landscape':.28}[state['flow']])
                shortfall=max(0,(minimum-target_height)/max(.01,minimum))
                terms['hero_prominence']+=shortfall if state['flow']=='square' else shortfall**2
        if u['semantic']=='cta' and u['role']=='action':
            tag_heights=[tb[3]-tb[1] for (tag,_),tb in zip(records,boxes) if tag['role']=='tag']
            source_h=max(1.,u['bounds'][3]-u['bounds'][1])
            preferred=max(source_h*desired*.82,
                          max(tag_heights,default=0.)*1.12,
                          min(height*.12,18.))
            terms['cta_prominence']+=max(0.,(preferred-(b-y))/max(1.,preferred))**2
        if u['semantic']=='logo' and width/height>=2.4:
            old=u['bounds'];ratio=(old[2]-old[0])/max(1e-9,old[3]-old[1])
            min_h=min(height*.22,(old[3]-old[1])*desired*.70)
            min_w=min(width*.055,(old[2]-old[0])*desired*.70)
            terms['brand_visibility']+=max(0,(min_h-(b-y))/max(1,min_h))**2
            terms['brand_visibility']+=max(0,(min_w-(r-x))/max(1,min_w))**2
            wanted=min(height*.60,width*.15/max(1e-9,ratio),(old[3]-old[1])*desired*1.8)
            terms['brand_visibility']+=max(0,(wanted-(b-y))/max(1,wanted))**2
        if state['flow']=='banner' and u['role'] in ('header','copy','tag','action','visual') and not _banner_vertical_neighbor(u,box,records):
            center=(y+b)/2
            terms['banner_centering']+=((center-_banner_centerline(records,width,height))/height)**2
        if _axis_active(u,source,width,height,elements,units):
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
    headlines=[(u,items) for u,items,_ in copy if u['semantic']=='headline']
    for head_unit,head_items in headlines:
        head_width=max((_text_info(e)[1] for e in head_items if e['type']=='text'),default=0.)
        if head_width<=0:continue
        for support,support_items,_ in copy:
            if support['semantic'] not in ('subheading','body'):continue
            support_width=max((_text_info(e)[1] for e in support_items if e['type']=='text'),default=0.)
            terms['hierarchy']+=max(0.,support_width/head_width-.95)**2
    main_copy=[b for (u,_),b in zip(records,boxes) if u['semantic'] in ('headline','subheading','body')]
    if state['flow'] in ('story','portrait','square'):
        # Preserve a clear source-evidenced reading order when a tall target
        # forces the design into one column (e.g. hero above headline).
        source_visual=[(u,b) for (u,_),b in zip(records,boxes) if u['role']=='visual']
        source_copy=[(u,b) for (u,_),b in zip(records,boxes) if u['role']=='copy']
        source_h=source['canvas']['height']
        for visual,vbox in source_visual:
            vy=(visual['bounds'][1]+visual['bounds'][3])/2/source_h
            for copy_unit,cbox in source_copy:
                cy=(copy_unit['bounds'][1]+copy_unit['bounds'][3])/2/source_h
                if abs(vy-cy)<.04:continue
                target_visual=(vbox[1]+vbox[3])/2
                target_copy=(cbox[1]+cbox[3])/2
                if (vy<cy and target_visual>target_copy) or (vy>cy and target_visual<target_copy):
                    terms['source_order']+=min(1.,abs(vy-cy)*2)
    if main_copy:
        copy_center=(min(b[0] for b in main_copy)+max(b[2] for b in main_copy))/2
        for (u,_),b in zip(records,boxes):
            if u['role']!='tag' or not u.get('source_upper'):
                continue
            source_center=(u['bounds'][0]+u['bounds'][2])/2/max(1.,source['canvas']['width'])
            current_center=(b[0]+b[2])/2/max(1.,width)
            # If a corner badge had to move inward, keep it on the copy's axis.
            moved_to_middle=(source_center<.30 or source_center>.70) and .30<=current_center<=.70
            if moved_to_middle:
                terms['badge_alignment']+=((current_center-copy_center/width)**2)
    if state['flow'] in ('story','portrait','square'):
        logos=[b for (u,_),b in zip(records,boxes) if u['semantic']=='logo']
        heroes=[b for (u,_),b in zip(records,boxes) if u['semantic']=='hero']
        for logo in logos:
            for hero in heroes:
                overlap=max(0.,min(logo[2],hero[2])-max(logo[0],hero[0]))
                if overlap<min(logo[2]-logo[0],hero[2]-hero[0])*.25:
                    continue
                gap=max(hero[1]-logo[3],logo[1]-hero[3],0.)
                preferred=min(width,height)*.045
                terms['brand_media_spacing']+=max(0.,(preferred-gap)/max(1.,preferred))**2
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
            tag_bands={}
            for index,(unit,_) in enumerate(records):
                if unit['role']=='tag':
                    key=('peer',unit['peer_ids']) if unit.get('peer_ids') else ('single',index)
                    tag_bands.setdefault(key,[]).append(index)
            for indexes in tag_bands.values():
                for row in _peer_row_groups(indexes,boxes):
                    row_left=min(boxes[index][0] for index in row)
                    terms['column_alignment']+=abs(row_left-axis)/width
            copy_top=min(box[1] for box in left_copy)
            for (unit,_),box in zip(records,boxes):
                if unit.get('semantic')!='logo' or unit['role']!='header':
                    continue
                overlap=max(0.,min(box[2],max(b[2] for b in left_copy))-
                            max(box[0],min(b[0] for b in left_copy)))
                if box[3]<=copy_top+height*.08 and overlap>0:
                    terms['column_alignment']+=abs(box[0]-axis)/width
            if state['flow']!='banner':
                actions=[b for (u,_),b in zip(records,boxes) if u['role']=='action']
                terms['column_alignment']+=sum(abs(b[0]-axis)/width for b in actions)
    scoring_roles={r:[u for u in units if u['role']==r]
                   for r in ('header','copy','action','tag','visual')}
    dense_short_banner=state['flow']=='banner' and _dense_short_banner(scoring_roles,width,height)
    if state['flow']=='banner':
        copy_boxes=[b for (u,_),b in zip(records,boxes) if u['role']=='copy']
        action_boxes=[b for (u,_),b in zip(records,boxes) if u['role']=='action']
        tag_count=sum(u['role']=='tag' for u,_ in records)
        if not dense_short_banner and len(copy_boxes)+len(action_boxes)+tag_count>=3:
            for action in action_boxes:
                aw=max(1.,action[2]-action[0])
                for copy in copy_boxes:
                    overlap=max(0.,min(action[2],copy[2])-max(action[0],copy[0]))
                    terms['banner_action_stack']+=overlap/min(aw,max(1.,copy[2]-copy[0]))
        columns=[]
        for roles in ({'logo'},{'headline','subheading','body','kicker'},
                      {'cta','badge'},{'hero','art'}):
            member_boxes=[box for (u,_),box in zip(records,boxes)
                          if u['semantic'] in roles or
                          roles=={'headline','subheading','body','kicker'} and u['role']=='copy' or
                          roles=={'cta','badge'} and u['role'] in ('tag','action')]
            if member_boxes:
                columns.append((min(box[0] for box in member_boxes),max(box[2] for box in member_boxes)))
        columns.sort()
        gaps=([max(0.,columns[0][0])]
              +[max(0.,b[0]-a[1]) for a,b in zip(columns,columns[1:])]
              +[max(0.,width-columns[-1][1])]) if columns else []
        if len(gaps)>=3:
            mean=sum(gaps)/len(gaps)
            normalizer=max(min(width,height)*.04,abs(mean),1.)
            terms['banner_gap_evenness']=sum(((gap-mean)/normalizer)**2 for gap in gaps)/len(gaps)
        heroes=[b for (u,_),b in zip(records,boxes) if u['semantic'] in ('hero','art')]
        actions=[b for (u,_),b in zip(records,boxes) if u['semantic']=='cta']
        preferred=min(width,height)*.09
        for hero in heroes:
            for action in actions:
                overlap_y=max(0.,min(hero[3],action[3])-max(hero[1],action[1]))
                if overlap_y<min(hero[3]-hero[1],action[3]-action[1])*.35:continue
                gap=max(hero[0]-action[2],action[0]-hero[2],0.)
                terms['primary_clearance']+=max(0.,(preferred-gap)/max(1.,preferred))**2
    def role_bounds(roles):
        return _bounds([e for (u,items) in records if u['semantic'] in roles for e in items])
    flow_roles=[{'logo'},{'headline','subheading','body','kicker'},{'cta'},{'hero','art'}]
    if state['flow']=='story':
        flow_roles=[{'logo'},{'hero','art'},{'headline','subheading','body','kicker'},{'cta'}]
    flow=[]
    for roles in flow_roles:
        if any(u['semantic'] in roles for u,_ in records):
            flow.append(role_bounds(roles))
    if state['flow'] in ('banner','story'):
        axis,extent=(0,width) if state['flow']=='banner' else (1,height)
        for a,b in zip(flow,flow[1:]):
            ac=(a[axis]+a[axis+2])/2;bc=(b[axis]+b[axis+2])/2
            terms['hierarchy']+=max(0,(ac-bc)/extent)*2
    main=[b for (u,_),b in zip(records,boxes) if u['role'] not in ('decoration','footer')]
    if main:
        left,top=min(b[0] for b in main),min(b[1] for b in main)
        right,bottom=max(b[2] for b in main),max(b[3] for b in main)
        terms['canvas_use']=(max(0,.80-(right-left)/width)**2+max(0,.76-(bottom-top)/height)**2)
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
                     canvas_use=35.,source_fidelity=18.,rag=5.,affinity=18.,whitespace=5.,background_busyness=7.,axial=160.,badge_flow=70.,footer_anchor=100.,hero_prominence=210.,cta_prominence=85.,brand_visibility=45.,column_alignment=120.,avoidable_wrap=25.,banner_action_stack=180.,banner_gap_evenness=12.,banner_centering=80.,stack_balance=300.,badge_alignment=80.,brand_media_spacing=55.,footer_visibility=240.,primary_clearance=65.,callout_migration=8.,source_order=65.,decoration_composition=48.)


def _energy(elements,units,source,width,height,desired):
    return sum(value*_ENERGY_WEIGHTS[name] for name,value in
               _energy_terms(elements,units,source,width,height,desired).items())


def _constraint_violation(elements, units, width, height):
    """Continuous repair cost, including required gaps that IoU alone misses."""
    element_records=_records(elements,units)
    records=[(u,_bounds(items)) for u,items in element_records]
    value=0.;gap=min(width,height)*.008
    for i,((u,items),(_,b)) in enumerate(zip(element_records,records)):
        for e in items:
            if e.get('type')=='image':
                continue
            value+=(max(0,-e['x'])+max(0,e['x']+e['width']-width))/width
            value+=(max(0,-e['y'])+max(0,e['y']+e['height']-height))/height
        margin=-min(width,height)*.20 if _can_bleed(u) else 0 if u['role']=='decoration' else _unit_margin(b,width,height,u)
        value+=(max(0,margin-b[0])+max(0,b[2]-width+margin))/width
        value+=(max(0,margin-b[1])+max(0,b[3]-height+margin))/height
        for v,c in records[:i]:
            if not _layered(u,v):
                value+=max(0,min((b[2]+gap-c[0])/width,(c[2]+gap-b[0])/width,
                                 (b[3]+gap-c[1])/height,(c[3]+gap-b[1])/height))
    return value


def _align_peer_rows(elements, units):
    """Keep same-row button peers on one centerline through geometry search."""
    bands={}
    by_id={e['id']:e for e in elements}
    for unit in units:
        if unit['role']=='tag' and unit.get('peer_ids'):
            items=[by_id[e['id']] for e in unit['items'] if e['id'] in by_id]
            if items:
                bands.setdefault(unit['peer_ids'],[]).append(items)
    for members in bands.values():
        if len(members)<2:continue
        boxes=[_bounds(items) for items in members]
        for row in _peer_row_groups(list(range(len(members))),boxes):
            if len(row)<2:continue
            row_members=[members[i] for i in row];row_boxes=[boxes[i] for i in row]
            center=sum((box[1]+box[3])/2 for box in row_boxes)/len(row_boxes)
            target_height=max(max(box[3]-box[1] for box in row_boxes),
                              max((e['height'] for items in row_members for e in items if e['type']=='text'),default=0.))
            for items,box in zip(row_members,row_boxes):
                _shift(items,0.,center-(box[1]+box[3])/2)
                backgrounds=[e for e in items if e['type']=='shape' and e.get('shape_type')=='rectangle']
                labels=[e for e in items if e['type']=='text']
                if len(backgrounds)==1 and labels:
                    back=backgrounds[0]
                    back['height']=target_height
                    back['y']=center-target_height/2
                    if 'corner_radius' in back:
                        back['corner_radius']=min(target_height/2,max(back['corner_radius'],target_height*.35))
                    for label in labels:
                        label['y']=center-label['height']/2
    return elements


def _polish(elements, units, source, width, height, desired, seed=0, iterations=36):
    """Bounded seeded annealing over atomic units, not per-element jitter.

    Wrapping and disjunctive separation are nonsmooth; a gradient-only optimizer
    would not enforce them. Keep the best feasible result, even if an annealing
    step temporarily accepts a slightly worse arrangement. Infeasible starts
    first reduce hard-constraint violations, including missing separation gaps.
    """
    rng=Random(seed)
    state=_aspect_state(source,width,height)
    current=_align_peer_rows(deepcopy(elements),units)
    current_energy=_energy(current,units,source,width,height,desired)
    current_feasible=_feasible(current,units,width,height)
    violation=_constraint_violation(current,units,width,height)
    best,best_energy=deepcopy(current),current_energy
    best_feasible=current_feasible
    centered_copy=_centered_copy_layout(current,units,width)
    # Decoration size and location have already been selected against the
    # occupied layout in _decorations. Annealing these units afterward used to
    # collapse or cluster them again, undoing that empty-space placement.
    # Keep them fixed while foreground units resolve any remaining conflicts.
    moving=[u for u in units if u['role']!='decoration'
            and not (u['role']=='footer' and not u.get('inflow_footer_asset'))
            and not (state['flow'] in ('banner','landscape') and
                     u['role']=='tag' and u.get('peer_ids'))]
    if not moving:
        return current,dict(before=current_energy,after=current_energy,accepted=0,iterations=0)
    accepted=0
    for step in range(iterations):
        u=moving[step%len(moving)] if step<len(moving)*2 else rng.choice(moving)
        ids={e['id'] for e in u['items']}
        block=[e for e in current if e['id'] in ids]
        bounds=_bounds(block); x,y,r,b=bounds
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
                box=_bounds(other)
                targets.extend([box[axis],box[axis+2]-size,(box[axis]+box[axis+2]-size)/2])
            targets=[v for v in targets if abs(v-old)>extent*.001]
            if axis==0 and u.get('footer_anchor') in ('left','right'):
                margin=min(width,height)*.05
                target=margin if u['footer_anchor']=='left' else width-margin-size
            elif axis==0 and _axis_active(u,source,width,height,current,units) and u['semantic'] in ('logo','hero'):
                target=(width-size)/2
            elif axis==0 and centered_copy and u['semantic'] in ('logo','hero','headline','subheading','body','cta'):
                target=(width-size)/2
            elif (axis==0 and not centered_copy and state['flow'] in ('banner','landscape')
                  and u['role']=='header' and u.get('semantic')=='logo'):
                copy_records=[(v,_bounds(other)) for v,other in current_records if v['role']=='copy'
                              and any(e.get('text_align','left')=='left' for e in other)]
                logo_box=_bounds(block)
                copy_left=min((box[0] for _,box in copy_records),default=None)
                copy_top=min((box[1] for _,box in copy_records),default=None)
                if (copy_left is not None and copy_top is not None and logo_box[3]<=copy_top+height*.08
                        and min(logo_box[2],max(box[2] for _,box in copy_records))>
                            max(logo_box[0],min(box[0] for _,box in copy_records))):
                    target=copy_left
                else:
                    target=old
            elif axis==0 and state['flow'] in ('banner','landscape') and u['role'] in ('copy','tag'):
                left_aligned_copy=any(v['role']=='copy' and any(e.get('text_align','left')=='left' for e in other)
                                      for v,other in current_records)
                is_left_column=any(e.get('text_align','left')=='left' for e in block) or (
                    u['role']=='tag' and left_aligned_copy and not centered_copy)
                anchors=[_bounds(other)[0] for v,other in current_records if v is not u and v['role'] in ('copy','tag')
                         and (any(e.get('text_align','left')=='left' for e in other) or
                              v['role']=='tag' and left_aligned_copy and not centered_copy)]
                target=min(anchors,key=lambda value:abs(value-old)) if is_left_column and anchors else old
            elif axis==1 and state['flow']=='banner' and not _banner_vertical_neighbor(u,bounds,current_records):
                target=_banner_centerline(current_records,width,height)-size/2
            else:
                target=min(targets,key=lambda v:abs(v-old)) if targets and rng.random()<.55 else old+rng.uniform(-1,1)*extent*(.04*cooling+.002)
            proposal=_move(proposal,target if axis==0 else x,target if axis==1 else y)
        elif motion==2 and len(block)==1 and block[0]['type']=='text':
            e=proposal[0]
            if width/height>=1.35 and '\n' not in e.get('content','') and rng.random()<.5:
                room=_text_right_limit(u,bounds,_records(current,units),width,height)-x
                e['width']=max(1,min(room,_text(e,1e9)[1]+e['font_size']*.02))
            else:
                e['width']=max(1,e['width']*rng.choice((.90,1.10)))
            e['font_size']=_fit_text(e,e['width'],height*.7,max(1,e['font_size']*rng.choice((1.,1.04))))
            e['height']=_text(e)[0]
            _tighten_text(e)
            new_box=_bounds(proposal);new_w=new_box[2]-new_box[0]
            align=e.get('text_align','left')
            left=(x+r-new_w)/2 if align=='center' else r-new_w if align=='right' else x
            proposal=_move(proposal,left,y)
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
            proposal=_move(proposal,left,(y+b-(b-y)*factor)/2)
        changed=[e for e in current if e['id'] not in ids]+proposal
        _sync_backing_panels(changed,units,width,height)
        _align_peer_rows(changed,units)
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
    state=_aspect_state(source,width,height)
    ratio_shift=max(abs(state['delta']),1/max(1e-9,abs(state['delta'])))
    has_reflow=any(c['name']!='proportional' for c in candidates)
    shortlist=[];families=set()
    for candidate in candidates:
        name=candidate['name']
        if ratio_shift>=1.6 and has_reflow and name=='proportional':
            continue
        family=re.sub(r'-\d+(?:\.\d+)?$','',name)
        if family in families:
            continue
        families.add(family);shortlist.append(candidate)
        if len(shortlist)>=8:
            break
    relationship_context=_visual_relationship_context(units,source,width,height)
    with TemporaryDirectory(prefix='chameleon-rank-') as directory:
        reference_image=None
        try:
            reference_path=Path(directory)/'source.png'
            render_design_to_png(source,Path(__file__).resolve().parents[1]/'brand_kit'/'assets',reference_path,max_size=640)
            with Image.open(reference_path) as image:
                reference_image=image.convert('RGB')
        except Exception:
            # Identity continuity is a soft cue; candidate diagnostics remain
            # useful if the source render fails for an unrelated renderer issue.
            pass
        for index,candidate in enumerate(shortlist):
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
                cost=cost/max(1,checked)+(8.0 if invisible else 0.0)+(4.0 if weak else 0.0)
                entries.append(dict(candidate=candidate,cost=cost,checked=checked,low_contrast=weak,busy_background=busy,invisible_text=invisible,warnings=warnings,image=full.copy()))
            except Exception as exc:
                # Diagnostics are optional; isolate third-party renderer faults.
                # Do not catch BaseException (interrupts and shutdown still work).
                entries.append(dict(candidate=candidate,cost=0.,error=str(exc)))
    if not entries or all('error' in item for item in entries):
        return candidates[0],dict(used=False,reason='Diagnostic rendering failed',errors=[e.get('error') for e in entries])
    valid=[e for e in entries if 'error' not in e]
    heuristic_cost=lambda e:e['candidate']['energy']+e['cost']
    perceptual=_siglip_rerank(valid,heuristic_cost,reference_image,relationship_context)
    index=perceptual.get('selected_index')
    selected=valid[index] if index is not None else min(valid,key=heuristic_cost)
    report=dict(used=True,
                method='rendered contrast/busyness/visible-text checks'+(' + SigLIP perceptual re-rank' if perceptual.get('used') else ''),
                perceptual=perceptual,
                candidates=[{k:v for k,v in e.items() if k not in ('candidate','image')}|{'name':e['candidate']['name']} for e in entries])
    return selected['candidate'],report


def _visual_relationship_context(units,source,width,height):
    """Summarize inferred semantic relationships for vision prompts."""
    flow=_aspect_state(source,width,height)['flow']
    bucket='wide' if flow=='landscape' else 'square' if flow=='preserve' else flow
    if bucket not in ('banner','wide','square','portrait','story'):
        ratio=width/height
        bucket='wide' if ratio>1.15 else 'portrait' if ratio<.87 else 'square'
    roles=set()
    for unit in units:
        semantic=unit.get('semantic')
        role={'kicker':'headline','badge':'cta','visual':'art'}.get(semantic,semantic)
        if role in ('logo','headline','subheading','body','cta','hero','art'):
            roles.add(role)
    density='dense' if len(units)/max(1,(width*height)/1_000_000)>=14 else 'open'
    sw,sh=source['canvas']['width'],source['canvas']['height']
    quadrants=[]
    for unit in units:
        if not unit.get('corner_callout'):
            continue
        x,y,r,b=unit['bounds'];cx=(x+r)/2;cy=(y+b)/2
        quadrants.append(('top' if cy<sh/2 else 'bottom')+' '+('left' if cx<sw/2 else 'right'))
    return dict(bucket=bucket,roles=sorted(roles),density=density,
                grouped_units=any(len(unit['items'])>1 for unit in units),
                callout_quadrants=sorted(set(quadrants)))


def _siglip_rerank(valid, heuristic_cost, reference_image=None, layout_context=None):
    """Optional learned SigLIP tie-break among already-valid candidate renders.

    Returns a diagnostic dict. When used, ``selected_index`` points at the chosen
    entry in ``valid``; otherwise it is ``None`` and the caller keeps the pure
    heuristic winner. SigLIP may only pick among candidates whose heuristic cost
    (layout energy + legibility/busyness cost) is within a near-tie band of the
    best -- so it reorders geometry-preference ties but can never rescue a
    materially worse (e.g. illegible or off-balance) layout. Degrades to a no-op
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
    tolerance=max(8.0,abs(best_cost)*.15)
    # The advertised band bounds how far the scorer can look; the effective
    # near-tie band is also relative to the winner so small absolute deltas do
    # not let a clearly costlier layout win on a low-cost canvas.
    effective_tolerance=min(tolerance,max(.20,abs(best_cost)*.15))
    band=[i for i,c in enumerate(costs) if c<=best_cost+effective_tolerance]
    # Let visual quality break close ties while retaining a small preference
    # for the lower-cost geometry inside the explicit hard band.
    chosen=max(band,key=lambda i:scores[i]-.10*(costs[i]-best_cost)/max(effective_tolerance,1e-9))
    return dict(used=True,note=note,selected_index=chosen,
                scores=[round(s,4) for s in scores],
                heuristic_costs=[round(c,4) for c in costs],
                tie_band=band,tolerance=round(tolerance,4),
                effective_tolerance=round(effective_tolerance,4),
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
        unit['target_aspect_ratio']=width/height
        unit['target_short_side']=min(width,height)
        if width/height<.68 and unit['role']=='footer' and unit.get('footer_anchor') in ('left','right'):
            # A corner watermark becomes a centered footer companion on tall
            # layouts; preserve its source anchor for migration decisions.
            unit['source_footer_anchor']=unit['footer_anchor']
            unit['footer_anchor']='center'
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
    if state['flow'] in ('banner','landscape'):
        for share in (.14,.22,.30):
            seeds.append(dict(name=f'inline-banner-{share}',elements=_banner_seed(units,width,height,desired*1.15,share,source_size=(sw,sh))))
        seeds.append(dict(name='left-hero-banner',elements=_banner_seed(units,width,height,desired*1.15,.24,True,source_size=(sw,sh))))
    if state['flow'] in ('story','portrait','square'):
        for hero_first in (True,False):
            shares=(.30,.38,.46,.54,.62) if .75<=width/height<=1.3 else (.34,.42,.48,.52)
            for share in shares:
                seeds.append(dict(name=f'semantic-stack-{hero_first}-{share}',elements=_stack_seed(units,width,height,desired*1.15,share,hero_first,(sw,sh))))
        has_movable_corner=any(u.get('corner_callout') for u in units)
        has_anchored_footer_asset=any(u['role']=='footer' and u.get('source_footer_anchor',u.get('footer_anchor')) in ('left','right')
                                      and any(e['type']=='image' for e in u['items']) for u in units)
        if width/height<.68 and (has_movable_corner or has_anchored_footer_asset):
            for hero_first in (True,False):
                seeds.append(dict(name=f'semantic-stack-inflow-callout-{hero_first}',
                                  elements=_stack_seed(units,width,height,desired*1.15,.42,
                                                       hero_first,(sw,sh),move_corner_callouts=True)))
    if context['affinities']:
        seeds.append(dict(name='affiliated-features',elements=_affiliated_seed(units,context,width,height,sw,sh,desired)))
    if context['background_map']:
        base=seeds[0]['elements']
        for cell in sorted(range(9),key=lambda i:context['background_map'][i])[:3]:
            seeds.append(dict(name=f'quiet-region-{cell}',elements=_quiet_seed(base,units,width,height,cell,desired*1.15)))
    seeds.append(dict(name='safe-grid',elements=_safe_grid(units,width,height,desired)))
    role_map={r:[u for u in units if u['role']==r]
              for r in ('header','copy','action','tag','visual','footer','decoration')}
    if _dense_short_banner(role_map,width,height):
        # Keep the specialized compact leaderboard candidate, but retain the
        # other generated families as fallbacks if this rigid arrangement is
        # infeasible or scores poorly for a particular content mix.
        dense_seed=dict(name='dense-short-banner',
                        elements=_dense_leaderboard_seed(role_map,width,height,desired*1.15,
                                                         min(width,height)*.035,(sw,sh)))

        def geometry_signature(seed):
            return tuple((e['id'],round(e['x'],2),round(e['y'],2),round(e['width'],2),
                          round(e['height'],2),round(e.get('rotation',0),2),
                          round(e.get('font_size',0),2),e.get('text_align'))
                         for e in seed['elements'])

        # Dense layouts are also returned by several generic banner builders.
        # Drop exact duplicate geometries so they cannot crowd distinct fallbacks
        # out of the small candidate set polished and visually re-ranked below.
        seen=set()
        distinct=[]
        dense_signature=geometry_signature(dense_seed)
        for seed in seeds:
            signature=geometry_signature(seed)
            if signature==dense_signature or signature in seen:
                continue
            seen.add(signature)
            distinct.append(seed)
        seeds=distinct+[dense_seed]
    feasible=[]
    rejected=[]
    for seed in seeds:
        _sync_backing_panels(seed['elements'],units,width,height)
        projected=None
        for topology in ('nearest','centers','alternate'):
            projected=_constrain(seed['elements'],units,width,height,topology)
            if projected is not None:
                if seed['name']=='dense-short-banner':
                    _equalize_control_geometry(projected,units)
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
        if candidate['name']=='dense-short-banner' and _feasible(candidate['elements'],units,width,height):
            # This role-based seed already has a measured one-row control band
            # and a shared media column. Independent local moves can undo
            # those relationships on very short canvases.
            elements=candidate['elements'];optimization=dict(before=candidate['energy'],after=candidate['energy'],accepted=0,iterations=0)
            _repair_copy_hierarchy(elements,units,width,height)
        elif _feasible(candidate['elements'],units,width,height):
            elements,optimization=_polish(candidate['elements'],units,source,width,height,desired,seed=90210+i,iterations=36)
            _repair_copy_hierarchy(elements,units,width,height)
        else:
            elements=candidate['elements'];optimization=dict(before=candidate['energy'],after=candidate['energy'],accepted=0,iterations=0)
        if candidate['name']!='dense-short-banner':
            _repair_footer_bottom_anchor(elements,units,width,height)
            _repair_primary_action_shape(elements,units,width,height)
            _repair_vertical_hero_spacing(elements,units,width,height)
            _repair_vertical_brand_clearance(elements,units,width,height)
        # Balance the logo by its visible artwork bounds for every landscape
        # family, including the compact dense-leaderboard seed. Transparent
        # PNG padding must not make a nominally left-aligned logo look flush
        # with one side of its brand column.
        _repair_banner_logo_column_balance(elements,units,width,height)
        _repair_banner_cta_copy_proximity(elements,units,width,height)
        _repair_sparse_vertical_stack(elements,units,width,height)
        _align_peer_rows(elements,units)
        optimized.append(dict(name=candidate['name'],elements=elements,energy=_energy(elements,units,source,width,height,desired),optimization=optimization))
    optimized.sort(key=lambda c:(c['energy'],c['name']))
    geometry_time=perf_counter()-start
    winner,visual=_visual_rerank(optimized,backgrounds,units,source,width,height,desired)
    output=_assemble(winner['elements'],backgrounds,source,width,height)
    report=dict(perceptual=context,aspect=state,roles=[dict(ids=[e['id'] for e in u['items']],role=u['semantic'],confidence=u['confidence']) for u in units],
                generated=len(seeds),feasible=len(feasible),rejected_topologies=len(rejected),recovered=recovered,starts=len(optimized),winner=winner['name'],
                energy_terms=_energy_terms(winner['elements'],units,source,width,height,desired),
                candidates=[{k:v for k,v in c.items() if k!='elements'} for c in optimized],
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
