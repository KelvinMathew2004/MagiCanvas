"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.

Implementation: offline v0.9 reflow adapted from Project Chameleon's
analysis -> candidate generation -> ranking pipeline. No editor/model runtime.
Explicit groups and source-evidenced overlapping layer clusters travel
together. Simple text/backplate groups can widen as a unit; artwork groups
receive one proportional transform. Inferred roles stay internal and do not
become output schema fields.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import ceil, cos, exp, isfinite, log, radians, sin, sqrt
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter
from pathlib import Path
import os
import re
from typing import Any

from renderer.converter.assets import visible_content_bounds
from engine.request_cache import asset_lookup_state, asset_lookup_request


_ASSETS_DIR=Path(__file__).resolve().parents[1]/'brand_kit'/'assets'
_DATASET_SPLIT_DIRS=tuple(Path(__file__).resolve().parents[1]/'data'/name for name in
                          ('crello_train','crello_validation','crello_test','crello_sample'))
_CRELLO_FONTS_PICKLE=Path(__file__).resolve().parents[1]/'data'/'crello_fonts.pickle'


def _rgb_pixels(image):
    """Read an RGB image as Python integer triples without Pillow's deprecated API."""
    data=image.tobytes()
    return list(zip(data[0::3],data[1::3],data[2::3]))


def _asset_root(element):
    """Find the root against which an element's asset_id is stored."""
    state=asset_lookup_state()
    if state is None:return _uncached_asset_root(element)
    key=(str(element.get('asset_id','')),os.environ.get('MAGICANVAS_DATASET_ASSETS_DIR'),id(_ASSETS_DIR),id(_DATASET_SPLIT_DIRS))
    state['calls']+=1
    if key in state['roots']:
        state['hits']+=1
        return state['roots'][key]
    result=_uncached_asset_root(element);state['roots'][key]=result
    return result


def _uncached_asset_root(element):
    """Source asset files are stable during one adaptation, then rediscovered."""
    asset_id=str(element.get('asset_id',''))
    if not asset_id:
        return _ASSETS_DIR
    if (_ASSETS_DIR/asset_id).is_file():
        return _ASSETS_DIR
    configured=os.environ.get('MAGICANVAS_DATASET_ASSETS_DIR')
    roots=([Path(configured)] if configured else [])+list(_DATASET_SPLIT_DIRS)
    for root_dir in roots:
        if (root_dir/asset_id).is_file():
            return root_dir
    return _ASSETS_DIR


@lru_cache(maxsize=1)
def _crello_font_manager():
    """Load the pinned Crello font resource for local dataset previews."""
    if not _CRELLO_FONTS_PICKLE.is_file():
        return None
    try:
        from cr_renderer.fonts import FontManager, normalize_family
        manager = FontManager(str(_CRELLO_FONTS_PICKLE))
        for alias, stored in (("Brusher", "Brusher Free Font"),
                              ("Exo 2", "Exo\\ 2")):
            if stored in manager._fonts:
                manager._fonts[normalize_family(alias)] = manager._fonts[stored]
        return manager
    except (ImportError, OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def _layout_font_manager():
    """Use the renderer's local lookup and merge brand faces over Crello faces."""
    try:
        from cr_renderer.fonts import FontManager
        from renderer.converter.brand_fonts import build_font_dict
        dataset=_crello_font_manager()
        fonts=dict(dataset._fonts) if dataset is not None else {}
        fonts.update(build_font_dict(str(Path(__file__).resolve().parents[1]/'brand_kit'/'fonts')))
        if not fonts:return None
        manager=FontManager.__new__(FontManager);manager._fonts=fonts
        return manager
    except (ImportError,OSError,ValueError):
        return None


@lru_cache(maxsize=16384)
def _skia_advances(content,family,bold,italic,size):
    manager=_layout_font_manager()
    if manager is None:return None
    try:
        from renderer.converter.wrap import char_widths
        style='bolditalic' if bold and italic else 'italic' if italic else 'bold' if bold else 'regular'
        return tuple(char_widths(manager,family,'bold' if bold else 'regular',style,size,content))
    except (ImportError,OSError,ValueError):
        return None


@lru_cache(maxsize=8192)
def _render_text_ink_cached(payload):
    """Rasterize a text probe with the actual converter and Skia text renderer.

    Pixel bounds include mixed styles, fractional tracking, integer renderer
    font sizes and italic overhang. No image assets or target labels are used.
    """
    import json
    from math import ceil
    e=json.loads(payload);manager=_layout_font_manager()
    if manager is None:return None
    try:
        import numpy as np
        import skia
        from renderer.converter.convert import convert_chameleon_to_crello
        from cr_renderer.schema import TextElement
        from cr_renderer.text_utils import generate_map,render_text
        # Source/exporter run offsets are end exclusive, as in the dataset
        # rendering adapter. The converter's internal style maps are inclusive.
        for run in e.get('style_runs') or []:run['end']=int(run['end'])-1
        example,_=convert_chameleon_to_crello({'canvas':{'width':1,'height':1},'elements':[e]},manager)
        prop=TextElement(uuid='',type='textElement',width=float(e['width']),height=float(e['height']),
            text=example['text'][0],fontSize=example['font_size'][0],font=example['font'][0],
            lineHeight=example['line_height'][0],textAlign=example['text_align'][0],
            capitalize=example['capitalize'][0],letterSpacing=example['letter_spacing'][0],
            boldMap=generate_map(example['font_bold'][0]),italicMap=generate_map(example['font_italic'][0]),
            colorMap=generate_map(['#ffffff']*len(example['text'][0])),lineMap=generate_map(example['text_line'][0]))
        pad=max(4,ceil(prop.fontSize*2));lines=max(1,example['text'][0].count('\n')+1)
        natural=_text_info(e,1e9)[1]
        w=ceil(max(prop.width,natural)+pad*2);h=ceil(lines*prop.fontSize*max(1,prop.lineHeight)+pad*2+prop.fontSize*2)
        if w*h>16_000_000 or max(w,h)>16384:return None
        surface=skia.Surface(max(1,w),max(1,h));canvas=surface.getCanvas();canvas.clear(0)
        canvas.translate(pad,pad);render_text(canvas,manager,prop)
        alpha=surface.makeImageSnapshot().toarray()[:,:,3]
        yy,xx=np.nonzero(alpha)
        if not len(xx):return (0.,0.,0.,0.)
        return (float(xx.min()-pad),float(yy.min()-pad),float(xx.max()+1-pad),float(yy.max()+1-pad))
    except (ImportError,OSError,ValueError,TypeError):
        return None


def _render_text_ink(e):
    import json
    probe={k:v for k,v in e.items() if k in ('type','content','font_family','font_size','font_weight',
        'font_style','italic','line_height','letter_spacing','text_align','capitalize','style_runs','width','height')}
    probe.update(id='font-probe',x=0,y=0,rotation=0,opacity=1,color='#ffffff')
    return _render_text_ink_cached(json.dumps(probe,sort_keys=True))


def _set_font_size(element,size):
    old=max(1e-6,float(element.get('font_size',1)))
    element['font_size']=max(1.,float(size))
    element['letter_spacing']=float(element.get('letter_spacing',0))*element['font_size']/old


def _source_ink_overhang(element,authored):
    """Allow authored font bearings, which the renderer paints without clipping."""
    ink=_render_text_ink(authored) if authored else None
    if ink is None:return (0.,0.)
    scale=float(element.get('font_size',1))/max(1.,float(authored.get('font_size',1)))
    return (float(ceil(max(0.,-ink[0])*scale)),float(ceil(max(0.,-ink[1])*scale)))


def _raw_bounds(items):
    if not items:return (0.,0.,0.,0.)
    boxes=[_rotated_bounds(e) for e in items]
    return (min(min(e['x'],b[0]) for e,b in zip(items,boxes)),
            min(min(e['y'],b[1]) for e,b in zip(items,boxes)),
            max(max(e['x']+e['width'],b[2]) for e,b in zip(items,boxes)),
            max(max(e['y']+e['height'],b[3]) for e,b in zip(items,boxes)))


def _fit_raw_block(items,width,height):
    """Fit transparent padding and rotated frames as well as painted pixels."""
    result=deepcopy(items);l,t,r,b=_raw_bounds(result)
    factor=min(1.,width/max(1e-6,r-l),height/max(1e-6,b-t))
    for e in result:
        e.update(x=(e['x']-l)*factor,y=(e['y']-t)*factor,
                 width=e['width']*factor,height=e['height']*factor)
        if e.get('type')=='text':_set_font_size(e,e['font_size']*factor)
    return result


def _center_control_labels(elements,source,width,height):
    from engine.layout_contracts import center_control_labels
    center_control_labels(elements,source,width,height)


def _unintended_control_overlaps(elements,source):
    """Source-disjoint controls cannot collide with another control or copy.

    Large photos, translucent overlays and artwork remain governed by painted
    hero masks, rather than treating transparent image padding as a collision.
    """
    src={e['id']:e for e in source['elements']};sw,sh=source['canvas']['width'],source['canvas']['height']
    result=[]
    from engine.layout_contracts import controls
    from engine.nested_structure import unpainted_ids
    unpainted=unpainted_ids(source)
    inferred=dict(controls(source))
    def control(e):
        return e['id'] in inferred or (e.get('type')=='shape' and e.get('group_id') is not None
                and e['width']*e['height']<.16*sw*sh and float(e.get('opacity',1.))>=.98)
    for i,a in enumerate(elements):
        sa=src.get(a['id'])
        if sa is None or a['id'] in unpainted:continue
        for b in elements[i+1:]:
            sb=src.get(b['id'])
            if sb is None or b['id'] in unpainted or not (control(sa) or control(sb)):continue
            if sa.get('group_id') and sa.get('group_id')==sb.get('group_id'):continue
            if float(sa.get('opacity',1.))<.98 or float(sb.get('opacity',1.))<.98:continue
            if any(e.get('type')!='text' and e['width']*e['height']>.8*sw*sh for e in (sa,sb)):continue
            if (a.get('type')!='text' and b.get('type')!='text'
                    and _overlap(_rotated_bounds(sa),_rotated_bounds(sb))>1.):continue
            if inferred.get(a['id'])==b['id'] or inferred.get(b['id'])==a['id']:continue
            ba=_text_visible_bounds(a) if a.get('type')=='text' else _bounds([a])
            bb=_text_visible_bounds(b) if b.get('type')=='text' else _bounds([b])
            overlap=_overlap(ba,bb)
            # Nested cards may contain another control's label. This source-
            # evidenced backing is intentional even across schema group IDs.
            # A newly introduced intersection receives no exemption.
            old_a=_text_visible_bounds(sa) if sa.get('type')=='text' else _bounds([sa])
            old_b=_text_visible_bounds(sb) if sb.get('type')=='text' else _bounds([sb])
            small=min(max(1.,(old_a[2]-old_a[0])*(old_a[3]-old_a[1])),max(1.,(old_b[2]-old_b[0])*(old_b[3]-old_b[1])))
            if (a.get('type')!=b.get('type') and 'text' in (a.get('type'),b.get('type'))
                    and _overlap(old_a,old_b)/small>=.92):continue
            if overlap>1.:result.append({'a':a['id'],'b':b['id'],'area':overlap})
    return result


def _restore_attached_ornaments(elements,source,width,height):
    """Carry compact source attachments with their closest painted text anchor.

    Translation alone cannot fix an ornament already detached by a seed. This
    pass restores its authored offset, scale and rotation before contrast moves.
    Large decorative artwork and full-canvas layers are deliberately excluded.
    """
    src=source['elements'];out={e['id']:e for e in elements};sw,sh=source['canvas']['width'],source['canvas']['height']
    texts=[e for e in src if e.get('type')=='text' and e['id'] in out and e.get('content','').strip()]
    for ornament in src:
        if ornament.get('type')=='text' or ornament['id'] not in out:continue
        if float(ornament.get('opacity',1.))<.98:continue
        area=ornament['width']*ornament['height']/(sw*sh)
        if ornament['type']=='image' and not ornament.get('group_id') and area<.12 and (ornament['x']<sw*.18 or ornament['x']+ornament['width']>sw*.82) and (ornament['y']<sh*.25 or ornament['y']+ornament['height']>sh*.8):continue
        thin=min(ornament['width']/sw,ornament['height']/sh)<.02
        if area>=.025 or not (thin or max(ornament['width']/sw,ornament['height']/sh)<.15):continue
        ob=_rotated_bounds(ornament);anchors=[]
        for t in texts:
            if t.get('group_id') and ornament.get('group_id') and t['group_id']!=ornament['group_id']:continue
            tb=_text_visible_bounds(t)
            dx=max(0.,ob[0]-tb[2],tb[0]-ob[2]);dy=max(0.,ob[1]-tb[3],tb[1]-ob[3])
            cx=abs((ob[0]+ob[2]-tb[0]-tb[2])/2)/sw
            cy=abs((ob[1]+ob[3]-tb[1]-tb[3])/2)/sh
            aligned=(cx<.065 and dy<.12*sh) or (cy<.03 and dx<.06*sw)
            if not aligned:continue
            # A backplate is sized around its label, not a free ornament.
            coverage=_overlap(ob,tb)/max(1.,(tb[2]-tb[0])*(tb[3]-tb[1]))
            if coverage>.8 and ornament.get('z_index',0)<t.get('z_index',0):continue
            anchors.append((dx/sw+dy/sh+.1*(cx+cy),t))
        if not anchors:continue
        anchor=min(anchors,key=lambda pair:pair[0])[1];target=out[anchor['id']]
        factor=target['font_size']/max(1.,anchor['font_size'])
        if not .01<=factor<=10:continue
        before=_text_visible_bounds(anchor);after=_text_visible_bounds(target)
        cx=(after[0]+after[2])/2+((ob[0]+ob[2]-before[0]-before[2])/2)*factor
        cy=(after[1]+after[3])/2+((ob[1]+ob[3]-before[1]-before[3])/2)*factor
        w,h=ornament['width']*factor,ornament['height']*factor
        moved=dict(out[ornament['id']],x=cx-w/2,y=cy-h/2,width=w,height=h,rotation=ornament.get('rotation',0.))
        l,t,r,b=_raw_bounds([moved])
        if l>=0 and t>=0 and r<=width and b<=height:out[ornament['id']].update(moved)


def _restore_callout_order(elements,source,width,height):
    from engine.qwen_refine import _source_callout_groups,project_container_tree,build_container_tree
    groups=_source_callout_groups(source)
    if len(groups)<2:return elements
    result=deepcopy(elements);out={e['id']:e for e in result};src={e['id']:e for e in source['elements']}
    sw,sh=source['canvas']['width'],source['canvas']['height']
    for ids in groups:
        if not ids<=out.keys():continue
        local={'canvas':source['canvas'],'elements':[src[eid] for eid in sorted(ids)]}
        tree=build_container_tree(local)
        tree['children']=[{'id':'source-callout','element_ids':sorted(ids),'mode':'rigid'}]
        projected,_=project_container_tree([out[eid] for eid in sorted(ids)],tree,width,height)
        for e in projected:out[e['id']].update(e)
    for side in ('left','right'):
        peers=[g for g in groups if g<=out.keys() and
               ((_raw_bounds([src[eid] for eid in g])[0]+_raw_bounds([src[eid] for eid in g])[2])/2<sw/2)==(side=='left')]
        ordered=sorted(peers,key=lambda ids:_raw_bounds([src[eid] for eid in ids])[1])
        centers=sorted((_raw_bounds([out[eid] for eid in ids])[1]+_raw_bounds([out[eid] for eid in ids])[3])/2 for ids in ordered)
        for ids,cy in zip(ordered,centers):
            box=_raw_bounds([out[eid] for eid in ids]);dy=cy-(box[1]+box[3])/2
            dy=min(height-box[3],max(-box[1],dy))
            for eid in ids:out[eid]['y']+=dy
    return result


def _balance_rule_peers(elements,source,width,height):
    """Balance source-symmetric flank ornaments against painted text bounds."""
    out={e['id']:e for e in elements};src=source['elements'];sw,sh=source['canvas']['width'],source['canvas']['height']
    rules=[e for e in src if e.get('type')!='text' and e['id'] in out
           and e['height']<=.03*sh and e['width']<=.2*sw and float(e.get('opacity',1.))>=.98]
    for text in src:
        if text.get('type')!='text' or text['id'] not in out or '\n' in text.get('content',''):continue
        sb=_text_visible_bounds(text);scx=(sb[0]+sb[2])/2;scy=(sb[1]+sb[3])/2
        left=[r for r in rules if r['x']+r['width']<=sb[0]+4 and abs(r['y']+r['height']/2-scy)<=.035*sh]
        right=[r for r in rules if r['x']>=sb[2]-4 and abs(r['y']+r['height']/2-scy)<=.035*sh]
        if len(left)!=1 or len(right)!=1:continue
        a,b=left[0],right[0]
        if abs(a['y']+a['height']/2-b['y']-b['height']/2)>4:continue
        lg=sb[0]-a['x']-a['width'];rg=b['x']-sb[2]
        if abs(lg-rg)>.04*sw:continue
        e=out[text['id']];ta,tb=out[a['id']],out[b['id']]
        ink=_text_visible_bounds(e);axis=(ta['x']+ta['width']/2+tb['x']+tb['width']/2)/2
        if abs((ink[0]+ink[2])/2-axis)>.08*width:continue
        shift=axis-(ink[0]+ink[2])/2
        if e['x']+shift<0 or e['x']+shift+e['width']>width:continue
        e['x']+=shift;ink=_text_visible_bounds(e)
        factor=min(ta['width']/max(1.,a['width']),tb['width']/max(1.,b['width']))
        gap=max(2.,(lg+rg)/2*factor)
        x1=ink[0]-gap-ta['width'];x2=ink[2]+gap
        if x1>=0 and x2+tb['width']<=width:
            ta['x']=x1;tb['x']=x2


def _simple_centered_stack_source(units,source):
    """A source brand/copy/action lockup has no independent hero or corner art."""
    sw=source['canvas']['width']
    return (bool(units) and all(u['role'] in ('header','copy','action') for u in units)
            and any(u['role']=='header' and any(e['type']=='image' for e in u['items']) for u in units)
            and all(abs((u['bounds'][0]+u['bounds'][2])/2-sw/2)<=.02*sw for u in units))


def _simple_stack_axis_error(elements,units,source,width,apply=False):
    """An unchanged vertical stack inherits the canvas axis, even on a banner.

    A true inline reflow has overlapping vertical bands and keeps its own axes.
    The source-only dispatch is intentionally independent of predicted boxes.
    """
    if not _simple_centered_stack_source(units,source):return 0.
    out={e['id']:e for e in elements};blocks=[]
    for u in units:
        members=[out[e['id']] for e in u['items'] if e['id'] in out]
        if not members:return 0.
        blocks.append((u,members,_raw_bounds(members)))
    headers=[b for u,_,b in blocks if u['role']=='header']
    copy=[b for u,_,b in blocks if u['role']!='header']
    if not headers or not copy or max(b[3] for b in headers)>min(b[1] for b in copy)+1:return 0.
    error=0.
    for _,members,box in blocks:
        delta=width/2-(box[0]+box[2])/2
        error+=abs(delta)/max(1.,width)
        if apply and box[0]+delta>=0 and box[2]+delta<=width:
            for e in members:e['x']+=delta
    return error/max(1,len(blocks))


def _source_geometry_safety(elements,source,units,width,height):
    """Normalize authored axes, control peers and circles before render selection."""
    from engine.qwen_refine import snap_source_axes
    out=deepcopy(elements);by={e['id']:e for e in out};src={e['id']:e for e in source['elements']}
    sw,sh=source['canvas']['width'],source['canvas']['height']
    for e in out:
        original=src[e['id']]
        if (original.get('type')=='shape' and original.get('shape_type')=='ellipse'
                and abs(original['width']-original['height'])<=2.):
            cx,cy=e['x']+e['width']/2,e['y']+e['height']/2
            side=min(width,height,sqrt(max(1.,e['width']*e['height'])))
            e.update(width=side,height=side,x=min(width-side,max(0.,cx-side/2)),y=min(height-side,max(0.,cy-side/2)))
        if (e.get('type')=='text' and width/height<1.8
                and abs(original['x']+original['width']/2-sw/2)<=.02*sw
                and abs(e['x']+e['width']/2-width/2)<=.12*width):
            margin=min(.04*width,max(0.,original['x'])/max(1.,sw)*width)
            e['width']=min(e['width'],width-2*margin);e['x']=(width-e['width'])/2
    # Pin explicit corner groups, not individual ribbon labels.
    for u in units:
        if u.get('corner_callout') and u['role']=='decoration':
            anchored=_fit_raw_block(_callout_block(u,width,height,sw,sh),width*.38,height*.16)
            box=_raw_bounds(anchored);edge=min(width,height)*.045
            x=width-edge-(box[2]-box[0]) if (u['bounds'][0]+u['bounds'][2])/2>sw/2 else edge
            anchored=_shift(anchored,x-box[0],edge-box[1])
            for e in anchored:
                if e['id'] in by:by[e['id']].update(e)
    # Restore colliding source peer badges together; tall formats use a column.
    peers=[u for u in units if u['role']=='tag' and not u.get('corner_callout')]
    if len(peers)>1:
        groups=[[by[e['id']] for e in u['items'] if e['id'] in by] for u in peers]
        boxes=[_bounds(g) for g in groups]
        collision=any(_overlap(a,b)>1 for i,a in enumerate(boxes) for b in boxes[:i])
        if collision or (width/height<=.45 and max(b[1] for b in boxes)-min(b[1] for b in boxes)<.08*height):
            vertical=width/height<=.45;gap=min(12.,.015*width)
            extent='height' if vertical else 'width';axis='y' if vertical else 'x'
            spans=[b[3]-b[1] if vertical else b[2]-b[0] for b in boxes]
            available=(height*.25 if vertical else width*.92)-gap*(len(groups)-1)
            scale=min(1.,available/max(1.,sum(spans)))
            center=sum((b[1]+b[3])/2 for b in boxes)/len(boxes)
            cursor=max(height*.04,min(center-(sum(spans)*scale+gap*(len(groups)-1))/2,
                                      height*.94-sum(spans)*scale-gap*(len(groups)-1))) if vertical else (width-sum(spans)*scale-gap*(len(groups)-1))/2
            for group,box,span in zip(groups,boxes,spans):
                for e in group:
                    e['x']=(e['x']-box[0])*scale+(width-(box[2]-box[0])*scale)/2 if vertical else (e['x']-box[0])*scale+cursor
                    e['y']=(e['y']-box[1])*scale+cursor if vertical else (e['y']-box[1])*scale+center-(box[3]-box[1])*scale/2
                    e['width']*=scale;e['height']*=scale
                    if e.get('type')=='text':_set_font_size(e,e['font_size']*scale)
                cursor+=span*scale+gap
    # Source axes are useful for modest aspect changes; extreme reflows own axes.
    if max((width/height)/(sw/sh),(sw/sh)/(width/height))<1.4:
        out,_=snap_source_axes(out,source,width,height)
    out=_restore_callout_order(out,source,width,height)
    _center_control_labels(out,source,width,height)
    _restore_attached_ornaments(out,source,width,height)
    _balance_rule_peers(out,source,width,height)
    _simple_stack_axis_error(out,units,source,width,apply=True)
    from engine.layout_contracts import repair
    return repair(out,source,units,width,height)


def _skyscraper_seed(units,width,height,source_size):
    """Reflow atomic foreground units down a narrow canvas, with one badge per row."""
    margin=width*.06;gap=max(4.,height*.012);cw=width-2*margin
    roles={r:sorted((u for u in units if u['role']==r),key=_semantic_order)
           for r in ('header','copy','action','tag','visual','footer','decoration')}
    output=[];blocks=[]
    specs=[('header',.09),('visual',.26),('tag',.14),('copy',.28),('action',.09),('footer',.06)]
    active=[(role,share) for role,share in specs if roles[role]]
    usable=height*.90-gap*max(0,len(active)-1);total=sum(share for _,share in active)
    scale=sqrt(width*height/max(1.,source_size[0]*source_size[1]))
    for role,share in active:
        room=usable*share/total
        members=roles[role];block=[]
        if role=='copy':
            block=_copy(members,cw,room,scale*1.7,gap*.6,align='center')
        else:
            per=(room-gap*max(0,len(members)-1))/max(1,len(members));cursor=0.
            for u in members:
                local=deepcopy(u)
                if role in ('action','tag'):
                    local['canvas_scale']=max(scale,.6);local['corner_callout']=False
                items=_unit(local,cw,per,scale*2,fill=role in ('header','visual'))
                items=_fit_raw_block(items,cw,per);box=_raw_bounds(items)
                block.extend(_shift(items,(cw-(box[2]-box[0]))/2-box[0],cursor-box[1]))
                cursor+=box[3]-box[1]+gap
        if block:
            block=_fit_raw_block(block,cw,room);box=_raw_bounds(block);blocks.append((block,box))
    used=sum(b[3]-b[1] for _,b in blocks)
    extra=max(gap,min(height*.055,(height*.88-used)/max(1,len(blocks)-1)))
    y=max(height*.04,(height-used-extra*max(0,len(blocks)-1))/2)
    for block,box in blocks:
        output.extend(_shift(block,margin+(cw-(box[2]-box[0]))/2-box[0],y-box[1]));y+=box[3]-box[1]+extra
    output.extend(_decorations(roles['decoration'],width,height,*source_size,output))
    return output


def _calibrate_text_boxes(elements,source,width,height,tree=None):
    """Widen into empty space, then adjust height, then shrink only as needed."""
    result=deepcopy(elements);out={e['id']:e for e in result};src={e['id']:e for e in source['elements']}
    notes=[]
    for e in result:
        if e.get('type')!='text' or not e.get('content','').strip():continue
        original=deepcopy(e);size=max(1.,float(e.get('font_size',1)))
        authored=src.get(e['id'],e)
        rotated_parent=next((out[sid] for sid,s in src.items() if sid in out and s['type']!='text'
            and authored.get('group_id') is not None and s.get('group_id')==authored.get('group_id')
            and s.get('z_index',0)<authored.get('z_index',0)
            and abs(float(e.get('rotation',0)))>.5
            and abs(float(out[sid].get('rotation',0))-float(e.get('rotation',0)))<.5),None)
        if rotated_parent is not None:
            # A shared rotated panel owns local glyph coordinates. Clamping
            # its label into the global axis-aligned panel box shifts and
            # shrinks correctly centered type on every fitting round.
            ink=_render_text_ink(e)
            if ink:
                e['width']=max(e['width'],ceil(ink[2])+2.)
                e['height']=max(e['height'],ceil(ink[3])+2.)
            notes.append(dict(element_id=e['id'],rotated_control_local_fit=True,font_before=size,font_after=e['font_size']))
            continue
        if abs(float(e.get('rotation',0)))>.5:
            # A vertical caption is measured in its local frame. Widening it
            # against global X obstacles changes its pivot and can swing ink
            # into a neighbouring headline even when the source lockup fits.
            before=_text_visible_bounds(e);ink=_render_text_ink(e)
            if ink:
                if ink[2]>round(e['width']):e['width']=ceil(ink[2])+2.
                if ink[3]>round(e['height']):e['height']=ceil(ink[3])+2.
                after=_text_visible_bounds(e)
                e['x']+=(before[0]+before[2]-after[0]-after[2])/2
                e['y']+=(before[1]+before[3]-after[1]-after[3])/2
            notes.append(dict(element_id=e['id'],rotated_text_local_fit=True,font_before=size,font_after=e['font_size']))
            continue
        left,right,top,bottom=0.,float(width),0.,float(height)
        authored=src.get(e['id'],e)
        sw,sh=source['canvas']['width'],source['canvas']['height']
        left=min(width*.04,max(0.,authored['x'])/sw*width)
        right=width-min(width*.04,max(0.,sw-authored['x']-authored['width'])/sw*width)
        # Respect the smallest source backing, rather than widening across it.
        backing=[]
        for sid,s in src.items():
            if sid==e['id'] or sid not in out or s.get('type')=='text':continue
            if s['width']>=.9*source['canvas']['width'] and s['height']>=.9*source['canvas']['height']:continue
            if (s['x']<=authored['x'] and s['y']<=authored['y'] and
                    s['x']+s['width']>=authored['x']+authored['width'] and
                    s['y']+s['height']>=authored['y']+authored['height'] and
                    s.get('z_index',0)<=authored.get('z_index',0)):
                backing.append((s['width']*s['height'],s,out[sid]))
        if backing:
            _,s,p=min(backing,key=lambda pair:pair[0])
            sx=p['width']/max(1.,s['width']);sy=p['height']/max(1.,s['height'])
            px=min(max(1.,size*.12),max(0.,authored['x']-s['x'])*sx)
            py=min(max(1.,size*.10),max(0.,authored['y']-s['y'])*sy)
            left=max(left,p['x']+px);right=min(right,p['x']+p['width']-px)
            top=max(top,p['y']+py);bottom=min(bottom,p['y']+p['height']-py)
        # Non-overlapping source neighbours are expansion obstacles. Existing
        # source overlap and a backing are intentional, not empty-space errors.
        for other in result:
            if other['id']==e['id']:continue
            a=src.get(other['id'])
            if a is None:continue
            sb=(authored['x'],authored['y'],authored['x']+authored['width'],authored['y']+authored['height'])
            ob=(a['x'],a['y'],a['x']+a['width'],a['y']+a['height'])
            if _overlap(sb,ob)>0:continue
            if other.get('type')!='text' and other['width']>.8*width and other['height']>.8*height:continue
            gap=max(1.,size*.04)
            if other['y']<e['y']+e['height'] and other['y']+other['height']>e['y']:
                if other['x']>=e['x']+e['width']-1:right=min(right,other['x']-gap)
                if other['x']+other['width']<=e['x']+1:left=max(left,other['x']+other['width']+gap)
            if other['x']<e['x']+e['width'] and other['x']+other['width']>e['x']:
                if other['y']>=e['y']+e['height']-1:bottom=min(bottom,other['y']-gap)
        # A misplaced alignment anchor cannot be repaired by reducing the font:
        # right-aligned ink keeps its right edge at the same coordinate at every
        # size. Move the frame into its available backing before fitting glyphs.
        if right>left and bottom>top:
            e['width']=min(e['width'],right-left)
            e['height']=min(e['height'],bottom-top)
            e['x']=min(max(e['x'],left),right-e['width'])
            e['y']=min(max(e['y'],top),bottom-e['height'])
        align=e.get('text_align','left');anchor=e['x']+(e['width']/2 if align=='center' else e['width'] if align=='right' else 0)
        room=(2*min(anchor-left,right-anchor) if align=='center' else anchor-left if align=='right' else right-anchor)
        natural=_text_info(e,1e9,size)[1]+max(1.,size*.03)
        preferred_lines=max(1,_text_info(authored)[2])
        if _text_info(e)[2]>preferred_lines or natural>e['width']:
            new_width=max(e['width'],min(room,natural))
            e['x']=anchor-(new_width/2 if align=='center' else new_width if align=='right' else 0)
            e['width']=new_width
        ink=_render_text_ink(e)
        if ink is None:
            h,w,_=_text(e);ink=(0.,0.,w,h)
        # Release an arbitrary sampled text anchor before reducing its font.
        if ink[2]+2>e['width'] and right-left>=ceil(ink[2])+2:
            new_width=ceil(ink[2])+2.
            e['x']=min(max(left,e['x']),right-new_width);e['width']=new_width
            ink=_render_text_ink(e) or ink
        if ink[3]+2>e['height'] and bottom-top>=ceil(ink[3])+2:
            e['height']=ceil(ink[3])+2.
            e['y']=min(max(top,e['y']),bottom-e['height'])
        def fits(probe):
            bounds=_render_text_ink(probe)
            if bounds is None:
                h,w,_=_text(probe);bounds=(0.,0.,w,h)
            bearing_x,bearing_y=_source_ink_overhang(probe,authored)
            return (bounds[0]>=-bearing_x-1e-6 and ceil(bounds[2])<=round(probe['width'])
                    and ceil(bounds[3])<=round(probe['height']) and probe['y']+bounds[3]<=bottom
                    and probe['x']+bounds[2]<=right
                    and probe['x']+bounds[0]>=max(0.,left-bearing_x)-1e-6
                    and bounds[1]>=-bearing_y-1e-6
                    and probe['y']+bounds[1]>=max(0.,top-bearing_y)-1e-6)
        if not fits(e):
            low,high=1.,size
            minimum=deepcopy(e);_set_font_size(minimum,low)
            if fits(minimum):
                for _ in range(12):
                    mid=(low+high)/2;probe=deepcopy(e);_set_font_size(probe,mid)
                    if fits(probe):low=mid
                    else:high=mid
                _set_font_size(e,low)
            else:
                # No feasible lower bound means the search is invalid. Preserve
                # readable copy and let the defect gate reject this candidate.
                notes.append({'element_id':e['id'],'unfittable':True,
                              'reason':'no feasible glyph fit in available region'})
        if any(abs(float(e.get(k,0))-float(original.get(k,0)))>1e-5 for k in ('x','y','width','height','font_size','letter_spacing')):
            notes.append({'element_id':e['id'],'widened':e['width']>original['width']+1e-5,
                          'font_before':size,'font_after':e['font_size'],
                          'measured_with':'skia-render' if _render_text_ink(e) is not None else 'font-metrics-fallback'})
    _center_control_labels(result,source,width,height)
    return result,notes


def _finish_candidate_text(elements,source,units,width,height,measured_stack=False):
    """Fit with renderer metrics, then restore the final painted alignment axes."""
    from engine.layout_contracts import _primary_axis,_copy_clearance
    result,notes=_calibrate_text_boxes(elements,source,width,height)
    if measured_stack:
        from engine.structural_reflow import resolve_stack
        result=resolve_stack(result,source,units,width,height)
    _primary_axis(result,source,units,width,height,apply=True)
    _copy_clearance(result,source,width,height)
    from engine.source_structure import coherent_copy_axes
    coherent_copy_axes(result,source,units,width,height)
    _center_control_labels(result,source,width,height)
    return result,notes


def _backing_containment_defects(elements, source):
    """Painted labels retain their explicit source backing, including bearings."""
    out={e['id']:e for e in elements};defects=[]
    for original in source['elements']:
        if original.get('type')!='text' or original['id'] not in out or original.get('group_id') is None:continue
        backings=[p for p in source['elements'] if p.get('type') in ('shape','image')
                  and p.get('group_id')==original['group_id'] and p['id'] in out
                  and p.get('z_index',0)<=original.get('z_index',0)
                  and p['x']<=original['x']+1 and p['y']<=original['y']+1
                  and p['x']+p['width']>=original['x']+original['width']-1
                  and p['y']+p['height']>=original['y']+original['height']-1]
        if not backings:continue
        backing=min(backings,key=lambda e:e['width']*e['height']);panel=out[backing['id']]
        before=_text_visible_bounds(original);after=_text_visible_bounds(out[original['id']])
        old=_rotated_bounds(backing);new=_rotated_bounds(panel)
        factor=min(panel['width']/max(1.,backing['width']),panel['height']/max(1.,backing['height']))
        before_gaps=(old[0]-before[0],old[1]-before[1],before[2]-old[2],before[3]-old[3])
        after_gaps=(new[0]-after[0],new[1]-after[1],after[2]-new[2],after[3]-new[3])
        if any(a>max(1.,max(0.,b)*factor+1.) for a,b in zip(after_gaps,before_gaps)):
            defects.append(f"{original['id']}: glyphs outside source backing")
    return defects


def _physical_layout_safe(elements, units, source, width, height):
    """Hard raster/geometry checks, without inferred unit gaps or margins."""
    from engine.layout_contracts import art_occlusions,image_collisions
    originals={e['id']:e for unit in units for e in unit['items']}
    if len(elements)!=len(originals) or {e['id'] for e in elements}!=set(originals):return False
    for e in elements:
        if not all(isfinite(float(e.get(k,0))) for k in ('x','y','width','height')):return False
        if min(e['width'],e['height'])<=0:return False
        if e['x']<-.00001 or e['y']<-.00001 or e['x']+e['width']>width+.00001 or e['y']+e['height']>height+.00001:return False
        bounds=_rotated_bounds(e)
        if bounds[0]<-.00001 or bounds[1]<-.00001 or bounds[2]>width+.00001 or bounds[3]>height+.00001:return False
        original=originals[e['id']]
        if (e.get('type')=='image' and original.get('fit')!='cover'
                and abs((e['width']/e['height'])/(original['width']/original['height'])-1)>.05):return False
    return (not _backing_containment_defects(elements,source)
            and not _unintended_text_overlaps(elements,units)
            and not _unintended_control_overlaps(elements,source)
            and not art_occlusions(elements,source)
            and not image_collisions(elements,source)
            and _objective_layout_cost(elements,units,source,width,height,1.,True)<=1.)


def _bounded_qwen_elements(elements, source, width, height):
    """Clamp raw proposals without moving their source-relative neighbours."""
    originals={e['id']:e for e in source['elements']};result=deepcopy(elements)
    for e in result:
        original=originals[e['id']];cx=e['x']+e['width']/2;cy=e['y']+e['height']/2
        if e.get('type')=='image' and original.get('fit')!='cover':
            # Interpret predicted width as the uniform asset scale. Height is
            # derived from the authored aspect ratio, so a noisy independent
            # height value cannot squash an otherwise useful image proposal.
            aspect=float(original['width'])/max(1e-6,float(original['height']))
            e['width']=min(max(1.,float(e['width'])),width)
            e['height']=e['width']/aspect
            if e['height']>height:
                e['height']=height;e['width']=height*aspect
        bounds=_rotated_bounds(e)
        factor=min(1.,width/max(e['width'],bounds[2]-bounds[0]),height/max(e['height'],bounds[3]-bounds[1]))
        e['width']*=factor;e['height']*=factor
        if e.get('type')=='text':_set_font_size(e,max(1.,e['font_size']*factor))
        e['x']=cx-e['width']/2;e['y']=cy-e['height']/2
        bounds=_rotated_bounds(e)
        hx=max(e['width'],bounds[2]-bounds[0])/2;hy=max(e['height'],bounds[3]-bounds[1])/2
        cx=min(width-hx,max(hx,cx));cy=min(height-hy,max(hy,cy))
        e['x']=cx-e['width']/2;e['y']=cy-e['height']/2
    return result


def _structural_qwen_projection(elements,source,units,width,height,tree=None):
    from engine.qwen_refine import build_container_tree,project_container_tree,snap_source_axes
    tree=tree or build_container_tree(source,units)
    tree=dict(tree,strict_canvas=True)
    projected,containers=project_container_tree(elements,tree,width,height)
    projected,fonts=_calibrate_text_boxes(projected,source,width,height,tree)
    projected,axes=snap_source_axes(projected,source,width,height,tree)
    return projected,{'containers':containers,'font_calibration':fonts,'axes':axes}


def _rotated_bounds(element):
    """Compute visual bounds for a rotated element around its center pivot."""
    x = float(element.get('x', 0.0))
    y = float(element.get('y', 0.0))
    w = float(element.get('width', 1.0))
    h = float(element.get('height', 1.0))
    rad = radians(float(element.get('rotation', 0.0)))
    cx, cy = x + w / 2.0, y + h / 2.0
    rw = w * abs(cos(rad)) + h * abs(sin(rad))
    rh = w * abs(sin(rad)) + h * abs(cos(rad))
    return (cx - rw / 2.0, cy - rh / 2.0, cx + rw / 2.0, cy + rh / 2.0)


def _bounds(items):
    boxes = []
    for e in items:
        rot = float(e.get('rotation', 0.0)) % 360.0
        is_rot = (0.5 < rot < 359.5) and abs(rot - 180.0) > 0.5
        if e.get('type')=='image':
            vb = visible_content_bounds(e,_asset_root(e))
            if is_rot:
                w, h, a = e['width'], e['height'], radians(rot)
                rw, rh = abs(w*cos(a))+abs(h*sin(a)), abs(w*sin(a))+abs(h*cos(a))
                cx, cy = e['x']+w/2, e['y']+h/2
                boxes.append((min(vb[0], e['x'], cx-rw/2), min(vb[1], e['y'], cy-rh/2),
                              max(vb[2], e['x']+w, cx+rw/2), max(vb[3], e['y']+h, cy+rh/2)))
            else:
                boxes.append(vb)
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


def _intentional_stack_pair(a, b, a_box=None, b_box=None):
    """Recognize substantial source overlap with an explicit paint order.

    A collision is only treated as intentional when the source gives us both
    a clear front/back order and meaningful visible-area overlap. This avoids
    turning ordinary proximity or alignment into a semantic group.
    """
    if a.get('z_index', 0) == b.get('z_index', 0):
        return False
    back, front = (a, b) if a.get('z_index', 0) < b.get('z_index', 0) else (b, a)
    # A translucent decorative wash may cross copy without becoming its
    # semantic parent. Explicit groups are already joined by the analyzer.
    from engine.geometry_policy import ambient_shape
    if ambient_shape(back):
        return False
    if back['type'] not in ('image', 'shape', 'text') or front['type'] not in ('image', 'shape', 'text'):
        return False
    def visible_box(item, box):
        if box is not None:return box
        return _text_visible_bounds(item) if item['type'] == 'text' else _bounds([item])
    back_box = visible_box(back, a_box if back is a else b_box)
    front_box = visible_box(front, b_box if front is b else a_box)
    back_area = max(1e-9, (back_box[2]-back_box[0])*(back_box[3]-back_box[1]))
    front_area = max(1e-9, (front_box[2]-front_box[0])*(front_box[3]-front_box[1]))
    fraction = _overlap(back_box, front_box) / min(back_area, front_area)
    threshold = .18 if 'text' in (a['type'], b['type']) else .12
    return fraction >= threshold


def _area(e):
    if e.get('type')=='image':
        x,y,r,b=_bounds([e])
        return max(1e-9,(r-x)*(b-y))
    return max(1e-9, e['width']*e['height'])


def _contains(a, b, tolerance=1):
    return (b['x'] >= a['x']-tolerance and b['y'] >= a['y']-tolerance
            and b['x']+b['width'] <= a['x']+a['width']+tolerance
            and b['y']+b['height'] <= a['y']+a['height']+tolerance)


@lru_cache(maxsize=256)
def _flat_image_asset_stats(path):
    """Return conservative pixel evidence for a raster shape/backplate."""
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
        image=Image.open(path).convert('RGBA')
        alpha=image.getchannel('A').point(lambda value:255 if value>=8 else 0)
        bbox=alpha.getbbox()
        if not bbox:return None
        crop=alpha.crop(bbox)
        mask_pixels=crop.histogram()[255]
        fill=mask_pixels/max(1,crop.width*crop.height)
        symmetry=[]
        for mirrored in (ImageOps.mirror(crop),ImageOps.flip(crop)):
            diff=ImageChops.difference(crop,mirrored)
            symmetry.append(1-sum(diff.histogram()[1:])/max(1,crop.width*crop.height))
        rgb=image.convert('RGB')
        std=ImageStat.Stat(rgb,mask=alpha).stddev
        rgb_std=sum(std[:3])/3
        edge=rgb.convert('L').filter(ImageFilter.FIND_EDGES).point(lambda value:255 if value>20 else 0)
        edge=ImageChops.multiply(edge,alpha)
        ehist=edge.histogram()
        edge_density=ehist[255]/max(1,edge.width*edge.height)
        return (rgb_std,fill,min(symmetry),edge_density,image.size,bbox)
    except (ImportError,OSError,ValueError):
        return None


def _flat_image_backplate(image, text):
    """Require uniform opaque pixels both globally and behind the text."""
    try:
        from PIL import Image, ImageStat
        path=(_asset_root(image)/image.get('asset_id','')).resolve()
        metrics=_flat_image_asset_stats(str(path))
        if metrics is None:return False
        rgb_std,fill,symmetry,edge_density,(iw,ih),alpha_bbox = metrics
        if rgb_std>12 or fill<.82 or symmetry<.90 or edge_density>.08:
            return False
        x,y,w,h=(float(image[k]) for k in ('x','y','width','height'))
        if w<=0 or h<=0:return False
        if image.get('fit','contain')!='contain':return False
        scale=min(w/iw,h/ih)
        rw,rh=iw*scale,ih*scale
        ox,oy=x+(w-rw)/2,y+(h-rh)/2
        if abs(float(image.get('rotation',0)))>.1 or abs(float(text.get('rotation',0)))>.1:return False
        text_h,_,_,line_widths,_=_text_info(text)
        align=text.get('text_align','left')
        line_lefts=[float(text['x'])+(float(text['width'])-line)/2 if align=='center'
                    else float(text['x'])+float(text['width'])-line if align=='right'
                    else float(text['x']) for line in line_widths]
        tx0=max(x,min(line_lefts,default=float(text['x'])))
        ty0=max(y,float(text['y']))
        tx1=min(x+w,max((left+line for left,line in zip(line_lefts,line_widths)),default=float(text['x'])+float(text['width'])))
        ty1=min(y+h,float(text['y'])+min(float(text['height']),text_h))
        if tx1<=tx0 or ty1<=ty0:return False
        vx0,vy0=ox+alpha_bbox[0]*scale,oy+alpha_bbox[1]*scale
        vx1,vy1=ox+alpha_bbox[2]*scale,oy+alpha_bbox[3]*scale
        text_w,text_h=tx1-tx0,ty1-ty0
        pad=max(1.,min(text_h*.12,min(vx1-vx0,vy1-vy0)*.02))
        if (vx1-vx0<text_w*1.02 or vy1-vy0<text_h*1.06
                or tx0<vx0+pad or ty0<vy0+pad or tx1>vx1-pad or ty1>vy1-pad):return False
        px0=max(0,min(iw-1,int((tx0-ox)/rw*iw)));py0=max(0,min(ih-1,int((ty0-oy)/rh*ih)))
        px1=max(px0+1,min(iw,int((tx1-ox)/rw*iw)));py1=max(py0+1,min(ih,int((ty1-oy)/rh*ih)))
        with Image.open(path) as source:
            patch=source.convert('RGBA').crop((px0,py0,px1,py1))
        pmask=patch.getchannel('A').point(lambda value:255 if value>=8 else 0)
        opaque=pmask.histogram()[255]/max(1,pmask.width*pmask.height)
        patch_std=sum(ImageStat.Stat(patch.convert('RGB'),mask=pmask).stddev[:3])/3 if opaque else 999
        return opaque>=.85 and patch_std<=12
    except (ImportError,OSError,ValueError,ZeroDivisionError):
        return False


@lru_cache(maxsize=128)
def _font(family, bold, italic):
    """Load actual supplied fonts locally; unknown fonts use an estimate."""
    try:
        from PIL import ImageFont
        from io import BytesIO
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
        manager = _crello_font_manager()
        if manager is not None:
            weight = 'bold' if bold else 'regular'
            style = 'bolditalic' if bold and italic else 'italic' if italic else 'bold' if bold else 'regular'
            return ImageFont.truetype(BytesIO(manager.lookup(family, weight, style)), 1000)
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
            try:
                advance = selected.getlength(c)/1000
            except OSError:
                # Some font programs exceed FreeType's execution limit. The
                # renderer's own metrics remain usable; don't abort a request
                # merely because the auxiliary Pillow measurement failed.
                rb, ri = bold, italic
                for start, end, run_bold, run_italic in runs:
                    if start <= i < end:
                        rb = bold if run_bold is None else run_bold
                        ri = italic if run_italic is None else run_italic
                measured = _skia_advances(c, family, rb, ri, 1000)
                advance = measured[0]/1000 if measured else (.30 if c.isspace() else .56)
            result.append(advance)
        else:
            result.append(.30 if c.isspace() or c in "ilI.,'!:;|" else .90 if c in 'MW@#%&' else .65 if c.isupper() else .56)
    return tuple(result)


def _text_info(e, width=None, size=None):
    """Cached actual-font word wrapping; preserve explicit newlines/style runs."""
    width = e['width'] if width is None else width
    base_size = max(1.0, float(e.get('font_size', 1.0)))
    size = float(e['font_size'] if size is None else size)
    spacing = float(e.get('letter_spacing', 0)) * (size / base_size)
    runs = tuple((r['start'], r['end'], None if 'font_weight' not in r else r['font_weight']=='bold',
                  None if 'font_style' not in r else r['font_style']=='italic') for r in (e.get('style_runs') or []))
    return _text_cached(e.get('content',''),e.get('font_family',''),e.get('font_weight')=='bold',
                        e.get('font_style')=='italic',e.get('line_height',1.2),
                        spacing,runs,width,size)


def _text(e, width=None, size=None):
    # Keep the public measurement helper's three-value contract.
    return _text_info(e,width,size)[:3]


@lru_cache(maxsize=65536)
def _text_cached(content,family,bold,italic,line_height,spacing,runs,width,size):
    # Match the supplied converter's space-replacement wrap rule. No hard
    # splitting of long words or text rewriting is hidden in the measurement.
    advances = _advances(content,family,bold,italic,runs)
    measured = _skia_advances(content,family,bold,italic,size)
    widths = list(measured) if measured is not None else [v*size for v in advances]
    drawn = _skia_advances(content,family,bold,italic,max(1,int(size)))
    glyph_widths = list(drawn) if drawn is not None else list(widths)
    for start,end,rb,ri in runs:
        styled = _skia_advances(content,family,bold if rb is None else rb,
                                italic if ri is None else ri,max(1,int(size)))
        if styled is not None:
            for i in range(max(0,start),min(len(content),end)):glyph_widths[i]=styled[i]
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
    for c, advance in zip(chars, glyph_widths):
        if c == '\n':
            lines.append(line_width+max(0, count-1)*spacing)
            line_width, count = 0, 0
        else:
            line_width, count = line_width+advance, count+1
    lines.append(line_width+max(0, count-1)*spacing)
    words=tuple(len(line.split()) for line in ''.join(chars).split('\n'))
    # The converter wraps with fractional-size advances, while cr-renderer
    # draws at int(fontSize). A fit must satisfy both, or a nominally fitting
    # single line can wrap before its narrower integer-size glyphs are drawn.
    wrap_lines=[];wrap_width=0.
    for c,advance in zip(chars,widths):
        if c=='\n':wrap_lines.append(wrap_width);wrap_width=0.
        else:wrap_width+=advance
    wrap_lines.append(wrap_width)
    required=max(max(lines,default=0),max(wrap_lines,default=0))
    return len(lines)*size*line_height+size*.12, required, len(lines), tuple(lines), words


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
            _set_font_size(e,max(1,e['font_size']))
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


def _edge_backdrop(element,source):
    """Recognize large lowest-layer images authored across a canvas edge.

    Use the rotated frame, not its unrotated width/height. A quarter-turn
    photograph may cover the whole height while only occupying one side.
    Grouped artwork and small edge ornaments remain foreground units.
    """
    if element.get('type')!='image' or element.get('group_id') is not None:return None
    sw,sh=source['canvas']['width'],source['canvas']['height']
    if element.get('z_index',0)!=min((e.get('z_index',0) for e in source['elements']),default=0):return None
    from engine.qwen_refine import _rotated_bounds
    l,t,r,b=_rotated_bounds(element)
    visible=max(0.,min(sw,r)-max(0.,l))*max(0.,min(sh,b)-max(0.,t))
    if visible<.25*sw*sh:return None
    if t<=.02*sh and b>=.98*sh:
        if l<-.02*sw and r<.98*sw:return 'left'
        if r>1.02*sw and l>.02*sw:return 'right'
    if l<=.02*sw and r>=.98*sw:
        if t<-.02*sh and b<.98*sh:return 'top'
        if b>1.02*sh and t>.02*sh:return 'bottom'
    return None


def _fit_edge_backdrop(element,source,width,height):
    """Keep a backdrop on its authored side with legal raw and rotated bounds."""
    from engine.qwen_refine import _rotated_bounds
    from engine.layout_contracts import edge_scene
    scene=edge_scene(source,width,height)
    if scene is not None and scene['photo']['id']==element['id']:return scene['photo']
    edge=_edge_backdrop(element,source)
    e=deepcopy(element)
    angle=abs(float(e.get('rotation',0.)))%180
    sw,sh=source['canvas']['width'],source['canvas']['height']
    if edge in ('left','right') and abs(angle-90)<=1 and width/height<=1.35 and sw/sh>=1.6:
        # A quarter turn cannot touch this edge while both frames are bounded.
        # Choose an unrotated, uniform backdrop alternative instead of an inset.
        e['rotation']=0.
    l,t,r,b=_rotated_bounds(e)
    factor=min(width/e['width'],height/e['height'],width/(r-l),height/(b-t))
    e['width']*=factor;e['height']*=factor
    rw,rh=(r-l)*factor,(b-t)*factor
    # Both raw and painted rectangles must fit. At quarter turns their
    # extents differ; the nearest legal center may leave an unavoidable inset.
    mx,my=max(e['width'],rw)/2,max(e['height'],rh)/2
    cx=mx if edge=='left' else width-mx if edge=='right' else width/2
    cy=my if edge=='top' else height-my if edge=='bottom' else height/2
    e.update(x=cx-e['width']/2,y=cy-e['height']/2)
    return e


def _analyze(source):
    from engine.geometry_policy import ambient_shape
    elements = source['elements']
    sw, sh = source['canvas']['width'], source['canvas']['height']
    max_font = max((e.get('font_size', 0) for e in elements), default=24)
    backgrounds, rest = [], []
    background_candidates = []
    for e in elements:
        full = e['x'] <= sw*.02 and e['y'] <= sh*.02 and e['x']+e['width'] >= sw*.98 and e['y']+e['height'] >= sh*.98
        behind = all(e.get('z_index', 0) <= f.get('z_index', 0) for f in elements if f['type']=='text')
        if (full and behind and not e.get('group_id') and e['type']!='text') or _edge_backdrop(e,source):
            background_candidates.append(e)
        else:
            rest.append(e)
    # Full-canvas overlays can be frames, borders, or transparent textures.
    # Keep them registered to the canvas, but preserve each layer's authored
    # fit mode so a contain-fitted border is never silently cropped to cover.
    backgrounds.extend(background_candidates)
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
    # Treat text-contained source layers as its parent, regardless of whether
    # the parent is a flat shape, ribbon image, or photo card. The small and
    # medium panel thresholds are intentionally conservative about containment.
    stack_boxes={e['id']:(_text_visible_bounds(e) if e['type']=='text' else _bounds([e]))
                 for e in rest}
    stack_links=[]
    source_parent_for_text={}
    for text_index,text in enumerate(rest):
        if text['type']!='text':continue
        text_box=stack_boxes[text['id']]
        text_area=max(1e-9,(text_box[2]-text_box[0])*(text_box[3]-text_box[1]))
        candidates=[]
        for parent_index,back in enumerate(rest):
            if (back['type'] not in ('image','shape') or ambient_shape(back,source)
                    or back.get('z_index',0)>=text.get('z_index',0)
                    or root(parent_index)==root(text_index)
                    or back.get('shape_type')=='ellipse' and float(back.get('opacity',1.))<.98):
                continue
            area_share=_area(back)/(sw*sh)
            containment=_overlap(stack_boxes[back['id']],text_box)/text_area
            if area_share<.12:
                minimum_containment=.85
            elif area_share<=.85:
                minimum_containment=.85 if area_share<.30 else .80
            else:
                continue
            if containment>=minimum_containment:
                candidates.append((area_share,parent_index,back))
        if candidates:
            _,parent_index,back=min(candidates,key=lambda row:row[0])
            source_parent_for_text[text['id']]=back['id']
            area_share=_area(back)/(sw*sh)
            text_h=max(1.,text_box[3]-text_box[1])
            font_size=max(1.,float(text.get('font_size',text_h)))
            # Only compact badges/ribbons are atomic. Larger panels stay as
            # independent containers so their children can reflow inside.
            if area_share<.16 and text_h<font_size*6:
                join(text_index,parent_index)
                stack_links.append((text_index,parent_index))

    # Source z-order plus substantial visible overlap is evidence of a collage
    # or layered illustration. Make connected layers one atomic layout unit.
    for i,a in enumerate(rest):
        for j,b in enumerate(rest[:i]):
            area_a=_area(a)/(sw*sh);area_b=_area(b)/(sw*sh)
            container_pair=(.12<=area_a<=.85 and _contains(a,b,2)) or (
                            .12<=area_b<=.85 and _contains(b,a,2))
            parent_pair=(source_parent_for_text.get(a['id'])==b['id']
                         or source_parent_for_text.get(b['id'])==a['id'])
            if (root(i)!=root(j) and not container_pair and not parent_pair
                    and _intentional_stack_pair(a,b,stack_boxes[a['id']],stack_boxes[b['id']])):
                join(i,j)
                stack_links.append((i,j))
    stacked_roots={root(index) for pair in stack_links for index in pair}
    stacked_ids={e['id'] for i,e in enumerate(rest) if root(i) in stacked_roots}
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
        if ambient_shape(panel,source):continue
        shape_panel=panel['type']=='shape'
        image_panel=panel['type']=='image'
        if (not (shape_panel or image_panel)
                or not sw*sh*.12<=_area(panel)<=sw*sh*.85):continue
        panel_box=_bounds([panel]);backed=[]
        contained_text_groups=0
        for child in groups:
            if child is items:continue
            child_backed=[]
            for element in child:
                if panel.get('z_index',0)>=element.get('z_index',0):continue
                child_box=stack_boxes[element['id']]
                child_area=max(1e-9,(child_box[2]-child_box[0])*(child_box[3]-child_box[1]))
                if _overlap(panel_box,child_box)/child_area>=.80:
                    child_backed.append(element['id'])
            if child_backed:
                backed.extend(child_backed)
                if any(e['type']=='text' and e['id'] in child_backed for e in child):
                    contained_text_groups+=1
        text_groups=contained_text_groups
        panel_area=_area(panel)
        if text_groups>=1:
            panel_candidates.append((panel_area,panel,tuple(sorted(set(backed)))) )
    # Nested cards often have a translucent tint/backplate directly beneath
    # the opaque white card. Treat the front card as the sole layout anchor
    # and carry the lower-z plate with it. If both are independent anchors,
    # Qwen can enlarge the tint to fill the canvas while leaving the actual
    # card and its contents behind in a different location.
    shadowed_panels=set()
    underlays_by_panel={}
    for _,underlay,underlay_children in panel_candidates:
        underlay_box=_bounds([underlay])
        underlay_area=max(1e-9,_area(underlay))
        underlay_text={eid for eid in underlay_children
                       if any(e['id']==eid and e['type']=='text' for group in groups for e in group)}
        parents=[]
        for _,front,front_children in panel_candidates:
            if front['id']==underlay['id'] or front.get('z_index',0)<=underlay.get('z_index',0):
                continue
            front_box=_bounds([front])
            if _overlap(front_box,underlay_box)/underlay_area<.80:
                continue
            front_text={eid for eid in front_children
                        if any(e['id']==eid and e['type']=='text' for group in groups for e in group)}
            if (underlay_text & front_text) or front['id'] in underlay_children:
                parents.append(front)
        if parents:
            front=max(parents,key=lambda item:float(item.get('z_index',0)))
            shadowed_panels.add(underlay['id'])
            underlays_by_panel.setdefault(front['id'],[]).append(underlay['id'])
    assigned=set()
    # Prefer the frontmost contained card as the text anchor. A translucent
    # tint/underlay may be microscopically smaller than the white panel above
    # it; area-only sorting wrongly detached copy from the visible card.
    for _,panel,backed in sorted(panel_candidates,
                                 key=lambda candidate:(-float(candidate[1].get('z_index',0)),
                                                       candidate[0])):
        if panel['id'] in shadowed_panels:
            continue
        children=tuple(eid for eid in backed if eid not in assigned)
        children=tuple(dict.fromkeys((*children,*underlays_by_panel.get(panel['id'],()))))
        if not children:continue
        panel_children[panel['id']]=children
        assigned.update(children)
    units = []
    # Rebuild components after adding inferred stack relationships.
    grouped={}
    for i,e in enumerate(rest):
        grouped.setdefault(root(i),[]).append(e)
    for items in grouped.values():
        x, y, r, b = _bounds(items)
        texts = [e for e in items if e['type']=='text']
        shapes = [e for e in items if e['type']=='shape']
        images = [e for e in items if e['type']=='image']
        label_back=shapes[0] if len(shapes)==1 else next((e for e in images if len(images)==1
                   and len(items)==2 and len(texts)==1 and _flat_image_backplate(e,texts[0])),None)
        if label_back is not None and label_back['type']=='image' and texts:
            lb,tb=_bounds([label_back]),_text_visible_bounds(texts[0])
            contained=_overlap(lb,tb)/max(1e-9,(tb[2]-tb[0])*(tb[3]-tb[1]))>=.85
        else:
            contained=label_back is not None and all(_contains(label_back,t,3) for t in texts)
        label = (label_back is not None and len(texts)==1 and len(items)==2 and contained)
        if label:
            role = 'action' if y/sh > .6 else 'tag'
        elif len(texts)==len(items):
            role = 'footer' if y/sh > .90 and max(t['font_size'] for t in texts) < max_font*.5 else 'copy'
        elif images and not texts and (r-x)/max(1, b-y)>2 and y/sh<.22 and (r-x)*(b-y)<sw*sh*.15:
            role = 'header'
        elif images and not texts and y/sh>.88 and (r-x)*(b-y)<sw*sh*.05:
            role = 'footer'
        elif len(items)==1 and items[0]['id'] in panel_children:
            role = 'decoration'
        elif len(items)==1 and shapes and (ambient_shape(shapes[0]) or shapes[0].get('shape_type') in ('ellipse', 'line')):
            role = 'decoration'
        else:
            role = 'visual'
        preserve_layer_stack=any(e['id'] in stacked_ids for e in items)
        unit=dict(items=items,bounds=(x,y,r,b),role=role,label=label,
                  preserve_layer_stack=preserve_layer_stack)
        source_parents=tuple(sorted({source_parent_for_text[e['id']] for e in items
                                     if e['id'] in source_parent_for_text}))
        if source_parents:
            unit['source_parent_ids']=source_parents
        backed_by=tuple(sorted(panel_id for panel_id,children in panel_children.items()
                               if any(e['id'] in children for e in items)))
        if backed_by:
            unit['backed_by_ids']=backed_by
        if len(items)==1 and items[0]['id'] in panel_children:
            panel=items[0]
            unit.update(backing_panel=True,backed_ids=panel_children[panel['id']],
                        backing_source_width=panel['width'],backing_source_height=panel['height'],
                        backing_source_x=panel['x'],backing_source_y=panel['y'],
                        backing_source_canvas_width=sw,backing_source_canvas_height=sh,
                        backing_source_radius=panel.get('corner_radius',0.))
        units.append(unit)
    units=_enrich_units(source, units)
    fonts=sorted(e['font_size'] for e in elements if e['type']=='text')
    median=fonts[len(fonts)//2] if fonts else 24
    for u in units:
        u['font_median']=median
        u['source_canvas']=(sw,sh)
    return backgrounds, units


def _unit(u, width, height, scale, fill=False):
    width, height = max(.0001, width), max(.0001, height)
    items, (x,y,r,b) = u['items'], u['bounds']
    small_visual=(u.get('role')=='visual' and u.get('source_area_share',1.)<.12)
    if small_visual:
        # Small source artwork is an icon/ornament, not a slot-filling hero.
        # Keep it at the canvas-proportional scale even when a seed asks to
        # enlarge visuals by 1.15x or fill its available cell.
        fill=False
    if len(items)==1 and items[0]['type']=='text':
        e = deepcopy(items[0])
        desired=_target_font(e,u,scale)
        if u.get('semantic') in ('headline','subheading'):
            e['width'],fitted_size=_balanced_text(e,width,height,desired)
            _set_font_size(e,fitted_size)
        else:
            e['width']=width
            _set_font_size(e, _fit_text(e, width, height, desired))
        e['height'] = min(height, _text(e)[0])
        e['x'], e['y'] = 0, 0
        return _contain([e], width, height)
    if u['label'] and not u.get('corner_callout'):
        shape = next(e for e in items if e['type'] in ('shape','image'))
        texts = sorted((e for e in items if e['type']=='text'), key=lambda e:(e['y'],e['x']))
        label_w = min(width, max((r-x)*scale, max(_text(e, 1e9, _target_font(e,u,scale))[1]+_target_font(e,u,scale)*1.6 for e in texts)))
        result = deepcopy(items)
        back = next(e for e in result if e['type'] in ('shape','image'))
        labels = sorted((e for e in result if e['type']=='text'), key=lambda e:(e['y'],e['x']))
        pad = min(label_w*.10, max(_target_font(e,u,scale) for e in texts)*.8)
        total_h = 0
        for old, e in zip(texts, labels):
            e['width'] = max(.01, label_w-2*pad)
            desired_font=_target_font(old,u,scale)
            fit_height=height/len(texts)*.75
            if '\n' not in old.get('content',''):
                fit_height=min(fit_height,desired_font*(old.get('line_height',1.2)+.12)+.001)
            _set_font_size(e, _fit_text(old, e['width'], fit_height, desired_font))
            e['height'] = _text(e)[0]
            total_h += e['height']
        label_h = min(height, max(shape['height']*scale, total_h*1.5))
        back.update(x=0, y=0, width=label_w, height=label_h)
        if back['type']=='image':back['fit']='fill'
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
        if small_visual:
            scale_cap=min(scale_cap,u.get('canvas_scale',scale)*.95)
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
            _set_font_size(e, _fit_text(e, e['width'], e['height'], e['font_size']))
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
    if (shared_peers and units[0].get('target_aspect_ratio',1.)<.35
            and width<=units[0].get('target_short_side',width)*1.5):
        row_gap=min(gap,width*.034)
        available=max(.001,(height-row_gap*(len(units)-1))/len(units))
        blocks=[_unit(u,width,available,scale) for u in units]
        total=sum(_bounds(b)[3]-_bounds(b)[1] for b in blocks)+row_gap*(len(blocks)-1)
        factor=min(1.,height/max(.001,total))
        output=[];cursor=0.
        for block in blocks:
            block=_scale(block,factor);box=_bounds(block)
            bw,bh=box[2]-box[0],box[3]-box[1]
            x=0. if align=='left' else width-bw if align=='right' else (width-bw)/2
            output.extend(_move(block,x,cursor));cursor+=bh+row_gap*factor
        return output,max(0.,cursor-row_gap*factor)
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
            vertical_hero=(u['semantic']=='hero' and u.get('target_aspect_ratio',1.)<.68
                           and not u.get('preserve_layer_stack')
                           and u.get('source_area_share',0.)>=.10)
            if vertical_hero:
                block=_unit(u,cw,ch,scale)
                full_slot=_unit(u,cw,ch,scale,fill=True)
                current=_bounds(block);maximum=_bounds(full_slot)
                current_w,current_h=max(1.,current[2]-current[0]),max(1.,current[3]-current[1])
                maximum_w,maximum_h=max(1.,maximum[2]-maximum[0]),max(1.,maximum[3]-maximum[1])
                growth=min(1.30,maximum_w/current_w,maximum_h/current_h)
                if growth>1.001:block=_scale(block,growth)
            else:
                block = _unit(u,cw,ch,scale,fill=not (u['semantic'] in ('hero','art') or anchored_media
                                                       or u.get('preserve_layer_stack')))
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


def _text_visible_bounds(element):
    """Approximate ink bounds using the same measured line widths as layout."""
    ink=_render_text_ink(element)
    if ink is not None:
        x,y=float(element['x']),float(element['y'])
        angle=radians(float(element.get('rotation',0)))
        cx,cy=x+element['width']/2,y+element['height']/2
        points=[]
        for px,py in ((ink[0],ink[1]),(ink[2],ink[1]),(ink[0],ink[3]),(ink[2],ink[3])):
            dx,dy=x+px-cx,y+py-cy
            points.append((cx+dx*cos(angle)-dy*sin(angle),cy+dx*sin(angle)+dy*cos(angle)))
        return (min(v[0] for v in points),min(v[1] for v in points),
                max(v[0] for v in points),max(v[1] for v in points))
    height,_,_,line_widths,_=_text_info(element)
    align=element.get('text_align','left')
    lefts=[element['x']+(element['width']-line)/2 if align=='center'
           else element['x']+element['width']-line if align=='right' else element['x']
           for line in line_widths]
    return (min(lefts,default=element['x']),element['y'],
            max((left+line for left,line in zip(lefts,line_widths)),default=element['x']+element['width']),
            element['y']+min(element['height'],height))


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
        source_w=max(1.,unit.get('backing_source_width',1.))
        source_h=max(1.,unit.get('backing_source_height',1.))
        source_cw=max(1.,unit.get('backing_source_canvas_width',width))
        source_ch=max(1.,unit.get('backing_source_canvas_height',height))
        source_x=float(unit.get('backing_source_x',0.))
        source_y=float(unit.get('backing_source_y',0.))
        # Keep large panels anchored to their source-relative canvas footprint;
        # only expand when reflowed children need more room. Fitting tightly to
        # copy bounds turns a full content card into a thin strip.
        mapped=(source_x/source_cw*width,source_y/source_ch*height,
                (source_x+source_w)/source_cw*width,
                (source_y+source_h)/source_ch*height)
        x=max(0.,min(mapped[0],left-pad_x));y=max(0.,min(mapped[1],top-pad_y))
        r=min(float(width),max(mapped[2],right+pad_x));b=min(float(height),max(mapped[3],bottom+pad_y))
        panel=by_id.get(unit['items'][0]['id'])
        if panel is None:
            panel=deepcopy(unit['items'][0]);elements.append(panel);by_id[panel['id']]=panel
        panel_w=max(1.,r-x);panel_h=max(1.,b-y)
        if panel['type']=='image':
            # An inferred image backplate may follow its children, but stretching
            # the raster to the new card bounds distorts the supplied artwork.
            # Expand a source-proportional frame to the child region where room
            # permits, then keep the original pixels in contain mode.
            scale=min(max(panel_w/source_w,panel_h/source_h),width/source_w,height/source_h)
            panel_w=max(1.,source_w*scale);panel_h=max(1.,source_h*scale)
            center_x=(x+r)/2;center_y=(y+b)/2
            x=min(max(0.,center_x-panel_w/2),max(0.,width-panel_w))
            y=min(max(0.,center_y-panel_h/2),max(0.,height-panel_h))
            panel['fit']='contain'
        panel.update(x=x,y=y,width=panel_w,height=panel_h)
        if 'corner_radius' in panel:
            panel['corner_radius']=min(panel['width']/2,panel['height']/2,
                                       unit.get('backing_source_radius',panel['corner_radius'])
                                       *min(width/source_w,height/source_h))
    return elements


def _separate_qwen_text_collisions(elements,units,width,height):
    """Nudge proposal-only touching text units apart without resizing type."""
    repaired=deepcopy(elements)
    changes=[]
    by_id={e['id']:e for e in repaired}
    # Model boxes occasionally close a source gap between adjacent logo or
    # subtitle lines by only a few pixels. Repair that small error locally so
    # a useful proposal is not discarded wholesale. Move the later-painted,
    # smaller text unit; keep its font hierarchy and all source geometry.
    for _ in range(max(1,len(repaired))):
        collisions=_unintended_text_overlaps(repaired,units)
        if not collisions:break
        moved=False
        for first_id,second_id in collisions:
            first,second=by_id.get(first_id),by_id.get(second_id)
            if first is None or second is None:continue
            a_box,b_box=_text_visible_bounds(first),_text_visible_bounds(second)
            # Prefer moving the smaller text, breaking ties toward later z.
            candidates=sorted((first,second),key=lambda e:(
                float(e.get('font_size',1.)), -float(e.get('z_index',0)), e['id']))
            for item in candidates:
                other=second if item is first else first
                item_box=_text_visible_bounds(item);other_box=_text_visible_bounds(other)
                if item_box[3]<=other_box[1] or other_box[3]<=item_box[1]:continue
                gap=max(2.,min(float(width),float(height))*.003)
                down=max(0.,other_box[3]-item_box[1])+gap
                up=max(0.,item_box[3]-other_box[1])+gap
                unit=next((u for u in units if any(e['id']==item['id'] for e in u['items'])),None)
                group_ids={e['id'] for e in unit['items']} if unit else {item['id']}
                group=[by_id[eid] for eid in group_ids if eid in by_id]
                for dy in (down,-up):
                    if any(e['y']+dy<0 or e['y']+dy+e['height']>height for e in group):continue
                    for e in group:e['y']+=dy
                    if not _unintended_text_overlaps(repaired,units):
                        changes.append(dict(kind='minimal-text-separation',element_ids=sorted(group_ids),dy=round(dy,3)))
                        moved=True
                        break
                    for e in group:e['y']-=dy
                if moved:break
            if moved:break
        if not moved:break
    return repaired,changes


def _separate_qwen_unit_collisions(elements,units,width,height,gap=None):
    """Resolve small Qwen box collisions between otherwise separate units."""
    repaired=deepcopy(elements)
    changes=[]
    gap=float(gap if gap is not None else min(width,height)*.004)
    by_id={e['id']:e for e in repaired}
    for _ in range(max(1,len(units)*2)):
        records=[(u,_bounds([by_id[e['id']] for e in u['items'] if e['id'] in by_id]))
                 for u in units]
        moved=False
        for i,(a,ab) in enumerate(records):
            for b,bb in records[:i]:
                if _qwen_layered(a,b):continue
                dx=max(0.,min(ab[2]+gap-bb[0],bb[2]+gap-ab[0]))
                dy=max(0.,min(ab[3]+gap-bb[1],bb[3]+gap-ab[1]))
                if dx<=1e-4 or dy<=1e-4:continue
                # Move the smaller text-bearing unit; otherwise keep the
                # larger/source-art unit fixed and move the smaller sibling.
                a_text=any(e['type']=='text' for e in a['items'])
                b_text=any(e['type']=='text' for e in b['items'])
                size_a=max(1.,(ab[2]-ab[0])*(ab[3]-ab[1]))
                size_b=max(1.,(bb[2]-bb[0])*(bb[3]-bb[1]))
                candidates=sorted(((a,ab,a_text,size_a),(b,bb,b_text,size_b)),
                                  key=lambda item:(not item[2],item[3]))
                applied=False
                for unit,box,_,_ in candidates:
                    group=[by_id[e['id']] for e in unit['items'] if e['id'] in by_id]
                    # Preserve source reading direction when one axis already
                    # separates the boxes; for true overlap prefer the text's
                    # nearest available vertical clearance.
                    if ab[3]<=bb[1]+gap:
                        options=[(0.,-dy if unit is a else dy)]
                    elif bb[3]<=ab[1]+gap:
                        options=[(0.,dy if unit is a else -dy)]
                    else:
                        other=bb if unit is a else ab
                        center=(box[1]+box[3])/2
                        other_center=(other[1]+other[3])/2
                        direction=1. if center>=other_center else -1.
                        options=[(0.,direction*dy),
                                 (direction*dx,0.)]
                    for mx,my in options:
                        if any(e['x']+mx<0 or e['y']+my<0 or
                               e['x']+mx+e['width']>width or
                               e['y']+my+e['height']>height for e in group):
                            continue
                        for e in group:e['x']+=mx;e['y']+=my
                        next_records=[(u,_bounds([by_id[e['id']] for e in u['items'] if e['id'] in by_id]))
                                      for u in units]
                        overlap=False
                        for j,(u,ub) in enumerate(next_records):
                            for v,vb in next_records[:j]:
                                if _qwen_layered(u,v):continue
                                if not (ub[2]+gap<=vb[0]+1e-5 or vb[2]+gap<=ub[0]+1e-5 or
                                        ub[3]+gap<=vb[1]+1e-5 or vb[3]+gap<=ub[1]+1e-5):
                                    overlap=True;break
                            if overlap:break
                        if not overlap:
                            changes.append(dict(kind='minimal-unit-separation',
                                                element_ids=sorted(e['id'] for e in group),
                                                dx=round(mx,3),dy=round(my,3)))
                            moved=applied=True
                            break
                        for e in group:e['x']-=mx;e['y']-=my
                    if applied:break
                if moved:break
            if moved:break
        if not moved:break
    return repaired,changes


def _micro_snap_qwen_proposal(elements, units, width, height,source=None):
    """Project whole source containers before any leaf-level fitting."""
    if source is None:
        sw,sh=next((u['source_canvas'] for u in units if u.get('source_canvas')),(width,height))
        source={'canvas':{'width':sw,'height':sh},
                'elements':list({e['id']:e for u in units for e in u['items']}.values())}
    result,report=_structural_qwen_projection(elements,source,units,width,height)
    return result,report['containers']+report['axes']+report['font_calibration']


@lru_cache(maxsize=2048)
def _text_pair_ink_overlap_cached(payload):
    """Measure shared painted pixels; intersecting glyph envelopes are insufficient."""
    import json
    import numpy as np
    import skia
    from math import floor,ceil
    from cr_renderer.schema import TextElement
    from cr_renderer.text_utils import generate_map,render_text
    from renderer.converter.convert import convert_chameleon_to_crello
    a,b=json.loads(payload);ba,bb=_text_visible_bounds(a),_text_visible_bounds(b)
    x,y=floor(max(ba[0],bb[0])),floor(max(ba[1],bb[1]))
    w,h=ceil(min(ba[2],bb[2]))-x,ceil(min(ba[3],bb[3]))-y
    if min(w,h)<=0:return 0
    if w*h>16_000_000:return _overlap(ba,bb)
    manager=_layout_font_manager()
    if manager is None:return _overlap(ba,bb)
    masks=[]
    for e in (a,b):
        e=deepcopy(e)
        for run in e.get('style_runs') or []:run['end']=int(run['end'])-1
        example,_=convert_chameleon_to_crello({'canvas':{'width':1,'height':1},'elements':[e]},manager)
        prop=TextElement(uuid='',type='textElement',width=e['width'],height=e['height'],
            text=example['text'][0],fontSize=example['font_size'][0],font=example['font'][0],
            lineHeight=example['line_height'][0],textAlign=example['text_align'][0],
            capitalize=example['capitalize'][0],letterSpacing=example['letter_spacing'][0],
            boldMap=generate_map(example['font_bold'][0]),italicMap=generate_map(example['font_italic'][0]),
            colorMap=generate_map(['#ffffff']*len(example['text'][0])),lineMap=generate_map(example['text_line'][0]))
        surface=skia.Surface(w,h);canvas=surface.getCanvas();canvas.clear(0)
        canvas.translate(e['x']-x+e['width']/2,e['y']-y+e['height']/2)
        canvas.rotate(float(e.get('rotation',0)))
        canvas.translate(-e['width']/2,-e['height']/2)
        render_text(canvas,manager,prop)
        masks.append(surface.makeImageSnapshot().toarray()[:,:,3]>32)
    return int(np.count_nonzero(masks[0]&masks[1]))


def _text_pair_ink_overlap(a,b):
    import json
    if _overlap(_text_visible_bounds(a),_text_visible_bounds(b))<=1.:return 0
    return _text_pair_ink_overlap_cached(json.dumps([a,b],sort_keys=True))


def _unintended_text_overlaps(elements, units):
    """Return overlapping text pairs unless that overlap was present in source."""
    source_text={e['id']:e for unit in units for e in unit['items']
                 if e.get('type')=='text' and e.get('content','').strip()
                 and e['width']>0 and e['height']>0 and float(e.get('opacity',1))>0}
    intentional=set()
    source_values=list(source_text.values())
    for i,a in enumerate(source_values):
        for b in source_values[:i]:
            if _text_pair_ink_overlap(a,b)>1.:
                intentional.add(frozenset((a['id'],b['id'])))
    candidate=[e for e in elements if e.get('type')=='text' and e.get('content','').strip()
               and e['id'] in source_text]
    collisions=[]
    for i,a in enumerate(candidate):
        box_a=_text_visible_bounds(a)
        for b in candidate[:i]:
            pair=frozenset((a['id'],b['id']))
            if pair in intentional:
                continue
            if _overlap(box_a,_text_visible_bounds(b))>1. and _text_pair_ink_overlap(a,b)>1.:
                collisions.append(tuple(sorted((a['id'],b['id']))))
    return collisions


def _decorations(units,width,height,sw,sh,occupied):
    result=[]
    occupied=list(occupied)
    for u in units:
        if u.get('backing_panel'):
            backed=set(u.get('backed_ids',()))
            children=[e for e in occupied if e['id'] in backed]
            if children:
                panel=deepcopy(u['items'][0])
                _sync_backing_panels(children+[panel],[u],width,height)
                result.append(panel)
                continue
            # A standalone/fallback render still has to retain the panel when
            # no adapted children were placed. Fall through to normal sizing.
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
               # The two-column treatment only lays out headline and
               # secondary copy. A kicker or other semantic in the same copy
               # stack must stay on the ordinary complete-copy path below;
               # otherwise it silently disappears from this seed.
               and len(copy_units)==1+len(secondary)
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
            _set_font_size(e, _fit_text(e, natural + chosen * 0.2, height, chosen))
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
    separate_cta=bool(roles['action'] and roles['tag'] and height<=120 and width/height>=3.2)
    cta_w=min(cw*.23,max(cw*.16,available*1.5)) if separate_cta else 0.
    if separate_cta:copy_w=max(1.,copy_w-cta_w-gap)
    actions=roles['tag'] if separate_cta else roles['tag']+roles['action']
    action_h=(available*.22 if separate_cta else min(available*.42,max(available*.32,min(width,height)*.31))) if actions else 0
    action_row,action_used=_row(actions,copy_w,action_h,scale*1.25,gap*.9,align='left',equalize_control_height=True) if actions else ([],0)
    text_h=max(1.,available-action_used-(gap if actions else 0))
    copy=_copy(roles['copy'],copy_w,text_h,scale*1.3,gap*.45,align='left')
    logo=_fit_raw_block(logo,logo_w,available*.82) if logo else []
    logo_h=_raw_bounds(logo)[3] if logo else 0.
    copy=_fit_raw_block(copy,copy_w,text_h) if copy else []
    action_row=_fit_raw_block(action_row,copy_w,action_h) if action_row else []
    media_items=_fit_raw_block(media_items,media_w,media_available) if media_items else []
    output=[]
    if logo:
        output.extend(_move(logo,margin,top+(available-logo_h)/2))
    if copy:
        output.extend(_move(copy,copy_x,top))
    if action_row:
        action_y=top+text_h+gap
        output.extend(_move(action_row,copy_x,action_y))
    if separate_cta:
        cta,_=_banner_actions(roles['action'],cta_w,available*.76,scale,gap,cw)
        if cta:
            cta=_fit_raw_block(cta,cta_w,available*.76);box=_raw_bounds(cta)
            output.extend(_shift(cta,copy_x+copy_w+gap-box[0],top+(available-(box[3]-box[1]))/2-box[1]))
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
    peer_right=max(box[2] for box in peer_boxes)
    for u in units:
        if u['role'] not in ('tag','action'):continue
        items=[by_id[e['id']] for e in u['items'] if e['id'] in by_id]
        box=_bounds(items) if items else None
        if (u['role']=='action' and box and box[0]>=peer_right+1
                and abs((box[1]+box[3])/2-row_center)>target_h*.75):
            continue  # A separate action column is not part of the badge row.
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
                _set_font_size(e, _fit_text(e, e['width'], e['height'], max(e['font_size'], target)))
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
    before_violation=_constraint_violation(elements,units,width,height)
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
    if moved and _constraint_violation(elements,units,width,height)>before_violation+1e-6:
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
        images=[e for e in u['items'] if e['type']=='image']
        text=[e for e in u['items'] if e['type']=='text']
        # Asset semantics supplement geometry across source orientations.
        # Missing small media cannot become the largest available hero.
        missing=images and all(e.get('asset_id') and not (_asset_root(e)/e['asset_id']).is_file() for e in images)
        watermark=images and any('watermark' in e['id'].lower() or ('badge' in e.get('asset_id','').lower() and e.get('z_index',0)<=2) for e in images)
        if images and (r-x)*(b-y)<sw*sh*.12 and (watermark or missing and (y>sh*.5 or x>sw*.8)):
            u.update(role='footer',semantic='footer',footer_anchor='right' if (x+r)/2>sw/2 else 'left',missing_asset=bool(missing),confidence=.95)
            continue
        named_logo=images and any('logo' in e.get('asset_id','').lower() or e['id'].lower()=='logo' for e in images)
        side_logo=bool(sw/sh>=2.5 and images and not text and x<sw*.18 and (r-x)/max(1.,b-y)>=1.8 and (r-x)*(b-y)<sw*sh*.12 and all(e.get('fit','contain')=='contain' for e in images))
        if named_logo or side_logo:
            u.update(role='header',semantic='logo',confidence=.95)
            continue
        if text and not images and not u['label'] and b>=sh*.82 and max(e['font_size'] for e in text)<=min(10.,head_font*.40) and all(float(e.get('opacity',1.))<.8 or _rgba(e.get('color','')) and _rgba(e['color'])[3]<.8 for e in text):
            u.update(role='footer',semantic='footer',confidence=.9)
            continue
        # Repeated compact pills on the same source row are badges, rather
        # than multiple CTAs. A taller, distinct button remains an action.
        angle=sum(e.get('rotation',0) for e in u['items'])/len(u['items'])
        size=max((e.get('font_size',0) for e in u['items']),default=0)
        peers = [v for v in labels if v is not u and (abs(v['bounds'][1]-y) < sh*.025 or abs(sum(v['bounds'][::2])/2-(x+r)/2)<sw*.025)
                 and .85 < (v['bounds'][3]-v['bounds'][1])/max(1,b-y) < 1.15
                 and abs(sum(e.get('rotation',0) for e in v['items'])/len(v['items'])-angle)<=2
                 and .85 < max((e.get('font_size',0) for e in v['items']),default=0)/max(1,size) < 1.15]
        # A single label placed in a source corner is an accent even when it
        # is not rotated. Keep it anchored to that corner when the canvas
        # changes shape; a repeated row of similar labels remains a badge set.
        source_corner_label=bool(u['label'] and y<sh*.28 and (x<sw*.18 or r>sw*.8))
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
        if u['label'] and peers:
            u['role'] = 'tag'
        images = [e for e in u['items'] if e['type'] == 'image']
        small_image = images and not any(e['type'] == 'text' for e in u['items']) and (r-x)*(b-y) < sw*sh*.09
        near_top = (y+b)/2 < sh*.22
        near_corner = min(abs(x), abs(sw-r)) < sw*.12 and y < sh*.28
        relative_small = ((r-x)*(b-y)<sw*sh*.12 if len(image_units)==1 else max((_area(e) for e in images), default=0) < largest_image*.55)
        if u['role']=='header' and images:u['role']='visual'
        if small_image and (near_top or near_corner) and relative_small:
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
        u['source_area_share']=max(0.,(r-x)*(b-y))/(sw*sh)
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
            block=_fit_raw_block(block,w,available)
            box=_raw_bounds(block);bw,bh=box[2]-box[0],box[3]-box[1]
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
        output.extend(_shift(block,px,py))
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
    if width/height<=.4:
        return _skyscraper_seed(units,width,height,next((u.get('source_canvas') for u in units if u.get('source_canvas')),(width,height)))
    margin = min(width,height)*.025
    count = len(units)
    cols = max(1, min(count, round(sqrt(count*width/height))))
    rows = (count+cols-1)//cols
    cw, ch = (width-2*margin)/cols, (height-2*margin)/rows
    output = []
    for i, u in enumerate(sorted(units,key=_semantic_order)):
        block = _unit(u, cw*.92, ch*.92, scale, fill=False)
        # A unit can have transparent/unused frame padding beyond its painted
        # bounds. Fit and translate raw frames (including rotated corners),
        # rather than centering only the ink and exporting an off-canvas box.
        block = _fit_raw_block(block,cw*.92,ch*.92)
        l,t,r,b = _raw_bounds(block);w,h=r-l,b-t
        output.extend(_shift(block,margin+(i%cols)*cw+(cw-w)/2-l,
                             margin+(i//cols)*ch+(ch-h)/2-t))
    return output


def _layered(a, b):
    """Allow source-backed containment and visually meaningful layer overlap."""
    a_ids={e['id'] for e in a['items']}
    b_ids={e['id'] for e in b['items']}
    if (set(a.get('backed_ids',())) & b_ids
            or set(b.get('backed_ids',())) & a_ids
            or set(a.get('source_parent_ids',())) & b_ids
            or set(b.get('source_parent_ids',())) & a_ids
            or set(a.get('backed_by_ids',())) & set(b.get('backed_by_ids',()))):
        return True
    if a['role'] == 'decoration' or b['role'] == 'decoration':
        # A text-bearing corner callout gets separation from main content.
        if (a.get('corner_callout') and b['role']!='decoration') or (b.get('corner_callout') and a['role']!='decoration'):
            return False
        return True
    if any(e['type']=='text' for e in a['items']+b['items']):
        return False
    ar, br = a['bounds'], b['bounds']
    fraction = _overlap(ar,br)/max(1e-9,min((ar[2]-ar[0])*(ar[3]-ar[1]),(br[2]-br[0])*(br[3]-br[1])))
    return fraction > .12


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


def _qwen_layered(a, b):
    """Allow only source-supported layering in model-proposed layouts.

    Qwen candidates used to bypass separation for every text/visual pair. That
    let CTAs drift underneath later-painted foreground artwork and let copy
    collide with unrelated images while still passing feasibility. `_analyze`
    already records contained backplates and groups intentional source stacks;
    layouts retain those overlaps through `_layered` without granting this
    blanket exception.
    """
    if _layered(a,b):
        return True
    # Rotated full-bleed artwork often has a large rectangular frame even
    # when most of its pixels are transparent. Permit foreground content to
    # sit over a source-scale backdrop only when the artwork is a source
    # cluster and the cluster has an image layer painted below that content.
    # Foreground artwork remains protected by _source_occlusion_defects().
    visual=a if a.get('role')=='visual' else b if b.get('role')=='visual' else None
    foreground=b if visual is a else a if visual is b else None
    if visual is not None and visual.get('preserve_layer_stack'):
        front_items=foreground.get('items',())
        front_z=min((float(e.get('z_index',0)) for e in front_items),default=0.)
        source_w,source_h=visual.get('source_canvas',(0.,0.))
        if source_w>0 and source_h>0:
            full_backdrop=any(
                e.get('type')=='image'
                and float(e.get('z_index',0))<front_z
                and float(e.get('width',0))*float(e.get('height',0))>=source_w*source_h*.80
                for e in visual.get('items',()))
            if full_backdrop:
                return True
    return False


def _authored_image_bleed(element, original, canvas, width, height):
    """Keep image bleed within source crop, allowing bounded adjacent-edge transfer for flat accents."""
    if not original or element.get('type')!='image' or not canvas:return False
    if abs(float(element.get('rotation',0.)))>.5:return False
    sw,sh=canvas
    before=(max(0.,-original['x'])/original['width'],
            max(0.,-original['y'])/original['height'],
            max(0.,original['x']+original['width']-sw)/original['width'],
            max(0.,original['y']+original['height']-sh)/original['height'])
    if not any(before):return False
    after=(max(0.,-element['x'])/max(1.,element['width']),
           max(0.,-element['y'])/max(1.,element['height']),
           max(0.,element['x']+element['width']-width)/max(1.,element['width']),
           max(0.,element['y']+element['height']-height)/max(1.,element['height']))
    caps=list(before)
    # A flat accent that already bleeds at one edge can meet an adjacent
    # edge after aspect reflow. Bound that crop by its existing crop fraction;
    # photographs and interior objects do not acquire a new bleed permission.
    stats=_flat_image_asset_stats(str(_asset_root(original)/original.get('asset_id','')))
    if stats and stats[0]<=18:
        near=(original['x']<=.06*sw,original['y']<=.06*sh,
              original['x']+original['width']>=.94*sw,
              original['y']+original['height']>=.94*sh)
        caps=[max(b,max(before)) if edge else b for b,edge in zip(before,near)]
    return all(a<=b+1e-5 for a,b in zip(after,caps)) and max(after)<1.


def _record_canvas_bounds(items, originals, canvas, width, height):
    boxes=[]
    for e in items:
        b=_bounds([e])
        if _authored_image_bleed(e,originals.get(e['id']),canvas,width,height):
            b=(max(0.,b[0]),max(0.,b[1]),min(width,b[2]),min(height,b[3]))
        boxes.append(b)
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes)) if boxes else (0.,0.,0.,0.)


def _feasible(elements, units, width, height, strict_margins=True, gap_factor=.008,
              allow_text_visual_overlap=False):
    originals={e['id']:e for u in units for e in u['items']}
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    if any(not all(isfinite(v) for v in (e.get('x',0),e.get('y',0),e.get('width',0),e.get('height',0)))
           or min(e['width'],e['height'])<=0
           or ((e['x'] < -1e-5 or e['y'] < -1e-5
           or e['x']+e['width'] > width+1e-5 or e['y']+e['height'] > height+1e-5
           ) and not _authored_image_bleed(e,originals.get(e['id']),canvas,width,height))
           for e in elements):
        return False
    if _unintended_text_overlaps(elements,units):
        return False
    records = []
    for u,items in _records(elements,units):
        if allow_text_visual_overlap:
            boxes=[_text_visible_bounds(e) if e.get('type')=='text' else
                   _record_canvas_bounds([e],originals,canvas,width,height) for e in items]
            box=(min(b[0] for b in boxes),min(b[1] for b in boxes),
                 max(b[2] for b in boxes),max(b[3] for b in boxes))
        else:
            box=_record_canvas_bounds(items,originals,canvas,width,height)
        records.append((u,box))
    gap = min(width,height)*gap_factor
    for i,(u,b) in enumerate(records):
        if not all(isfinite(v) for v in b):
            return False
        margin = (-min(width,height)*.20 if _can_bleed(u) else
                  0 if not strict_margins or u['role']=='decoration' else
                  _unit_margin(b,width,height,u))
        if b[0]<margin-1e-5 or b[1]<margin-1e-5 or b[2]>width-margin+1e-5 or b[3]>height-margin+1e-5:
            return False
        for v,c in records[:i]:
            if (_qwen_layered(u,v) if allow_text_visual_overlap else _layered(u,v)):
                continue
            pair_gap=gap
            if allow_text_visual_overlap and canvas:
                a0,b0=u['bounds'],v['bounds']
                source_gap=max(0.,a0[0]-b0[2],b0[0]-a0[2],a0[1]-b0[3],b0[1]-a0[3])
                if 0<source_gap<.008*min(canvas):
                    pair_gap=min(gap,source_gap*min(width/canvas[0],height/canvas[1]))
            if not (b[2]+pair_gap<=c[0]+1e-5 or c[2]+pair_gap<=b[0]+1e-5 or b[3]+pair_gap<=c[1]+1e-5 or c[3]+pair_gap<=b[1]+1e-5):
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


def _constrain(elements, units, width, height, topology='nearest', strict_margins=True,
               allow_text_visual_overlap=False):
    """Cassowary placement with required bounds/separation and soft preferences.

    Non-overlap is disjunctive, not one linear equation. Each candidate chooses
    its separating axes first; Kiwi then solves that particular topology. An
    infeasible topology is rejected rather than silently weakening constraints.
    """
    try:
        import kiwisolver as kiwi
    except ImportError:
        return None
    elements=deepcopy(elements)
    # Alpha-cropped artwork can fit visibly while its transparent raw frame
    # is taller than the canvas. Translation cannot fix that contradiction.
    # Uniformly fit the complete unit before solving its placement.
    for unit,items in _records(elements,units):
        left=min(e['x'] for e in items);top=min(e['y'] for e in items)
        right=max(e['x']+e['width'] for e in items)
        bottom=max(e['y']+e['height'] for e in items)
        margin=min(width,height)*.055 if strict_margins and unit['role']!='decoration' else 0.
        factor=min(1.,(width-2*margin)/max(.001,right-left),
                   (height-2*margin)/max(.001,bottom-top))
        if factor<1.:
            scaled=_scale(items,factor)
            for item,fitted in zip(items,scaled):item.update(fitted)
    records = _records(elements, units)
    boxes = [_bounds(items) for _,items in records]
    solver = kiwi.Solver()
    xs = [kiwi.Variable(f'x{i}') for i in range(len(records))]
    ys = [kiwi.Variable(f'y{i}') for i in range(len(records))]
    # Qwen uses a smaller separation allowance, but units must still have a
    # positive gap and every element remains strictly on-canvas.
    gap_factor=.008 if strict_margins else .004
    gap = min(width,height)*gap_factor
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
            margin = (-min(width,height)*.20 if _can_bleed(u) else
                      0 if not strict_margins or u['role']=='decoration' else
                      _unit_margin(box,width,height,u))
            solver.addConstraint(xs[i]>=margin)
            solver.addConstraint(ys[i]>=margin)
            solver.addConstraint(xs[i]+w<=width-margin)
            solver.addConstraint(ys[i]+h<=height-margin)
            for item in records[i][1]:
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
                if (_qwen_layered(u,v) if allow_text_visual_overlap else _layered(u,v)):
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
    return output if _feasible(output,units,width,height,strict_margins,gap_factor,
                               allow_text_visual_overlap) else None


def _background_fit(element,source,width,height):
    """Fill the background by a uniform cover crop, never by pixel stretching."""
    asset=Path(_asset_root(element))/str(element.get('asset_id',''))
    if not asset.is_file():
        return 'cover' if element.get('fit')=='cover' else 'contain'
    return 'cover'


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
            if e['type']=='image':e['fit']=_background_fit(e,source,width,height)
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


@lru_cache(maxsize=2048)
def _text_visual_coverage_stats_cached(payload):
    """Intersect glyph pixels with art; retain peak line coverage as well."""
    import json
    import numpy as np
    import skia
    from PIL import Image
    from math import floor,ceil,radians,cos,sin
    from cr_renderer.schema import TextElement
    from cr_renderer.text_utils import generate_map,render_text
    from renderer.converter.convert import convert_chameleon_to_crello
    text,visual=json.loads(payload)
    box=_text_visible_bounds(text);x,y=floor(box[0]),floor(box[1]);w,h=ceil(box[2])-x,ceil(box[3])-y
    if min(w,h)<=0:return (0.,0.,0)
    manager=_layout_font_manager()
    if manager is None or w*h>16_000_000:
        fraction=_overlap(box,_bounds([visual]))/max(1,w*h)
        return (fraction,fraction,0)
    e=deepcopy(text)
    for run in e.get('style_runs') or []:run['end']=int(run['end'])-1
    example,_=convert_chameleon_to_crello({'canvas':{'width':1,'height':1},'elements':[e]},manager)
    prop=TextElement(uuid='',type='textElement',width=e['width'],height=e['height'],
        text=example['text'][0],fontSize=example['font_size'][0],font=example['font'][0],
        lineHeight=example['line_height'][0],textAlign=example['text_align'][0],
        capitalize=example['capitalize'][0],letterSpacing=example['letter_spacing'][0],
        boldMap=generate_map(example['font_bold'][0]),italicMap=generate_map(example['font_italic'][0]),
        colorMap=generate_map(['#ffffff']*len(example['text'][0])),lineMap=generate_map(example['text_line'][0]))
    surface=skia.Surface(w,h);canvas=surface.getCanvas();canvas.clear(0)
    canvas.translate(e['x']-x+e['width']/2,e['y']-y+e['height']/2)
    canvas.rotate(float(e.get('rotation',0)));canvas.translate(-e['width']/2,-e['height']/2)
    render_text(canvas,manager,prop)
    yy,xx=np.nonzero(surface.makeImageSnapshot().toarray()[:,:,3]>32)
    if not len(xx):return (0.,0.,0)
    try:
        if visual.get('type')=='shape':
            from renderer.converter.shapes import render_shape,shape_fill_alpha
            alpha=np.asarray(render_shape(visual).getchannel('A'))*shape_fill_alpha(visual)
        else:
            with Image.open(_asset_root(visual)/visual['asset_id']) as im:
                alpha=np.asarray(im.convert('RGBA').getchannel('A'))
    except (OSError,KeyError):
        fraction=_overlap(box,_bounds([visual]))/max(1,w*h)
        return (fraction,fraction,0)
    vh,vw=alpha.shape;ew,eh=visual['width'],visual['height']
    dx=xx+x+.5-(visual['x']+ew/2);dy=yy+y+.5-(visual['y']+eh/2)
    a=radians(float(visual.get('rotation',0)));lx=dx*cos(a)+dy*sin(a)+ew/2;ly=-dx*sin(a)+dy*cos(a)+eh/2
    inside=(lx>=0)&(lx<ew)&(ly>=0)&(ly<eh)
    fit=visual.get('fit','contain')
    if fit=='fill':px=lx*vw/ew;py=ly*vh/eh
    else:
        scale=(max if fit=='cover' else min)(ew/vw,eh/vh)
        px=(lx-(ew-vw*scale)/2)/scale;py=(ly-(eh-vh*scale)/2)/scale
    inside &= (px>=0)&(px<vw)&(py>=0)&(py<vh)
    active=np.zeros(len(xx),dtype=bool)
    active[inside]=alpha[py[inside].astype(int),px[inside].astype(int)]*float(visual.get('opacity',1.))>32
    peak=float(active.mean())
    if abs(float(text.get('rotation',0)))<1:
        rows=np.unique(yy)
        cuts=np.flatnonzero(np.diff(rows)>max(2.,float(text.get('font_size',12))*.20))+1
        for band in np.split(rows,cuts):
            mask=(yy>=band[0])&(yy<=band[-1])
            # Ignore a few anti-aliased pixels or a detached dot/accent.
            if mask.sum()>=16 and active[mask].sum()>=4:
                peak=max(peak,float(active[mask].mean()))
    return (float(active.mean()),peak,int(active.sum()))


def _text_visual_coverage_stats(text,visual):
    import json
    if _overlap(_text_visible_bounds(text),_bounds([visual]))<=0:return (0.,0.,0)
    return _text_visual_coverage_stats_cached(json.dumps([text,visual],sort_keys=True))


def _text_visual_coverage(text,visual):
    return _text_visual_coverage_stats(text,visual)[0]


def _source_occlusion_defects(elements,source):
    """Detect text newly buried by a later-painted visual layer.

    Compare visible text/image bounds with the authored design so intentional
    source overlaps remain valid. A candidate is defective only when a visual
    element painted after the text newly covers a substantial share of its
    visible bounds.
    """
    candidate={e['id']:e for e in elements}
    originals={e['id']:e for e in source.get('elements',[])}
    visuals=[e for e in source.get('elements',[]) if e.get('type') in ('image','shape')]
    defects=[]
    for text in elements:
        if text.get('type')!='text' or not text.get('content','').strip():continue
        original_text=originals.get(text['id'])
        if original_text is None:continue
        text_box=_text_visible_bounds(text)
        text_area=max(1e-9,(text_box[2]-text_box[0])*(text_box[3]-text_box[1]))
        source_text_box=_text_visible_bounds(original_text)
        source_text_area=max(1e-9,(source_text_box[2]-source_text_box[0])*(source_text_box[3]-source_text_box[1]))
        for original_visual in visuals:
            if original_visual.get('z_index',0)<=original_text.get('z_index',0):continue
            visual=candidate.get(original_visual['id'])
            if visual is None:continue
            source_fraction=_text_visual_coverage(original_text,original_visual)
            candidate_fraction=_text_visual_coverage(text,visual)
            allowed=max(source_fraction*1.5,source_fraction+.20)
            if candidate_fraction>allowed and candidate_fraction>.15:
                defects.append(dict(text_id=text['id'],visual_id=visual['id'],
                    fraction=round(candidate_fraction,4),allowed_fraction=round(allowed,4)))
    return defects


def _energy_terms(elements, units, source, width, height, desired,
                  allow_text_visual_overlap=False):
    records = _records(elements,units)
    originals={e['id']:e for e in source['elements']}
    canvas=(source['canvas']['width'],source['canvas']['height'])
    boxes = [_record_canvas_bounds(items,originals,canvas,width,height) for _,items in records]
    state=_aspect_state(source,width,height)
    terms = dict(overlap=0.,occlusion=0.,boundary=0.,edge_margin=0.,text_fit=0.,readability=0.,balance=0.,
                 hierarchy=0.,alignment=0.,spacing=0.,canvas_use=0.,source_fidelity=0.)
    terms.update(rag=0.,affinity=0.,whitespace=0.,background_busyness=0.,axial=0.,badge_flow=0.,footer_anchor=0.,hero_prominence=0.,cta_prominence=0.,brand_visibility=0.,column_alignment=0.,avoidable_wrap=0.,banner_action_stack=0.,banner_gap_evenness=0.,banner_centering=0.,stack_balance=0.,badge_alignment=0.,brand_media_spacing=0.,footer_visibility=0.,primary_clearance=0.,callout_migration=0.,source_order=0.,decoration_composition=0.)
    terms['occlusion']=sum(defect['fraction']-defect['allowed_fraction']
                           for defect in _source_occlusion_defects(elements,source))
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
            preferred=min(width,height)*(.055 if u['role'] in ('header','copy','tag','action','visual') else .045)
            clearances=(x,y,width-r,height-b)
            terms['edge_margin']+=sum((max(0.,preferred-clearance)/max(1.,preferred))**2
                                      for clearance in clearances)
        if u.get('inflow_callout'):
            terms['callout_migration']+=1.
        for (v,_),other in zip(records[:i],boxes[:i]):
            if not (_qwen_layered(u,v) if allow_text_visual_overlap else _layered(u,v)):
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
                flow_floor={'square':.32,'portrait':.27,'story':.25,'landscape':.28}[state['flow']]
                minimum=(max(source_height*.72,flow_floor) if source_share>=.12
                         else min(source_height*.85,flow_floor))
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


_ENERGY_WEIGHTS=dict(overlap=1000.,occlusion=180.,boundary=1500.,edge_margin=8.,text_fit=500.,readability=13.,
                     balance=18.,hierarchy=30.,alignment=16.,spacing=8.,
                     canvas_use=35.,source_fidelity=18.,rag=5.,affinity=18.,whitespace=5.,background_busyness=7.,axial=160.,badge_flow=70.,footer_anchor=100.,hero_prominence=210.,cta_prominence=85.,brand_visibility=45.,column_alignment=120.,avoidable_wrap=25.,banner_action_stack=180.,banner_gap_evenness=12.,banner_centering=80.,stack_balance=300.,badge_alignment=80.,brand_media_spacing=55.,footer_visibility=240.,primary_clearance=65.,callout_migration=8.,source_order=65.,decoration_composition=48.)


def _energy(elements,units,source,width,height,desired):
    return sum(value*_ENERGY_WEIGHTS[name] for name,value in
               _energy_terms(elements,units,source,width,height,desired).items())


def _objective_layout_cost(elements,units,source,width,height,desired,
                           allow_text_visual_overlap=False):
    """Score geometry and text defects without template or style preferences."""
    originals={e['id']:e for e in source['elements']}
    from engine.nested_structure import unpainted_ids
    unpainted=unpainted_ids(source)
    from engine.layout_contracts import issues
    cost=1000.*len(_unintended_text_overlaps(elements,units))
    cost+=100.*len(issues(elements,source,width,height,units))
    cost+=250.*len(_unintended_control_overlaps(elements,source))
    cost+=180.*sum(max(0.,d['fraction']-d['allowed_fraction'])
                   for d in _source_occlusion_defects(elements,source))
    # Inferred emblem similarity is reported separately by the selector. A
    # legitimate reflow can change its offsets without clipping or collisions;
    # folding that preference into this cost falsely labels clean layouts.
    for e in elements:
        if min(e['width'],e['height'])<=0:return 1e9
        if not _authored_image_bleed(e,originals.get(e['id']),
                                    (source['canvas']['width'],source['canvas']['height']),width,height):
            cost+=1500.*(max(0.,-e['x'],e['x']+e['width']-width)/width
                         +max(0.,-e['y'],e['y']+e['height']-height)/height)
        if e.get('type')=='text' and e['id'] not in unpainted:
            ink=_render_text_ink(e)
            if ink is None:
                h,w,_=_text(e);ink=(0.,0.,w,h)
            # Raster ink, not nominal line-height, determines clipping. Small
            # antialiasing overhangs receive the same one-pixel tolerance.
            bearing_x,bearing_y=_source_ink_overhang(e,originals.get(e['id']))
            cost+=500.*(max(0.,-ink[0]-bearing_x,ink[2]-round(e['width']))/max(1.,e['width'])
                         +max(0.,-ink[1]-bearing_y,ink[3]-round(e['height']))/max(1.,e['height']))
            cost+=500.*(max(0.,-e['x']-ink[0],e['x']+ink[2]-width)/width
                         +max(0.,-e['y']-ink[1],e['y']+ink[3]-height)/height)
            if e.get('max_lines'):
                cost+=500.*max(0.,_text(e)[2]-e['max_lines'])
    return cost


def _constraint_violation(elements, units, width, height):
    """Continuous repair cost, including required gaps that IoU alone misses."""
    element_records=_records(elements,units)
    records=[(u,_bounds(items)) for u,items in element_records]
    value=0.;gap=min(width,height)*.008
    for i,((u,items),(_,b)) in enumerate(zip(element_records,records)):
        for e in items:
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


def _even_source_peer_spacing(elements, units, width, height):
    """Offer equal gaps only for a source-evidenced, already-horizontal row."""
    changed=deepcopy(elements);by_id={e['id']:e for e in changed};bands={}
    for unit in units:
        if unit.get('peer_ids') and unit.get('role')=='tag':
            bands.setdefault(unit['peer_ids'],[]).append(unit)
    adjusted=False
    for peers in bands.values():
        if len(peers)<3:continue
        peers.sort(key=lambda u:u['bounds'][0])
        source_boxes=[u['bounds'] for u in peers]
        gaps=[b[0]-a[2] for a,b in zip(source_boxes,source_boxes[1:])]
        if min(gaps)<0 or max(gaps)-min(gaps)>max(2.,sum(gaps)/len(gaps)*.15):continue
        members=[[by_id[e['id']] for e in u['items'] if e['id'] in by_id] for u in peers]
        if any(not items for items in members):continue
        boxes=[_bounds(items) for items in members]
        centers=[(b[1]+b[3])/2 for b in boxes]
        if max(centers)-min(centers)>min(width,height)*.03:continue
        if any(b[0]<a[2] for a,b in zip(boxes,boxes[1:])):continue
        gap=(boxes[-1][2]-boxes[0][0]-sum(b[2]-b[0] for b in boxes))/(len(boxes)-1)
        if gap<0:continue
        source_height=sum(b[3]-b[1] for b in source_boxes)/len(source_boxes)
        target_height=sum(b[3]-b[1] for b in boxes)/len(boxes)
        source_ratio=(sum(gaps)/len(gaps))/max(1e-6,source_height)
        gap=min(gap,max(source_ratio,.45)*target_height)
        x=boxes[0][0]
        # Use a shared source left edge only when the original explicitly
        # establishes it, and the corresponding target copy is available.
        copy=[u for u in units if u.get('role')=='copy' and
              abs(u['bounds'][0]-source_boxes[0][0])<=max(4.,source_height*.12)]
        row_width=sum(b[2]-b[0] for b in boxes)+gap*(len(boxes)-1)
        if copy:
            target_items=[by_id[e['id']] for u in copy for e in u['items'] if e['id'] in by_id]
            if target_items:x=_bounds(target_items)[0]
        else:
            # Preserve a source-evidenced column, not the compacted row's old
            # left edge. Median peer centers retain the authored axis even
            # with unequal outer widths; every control still moves rigidly.
            from statistics import median
            axis=median((b[0]+b[2])/2 for b in source_boxes)
            columns=[u for u in units if u.get('role')=='copy' and
                     abs((u['bounds'][0]+u['bounds'][2])/2-axis)<=4. and
                     all(e.get('text_align','left')=='center'
                         for e in u['items'] if e['type']=='text')]
            targets=[];target_boxes=[];target_alignments=[]
            for u in columns:
                items=[by_id[e['id']] for e in u['items'] if e['id'] in by_id]
                if items:
                    b=_bounds(items);targets.append((b[0]+b[2])/2);target_boxes.append(b)
                    target_alignments.append({e.get('text_align','left') for e in items if e['type']=='text'})
            # Keep a shared center when it survives reflow. Otherwise an
            # explicitly left/right-aligned target copy column can establish
            # its new edge; unequal copy widths must not hide that evidence.
            tolerance=max(2.,width*.005)
            if len(targets)>=2 and max(targets)-min(targets)<=tolerance:
                cursor=0.;centers=[]
                for b in boxes:
                    centers.append(cursor+(b[2]-b[0])/2)
                    cursor+=b[2]-b[0]+gap
                aligned=median(targets)-median(centers)
                if 0.<=aligned<=width-row_width:x=aligned
            elif len(targets)>=2:
                for alignment,edge in (('left',0),('right',2)):
                    axes=[b[edge] for b in target_boxes]
                    if (all(a=={alignment} for a in target_alignments)
                            and max(axes)-min(axes)<=tolerance):
                        aligned=median(axes)-(row_width if alignment=='right' else 0.)
                        if 0.<=aligned<=width-row_width:x=aligned
                        break
        x=max(0.,min(x,width-row_width))
        for items,box in zip(members,boxes):
            dx=x-box[0]
            if abs(dx)>1e-6:_shift(items,dx,0.);adjusted=True
            x+=box[2]-box[0]+gap
    return changed if adjusted else None


def _peer_gap_cost(elements, units):
    """Soft row-spacing cost, only for peers compactly spaced in the source."""
    by_id={e['id']:e for e in elements};bands={};cost=0.
    for u in units:
        if u.get('role')=='tag' and u.get('peer_ids'):
            bands.setdefault(u['peer_ids'],[]).append(u)
    for peers in bands.values():
        if len(peers)<3:continue
        peers.sort(key=lambda u:u['bounds'][0]);source=[u['bounds'] for u in peers]
        source_gaps=[b[0]-a[2] for a,b in zip(source,source[1:])]
        h=sum(b[3]-b[1] for b in source)/len(source)
        if min(source_gaps)<0 or max(source_gaps)-min(source_gaps)>max(2.,sum(source_gaps)/len(source_gaps)*.15):continue
        if sum(source_gaps)/len(source_gaps)>.8*h:continue
        members=[[by_id[e['id']] for e in u['items'] if e['id'] in by_id] for u in peers]
        if any(not row for row in members):continue
        boxes=[_bounds(row) for row in members];height=sum(b[3]-b[1] for b in boxes)/len(boxes)
        centers=[(b[1]+b[3])/2 for b in boxes]
        if max(centers)-min(centers)>max(2.,height*.2):continue
        gaps=[b[0]-a[2] for a,b in zip(boxes,boxes[1:])]
        if min(gaps)<0:continue
        cost+=min(2.,max(0.,max(gaps)-min(gaps)-2.)/max(1.,height))
        cost+=min(2.,max(0.,sum(gaps)/len(gaps)-1.5*height)/max(1.,height))
    return cost


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
    from engine.source_structure import copy_axis_contracts, apply_copy_axis_contracts
    column_contracts=copy_axis_contracts(current,units,width,height)
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
            _set_font_size(e, _fit_text(e, e['width'], height * 0.7, max(1, e['font_size'] * rng.choice((1.0, 1.04)))))
            e['height']=_text(e)[0]
            _tighten_text(e)
            new_box=_bounds(proposal);new_w=new_box[2]-new_box[0]
            align=e.get('text_align','left')
            left=(x+r-new_w)/2 if align=='center' else r-new_w if align=='right' else x
            proposal=_move(proposal,left,y)
        else:
            factor=rng.choice((.95,1.05))
            if u['role']=='visual':
                source_width=max(1.,u['bounds'][2]-u['bounds'][0])
                source_height=max(1.,u['bounds'][3]-u['bounds'][1])
                current_width=max(1.,r-x);current_height=max(1.,b-y)
                visual_scale_limit=(.95 if u.get('source_area_share',1.)<.12 else 1.20)
                maximum_factor=min(source_width*desired*visual_scale_limit/current_width,
                                   source_height*desired*visual_scale_limit/current_height)
                factor=min(factor,maximum_factor)
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
        apply_copy_axis_contracts(changed,column_contracts,units,width,height)
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


def _contrast_floor_deficit(source_value,target_value):
    """Do not relocate weak authored type for a sub-0.4 contrast variation."""
    if source_value<3. and target_value>=max(1.75,source_value-.40):return 0.
    return max(0.,source_value*.90-target_value)


def _qwen_contrast_safe(candidate, baseline, qwen_primary=False, source_contrast=None):
    """Compare soft warnings against available alternatives; invisible ink is hard."""
    if candidate is None or candidate.get('invisible_text') or 'error' in candidate:
        return False
    baselines=baseline if isinstance(baseline,(list,tuple)) else [baseline]
    baselines=[e for e in baselines if e is not None and not e.get('invisible_text') and 'error' not in e]
    if candidate.get('physical_eligible') is not None:
        return candidate['physical_eligible'] and not candidate.get('contrast_regressions')
    if not baselines:
        return True
    weak=min(len(e.get('low_contrast',[])) for e in baselines)
    cost=min(e.get('cost',0.) for e in baselines)
    if (len(candidate.get('low_contrast',[]))<=weak
            or candidate.get('cost',0.)<=cost+1.25):
        return True
    # Don't reject a better reflow solely for retaining text that was already
    # low-contrast in the source. Require every weak output label to be weak
    # in the rendered source too, and keep its measured glyph contrast close
    # to that source value. New contrast failures remain disqualifying.
    if source_contrast:
        metrics=candidate.get('contrast_by_text',{})
        weak_ids=candidate.get('low_contrast',[])
        inherited=bool(weak_ids) and all(
            eid in source_contrast
            and source_contrast[eid].get('near_threshold')
            and metrics.get(eid,{}).get('lower_quintile') is not None
            and metrics[eid]['lower_quintile'] >=
                max(1.0,source_contrast[eid].get('lower_quintile',0.0)*.88)
            for eid in weak_ids)
        if inherited:
            return True
    return False


def _needs_contrast_repair(entry):
    """Do not relocate authored groups for a marginal contrast threshold miss."""
    elements={e['id']:e for e in entry.get('candidate',{}).get('elements',[])}
    measured=entry.get('contrast_by_text',{})
    for eid in entry.get('low_contrast',[]):
        values=measured.get(eid,{})
        minimum,lower=values.get('minimum'),values.get('lower_quintile')
        if minimum is None or lower is None:return True
        e=elements.get(eid,{})
        target=3. if e.get('font_size',0)>=24 or e.get('font_weight')=='bold' and e.get('font_size',0)>=18 else 4.5
        if minimum<3.0 or lower<target*.90:return True
    return bool(entry.get('invisible_text') or entry.get('busy_background'))


def _visual_rerank(candidates, backgrounds, units, source, width, height, desired):
    """Rendered diagnostics plus an optional learned SigLIP perceptual judge.

    Analyze top geometry-safe candidates. A text-free diagnostic rendition lets
    us inspect actual backing pixels without measuring the glyph's own color.
    When the baked SigLIP weights are present, their image/text score breaks
    ties inside the measured defect band. Otherwise deterministic ranking is
    used. Either way this stage
    retains existing candidates and can add bounded source-local Qwen contrast
    variants. Every added layout passes geometry checks and is rendered again.
    """
    if not candidates or width*height>16000000:
        return candidates[0],dict(used=False,reason='Canvas exceeds diagnostic raster budget')
    try:
        from renderer.converter.convert import render_design_to_png
        from PIL import Image, ImageChops
        import numpy as np
    except ImportError as exc:
        return candidates[0],dict(used=False,reason=f'Renderer unavailable: {exc}')
    diagnostic_assets=Path(__file__).resolve().parents[1]/'brand_kit'/'assets'
    render_diagnostic=render_design_to_png
    if (os.environ.get('MAGICANVAS_DATASET_RENDERING','').strip().lower() in ('1','true','yes','on')
            and _CRELLO_FONTS_PICKLE.is_file()):
        try:
            from scripts.dataset_renderer import render_design_to_png as render_crello_dataset
            data_root=Path(__file__).resolve().parents[1]/'data'
            dataset_roots=[data_root/f'crello_{split}'
                           for split in ('train','validation','test')]
            asset_ids=[str(e.get('asset_id','')) for e in source.get('elements',[])
                       if e.get('type')=='image' and e.get('asset_id')]
            existing_roots=[root for root in dataset_roots if root.is_dir()]
            dataset_root=max(existing_roots,
                             key=lambda root:sum((root/asset_id).is_file() for asset_id in asset_ids),
                             default=None)
            if dataset_root is not None:
                render_diagnostic=lambda doc,assets,path,max_size=640: render_crello_dataset(
                    doc,dataset_root,path,_CRELLO_FONTS_PICKLE,max_size=max_size,adjust_tight_text_widths=False)
                diagnostic_assets=dataset_root
        except ImportError:
            pass
    entries=[]
    state=_aspect_state(source,width,height)
    ratio_shift=max(abs(state['delta']),1/max(1e-9,abs(state['delta'])))
    has_reflow=any(c['name']!='proportional' for c in candidates)
    shortlist=[];families=set()
    qwen_candidates=[c for c in candidates if c['name'].startswith('qwen-') and (not c.get('macro_locked') or c.get('learned_macro'))]
    for candidate in candidates:
        name=candidate['name']
        if name.startswith('qwen-'):continue
        if ratio_shift>=1.6 and has_reflow and name=='proportional':continue
        family=re.sub(r'-\d+(?:\.\d+)?$','',name)
        if family in families:continue
        families.add(family);shortlist.append(candidate)
        if len(shortlist)>=(4 if qwen_candidates else 8):break
    # Keep combined font+position siblings available to the visual judge.
    shortlist.extend(c for c in candidates if c.get('measured_stack') and c not in shortlist)
    shortlist.extend(c for c in candidates if c.get('source_attached') and c not in shortlist)
    shortlist.extend(qwen_candidates[:6])
    shortlist.extend(c for c in qwen_candidates
                     if c['name'] in ('qwen-pre-container-font-fitted','qwen-raw-constraint-repaired') and c not in shortlist)
    # A preserved source-stack layout can have a higher geometric objective
    # because it retains authored composition. Always compare it visually.
    shortlist.extend(c for c in qwen_candidates
                     if c['name']=='qwen-layer-preserved' and c not in shortlist)
    # Typography-floor siblings survive the family cap for direct comparison.
    shortlist.extend(c for c in candidates
                     if c['name'].startswith(('qwen-readable-text-floor',
                                               'readable-text-floor-'))
                     if c not in shortlist)
    shortlist.extend(c for c in qwen_candidates
                     if c['name']=='qwen-source-contained-proposal' and c not in shortlist)
    # Reserve one representative of each available independent path before
    # rendering. Macro proposals are generated only when their quality gate
    # warrants them; never manufacture a candidate for an inapplicable family.
    for predicate in (
        lambda c: c['name'] in ('qwen-pre-container-font-fitted', 'qwen-raw-constraint-repaired'),
        lambda c: c['name'].startswith('macro-band-reflow'),
        lambda c: c['name'].startswith('source-attached-proportional'),
        lambda c: any(part in c['name'] for part in ('semantic-stack', 'editorial-split')),
    ):
        representative=next((c for c in candidates if predicate(c)),None)
        if representative is not None and representative not in shortlist:
            shortlist.append(representative)
    # Compare a source-evidenced spacing repair as an additional candidate;
    # leave the incumbent intact and require the normal geometry/ink checks.
    for parent in shortlist[:4]:
        spaced=_even_source_peer_spacing(parent['elements'],units,width,height)
        if (spaced is not None and
                _feasible(spaced,units,width,height,strict_margins=False,gap_factor=.004,
                          allow_text_visual_overlap=True) and
                not _unintended_text_overlaps(spaced,units)):
            shortlist.append(dict(name=parent['name']+'-source-peer-spacing',elements=spaced,
                                  energy=_energy(spaced,units,source,width,height,desired)))
    # Reserve functional families explicitly; energy ordering must not let
    # several near-identical Qwen siblings evict the aspect-bucket baseline.
    required_families={}
    primary=next((c for name in ('qwen-font-fitted','qwen-proposal','qwen-original','qwen-raw-constraint-repaired')
                  for c in qwen_candidates if c['name']==name),None)
    if primary is None and qwen_candidates:primary=qwen_candidates[0]
    if primary is not None:
        locked=dict(primary,name='qwen-container-locked-font-fitted')
        shortlist.insert(0,locked)
        required_families['qwen-container-locked-font-fitted']=locked['name']
    bucket_prefix=('deterministic-aspect-reflow','inline-banner','editorial','split','left-hero','dense-short') if width/height>=1.35 else ('deterministic-aspect-reflow','semantic-stack','copy-first','visual-first')
    bucket=next((c for c in candidates if not c['name'].startswith('qwen-')
                 and c['name'].startswith(bucket_prefix)),None)
    bucket=next((c for c in candidates if c['name'].startswith('deterministic-aspect-reflow-')),bucket)
    if bucket is not None:
        if bucket not in shortlist:shortlist.append(bucket)
        required_families['deterministic-aspect-reflow']=bucket['name']
    if ratio_shift<1.35:
        proportional=next((c for c in candidates if c['name']=='proportional'),None)
        if proportional is not None:
            centered=dict(proportional,name='proportional-centered')
            shortlist.append(centered)
            required_families['proportional-centered']=centered['name']
    from engine.qwen_refine import relationship_variant
    repaired_signatures=set()
    for name in ('qwen-proposal','qwen-font-fitted','qwen-original'):
        parent=next((c for c in qwen_candidates if c['name']==name),None)
        if parent is None:continue
        repaired,structure=_structural_qwen_projection(parent['elements'],source,units,width,height)
        changes=structure['containers']+structure['font_calibration']+structure['axes']
        before={e['id']:e for e in parent['elements']}
        error=sum(abs(e[k]-before[e['id']][k]) for e in repaired for k in ('x','y','width','height'))/max(1.,len(repaired)*(width+height))
        signature=tuple((e['id'],*(round(float(e.get(k,0)),2)
                        for k in ('x','y','width','height','font_size'))) for e in repaired)
        if error<.006 or signature in repaired_signatures:continue
        if (not _feasible(repaired,units,width,height,strict_margins=False,
                          gap_factor=.004,allow_text_visual_overlap=True)
                or _unintended_text_overlaps(repaired,units)):continue
        repaired_signatures.add(signature)
        shortlist.append(dict(name='qwen-source-relationships-'+name.removeprefix('qwen-'),
            elements=repaired,energy=_energy(repaired,units,source,width,height,desired),
            variant_of=name,relationship_repairs=changes,structure=structure))
    relationship_context=_visual_relationship_context(units,source,width,height)
    primary_regressions=[]
    from engine.render_cache import DiagnosticRenderCache
    render_mode='dataset' if diagnostic_assets.name.startswith('crello_') else 'brand'
    with TemporaryDirectory(prefix='chameleon-rank-') as directory, DiagnosticRenderCache(render_diagnostic,diagnostic_assets,render_mode) as raster_cache:
        reference_image=None;source_under=None;source_contrast={};clean_zones=[]
        try:
            reference_path=Path(directory)/'source.png'
            raster_cache.render(source,reference_path)
            with Image.open(reference_path) as image:
                reference_image=image.convert('RGB')
            if qwen_candidates or any(c.get('macro_locked') for c in candidates):
                source_backing=dict(canvas=source['canvas'],elements=[e for e in source['elements'] if e['type']!='text'])
                source_backing_path=Path(directory)/'source-backing.png'
                raster_cache.render(source_backing,source_backing_path)
                with Image.open(source_backing_path) as image:
                    source_under=image.convert('RGB')
                # Measure whether source text already has low contrast at its
                # authored location. This lets the selector preserve an
                # existing style without tolerating newly introduced failures.
                source_ink=ImageChops.difference(reference_image,source_under).convert('L')
                sw=float(source['canvas']['width']);sh=float(source['canvas']['height'])
                source_under_array=np.asarray(source_under);source_ink_array=np.asarray(source_ink)
                for e in source.get('elements',[]):
                    if e.get('type')!='text' or not e.get('content','').strip():continue
                    color=_rgba(e.get('color',''))
                    if not color or abs(float(e.get('rotation',0)))>1:continue
                    left=max(0,round(e['x']/sw*source_under.width))
                    top=max(0,round(e['y']/sh*source_under.height))
                    right=min(source_under.width,max(left+1,round((e['x']+e['width'])/sw*source_under.width)))
                    bottom=min(source_under.height,max(top+1,round((e['y']+e['height'])/sh*source_under.height)))
                    if right<=left or bottom<=top:continue
                    from engine.raster_metrics import glyph_statistics
                    stats=glyph_statistics(source_under_array[top:bottom,left:right],source_ink_array[top:bottom,left:right],color,e.get('opacity',1))
                    if stats is None:continue
                    lower=stats['lower_quintile']
                    threshold=3. if e.get('font_size',0)>=24 or e.get('font_weight')=='bold' and e.get('font_size',0)>=18 else 4.5
                    source_contrast[e['id']]=dict(lower_quintile=lower,p05=stats['p05'],
                        low=lower<threshold*.97,near_threshold=lower<threshold*1.20)
        except Exception:
            # Identity continuity is a soft cue; candidate diagnostics remain
            # useful if the source render fails for an unrelated renderer issue.
            pass
        try:
            from engine.clean_zones import source_zones
            source_background=dict(canvas=source['canvas'],elements=[e for e in source['elements'] if e['id'] in {b['id'] for b in backgrounds}])
            source_background_path=Path(directory)/'source-background-only.png'
            raster_cache.render(source_background,source_background_path)
            with Image.open(source_background_path) as image:
                clean_zones=source_zones(source,image.convert('RGB'),source_contrast,units)
        except (OSError,ValueError,ImportError):
            clean_zones=[]
        for index,candidate in enumerate(shortlist):
            from engine.source_structure import copy_axis_contracts, apply_copy_axis_contracts
            # These candidates already own a complete container arrangement.
            # Do not acquire a competing leaf-level alignment contract.
            column_contracts=([] if candidate.get('source_attached') or candidate.get('macro_locked')
                              else copy_axis_contracts(candidate["elements"],units,width,height))
            if candidate.get('macro_locked'):
                candidate=dict(candidate,elements=deepcopy(candidate['elements']))
                shortlist[index]=candidate
            if not candidate.get('macro_locked'):
                from engine.layout_contracts import repair
                repaired=(deepcopy(candidate['elements']) if candidate.get('source_attached')
                          else repair(candidate['elements'],source,units,width,height))
                from engine.qwen_refine import snap_source_axes
                if not candidate.get('source_attached'):
                    repaired,_=snap_source_axes(repaired,source,width,height)
                if candidate.get('measured_stack'):
                    from engine.structural_reflow import resolve_stack
                    repaired=resolve_stack(repaired,source,units,width,height)
                if candidate.get('source_attached'):
                    repaired,contract_fonts=_calibrate_text_boxes(repaired,source,width,height)
                    from engine.source_structure import fit_groups_to_canvas
                    from engine.layout_contracts import _primary_axis
                    _primary_axis(repaired,source,units,width,height,apply=True)
                    fit_groups_to_canvas(repaired,source,width,height)
                else:
                    repaired,contract_fonts=_finish_candidate_text(repaired,source,units,width,height,candidate.get('measured_stack',False))
                candidate=dict(candidate,elements=repaired,contract_font_calibration=contract_fonts)
                shortlist[index]=candidate
            # Every candidate, including raw Qwen and macro layouts, obeys the
            # same aspect policy before its actual pixels are scored.
            from engine.geometry_policy import preserve_aspects
            from engine.layout_contracts import _separate_art
            repaired=preserve_aspects(candidate['elements'],source,width,height)
            # Source-attached and macro layouts own whole-container geometry.
            # Never make their leaves independently movable after calibration.
            if not candidate.get('source_attached') and not candidate.get('macro_locked'):
                _separate_art(repaired,source,width,height,new_contacts_only=True)
            # Last geometry pass before rasterization: keep compact, source-
            # evidenced peers together. Reject a move that introduces a new
            # geometry violation; never alter children within each control.
            column_repairs=apply_copy_axis_contracts(repaired,column_contracts,units,width,height)
            candidate=dict(candidate,copy_axis_contracts=column_contracts,copy_axes_preserved=len(column_repairs))
            spaced=_even_source_peer_spacing(repaired,units,width,height)
            if spaced is not None and not _unintended_text_overlaps(spaced,units):
                before={e['id']:e for e in repaired};after={e['id']:e for e in spaced}
                records=[]
                for u in units:
                    old=[before[e['id']] for e in u['items'] if e['id'] in before]
                    new=[after[e['id']] for e in u['items'] if e['id'] in after]
                    if old and new:records.append((u,_bounds(old),_bounds(new)))
                def overlap(a,b):
                    return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
                introduces_contact=any(not _layered(u,v) and overlap(a2,b2)>overlap(a1,b1)+1e-6
                    for i,(u,a1,a2) in enumerate(records) for v,b1,b2 in records[:i])
                if not introduces_contact:repaired=spaced
            candidate=dict(candidate,elements=repaired)
            shortlist[index]=candidate
            candidate_backgrounds=candidate.get('backgrounds',backgrounds)
            doc=_assemble(candidate['elements'],candidate_backgrounds,source,width,height)
            full_path=Path(directory)/f'{index}.png'
            backing_path=Path(directory)/f'{index}-backing.png'
            try:
                backing=dict(canvas=doc['canvas'],elements=[e for e in doc['elements'] if e['type']!='text'])
                warnings,_=raster_cache.render_many([(doc,full_path),(backing,backing_path)])
                with Image.open(full_path) as image:
                    full=image.convert('RGB')
                with Image.open(backing_path) as image:
                    under=image.convert('RGB')
                ink=ImageChops.difference(full,under).convert('L')
                import numpy as np
                under_array=np.asarray(under);ink_array=np.asarray(ink)
                cost,checked,weak,busy,invisible=0.,0,[],[],[]
                contrast_by_text={}
                from engine.nested_structure import unpainted_ids
                source_unpainted=unpainted_ids(source)
                for e in doc['elements']:
                    if e['type']!='text' or e['id'] in source_unpainted or not e.get('content','').strip():
                        continue
                    color=_rgba(e.get('color',''))
                    if not color or abs(e.get('rotation',0))>1:
                        continue
                    left=max(0,round(e['x']/width*under.width));top=max(0,round(e['y']/height*under.height))
                    right=min(under.width,max(left+1,round((e['x']+e['width'])/width*under.width)))
                    bottom=min(under.height,max(top+1,round((e['y']+e['height'])/height*under.height)))
                    if right<=left or bottom<=top:
                        continue
                    under_crop=under.crop((left,top,right,bottom))
                    ink_crop=ink.crop((left,top,right,bottom))
                    from engine.raster_metrics import glyph_statistics
                    stats=glyph_statistics(under_array[top:bottom,left:right],ink_array[top:bottom,left:right],color,e.get('opacity',1))
                    if stats is None:
                        cost+=3.;invisible.append(e['id']);weak.append(e['id']);checked+=1
                        contrast_by_text[e['id']]=dict(minimum=None,lower_quintile=None,ink_pixels=0)
                        continue
                    target=3 if e['font_size']>=24 or e.get('font_weight')=='bold' and e['font_size']>=18 else 4.5
                    source_floor=source_contrast.get(e['id'],{}).get('lower_quintile')
                    if source_floor is not None:target=min(target,max(1.,source_floor*.90))
                    low=stats['lower_quintile'];minimum=stats['minimum']
                    contrast_by_text[e['id']]=dict(minimum=round(minimum,3),lower_quintile=round(low,3),p05=round(stats['p05'],3),background_rgb_jump=stats['background_rgb_jump'],ink_pixels=stats['ink_pixels'])
                    deficit=max(0,(target-low)/target)
                    cost+=deficit*5
                    # The 20th percentile represents contrast across a
                    # meaningful share of glyph pixels. A single anti-aliased
                    # edge pixel should not trigger a repair when the glyph
                    # itself clears its contrast target.
                    if deficit>.03:
                        weak.append(e['id'])
                    roughness=stats['roughness']
                    from PIL import ImageFilter
                    edge_crop=under_crop.convert('L').filter(ImageFilter.FIND_EDGES)
                    edge_data=edge_crop.tobytes()
                    edge_density=float(np.frombuffer(edge_data,dtype=np.uint8).mean()/255.) if edge_data else 0.
                    luma_std=stats['luma_std']
                    if roughness>.08 or edge_density>.07 or (luma_std>.15 and edge_density>.04):
                        busy.append(e['id'])
                    checked+=1
                cost=cost/max(1,checked)+(8.0 if invisible else 0.0)
                occlusion_defects=_source_occlusion_defects(doc['elements'],source)
                structural_errors=[]
                source_by_id={e['id']:e for e in source['elements']}
                if (set(source_by_id)!={e['id'] for e in doc['elements']}
                        or len(doc['elements'])!=len(source_by_id)):
                    structural_errors.append('element IDs differ')
                background_ids={e['id'] for e in backgrounds}
                for element in doc['elements']:
                    if (element['x']<-.01 or element['y']<-.01
                            or element['x']+element['width']>width+.01
                            or element['y']+element['height']>height+.01):
                        structural_errors.append(f"{element['id']}: raw frame outside canvas")
                    original=source_by_id.get(element['id'])
                    if (original and element.get('type')=='image' and element['id'] not in background_ids
                            and original.get('fit')!='cover' and min(original['width'],original['height'],element['height'])>0
                            and abs((element['width']/element['height'])/(original['width']/original['height'])-1)>.05):
                        structural_errors.append(f"{element['id']}: image aspect distortion")
                structural_errors.extend(_backing_containment_defects(doc['elements'],source))
                entries.append(dict(candidate=candidate,cost=cost,checked=checked,low_contrast=weak,busy_background=busy,invisible_text=invisible,contrast_by_text=contrast_by_text,occlusion_defects=occlusion_defects,structural_errors=structural_errors,warnings=warnings,image=full.copy()))
                if not candidate.get('macro_locked') and clean_zones and 'clean-zone-fitted' not in candidate['name'] and sum('clean-zone-fitted' in c['name'] for c in shortlist)<3:
                    from engine.clean_zones import fitted_variant
                    fitted=fitted_variant(candidate['elements'],source,units,
                        [e for e in doc['elements'] if e['id'] in {b['id'] for b in backgrounds}],
                        clean_zones,contrast_by_text,width,height,diagnostic_assets)
                    if fitted:
                        repaired,change=fitted
                        shortlist.append(dict(candidate,name=candidate['name']+'-clean-zone-fitted',
                                              elements=repaired,clean_zone_repairs=[change]))
                regressions=[eid for eid in weak if not source_contrast.get(eid,{}).get('low',True)]
                if candidate['name']=='qwen-container-locked-font-fitted':
                    primary_regressions=regressions
                # Independent deterministic fallbacks need the same measured
                # repair opportunity as the learned family. Never recursively
                # repair a repair, and retain the primary learned slot.
                if (not candidate.get('macro_locked') and regressions and not invisible and 'contrast-repaired' not in candidate['name']
                        and (candidate['name']=='qwen-container-locked-font-fitted'
                             or candidate['name']=='qwen-pre-container-font-fitted'
                             or not candidate['name'].startswith('qwen-'))):
                    from engine.qwen_refine import whole_lockup_contrast_variants
                    calm=whole_lockup_contrast_variants(candidate['elements'],units,source,width,height,regressions,under)
                    for cell_index,(repaired,changes) in enumerate(calm,1):
                        name=f"{candidate['name']}-contrast-repaired-cell-{cell_index:02d}"
                        shortlist.append(dict(candidate,name=name,elements=repaired,contrast_repairs=changes,
                                              variant_of=candidate['name']))
                    if calm and candidate['name']=='qwen-container-locked-font-fitted':
                        required_families['qwen-contrast-repaired']='proposed and rerendered'
                target_repair_ids=sorted(set(weak)|set(busy))
                if (not candidate.get('macro_locked') and candidate['name'].startswith('qwen-') and target_repair_ids and not invisible
                        and _needs_contrast_repair(entries[-1])
                        and 'contrast-repaired' not in candidate['name']
                        and sum('contrast-repaired' in c['name'] for c in shortlist)<10):
                    from engine.qwen_refine import contrast_variant
                    repairs,_=contrast_variant(candidate['elements'],units,width,height,target_repair_ids,under,full,
                                               return_alternatives=True,source_under=source_under,
                                               source_elements=source['elements'])
                    # Share the render budget across parent proposals. A first
                    # parent's many nudges must not crowd out a sound raw-layout
                    # column repair from a later sibling.
                    for repair_index,(repaired,changes) in enumerate(repairs[:3],1):
                        if sum('contrast-repaired' in c['name'] for c in shortlist)>=10:break
                        repaired,structure=_structural_qwen_projection(
                            repaired,source,units,width,height)
                        if (not _feasible(repaired,units,width,height,strict_margins=False,
                                          gap_factor=.001,allow_text_visual_overlap=True)
                                or _unintended_text_overlaps(repaired,units)):continue
                        shortlist.append(dict(name=f"{candidate['name']}-contrast-repaired-{repair_index:02d}",
                            structure=structure,
                            elements=repaired,energy=_energy(repaired,units,source,width,height,desired),
                            contrast_repairs=changes,
                            readability_floor=candidate.get('readability_floor'),
                            readability_floor_met=candidate.get('readability_floor_met',False)))

            except Exception as exc:
                # Diagnostics are optional; isolate third-party renderer faults.
                # Do not catch BaseException (interrupts and shutdown still work).
                entries.append(dict(candidate=candidate,cost=0.,error=str(exc)))
    if not entries or all('error' in item for item in entries):
        return candidates[0],dict(used=False,reason='Diagnostic rendering failed',errors=[e.get('error') for e in entries])
    valid=[e for e in entries if 'error' not in e]
    # Contrast repair is only an alternative when the actual full-raster
    # diagnostics confirm it improved the parent. Pixel-sampled contrast inside
    # the proposal helper alone can miss new low-contrast glyphs elsewhere.
    by_name={e['candidate']['name']:e for e in valid}
    rejected_contrast=[]
    filtered=[]
    for entry in valid:
        candidate=entry['candidate'];name=candidate['name']
        if '-contrast-repaired-' not in name:
            filtered.append(entry);continue
        parent_name=name.split('-contrast-repaired-',1)[0]
        parent=by_name.get(parent_name)
        large_lockup_moves=[change for change in candidate.get('contrast_repairs',[])
            if change.get('repair_type') in ('lockup-repositioned-2d','lockup-repositioned')
            and (abs(float(change.get('dx',0)))>.12*width
                 or abs(float(change.get('dy',0)))>.12*height)]
        if large_lockup_moves:
            rejected_contrast.append(dict(name=name,
                parent=parent['candidate']['name'] if parent else parent_name,
                reason='contrast repair moved the lockup beyond the 12% canvas movement limit'))
            continue
        moved={eid for change in candidate.get('contrast_repairs',[])
               for eid in (change.get('element_ids',[])+change.get('backing_for',[]))}
        before=parent.get('contrast_by_text',{}) if parent else {}
        after=entry.get('contrast_by_text',{})
        def source_floor(eid):
            element=next((e for e in candidate['elements'] if e['id']==eid),{})
            target=3. if element.get('font_size',0)>=24 or element.get('font_weight')=='bold' and element.get('font_size',0)>=18 else 4.5
            floor=source_contrast.get(eid,{}).get('lower_quintile')
            return min(target,max(1.,floor*.90)) if floor is not None else target
        before_count=len(parent.get('low_contrast',[])) if parent else 0
        after_count=len(entry.get('low_contrast',[]))
        before_busy=len(parent.get('busy_background',[])) if parent else 0
        after_busy=len(entry.get('busy_background',[]))
        zone_floors={eid:floor for change in candidate.get('contrast_repairs',[])
                     if change.get('repair_type')=='background-column-fit'
                     for eid,floor in change.get('source_contrast_floor',{}).items()}
        def clears_floor(metrics,floor):
            # Prefer the lower glyph percentile; tolerate a few edge pixels
            # below the floor when the rest of the glyph is readable.
            lower=metrics.get('lower_quintile')
            minimum=metrics.get('minimum')
            return (lower is not None and lower>=floor) or (
                minimum is not None and minimum>=2.2 and lower is not None and lower>=3.0)
        restored_zone=bool(zone_floors) and after_count<=before_count and all(
            eid in before and eid in after and before[eid].get('minimum') is not None
            and after[eid].get('minimum') is not None
            and after[eid]['minimum']>=floor
            and after[eid]['minimum']>=before[eid]['minimum']+.5
            for eid,floor in zone_floors.items())
        is_quiet_2d = any(change.get('repair_type') in ('lockup-repositioned-2d', 'lockup-repositioned','container-calm-cell')
                          for change in candidate.get('contrast_repairs', []))
        is_backdrop_rotation=any(change.get('repair_type')=='backdrop-rotation-variant'
                                 for change in candidate.get('contrast_repairs',[]))
        contrast_ids=(set(before)&set(after)) if is_backdrop_rotation else moved
        all_above_min = all(clears_floor(after.get(eid,{}),zone_floors.get(eid,source_floor(eid)))
                            for eid in contrast_ids if eid in after)
        improved_quiet = bool(is_quiet_2d and after_count <= before_count and all_above_min and
                              (after_busy < before_busy or entry['cost'] < (parent['cost'] if parent else 0.0) - 0.1))
        worsened=[eid for eid in contrast_ids if eid in before and eid in after
                  and before[eid].get('lower_quintile') is not None
                  and after[eid].get('lower_quintile') is not None
                  and after[eid]['lower_quintile']+1e-4<before[eid]['lower_quintile']
                  and not (improved_quiet and clears_floor(after[eid],zone_floors.get(eid,source_floor(eid))))]
        failed_minimum=[eid for eid in moved if eid in before and eid in after
            and before[eid].get('minimum') is not None and before[eid]['minimum']<3.0
            and not clears_floor(after[eid],zone_floors.get(eid,source_floor(eid)))]
        improved_backdrop=(bool(is_backdrop_rotation and parent and after_count<=before_count
                                and entry['cost']<parent['cost']-.1 and not worsened))
        if (parent is None or (after_count>=before_count and not restored_zone and not improved_quiet
                               and not improved_backdrop) or worsened or failed_minimum):
            rejected_contrast.append(dict(name=name,parent=parent['candidate']['name'] if parent else None,
                reason='low-contrast count did not decrease' if parent and after_count>=before_count and not restored_zone and not improved_quiet and not improved_backdrop
                       else 'moved text contrast stayed below its glyph floor' if failed_minimum
                       else 'moved text contrast regressed' if worsened else 'parent unavailable'))
        else:filtered.append(entry)
    valid=filtered
    objective_layout_costs=[_objective_layout_cost(e['candidate']['elements'],units,source,
                width,height,desired,allow_text_visual_overlap=e['candidate']['name'].startswith('qwen-'))
                +1500.*len(e.get('structural_errors',[]))
                for e in valid]
    from engine.qwen_refine import source_emblem_error
    for entry,layout_cost in zip(valid,objective_layout_costs):
        entry['layout_defect_cost']=layout_cost
        entry['source_lockup_error']=source_emblem_error(entry['candidate']['elements'],units)
        entry['source_relationship_error']=relationship_variant(
            entry['candidate']['elements'],units,source,width,height)[2]
        from engine.source_structure import relationship_cost
        entry['attachment_error']=relationship_cost(entry['candidate']['elements'],source,width,height)
        entry['peer_gap_cost']=_peer_gap_cost(entry['candidate']['elements'],units)
        entry['cost']+=entry['peer_gap_cost']
        from engine.macro_layout import quality
        entry['layout_quality']=quality(entry['candidate']['elements'],source,width,height)
        entry['contrast_regressions']=[];entry['mild_contrast_regressions']=[]
        penalty=0.
        for eid,metrics in entry.get('contrast_by_text',{}).items():
            floor=source_contrast.get(eid,{}).get('lower_quintile')
            if floor is not None:floor=min(4.5,floor)
            low=metrics.get('lower_quintile');minimum=metrics.get('minimum')
            if floor is None or low is None:continue
            deficit=_contrast_floor_deficit(floor,low)
            penalty+=12.*deficit
            if eid in {z['text_id'] for z in clean_zones} and metrics.get('p05',99)<2.0:
                entry['contrast_regressions'].append(eid)
                penalty+=25.*(2.-metrics['p05'])
            if not source_contrast.get(eid,{}).get('low',True):
                if low<min(2.2,floor*.72):
                    entry['contrast_regressions'].append(eid)
                elif deficit>0:entry['mild_contrast_regressions'].append(eid)
        entry['contrast_penalty']=penalty
        entry['cost']+=penalty
        # Compact banners still need readable type when every candidate is
        # structurally clean. This is a soft selection cost, not a hard gate.
        entry['primary_scales']={}
        if width/height>=3.2:
            semantics={e['id']:u.get('semantic') for u in units for e in u['items']}
            for e in entry['candidate']['elements']:
                semantic=semantics.get(e['id'])
                if e.get('type')=='text' and semantic in ('headline','kicker','cta'):
                    entry['primary_scales'][e['id']]=float(e['font_size'])
                elif e.get('type')=='image' and semantic in ('hero','art','logo'):
                    entry['primary_scales'][e['id']]=sqrt(e['width']*e['height'])
        entry['readability_penalty']=0.
        if width/height>=2.5:
            semantics={e['id']:u.get('semantic') for u in units for e in u['items']}
            main=[e for e in entry['candidate']['elements'] if e.get('type')=='text'
                  and semantics.get(e['id']) in ('headline','subheading','body','cta')]
            for e in main:
                role=semantics[e['id']]
                floor=min(42.,height*(.08 if role=='headline' else .04 if role in ('subheading','body') else .05))
                deficit=max(0.,1.-float(e['font_size'])/max(1.,floor))
                entry['readability_penalty']+=12.*deficit
            entry['cost']+=entry['readability_penalty']
        if min(width,height)<=200:
            semantics={e['id']:u.get('semantic') for u in units for e in u['items']}
            deficits=[]
            for e in entry['candidate']['elements']:
                if e.get('type')!='text' or not e.get('content','').strip():continue
                semantic=semantics.get(e['id'])
                share=.16 if semantic=='headline' else .05 if semantic in ('footer','callout') else .09
                floor=min(24.,max(1.,min(width,height)*share))
                deficits.append(max(0.,1.-float(e['font_size'])/floor))
            compact_penalty=30.*sum(deficits)/max(1,len(deficits))
            entry['readability_penalty']+=compact_penalty
            entry['cost']+=compact_penalty
        if width/height<=.45:
            foreground=[e for e in entry['candidate']['elements'] if e.get('type')=='text' or
                        any(e['id'] in {p['id'] for p in u['items']} and u['role']!='decoration' for u in units)]
            if foreground and len(units)>=3:
                span=(max(e['y']+e['height'] for e in foreground)-min(e['y'] for e in foreground))/height
                entry['readability_penalty']+=45.*max(0.,.55-span)/.55
                entry['cost']+=45.*max(0.,.55-span)/.55
        # Soft type hierarchy cost applies to every format, including enlarged
        # canvases where a bounded but tiny container otherwise wins by default.
        original_text={e['id']:e for e in source['elements'] if e.get('type')=='text'}
        hierarchy=[]
        reference=min(width/source['canvas']['width'],height/source['canvas']['height'])
        for e in entry['candidate']['elements']:
            old=original_text.get(e['id'])
            if old and float(old.get('font_size',0))>0:
                ratio=float(e['font_size'])/max(1.,old['font_size']*reference)
                hierarchy.append(max(0.,.60-ratio)/.60)
        hierarchy_cost=20.*sum(hierarchy)/max(1,len(hierarchy))
        # Aspect changes should not reward uniformly tiny source-preserving
        # cards. Compare type scale to a bounded area-based scale, retaining
        # the source hierarchy rather than enlarging every caption to 14px.
        if ratio_shift>=1.8:
            readable_reference=min(sqrt(width*height/(source['canvas']['width']*source['canvas']['height'])),reference*1.8)*.8
            relative=[max(0.,1.-float(e['font_size'])/max(1.,original_text[e['id']]['font_size']*readable_reference))
                      for e in entry['candidate']['elements'] if e.get('type')=='text' and e['id'] in original_text]
            hierarchy_cost+=18.*sum(relative)/max(1,len(relative))
        entry['readability_penalty']+=hierarchy_cost;entry['cost']+=hierarchy_cost
        entry['physical_eligible']=(not entry.get('structural_errors')
                and not _unintended_text_overlaps(entry['candidate']['elements'],units)
                and not entry.get('invisible_text') and not entry.get('occlusion_defects')
                and layout_cost<=1.)
        entry['has_layout_defects']=(not entry['physical_eligible'] or bool(entry['contrast_regressions']))
    objective_cost=lambda e:(e['layout_defect_cost']+e['cost']+100.*e.get('attachment_error',0.)
                             +60.*_simple_stack_axis_error(e['candidate']['elements'],units,source,width))
    qwen_feasible={e['candidate']['name']:e['physical_eligible']
                   for e in valid if e['candidate']['name'].startswith('qwen-')}
    display_text_ids={e['id'] for unit in units
                      if unit.get('semantic') in ('headline','kicker')
                      for e in unit['items'] if e.get('type')=='text'}
    qwen_primary=abs(log(state['delta']))>=.12
    perceptual=_siglip_rerank(valid,objective_cost,reference_image,relationship_context,
                              qwen_feasible=qwen_feasible,qwen_primary=qwen_primary,
                              display_text_ids=display_text_ids,source_contrast=source_contrast)
    if perceptual.get('selected_index') is None:
        fallback=_select_visual_candidate(valid,[objective_cost(e) for e in valid],[0.]*len(valid),
                                           qwen_feasible,qwen_primary,False,
                                           display_text_ids=display_text_ids,
                                           source_contrast=source_contrast)
        perceptual.update(fallback,used=False)
    selected=valid[perceptual['selected_index']]
    family_rules={
        'qwen-container-locked-font-fitted':(bool(qwen_candidates),lambda n:n=='qwen-container-locked-font-fitted'),
        'qwen-contrast-repaired':(bool(primary_regressions),lambda n:n.startswith('qwen-container-locked-font-fitted-contrast-repaired-')),
        'deterministic-aspect-reflow':(True,lambda n:n==required_families.get('deterministic-aspect-reflow')),
        'proportional-centered':(ratio_shift<1.35,lambda n:n=='proportional-centered'),
    }
    coverage={family:dict(required=required,
                         rendered=[e['candidate']['name'] for e in entries if 'error' not in e and predicate(e['candidate']['name'])],
                         eligible=[e['candidate']['name'] for e in valid if predicate(e['candidate']['name'])])
              for family,(required,predicate) in family_rules.items()}
    report=dict(used=True,
                raster_cache=dict(raster_cache.stats),
                required_families=required_families,
                shortlist_coverage=coverage,
                method='rendered glyph contrast/overflow/source-lockup/structural gates'+(' + SigLIP tie-break' if perceptual.get('used') else ''),
                perceptual=perceptual,qwen_primary=qwen_primary,
                source_contrast=source_contrast,
                objective_cost_components=[dict(name=e['candidate']['name'],
                    visual_cost=round(e['cost'],4),layout_defect_cost=round(layout_cost,4),
                    total=round(e['cost']+layout_cost,4))
                    for e,layout_cost in zip(valid,objective_layout_costs)],
                candidates=[{k:v for k,v in e.items() if k not in ('candidate','image')}
                            |{'name':e['candidate']['name'],
                              'readability_floor':e['candidate'].get('readability_floor'),
                              'readability_floor_met':e['candidate'].get('readability_floor_met',False),
                              'contrast_repairs':e['candidate'].get('contrast_repairs',[])}
                            for e in entries],
                rejected_contrast_variants=rejected_contrast)
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


def _readable_text_floor_variant(elements,units,width,height):
    """Offer a hierarchy-preserving sibling when all type can grow safely.

    This is deliberately a candidate, not an unconditional edit: an authored
    tiny footer or a crowded canvas may not have room for larger type. The
    under-floor text group receives a common additive increase, preserving its
    internal size gaps; the next larger size caps the floor so hierarchy order
    cannot invert. Tracking scales with each font, and affected frames grow
    around their existing centers. Normal geometry checks decide if the result
    can be used.
    """
    short=min(float(width),float(height))
    floor=min(16.,max(8.,short*.018))
    texts=[e for e in elements if e.get('type')=='text' and e.get('content','').strip()]
    if not texts:return None
    smallest=min(float(e.get('font_size',1.)) for e in texts)
    if smallest>=floor-1e-4:return None
    below=[e for e in texts if float(e.get('font_size',1.))<floor]
    largest_below=max(float(e.get('font_size',1.)) for e in below)
    next_size=min((float(e.get('font_size',1.)) for e in texts
                   if float(e.get('font_size',1.))>=floor),default=None)
    target_floor=floor
    if next_size is not None:
        target_floor=min(target_floor,max(smallest,next_size*.90))
    delta=min(target_floor-smallest,
              max(0.,(next_size*.90-largest_below) if next_size is not None else target_floor-smallest))
    # Keep this an exploration step, not a wholesale change to the layout.
    if delta<=1e-4 or delta>min(12.,short*.025):return None
    below_ids={e['id'] for e in below}
    result=deepcopy(elements)
    for e in result:
        if e.get('id') not in below_ids:continue
        old_size=max(1.,float(e.get('font_size',1.)))
        new_size=old_size+delta
        old_cx=e['x']+e['width']/2.;old_cy=e['y']+e['height']/2.
        e['font_size']=new_size
        e['letter_spacing']=float(e.get('letter_spacing',0.))*(new_size/old_size)
        measured_h,measured_w,lines=_text_info(e,max(e['width'],1e6),new_size)[:3]
        new_width=max(e['width'],measured_w+new_size*.12)
        if new_width>width or (e.get('max_lines') and lines>e['max_lines']):return None
        measured_h=_text_info(e,new_width,new_size)[0]
        new_height=max(e['height'],measured_h)
        e.update(width=new_width,height=new_height,
                 x=old_cx-new_width/2.,y=old_cy-new_height/2.)
    return result


def _source_scaled_art_variant(elements,units,source,width,height):
    """Offer a proportional source-layout sibling for heavily shrunken art.

    Qwen can identify the right image group but sometimes compresses the whole
    group far below the source-to-target area scale. This candidate restores
    the group's relative geometry using one uniform scale around the canvas
    centers. It is considered only for large, unboxed image groups; ranking
    and feasibility checks still decide whether it survives.
    """
    sw=float(source['canvas']['width']);sh=float(source['canvas']['height'])
    scale=sqrt(float(width)*float(height)/max(1.,sw*sh))
    by_id={e['id']:e for e in elements}
    source_by_id={e['id']:e for e in source['elements']}
    result=deepcopy(elements);changed=False
    canvas_area=max(1.,sw*sh)
    for unit in units:
        items=unit.get('items',())
        if not items or any(e.get('type')!='image' for e in items):
            continue
        if unit.get('backed_ids') or unit.get('backed_by_ids'):
            continue
        source_bounds=unit.get('bounds')
        if not source_bounds:
            continue
        sx0,sy0,sx1,sy1=map(float,source_bounds)
        if max(0.,sx1-sx0)*max(0.,sy1-sy0)/canvas_area<.045:
            continue
        ids=[e['id'] for e in items]
        if any(eid not in by_id or eid not in source_by_id for eid in ids):
            continue
        current=[by_id[eid] for eid in ids]
        if any(abs(float(e.get('rotation',0.)))>.5 for e in current):
            continue
        cx0=min(float(e['x']) for e in current)
        cy0=min(float(e['y']) for e in current)
        cx1=max(float(e['x'])+float(e['width']) for e in current)
        cy1=max(float(e['y'])+float(e['height']) for e in current)
        expected_w=(sx1-sx0)*scale;expected_h=(sy1-sy0)*scale
        if expected_w<=1 or expected_h<=1:
            continue
        # Avoid touching ordinary model sizing; only explore a substantial
        # correction when both dimensions show that the group collapsed.
        if ((cx1-cx0)>=expected_w*.78
                or (cy1-cy0)>=expected_h*.78):
            continue
        proposed=[]
        for original in items:
            source_item=source_by_id[original['id']]
            proposed.append({
                'id':original['id'],
                'x':(float(source_item['x'])-sw/2.)*scale+float(width)/2.,
                'y':(float(source_item['y'])-sh/2.)*scale+float(height)/2.,
                'width':float(source_item['width'])*scale,
                'height':float(source_item['height'])*scale,
            })
        min_x=min(e['x'] for e in proposed);min_y=min(e['y'] for e in proposed)
        max_x=max(e['x']+e['width'] for e in proposed)
        max_y=max(e['y']+e['height'] for e in proposed)
        if max_x-min_x>width or max_y-min_y>height:
            continue
        dx=dy=0.
        # Apply one shared boundary correction so internal spacing is exact.
        if min_x<0:dx=-min_x
        elif max_x>width:dx=float(width)-max_x
        if min_y<0:dy=-min_y
        elif max_y>height:dy=float(height)-max_y
        for e in proposed:
            target=next(item for item in result if item['id']==e['id'])
            target.update(x=e['x']+dx,y=e['y']+dy,
                          width=e['width'],height=e['height'])
        changed=True
    return result if changed else None


def _rotation_candidate_variants(elements,units,width,height,limit=6):
    """Generate small rotation alternatives for canvas-registered backgrounds."""
    from engine.qwen_refine import _rotated_bounds
    """Generate bounded background rotations; leave foreground art and rules alone."""
    variants=[]
    # This helper is retained for compatibility with older callers. Foreground
    # rotation was the source of tilted separator rules; background alternatives
    # are created separately where the renderer can expand their crop safely.
    return variants



def _busy_background_font_shrink_only(candidate, baselines, display_text_ids):
    """Find a strong Qwen layout penalized only for display text on busy pixels.

    The exception is deliberately narrow: the Qwen render must have no measured
    text/layout defects, all checked text must have strong contrast, and a
    cheaper heuristic must remove busyness on a display element while keeping
    that element's box fixed and shrinking only display fonts.
    """
    if (candidate.get('low_contrast') or candidate.get('invisible_text')
            or abs(float(candidate.get('layout_defect_cost',float('inf'))))>1e-8):
        return []
    contrast=candidate.get('contrast_by_text') or {}
    if not contrast or any(value.get('lower_quintile') is None
                           or value['lower_quintile']<4.5
                           for value in contrast.values()):
        return []
    q_elements={e.get('id'):e for e in candidate.get('candidate',{}).get('elements',[])}
    if not q_elements:return []
    display_text_ids=set(display_text_ids or ())
    q_busy=set(candidate.get('busy_background',[]))
    qualifying=[]
    for index,baseline in enumerate(baselines):
        if (baseline.get('invisible_text') or baseline.get('error')
                or baseline.get('cost',0.)>=candidate.get('cost',0.)):
            continue
        h_elements={e.get('id'):e for e in baseline.get('candidate',{}).get('elements',[])}
        if q_elements.keys()!=h_elements.keys():
            continue
        h_busy=set(baseline.get('busy_background',[]))
        changed_display=False
        only_display_fonts_shrank=True
        for eid,q in q_elements.items():
            h=h_elements[eid]
            if q.get('type')!=h.get('type') or any(
                    abs(float(q.get(k,0.))-float(h.get(k,0.)))>1.0
                    for k in ('x','y','width','height')):
                only_display_fonts_shrank=False
                break
            if q.get('type')!='text':
                continue
            if q.get('content')!=h.get('content'):
                only_display_fonts_shrank=False
                break
            q_size=float(q.get('font_size',0.));h_size=float(h.get('font_size',0.))
            if q_size>0 and abs(h_size-q_size)>max(1.,q_size*.05):
                if eid not in display_text_ids or h_size>=.92*q_size:
                    only_display_fonts_shrank=False
                    break
                q_spacing=float(q.get('letter_spacing',0.))/q_size
                h_spacing=float(h.get('letter_spacing',0.))/max(1.,h_size)
                if abs(q_spacing-h_spacing)>max(.25,abs(q_spacing)*.05):
                    only_display_fonts_shrank=False
                    break
                if eid in q_busy-h_busy:
                    changed_display=True
        if changed_display and only_display_fonts_shrank:
            qualifying.append(index)
    return qualifying


def _select_visual_candidate(valid,costs,scores,qwen_feasible=False,qwen_primary=False,
                             ranker_trained=False,display_text_ids=None,source_contrast=None):
    """Gate by measured defects; use image scores only inside a defect tie."""
    qwen=[i for i,e in enumerate(valid) if e['candidate']['name'].startswith('qwen-')]
    heur=[i for i in range(len(valid)) if i not in qwen]
    baselines=[valid[i] for i in heur]
    feasible=lambda i:(valid[i]['physical_eligible'] if 'physical_eligible' in valid[i] else
                       qwen_feasible.get(valid[i]['candidate']['name'],False)
                       if isinstance(qwen_feasible,dict) else bool(qwen_feasible))
    tolerated={i:_busy_background_font_shrink_only(valid[i],baselines,display_text_ids)
               for i in qwen}
    safe=[i for i in qwen if feasible(i) and
          (_qwen_contrast_safe(valid[i],baselines,qwen_primary,source_contrast) or tolerated[i])]
    coherent=[i for i in safe if valid[i].get('source_lockup_error',0.)<=.025
              and not valid[i].get('has_layout_defects',False)]
    fractured=[]
    if coherent:
        fractured=[valid[i]['candidate']['name'] for i in safe
                   if valid[i].get('source_lockup_error',0.)>.10]
        safe=[i for i in safe if valid[i].get('source_lockup_error',0.)<=.10]
    clean_base=[i for i in safe if valid[i]['candidate']['name'] in
                ('qwen-original','qwen-proposal','qwen-font-fitted','qwen-container-locked-font-fitted',
                 'qwen-pre-container-original','qwen-pre-container-proposal',
                 'qwen-pre-container-font-fitted')
                and not _needs_contrast_repair(valid[i])
                and not valid[i].get('invisible_text')
                and not valid[i].get('has_layout_defects',False)]
    suppressed=[]
    if clean_base:
        suppressed=[valid[i]['candidate']['name'] for i in safe
                    if '-contrast-repaired-' in valid[i]['candidate']['name']]
        safe=[i for i in safe if valid[i]['candidate']['name'] not in suppressed]
    selection_costs=list(costs)
    # A clean reflow that makes every primary object at least as readable,
    # with a substantial shared size gain, can beat a smaller learned stack.
    # Compare the same IDs; no template coordinates or preferred columns.
    dominated={}
    for qi in safe:
        scales=valid[qi].get('primary_scales',{})
        if len(scales)<3:continue
        for hi in heur:
            h=valid[hi]
            if h.get('has_layout_defects',False):continue
            other=h.get('primary_scales',{})
            if not scales.keys()<=other.keys():continue
            ratios=[other[eid]/max(1e-6,value) for eid,value in scales.items()]
            if min(ratios)>=.95 and sum(r>=1.15 for r in ratios)>=2 and sum(log(max(1e-6,r)) for r in ratios)/len(ratios)>=log(1.25):
                dominated[qi]=hi
                selection_costs[qi]+=25.*sum(log(max(1.,r)) for r in ratios)/len(ratios)
                break
    best=min(range(len(costs)),key=costs.__getitem__)
    tolerance=max(1.25,abs(costs[best])*.03)
    for i in qwen:
        if tolerated[i]:selection_costs[i]=min(selection_costs[i],costs[best]+tolerance)

    def hard_key(i):
        e=valid[i]
        return (bool(e.get('invisible_text')),bool(e.get('has_layout_defects')),
                len(e.get('occlusion_defects',[])),len(e.get('contrast_regressions',[])))

    def tie_band(indices):
        if not indices:return []
        tier=min(hard_key(i) for i in indices)
        pool=[i for i in indices if hard_key(i)==tier]
        # Contrast counts and raster geometry rank before the VLM score.
        minimum=min(selection_costs[i] for i in pool)
        return [i for i in pool if selection_costs[i]<=minimum+max(1.25,abs(minimum)*.03)]

    eligible=heur+safe
    if not eligible:eligible=list(range(len(valid)))
    global_band=tie_band(eligible)
    family_band=tie_band(safe)
    clean_qwen=[i for i in family_band if not any(hard_key(i))
                and valid[i].get('layout_defect_cost',0.)<=1.]
    # A tiny clean poster is not a readable leaderboard. Keep the learned
    # preference only while its soft readability loss matches clean reflows.
    readable_heur=[valid[i].get('readability_penalty',0.) for i in heur
                   if not valid[i].get('has_layout_defects',False)]
    if readable_heur:
        floor=min(readable_heur)+1.25
        clean_qwen=[i for i in clean_qwen if valid[i].get('readability_penalty',0.)<=floor]
    # Learned preference is a tie-break, not a bypass of measured attachment,
    # readability and raster costs. Otherwise a "clean" but fractured model
    # layout hides a substantially better source-consistent alternative.
    clean_qwen=[i for i in clean_qwen if i not in dominated]
    tied_qwen=[i for i in clean_qwen if i in global_band]
    pool=tied_qwen if qwen_primary and tied_qwen else global_band
    chosen=max(pool,key=lambda i:(scores[i],-selection_costs[i],-i))
    qchoice=max(family_band,key=lambda i:(scores[i],-selection_costs[i],-i)) if family_band else None
    best_heuristic=min(heur,key=lambda i:costs[i]) if heur else None
    checked=qchoice if qchoice is not None else (qwen[0] if qwen else None)
    return dict(selected_index=chosen,objective_costs=[round(c,4) for c in costs],
        selection_costs=[round(c,4) for c in selection_costs],
        source_lockup_rejected_names=fractured,contrast_repair_suppressed_names=suppressed,
        primary_scale_dominated_names=[valid[i]['candidate']['name'] for i in dominated],
        busy_background_tolerated_names=[valid[i]['candidate']['name'] for i in qwen if tolerated[i]],
        tie_band=global_band,qwen_family_band=family_band,tolerance=round(tolerance,4),
        qwen_eligible=bool(safe),eligible_qwen_names=[valid[i]['candidate']['name'] for i in safe],
        trained_ranker_loaded=ranker_trained,relationship_override=None,
        qwen_safety_checks=dict(feasible=bool(checked is not None and feasible(checked)),
            no_invisible_text=bool(checked is not None and not valid[checked].get('invisible_text')),
            contrast_within_tolerance=checked in safe,qwen_primary=qwen_primary,
            objective_within_band=checked in family_band,objective_within_global_band=checked in global_band,
            siglip_margin=bool(checked is not None and best_heuristic is not None
                              and scores[checked]>scores[best_heuristic]+.03),
            busy_background_tolerated=bool(checked is not None and tolerated.get(checked))),
        overrode_heuristic=chosen!=best_heuristic)


def _siglip_rerank(valid, objective_cost, reference_image=None, layout_context=None,
                   qwen_feasible=False, qwen_primary=False,display_text_ids=None,
                   source_contrast=None):
    """Optional visual scoring with a deterministic family-aware fallback."""
    if len(valid)<2:
        return dict(used=False,selected_index=None,reason='Fewer than two candidates to compare')
    ranker_trained=False;scores=None;note='SigLIP unavailable'
    try:
        from engine.vision import score_layouts, _load_ranker
        ranker_trained=_load_ranker() is not None
        result=score_layouts([e['image'] for e in valid],reference_image=reference_image,
                            layout_context=layout_context) if len(valid)>1 else None
        if result is not None:scores,note=result
    except Exception as exc:
        note=f'Vision module unavailable: {type(exc).__name__}: {exc}'
    used=scores is not None and len(scores)==len(valid)
    if not used:return dict(used=False,selected_index=None,reason=note)
    report=_select_visual_candidate(valid,[objective_cost(e) for e in valid],scores,
                                    qwen_feasible,qwen_primary,ranker_trained,
                                    display_text_ids=display_text_ids,
                                    source_contrast=source_contrast)
    return dict(report,used=used,note=note,scores=[round(s,4) for s in scores])


def _assemble(elements,backgrounds,source,width,height):
    by_id={e['id']:deepcopy(e) for e in elements}
    for old in backgrounds:
        e=deepcopy(old)
        if _edge_backdrop(old,source):
            e=_fit_edge_backdrop(old,source,width,height)
        elif e['type']=='image':
            # JSON frame bounds remain legal even when the original image
            # rotates. The renderer clips rotated pixels at canvas edges.
            e.update(x=0.,y=0.,width=float(width),height=float(height))
            e['fit']=_background_fit(old,source,width,height)
        else:
            e.update(x=0,y=0,width=width,height=height)
        by_id[e['id']]=e
    from engine.geometry_policy import preserve_aspects
    return dict(canvas=dict(width=int(width),height=int(height)),
                elements=preserve_aspects([by_id[e['id']] for e in source['elements']],source,width,height))


def _broken_source_stack_units(proposal, units):
    """Find source-evidenced stacks whose overlap Qwen materially worsened.

    A simple overlap-present/absent check misses harmful cases: a source pair
    that just touches can still count as layered after Qwen moves one element
    completely underneath another. Compare overlap as a fraction of the
    smaller visible element and preserve the authored shared transform when
    the proposal increases that fraction substantially.
    """
    proposed_by_id={e['id']:e for e in proposal}
    def visible_box(element):
        return (_text_visible_bounds(element) if element['type']=='text'
                else _bounds([element]))
    def overlap_fraction(a,b):
        a_box,b_box=visible_box(a),visible_box(b)
        a_area=max(1e-9,(a_box[2]-a_box[0])*(a_box[3]-a_box[1]))
        b_area=max(1e-9,(b_box[2]-b_box[0])*(b_box[3]-b_box[1]))
        return _overlap(a_box,b_box)/min(a_area,b_area)

    broken=[]
    for unit in units:
        if not unit.get('preserve_layer_stack') or len(unit['items'])<2:
            continue
        source_items=unit['items']
        stack_broken=False
        for i,a in enumerate(source_items):
            pa=proposed_by_id.get(a['id'])
            if pa is None:
                continue
            for b in source_items[:i]:
                pb=proposed_by_id.get(b['id'])
                if pb is None:
                    continue
                if not _intentional_stack_pair(a,b,visible_box(a),visible_box(b)):
                    continue
                source_fraction=overlap_fraction(a,b)
                proposed_fraction=overlap_fraction(pa,pb)
                # Do not accept a much denser collision just because the two
                # frames still overlap. This is especially important for
                # small controls adjacent to a foreground image edge.
                if (not _intentional_stack_pair(pa,pb,visible_box(pa),visible_box(pb))
                        or proposed_fraction>max(source_fraction*1.5,source_fraction+.20)):
                    stack_broken=True
                    break
            if stack_broken:
                break
        if stack_broken:
            broken.append(unit)
    return broken


def _preserve_layered_proposals(proposal, units, return_anchor_ids=False):
    """Project model proposals for source stacks onto one shared transform.

    A layout model may choose where and how large a composite should be, but
    its independent per-element boxes cannot tear apart known source layers.
    """
    result=deepcopy(proposal)
    proposed_by_id={e['id']:e for e in result}
    geometry_keys=('x','y','width','height','font_size','letter_spacing',
                   'corner_radius','stroke_width')
    anchor_ids = []
    for unit in units:
        if not unit.get('preserve_layer_stack') or len(unit['items'])<2:
            continue
        source_items = unit['items']
        suggested=[proposed_by_id[e['id']] for e in source_items if e['id'] in proposed_by_id]
        if len(suggested)!=len(source_items):
            continue
        visual_items = [e for e in source_items if e['type'] in ('image', 'shape')]
        text_items = [e for e in source_items if e['type']=='text']
        largest_visual_area=max((float(e.get('width',0.0))*float(e.get('height',0.0))
                                 for e in visual_items),default=0.0)
        largest_text_area=max((float(e.get('width',0.0))*float(e.get('height',0.0))
                               for e in text_items),default=0.0)
        # A small ornament can overlap a headline without representing the
        # scale or position of the whole lockup. Use it as the transform anchor
        # only when its area is substantial relative to the largest text item.
        if visual_items and (not text_items or largest_visual_area>=.5*largest_text_area):
            anchor = max(visual_items, key=lambda e: float(e.get('width', 0.0)) * float(e.get('height', 0.0)))
            anchor_id = anchor['id']
            anchor_ids.append(anchor_id)
            prop_anchor = proposed_by_id[anchor_id]
            factor = min(float(prop_anchor['width']) / max(1e-6, float(anchor['width'])),
                         float(prop_anchor['height']) / max(1e-6, float(anchor['height'])))
            anchor_cx = float(prop_anchor['x']) + float(prop_anchor['width']) / 2.0
            anchor_cy = float(prop_anchor['y']) + float(prop_anchor['height']) / 2.0
            src_anchor_cx = float(anchor['x']) + float(anchor['width']) / 2.0
            src_anchor_cy = float(anchor['y']) + float(anchor['height']) / 2.0
        else:
            scales = [proposed_by_id[e['id']]['width'] / max(1e-6, float(e.get('width', 1.0)))
                      for e in source_items if e['id'] in proposed_by_id and float(e.get('width', 0)) > 0]
            from statistics import median
            factor = median(scales) if scales else 1.0
            anchor_cx = median(proposed_by_id[e['id']]['x'] + proposed_by_id[e['id']]['width'] / 2.0
                               for e in source_items if e['id'] in proposed_by_id)
            anchor_cy = median(proposed_by_id[e['id']]['y'] + proposed_by_id[e['id']]['height'] / 2.0
                               for e in source_items if e['id'] in proposed_by_id)
            src_anchor_cx = (unit['bounds'][0] + unit['bounds'][2]) / 2.0
            src_anchor_cy = (unit['bounds'][1] + unit['bounds'][3]) / 2.0
            anchor_id = source_items[0]['id']
            anchor_ids.append(anchor_id)

        for item in source_items:
            target = proposed_by_id[item['id']]
            src_cx = item['x'] + item['width'] / 2.0
            src_cy = item['y'] + item['height'] / 2.0
            tgt_cx = anchor_cx + (src_cx - src_anchor_cx) * factor
            tgt_cy = anchor_cy + (src_cy - src_anchor_cy) * factor
            tgt_w = item['width'] * factor
            tgt_h = item['height'] * factor
            target['width'] = tgt_w
            target['height'] = tgt_h
            target['x'] = tgt_cx - tgt_w / 2.0
            target['y'] = tgt_cy - tgt_h / 2.0
            for k in ('corner_radius', 'stroke_width'):
                if k in item:
                    target[k] = item[k] * factor
            if item['type'] == 'text':
                src_font = max(1.0, float(item.get('font_size', 1.0)))
                scaled_size = max(1.0, src_font * factor)
                fitted_size = _fit_text(target, tgt_w, tgt_h, scaled_size)
                target['font_size'] = fitted_size
                target['letter_spacing'] = float(item.get('letter_spacing', 0.0)) * (fitted_size / src_font)
    if return_anchor_ids:
        return result, anchor_ids
    return result


def _preserve_backed_containment(proposal,units,width,height):
    """Keep source-contained panel contents inside their proposed backing panel.

    Large source panels are intentionally separate units so a model can place
    them independently. This sibling repairs only children that Qwen has moved
    substantially outside the panel, mapping their source-relative positions
    into the panel's proposed frame while retaining Qwen's font choice when it
    still fits.
    """
    result=deepcopy(proposal)
    by_id={e['id']:e for e in result}
    source_by_id={e['id']:e for unit in units for e in unit['items']}
    repaired=[]
    for panel_unit in units:
        if not panel_unit.get('backing_panel'):
            continue
        source_panel=panel_unit['items'][0]
        panel=by_id.get(source_panel['id'])
        if panel is None:
            continue
        sw=max(1.,float(source_panel.get('width',1.)))
        sh=max(1.,float(source_panel.get('height',1.)))
        sx=max(.05,float(panel.get('width',sw))/sw)
        sy=max(.05,float(panel.get('height',sh))/sh)
        panel_box=_bounds([panel])
        panel_area=max(1.,(panel_box[2]-panel_box[0])*(panel_box[3]-panel_box[1]))
        local=[]
        for eid in panel_unit.get('backed_ids',()):
            original=source_by_id.get(eid);current=by_id.get(eid)
            if original is None or current is None:
                continue
            current_box=(_text_visible_bounds(current) if current.get('type')=='text'
                         else _bounds([current]))
            current_area=max(1.,(current_box[2]-current_box[0])*(current_box[3]-current_box[1]))
            containment=_overlap(panel_box,current_box)/current_area
            # Preserve a model's intentional reflow if it kept the content in
            # the source backing. Repair only the detached children.
            if containment>=.97:
                continue
            x_offset=float(original['x'])-float(source_panel['x'])
            y_offset=float(original['y'])-float(source_panel['y'])
            current['x']=float(panel['x'])+x_offset*sx
            current['y']=float(panel['y'])+y_offset*sy
            if current.get('type')=='text':
                current['width']=max(1.,float(original['width'])*sx)
                current['height']=max(1.,float(original['height'])*sy)
                source_size=max(1.,float(original.get('font_size',1.)))
                size=max(1.,float(current.get('font_size',source_size)))
                size=min(size,source_size*min(sx,sy))
                fitted=_fit_text(current,current['width'],current['height'],size)
                current['font_size']=max(1.,fitted)
                current['letter_spacing']=float(original.get('letter_spacing',0.))*current['font_size']/source_size
            else:
                scale=min(sx,sy)
                current['width']=max(1.,float(original['width'])*scale)
                current['height']=max(1.,float(original['height'])*scale)
            local.append(eid)
        if local:
            repaired.extend(local)
    # Reject this sibling if the recovered card no longer fits on the canvas.
    for e in result:
        if e['id'] in repaired:
            e['x']=min(max(0.,float(e['x'])),max(0.,width-float(e['width'])))
            e['y']=min(max(0.,float(e['y'])),max(0.,height-float(e['height'])))
    return result,sorted(set(repaired))


@asset_lookup_request
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
    if width==sw and height==sh:
        result=deepcopy(source);groups={};adjustments=[]
        for e in result['elements']:groups.setdefault(e.get('group_id') or ('element',e['id']),[]).append(e)
        for members in groups.values():
            l,t,r,b=_raw_bounds(members)
            factor=min(1.,width/max(1.,r-l),height/max(1.,b-t))
            if factor<1.:
                for e in members:
                    e.update(x=l+(e['x']-l)*factor,y=t+(e['y']-t)*factor,
                             width=e['width']*factor,height=e['height']*factor)
                    if e.get('type')=='text':_set_font_size(e,e['font_size']*factor)
                l,t,r,b=_raw_bounds(members)
            dx=max(0.,-l)-max(0.,r-width);dy=max(0.,-t)-max(0.,b-height)
            if factor<1. or abs(dx)+abs(dy)>1e-7:
                for e in members:e['x']+=dx;e['y']+=dy
                adjustments.append([e['id'] for e in members])
        # Preserve authored art overlap, but a separate paragraph touching a
        # control is a readability defect, not a required artistic lockup.
        from engine.layout_contracts import _clear_controls
        before_clearance=deepcopy(result['elements'])
        _clear_controls(result['elements'],source,width,height)
        readability_changed=result['elements']!=before_clearance
        if readability_changed:
            bg,identity_units=_analyze(source)
            candidate=[e for e in result['elements'] if e['id'] not in {b['id'] for b in bg}]
            if _unintended_text_overlaps(candidate,identity_units):
                result['elements']=before_clearance;readability_changed=False
        name='identity-control-clearance' if readability_changed else 'identity-source-bounds' if adjustments else 'identity'
        return result,dict(winner=name,generated=1,feasible=1,starts=1,
            candidates=[],visual={'used':False,'reason':'Identity: preserve source art; legalize frames and separate unrelated copy from controls'},
            qwen='not needed for identity resize',qwen_diagnostics={'proposal_received':False,'bounded_source_groups':adjustments},
            hard_constraints_passed=True,seconds={'total':round(perf_counter()-start,4)})
    backgrounds,units=_analyze(source)
    desired=sqrt(width*height/(sw*sh))
    for unit in units:
        unit['target_aspect_ratio']=width/height
        unit['canvas_scale']=desired
        unit['target_short_side']=min(width,height)
        if width/height<.68 and unit['role']=='footer' and unit.get('footer_anchor') in ('left','right'):
            # A corner watermark becomes a centered footer companion on tall
            # layouts; preserve its source anchor for migration decisions.
            unit['source_footer_anchor']=unit['footer_anchor']
            unit['footer_anchor']='center'
    state=_aspect_state(source,width,height)
    context=_perceptual_context(source,backgrounds,units,width,height)
    context['typography_gamma']=min(1.,.65+.7*desired)
    for u in units:
        u['perceptual']=context
        u['font_gamma']=min(1.,.65+.7*desired)
    seeds=[dict(name='proportional',elements=_proportional(units,width,height,sw,sh))]
    from engine.qwen_refine import build_container_tree
    container_tree=build_container_tree(source,units)
    qwen_status = 'disabled (set MAGICANVAS_QWEN_ENABLED=1 to enable)'
    qwen_diagnostics = dict(status=qwen_status, proposal_received=False,
                            constraint_feasible=False, optimized=False,
                            structural_pass_version=1,
                            container_tree={k:v for k,v in container_tree.items() if k!='source_elements'})
    if os.environ.get('MAGICANVAS_QWEN_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        try:
            from engine.qwen_seed import generate_seed
            generation={}
            proposal, qwen_status = generate_seed(source, target_canvas, seeds[0]['elements'],diagnostics=generation)
            qwen_diagnostics['generation']={k:v for k,v in generation.items() if k!='model_elements'}
            qwen_diagnostics['model_elements']=[dict(id=eid,**values) for eid,values in generation.get('model_elements',{}).items()]
            qwen_diagnostics['status'] = qwen_status
            if proposal is not None:
                from engine.qwen_refine import anchor_superscripts, font_variant
                original_proposal=deepcopy(proposal)
                from engine.container_flow import variants as sibling_variants
                for sibling_items,sibling_report in sibling_variants(proposal,source):
                    seeds.append(dict(name='qwen-sibling-grid-'+str(len(seeds)),elements=sibling_items,sibling_flow=sibling_report))
                if generation.get('completed_proposal'):
                    seeds.append(dict(name='qwen-completed-source-attachments',elements=deepcopy(proposal),completion=deepcopy(generation.get('completion',{}))))
                # Decode the model boxes independently of source-lockup repair.
                raw=[{**deepcopy(e),**generation.get('model_elements',{}).get(e['id'],{})}
                     for e in (seeds[0]['elements'] if generation.get('model_elements') else original_proposal)]
                source_by_id={e['id']:e for e in source['elements']}
                for e in raw:
                    if e.get('type')=='text':
                        original=source_by_id[e['id']]
                        e['letter_spacing']=float(original.get('letter_spacing',0))*e['font_size']/max(1.,original['font_size'])
                raw=_bounded_qwen_elements(raw,source,width,height)
                raw,raw_fonts=_calibrate_text_boxes(raw,source,width,height,None)
                seeds.append(dict(name='qwen-pre-container-font-fitted',elements=raw))
                if not _physical_layout_safe(raw,units,source,width,height):
                    # Preserve useful geometric repair as a distinctly named
                    # sibling; never relabel a constrained layout as raw.
                    seeds.append(dict(name='qwen-raw-constraint-repaired',elements=deepcopy(raw)))
                qwen_diagnostics['raw_font_calibration']=raw_fonts
                proposal,snapped_labels=_micro_snap_qwen_proposal(
                    proposal,units,width,height,source=source)
                # Keep compact source-linked superscripts attached in the
                # primary proposal as well as in the calibrated sibling.
                proposal,anchor_changes=anchor_superscripts(proposal,units)
                # Superscript anchoring may change a text box after the first
                # overlap cleanup. Recheck the final proposal geometry before
                # feasibility validation so a repair cannot reintroduce a
                # collision with adjacent logo/subtitle text.
                proposal,post_anchor_changes=_separate_qwen_text_collisions(
                    proposal,units,width,height)
                proposal,post_anchor_unit_changes=_separate_qwen_unit_collisions(
                    proposal,units,width,height)
                decoded_candidate,decoded_separations=_separate_qwen_text_collisions(
                    original_proposal,units,width,height)
                decoded_candidate,decoded_unit_changes=_separate_qwen_unit_collisions(
                    decoded_candidate,units,width,height)
                qwen_diagnostics['proposal_received'] = True
                qwen_diagnostics['micro_snapped_labels']=snapped_labels
                qwen_diagnostics['superscript_anchored_ids']=anchor_changes
                qwen_diagnostics['post_anchor_text_separations']=post_anchor_changes
                qwen_diagnostics['post_anchor_unit_separations']=post_anchor_unit_changes
                seeds.append(dict(name='qwen-proposal', elements=proposal))
                contained_candidate,contained_ids=_preserve_backed_containment(
                    proposal,units,width,height)
                if (contained_ids and contained_candidate!=proposal
                        and _feasible(contained_candidate,units,width,height,
                                      strict_margins=False,gap_factor=.004,
                                      allow_text_visual_overlap=True)):
                    seeds.append(dict(name='qwen-source-contained-proposal',
                                      elements=contained_candidate))
                    qwen_diagnostics['source_containment_repaired_ids']=contained_ids
                if decoded_candidate!=original_proposal:
                    # Preserve the model's decoded composition independently
                    # from source-lockup repairs, which can over-scale a
                    # rotated composite that already contains the layout.
                    seeds.append(dict(name='qwen-decoded-repaired',
                                      elements=decoded_candidate))
                    qwen_diagnostics['decoded_text_separations']=decoded_separations
                    qwen_diagnostics['decoded_unit_separations']=decoded_unit_changes
                # Preserve a source-evidenced overlapping artwork stack only
                # when Qwen separated one of its layers. Keep the model layout
                # above as a sibling so the renderer/ranker can still prefer
                # an intentional reflow when it renders better.
                broken_stack_units=_broken_source_stack_units(proposal,units)
                if broken_stack_units:
                    stacked,stack_anchor_ids=_preserve_layered_proposals(proposal,broken_stack_units,return_anchor_ids=True)
                    stacked,stack_anchor_changes=anchor_superscripts(stacked,units)
                    raw_stack_cost=_objective_layout_cost(
                        proposal,units,source,width,height,desired,
                        allow_text_visual_overlap=True)
                    preserved_stack_cost=_objective_layout_cost(
                        stacked,units,source,width,height,desired,
                        allow_text_visual_overlap=True)
                    preserved_candidate=stacked
                    if not _feasible(preserved_candidate,units,width,height,
                                     strict_margins=False,gap_factor=.004,
                                     allow_text_visual_overlap=True):
                        for topology in ('nearest','centers','alternate'):
                            projected=_constrain(stacked,units,width,height,topology,
                                                 strict_margins=False,
                                                 allow_text_visual_overlap=True)
                            if projected is not None and _feasible(
                                    projected,units,width,height,strict_margins=False,
                                    gap_factor=.004,allow_text_visual_overlap=True):
                                preserved_candidate=projected
                                break
                    if (stacked!=proposal and preserved_candidate is not None
                            and _feasible(preserved_candidate,units,width,height,
                                          strict_margins=False,gap_factor=.004,
                                          allow_text_visual_overlap=True)):
                        # This sibling is intentionally retained even when its
                        # geometric score is higher. The rendered judge decides
                        # whether preserving the source overlap improves design.
                        seeds.append(dict(name='qwen-layer-preserved',elements=preserved_candidate))
                        qwen_diagnostics['layer_preserved_elements']=deepcopy(preserved_candidate)
                        qwen_diagnostics['layer_stack_repaired_ids']=sorted({
                            e['id'] for unit in broken_stack_units for e in unit['items']})
                        qwen_diagnostics['layer_stack_anchor_ids']=stack_anchor_ids if stack_anchor_ids else stack_anchor_changes
                        qwen_diagnostics['layer_stack_defect_cost']={
                            'proposal':round(raw_stack_cost,6),
                            'preserved':round(preserved_stack_cost,6)}
                if original_proposal!=proposal:
                    seeds.append(dict(name='qwen-original',elements=original_proposal))
                calibrated,font_changes=font_variant(proposal,units,width,height)
                qwen_diagnostics['font_calibrated_ids']=font_changes
                if calibrated!=proposal:
                    # Relationship repair sets shared transforms and source
                    # offsets. Running it again would overwrite measured font
                    # sizes with source-scaled fonts, silently undoing the fit.
                    seeds.append(dict(name='qwen-font-fitted',elements=calibrated))
        except Exception as exc:
            if os.environ.get('MAGICANVAS_REQUIRE_QWEN_ADAPTER')=='1':raise
            qwen_status = f'Qwen proposal unavailable: {type(exc).__name__}: {exc}'
            qwen_diagnostics['status'] = qwen_status
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
    if width/height<=.45:
        seeds.append(dict(name='skyscraper-vertical-stack',elements=_skyscraper_seed(units,width,height,(sw,sh))))
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
    if context['background_map'] and max(context['background_map'])-min(context['background_map'])>=.12:
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
    source_element_ids={e['id'] for unit in units for e in unit['items']}
    incomplete_seed_names=[]
    for seed in seeds:
        # A generated layout is an all-elements candidate. Reject malformed
        # seeds before geometric constraints index unit members; one omitted
        # element must not abort the entire adaptation or reach the renderer.
        seed_element_ids={e['id'] for e in seed['elements']}
        if (seed_element_ids!=source_element_ids
                or len(seed['elements'])!=len(seed_element_ids)):
            incomplete_seed_names.append({
                'name':seed['name'],
                'missing_ids':sorted(source_element_ids-seed_element_ids),
                'extra_ids':sorted(seed_element_ids-source_element_ids),
                'duplicate_ids':sorted({e['id'] for e in seed['elements']
                                        if sum(candidate['id']==e['id']
                                               for candidate in seed['elements'])>1}),
            })
            continue
        projected=None
        qwen_seed=seed['name'].startswith('qwen-')
        if not qwen_seed:
            _sync_backing_panels(seed['elements'],units,width,height)
        elif seed['name']=='qwen-proposal' and _unintended_text_overlaps(seed['elements'],units):
            qwen_diagnostics['constraint_reason']='unintended text-on-text overlap'
        if seed['name']=='qwen-pre-container-font-fitted':
            # Keep this ablation exact even if raster checks later reject it.
            # A failed raw layout must not silently become a constrained one.
            projected=seed['elements']
        elif qwen_seed and _physical_layout_safe(seed['elements'],units,source,width,height):
            # Retain a valid model layout intact. Only project it when it
            # violates on-canvas bounds or separation constraints.
            projected=seed['elements']
        else:
            for topology in ('nearest','centers','alternate'):
                projected=_constrain(seed['elements'],units,width,height,topology,
                                     strict_margins=not qwen_seed,
                                     allow_text_visual_overlap=qwen_seed)
                if projected is not None:
                    if seed['name']=='dense-short-banner':
                        _equalize_control_geometry(projected,units)
                    break
            if projected is None and _feasible(
                    seed['elements'],units,width,height,
                    strict_margins=not qwen_seed,
                    gap_factor=.001 if qwen_seed else .008,
                    allow_text_visual_overlap=qwen_seed):
                projected=seed['elements']
        if projected is None:
            rejected.append(seed)
            if seed['name'] == 'qwen-proposal' and 'constraint_reason' not in qwen_diagnostics:
                qwen_diagnostics['constraint_reason'] = 'rejected by layout constraints'
        else:
            energy=_energy(projected,units,source,width,height,desired)
            feasible.append(dict(name=seed['name'],elements=projected,energy=energy))
            if seed['name'] == 'qwen-proposal':
                qwen_diagnostics.update(constraint_feasible=True,
                                        constrained_elements=deepcopy(projected),
                                        constrained_energy=energy)
    # Spend a bounded repair budget on the most promising rejected seeds.
    rejected=[seed for seed in rejected if not seed['name'].startswith('qwen-')]
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
    # Make bounded exploratory siblings from the best distinct layouts. The
    # parent always remains in the candidate set, so these repairs can never
    # force a rotation or larger typography when the original scores better.
    exploratory=[]
    bases=[]
    seen_bases=set()
    preferred_bases=([c for c in feasible if c['name'] in
                      ('qwen-proposal','qwen-font-fitted','qwen-original',
                       'qwen-local-refined','qwen-local-refined-font-fitted')]+
                     feasible)
    for candidate in preferred_bases:
        if candidate['name'].endswith(('-readable-text-floor',)) or '-rotation-' in candidate['name']:
            continue
        signature=tuple((e['id'],round(e['x'],1),round(e['y'],1),round(e['width'],1),
                         round(e['height'],1),round(e.get('rotation',0.),1))
                        for e in candidate['elements'])
        if signature in seen_bases:continue
        seen_bases.add(signature);bases.append(candidate)
        if len(bases)>=5:break
    for parent in bases:
        is_qwen=parent['name'].startswith('qwen-')
        label=parent['name'].removeprefix('qwen-') if is_qwen else parent['name']
        readable=_readable_text_floor_variant(parent['elements'],units,width,height)
        if readable is not None:
            readable_candidate=readable
            if not _feasible(readable_candidate,units,width,height,strict_margins=False,
                             gap_factor=.004,allow_text_visual_overlap=True):
                for topology in ('nearest','centers','alternate'):
                    projected=_constrain(readable,units,width,height,topology,
                                          strict_margins=False,allow_text_visual_overlap=True)
                    if projected is not None and _feasible(
                            projected,units,width,height,strict_margins=False,
                            gap_factor=.004,allow_text_visual_overlap=True):
                        readable_candidate=projected
                        break
            if (_feasible(readable_candidate,units,width,height,strict_margins=False,
                          gap_factor=.004,allow_text_visual_overlap=True)
                    and not set(_unintended_text_overlaps(readable_candidate,units))-
                               set(_unintended_text_overlaps(parent['elements'],units))):
                name=('qwen-' if is_qwen else '')+'readable-text-floor-'+label
                floor=min(16.,max(8.,min(float(width),float(height))*.018))
                minimum_font=min(float(e.get('font_size',1.)) for e in readable_candidate
                                 if e.get('type')=='text' and e.get('content','').strip())
                exploratory.append(dict(name=name,elements=readable_candidate,
                    energy=_energy(readable_candidate,units,source,width,height,desired),
                    variant_of=parent['name'],readability_floor=floor,
                    readability_floor_met=minimum_font>=floor-1e-4))
        if is_qwen:
            structured,structure=_structural_qwen_projection(
                parent['elements'],source,units,width,height,container_tree)
            if (_feasible(structured,units,width,height,strict_margins=False,
                          gap_factor=.001,allow_text_visual_overlap=True)
                    and not _unintended_text_overlaps(structured,units)):
                exploratory.append(dict(name='qwen-container-projected-'+label,
                    elements=structured,structure=structure,
                    energy=_energy(structured,units,source,width,height,desired),
                    variant_of=parent['name']))
        # Generic rotation of foreground images tilts decorative lines and
        # ornaments. Rotation candidates are generated for the background only.
    # Limit total renderer work while ensuring at least one model-first source
    # layout, when available, contributes exploratory siblings.
    if exploratory:
        qwen_exploratory=[c for c in exploratory if c['name'].startswith('qwen-')]
        other_exploratory=[c for c in exploratory if not c['name'].startswith('qwen-')]
        selected_qwen=qwen_exploratory[:12]
        feasible.extend(selected_qwen+other_exploratory[:12])
    feasible.sort(key=lambda c:(c['energy'],c['name']))
    optimized=[]
    optimize_candidates=feasible[:10]
    aspect_prefix=('inline-banner','editorial','split','left-hero','dense-short') if width/height>=1.35 else ('semantic-stack','copy-first','visual-first')
    reserved=next((c for c in feasible if not c['name'].startswith('qwen-')
                   and c['name'].startswith(aspect_prefix)),None)
    if reserved is not None and reserved not in optimize_candidates:optimize_candidates.append(reserved)
    if max(width/height/(sw/sh),(sw/sh)/(width/height))<1.35:
        proportional=next((c for c in feasible if c['name']=='proportional'),None)
        if proportional is not None and proportional not in optimize_candidates:optimize_candidates.append(proportional)
    for candidate in feasible:
        if candidate['name'].startswith('qwen-') and candidate not in optimize_candidates:
            optimize_candidates.append(candidate)
    for i,candidate in enumerate(optimize_candidates):
        is_qwen=candidate['name'].startswith('qwen-')
        if is_qwen:
            # Retain this geometry; bounded source-local refinement is added
            # below as a sibling rather than overwriting the proposal.
            elements=deepcopy(candidate['elements'])
            optimization=dict(before=candidate['energy'],after=candidate['energy'],accepted=0,iterations=0)
        elif candidate['name']=='dense-short-banner' and _feasible(candidate['elements'],units,width,height):
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
        if candidate['name']!='dense-short-banner' and not is_qwen:
            _repair_footer_bottom_anchor(elements,units,width,height)
            _repair_primary_action_shape(elements,units,width,height)
            _repair_vertical_hero_spacing(elements,units,width,height)
            _repair_vertical_brand_clearance(elements,units,width,height)
        # Balance the logo by its visible artwork bounds for every landscape
        # family, including the compact dense-leaderboard seed. Transparent
        # PNG padding must not make a nominally left-aligned logo look flush
        # with one side of its brand column.
        if not is_qwen:
            _repair_banner_logo_column_balance(elements,units,width,height)
            _repair_banner_cta_copy_proximity(elements,units,width,height)
            _repair_sparse_vertical_stack(elements,units,width,height)
            _align_peer_rows(elements,units)
        optimized.append(dict(name=candidate['name'],elements=elements,
            energy=_energy(elements,units,source,width,height,desired),optimization=optimization,
            readability_floor=candidate.get('readability_floor'),
            readability_floor_met=candidate.get('readability_floor_met',False),
            rotation_change=candidate.get('rotation_change'),
            **({'backgrounds':deepcopy(candidate['backgrounds'])}
               if candidate.get('backgrounds') is not None else {})))
        if candidate['name'] in ('qwen-proposal','qwen-font-fitted'):
            from engine.qwen_refine import polish_variant
            local,search=polish_variant(elements,units,source,width,height,desired)
            if candidate['name']=='qwen-proposal':
                qwen_diagnostics.update(optimized=search['accepted']>0,
                                        optimized_elements=deepcopy(local),
                                        optimized_energy=_energy(local,units,source,width,height,desired),
                                        optimization=search)
                local_name='qwen-local-refined'
            else:
                local_name='qwen-local-refined-font-fitted'
            # Keep the combined font-and-position sibling even when the bounded
            # position search accepts no move: calibration itself is a distinct
            # candidate and must be judged against its uncalibrated parent.
            if search['accepted'] or candidate['name']=='qwen-font-fitted':
                optimized.append(dict(name=local_name,elements=local,
                    energy=_energy(local,units,source,width,height,desired),optimization=search))
    # Learned candidates can compete with a container-projected sibling.
    # Deterministic reflows only receive glyph calibration; source projection
    # would replace their chosen reflow with a proportional source lockup.
    structural_rejections=[]
    structured_candidates=[]
    rejected_structures=[]
    # Preserve one genuine unit-based reflow for comparison. Projecting every
    # deterministic seed through the learned container envelope can turn all
    # families into the same source-proportional arrangement or reject them.
    baseline_for_render=None
    for candidate in sorted(optimized,key=lambda c:(c['energy'],c['name'])):
        if candidate['name'].startswith('qwen-') or not candidate['name'].startswith(aspect_prefix):continue
        fitted,fonts=_calibrate_text_boxes(candidate['elements'],source,width,height,None)
        if (_feasible(fitted,units,width,height,strict_margins=False,gap_factor=.004,
                      allow_text_visual_overlap=True) and not _unintended_text_overlaps(fitted,units)):
            baseline_for_render=dict(candidate,name='deterministic-aspect-reflow-'+candidate['name'],
                                     elements=fitted,structure={'font_calibration':fonts,'mode':'atomic-unit-reflow'})
            break
    # Keep bounded, font-fitted raw model boxes independently of container
    # projection so raster selection can reject a damaging projection.
    pre_container_qwen=[]
    from engine.qwen_refine import snap_source_axes
    for candidate in optimized:
        name=candidate['name']
        if name in ('qwen-pre-container-font-fitted','qwen-raw-constraint-repaired'):
            pre_container_qwen.append(candidate)
            continue
        if name not in ('qwen-original','qwen-proposal','qwen-font-fitted'):
            continue
        if not _physical_layout_safe(candidate['elements'],units,source,width,height):
            continue
        variant=name.removeprefix('qwen-')
        preserved=dict(candidate,
            name='qwen-pre-container-'+variant,
            elements=deepcopy(candidate['elements']),
            structure={'mode':'pre-container-qwen','variant':name})
        pre_container_qwen.append(preserved)
        axis_elements,axis_notes=snap_source_axes(
            candidate['elements'],source,width,height,container_tree)
        if (axis_notes and axis_elements!=candidate['elements']
                and _feasible(axis_elements,units,width,height,strict_margins=False,
                              gap_factor=.001,allow_text_visual_overlap=True)
                and not _unintended_text_overlaps(axis_elements,units)):
            pre_container_qwen.append(dict(candidate,
                name='qwen-pre-container-axis-snapped-'+variant,
                elements=axis_elements,
                structure={'mode':'pre-container-qwen-axis-snapped',
                           'variant':name,'axis_repairs':axis_notes}))
    for candidate in optimized:
        if not candidate['name'].startswith('qwen-'):
            structured,fonts=_calibrate_text_boxes(candidate['elements'],source,width,height,None)
            structure={'mode':'deterministic-reflow','font_calibration':fonts}
            accepted=(_feasible(structured,units,width,height,strict_margins=False,
                       gap_factor=.001,allow_text_visual_overlap=True)
                      and not _unintended_text_overlaps(structured,units))
        elif candidate['name'].startswith('qwen-pre-container-') or candidate['name']=='qwen-raw-constraint-repaired':
            structured=candidate['elements'];structure={'mode':'constraint-repaired-qwen' if candidate['name']=='qwen-raw-constraint-repaired' else 'raw-font-fitted'}
            accepted=_physical_layout_safe(structured,units,source,width,height)
        else:
            structured,structure=_structural_qwen_projection(
                candidate['elements'],source,units,width,height,container_tree)
            accepted=_physical_layout_safe(structured,units,source,width,height)
        if not accepted:
            structural_rejections.append(candidate['name'])
            rejected_structures.append(dict(candidate,elements=structured,structure=structure,
                energy=_energy(structured,units,source,width,height,desired)))
            continue
        structured_candidates.append(dict(candidate,elements=structured,structure=structure,
            energy=_energy(structured,units,source,width,height,desired)))
    if structured_candidates:
        optimized=structured_candidates
    else:
        # An impossible target must retain all elements and expose the failure;
        # do not mislabel an unverified structure as accepted.
        optimized=sorted(rejected_structures,key=lambda c:(
            _constraint_violation(c['elements'],units,width,height),c['name']))[:4]
        qwen_diagnostics['structural_fallback_reason']='No projected candidate satisfies geometry and text constraints; retaining the least-violating complete container layout'
    if optimized:
        signatures={tuple((e['id'],round(e['x'],3),round(e['y'],3),round(e['width'],3),
                           round(e['height'],3),round(e.get('font_size',0.),3))
                          for e in candidate['elements']) for candidate in optimized}
        for candidate in pre_container_qwen:
            signature=tuple((e['id'],round(e['x'],3),round(e['y'],3),round(e['width'],3),
                             round(e['height'],3),round(e.get('font_size',0.),3))
                            for e in candidate['elements'])
            if (signature not in signatures or
                    (candidate['name'] in ('qwen-pre-container-font-fitted','qwen-raw-constraint-repaired')
                     and not any(c['name']==candidate['name'] for c in optimized))):
                optimized.append(candidate);signatures.add(signature)
    if baseline_for_render is not None:optimized.append(baseline_for_render)
    if width/height<=.45:
        tall=_source_geometry_safety(_skyscraper_seed(units,width,height,(sw,sh)),source,units,width,height)
        tall,_=_calibrate_text_boxes(tall,source,width,height)
        optimized.append(dict(name='skyscraper-vertical-stack',elements=tall,
                             energy=_energy(tall,units,source,width,height,desired)))
    qwen_diagnostics['structural_rejected_candidates']=structural_rejections
    if incomplete_seed_names:
        qwen_diagnostics['incomplete_seed_candidates']=incomplete_seed_names
    from engine.structural_reflow import stack_seed
    measured_stack=stack_seed(source,units,width,height)
    if measured_stack:
        optimized.append(dict(name='measured-semantic-stack',elements=measured_stack,
                              energy=_energy(measured_stack,units,source,width,height,desired),measured_stack=True))
    optimized.sort(key=lambda c:(c['energy'],c['name']))
    for candidate in optimized:
        if candidate['name']=='qwen-pre-container-font-fitted':continue
        candidate['elements']=_source_geometry_safety(candidate['elements'],source,units,width,height)
        candidate['elements'],_= _calibrate_text_boxes(candidate['elements'],source,width,height)
        candidate['energy']=_energy(candidate['elements'],units,source,width,height,desired)
    # Independent source-attached alternatives bypass leaf-level topology
    # search. Their groups still undergo the same render and defect gates.
    from engine.source_structure import seed as attached_seed
    for flow in (False,True):
        attached=attached_seed(source,units,width,height,flow=flow)
        if attached:
            optimized.append(dict(name='source-attached-flow' if flow else 'source-attached-proportional',
                                  elements=attached,source_attached=True,
                                  energy=_energy(attached,units,source,width,height,desired)))
    geometry_time=perf_counter()-start-generation_time
    verification_started=perf_counter()
    winner,visual=_visual_rerank(optimized,backgrounds,units,source,width,height,desired)
    from engine.macro_layout import upgrade
    winner,visual,macro_diagnostics=upgrade(winner,visual,backgrounds,units,source,width,height,desired,_visual_rerank,proposal=qwen_diagnostics.get('model_elements'))
    qwen_diagnostics['macro_reflow']=macro_diagnostics
    # No geometry mutations after render verification. Store the selected
    # structural evidence and deliver the exact geometry that was judged.
    qwen_diagnostics['selected_structure']=winner.get('structure',{})
    qwen_diagnostics['winner'] = winner['name']
    qwen_diagnostics['variants']=sorted({c['name'] for c in optimized+visual.get('candidates',[])
                                        if c['name'].startswith('qwen-')})
    # Deliver the exact background layers that accompanied the judged layout.
    output=_assemble(winner['elements'],winner.get('backgrounds',backgrounds),
                     source,width,height)
    report=dict(perceptual=context,aspect=state,roles=[dict(ids=[e['id'] for e in u['items']],role=u['semantic'],confidence=u['confidence']) for u in units],
                generated=len(seeds),feasible=len(feasible),rejected_topologies=len(rejected),recovered=recovered,starts=len(optimized),winner=winner['name'],
                energy_terms=_energy_terms(winner['elements'],units,source,width,height,desired),
                candidates=[{k:v for k,v in c.items() if k!='elements'} for c in optimized],
                visual=visual,qwen=qwen_status,qwen_diagnostics=qwen_diagnostics,
                hard_constraints_passed=_physical_layout_safe(
                    winner['elements'],units,source,width,height),
                seconds=dict(generation=round(generation_time,4),geometry=round(geometry_time,4),render_verify=round(perf_counter()-verification_started,4),total=round(perf_counter()-start,4)))
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
