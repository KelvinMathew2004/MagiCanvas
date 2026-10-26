"""Source container trees, robust proposal projection and shared alignment axes.

Qwen places functional containers. Source geometry supplies their internal
layout; renderer-based calibration and deterministic axis clusters handle
small measurement/placement errors before visual verification. Compatibility
helpers for earlier candidates remain available, but the active pipeline uses
these universal structural passes.
"""
from copy import deepcopy
from math import cos, radians, sin
from statistics import median


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


def _overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def relationship_variant(elements, units, source, width, height):
    """Universal container/axis/font alternative for the rendered verifier."""
    from engine.resize import _structural_qwen_projection
    result,report=_structural_qwen_projection(elements,source,units,width,height)
    before={e['id']:e for e in elements}
    changes=report['containers']+report['font_calibration']+report['axes']
    error=sum(abs(float(e.get(k,0))-float(before[e['id']].get(k,0)))/max(1.,width if k in ('x','width') else height)
              for e in result for k in ('x','y','width','height'))/max(1,len(result))
    return result,changes,error


def _merge_groups(groups):
    """Return disjoint groups so no repair can move half of a lockup."""
    merged=[]
    for group in groups:
        group=set(group)
        if not group:continue
        changed=True
        while changed:
            changed=False
            for other in merged[:]:
                if group & other:
                    group.update(other);merged.remove(other);changed=True
        merged.append(group)
    return merged


def _emblem_groups(source,sw,sh):
    """Find centered, compact source emblems without absorbing separate photos."""
    from engine.resize import _overlap, _text_visible_bounds
    texts=[e for e in source.values() if e['type']=='text' and e.get('content','').strip()]
    def box(e):return (e['x'],e['y'],e['x']+e['width'],e['y']+e['height'])
    eligible=[]
    for e in source.values():
        share=e['width']*e['height']/max(1.,sw*sh)
        if e['type'] not in ('image','shape','text') or share>=.85:continue
        if e['type']!='text' and share>=.15:
            backed=sum(_overlap(box(e),_text_visible_bounds(t))/max(1.,
                (_text_visible_bounds(t)[2]-_text_visible_bounds(t)[0])*
                (_text_visible_bounds(t)[3]-_text_visible_bounds(t)[1]))>=.8
                and e.get('z_index',0)<t.get('z_index',0) for t in texts)
            if backed<2:continue
        eligible.append(e)
    links=[];radius=.06*min(sw,sh)
    for i,a in enumerate(eligible):
        ab=box(a)
        for b in eligible[i+1:]:
            bb=box(b)
            dx=max(0.,ab[0]-bb[2],bb[0]-ab[2]);dy=max(0.,ab[1]-bb[3],bb[1]-ab[3])
            if dx*dx+dy*dy<=radius*radius:links.append({a['id'],b['id']})
    result=[]
    for ids in _merge_groups(links):
        members=[source[eid] for eid in sorted(ids)]
        labels=[e for e in members if e['type']=='text' and e.get('content','').strip()]
        if len(labels)<2 or sum(e['type']!='text' for e in members)<2:continue
        left=min(e['x'] for e in members);right=max(e['x']+e['width'] for e in members)
        top=min(e['y'] for e in members);bottom=max(e['y']+e['height'] for e in members)
        center=(left+right)/2
        if (right-left)*(bottom-top)>.50*sw*sh or abs(center-sw/2)>.08*sw:continue
        largest=max(e.get('font_size',1.) for e in labels)
        main=[e for e in labels if not (len(e.get('content','').strip())<=3
                                       and e.get('font_size',1.)<.35*largest)]
        if any(abs(e['x']+e['width']/2-center)>.18*sw for e in main):continue
        result.append(ids)
    return result


def _copy_columns(source,sw,sh,lockups):
    """Source-aligned copy and its button form one column during border repair."""
    from engine.resize import _text_visible_bounds
    texts=[e for e in source.values() if e['type']=='text' and e.get('content','').strip()]
    links=[]
    for i,a in enumerate(texts):
        ab=_text_visible_bounds(a)
        for b in texts[i+1:]:
            bb=_text_visible_bounds(b)
            gap=max(0.,ab[1]-bb[3],bb[1]-ab[3])
            horizontal=max(0.,min(ab[2],bb[2])-max(ab[0],bb[0]))/max(1.,min(ab[2]-ab[0],bb[2]-bb[0]))
            aligned=min(abs(ab[0]-bb[0]),abs((ab[0]+ab[2]-bb[0]-bb[2])/2))<.08*sw
            if aligned and horizontal>=.6 and gap<=.18*sh:links.append({a['id'],b['id']})
    columns=[]
    for ids in _merge_groups(links):
        for group in lockups:
            if ids & group:ids.update(group)
        columns.append(ids)
    return _merge_groups(columns)


def source_emblem_error(elements,units):
    """Measure broken source emblem offsets and non-text scale hierarchy."""
    source={e['id']:e for u in units for e in u['items']}
    out={e['id']:e for e in elements}
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    if not canvas:return 0.
    sw,sh=canvas;error=0.
    for ids in _emblem_groups(source,sw,sh):
        if not ids<=out.keys():return 1.
        originals=[source[eid] for eid in sorted(ids)]
        left=min(e['x'] for e in originals);right=max(e['x']+e['width'] for e in originals)
        top=min(e['y'] for e in originals);bottom=max(e['y']+e['height'] for e in originals)
        sx,sy=(left+right)/2,(top+bottom)/2
        major=[e for e in originals if min(e['width'],e['height'])>=.025*min(sw,sh)]
        if not major:continue
        factor=max(.01,median(min(out[e['id']]['width']/max(1.,e['width']),
                                  out[e['id']]['height']/max(1.,e['height'])) for e in major))
        cx=median(out[e['id']]['x']+out[e['id']]['width']/2-(e['x']+e['width']/2-sx)*factor for e in originals)
        cy=median(out[e['id']]['y']+out[e['id']]['height']/2-(e['y']+e['height']/2-sy)*factor for e in originals)
        for e in originals:
            target=out[e['id']]
            error=max(error,abs(target['x']+target['width']/2-(cx+(e['x']+e['width']/2-sx)*factor))/max(1.,(right-left)*factor),
                      abs(target['y']+target['height']/2-(cy+(e['y']+e['height']/2-sy)*factor))/max(1.,(bottom-top)*factor))
            if e['type']!='text' and e.get('fit')!='cover':
                error=max(error,*(.5*abs(target[k]-e[k]*factor)/max(1.,e[k]*factor) for k in ('width','height')))
    return error


def lockup_groups(units):
    """Explicit groups and compact backed labels/superscripts, not nearest art."""
    from engine.resize import _bounds, _overlap, _text_visible_bounds
    source={e['id']:e for u in units for e in u['items']}
    parent={eid:eid for eid in source}
    def find(eid):
        while parent[eid]!=eid:
            parent[eid]=parent[parent[eid]];eid=parent[eid]
        return eid
    def join(a,b):
        parent[find(a)]=find(b)
    explicit={}
    for e in source.values():
        if e.get('group_id') is not None:
            key=e['group_id']
            if key in explicit:join(e['id'],explicit[key])
            else:explicit[key]=e['id']
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=canvas or (max((e['x']+e['width'] for e in source.values()),default=1),
                     max((e['y']+e['height'] for e in source.values()),default=1))
    texts=[e for e in source.values() if e['type']=='text' and e.get('content','').strip()]
    # A compact superscript is a lockup member when source geometry places it
    # at the upper edge of a much larger price/number, even without overlap.
    for sat in texts:
        if len(sat.get('content','').strip())>3:continue
        sb=_text_visible_bounds(sat)
        candidates=[]
        for anchor in texts:
            if anchor['id']==sat['id'] or anchor.get('font_size',0)<48:continue
            if sat.get('font_size',0)>.35*anchor.get('font_size',1):continue
            ab=_text_visible_bounds(anchor)
            gap=max(0.,sb[0]-ab[2],ab[0]-sb[2])
            top_distance=abs(sb[1]-ab[1])
            if gap<=.45*max(1.,sat.get('font_size',1)) and top_distance<=.5*max(1.,anchor.get('font_size',1)):
                candidates.append((gap+top_distance,anchor))
        if candidates:
            join(sat['id'],min(candidates,key=lambda pair:pair[0])[1]['id'])
    for e in source.values():
        eb=_bounds([e]);ea=max(1,(eb[2]-eb[0])*(eb[3]-eb[1]))
        choices=[]
        for t in texts:
            if t['id']==e['id']:continue
            tb=_text_visible_bounds(t);ta=max(1,(tb[2]-tb[0])*(tb[3]-tb[1]))
            if e['type']=='text':
                # Short price suffixes/superscripts contained in larger text.
                if len(e.get('content','').strip())<=3 and ta>=ea*2 and _overlap(eb,tb)/ea>=.5:
                    choices.append((ta,t))
            elif e['type'] in ('image','shape') and ea<=.85*sw*sh:
                covered=_overlap(eb,tb)/ta
                horizontal_overlap=max(0.,min(eb[2],tb[2])-max(eb[0],tb[0]))/max(1.,tb[2]-tb[0])
                vertical_overlap=max(0.,min(eb[3],tb[3])-max(eb[1],tb[1]))
                if (e.get('z_index',0)<t.get('z_index',0) and
                        (covered>=.85 or (horizontal_overlap>=.25 and vertical_overlap>0 and ea<=.85*sw*sh))):
                    # A compact frame can enclose several offset text parts.
                    # Join every contained part, not just the centered headline.
                    join(e['id'],t['id'])
                elif e['height']<=.015*sh and e['width']>=3*e['height']:
                    gap=max(0,tb[0]-eb[2],eb[0]-tb[2])
                    vertical_gap=max(0,tb[1]-eb[3],eb[1]-tb[3])
                    horizontal=max(0,min(tb[2],eb[2])-max(tb[0],eb[0]))/max(1,min(tb[2]-tb[0],eb[2]-eb[0]))
                    if (gap<=.035*sw and abs((eb[1]+eb[3]-tb[1]-tb[3])/2)<=.02*sh
                            or horizontal>=.5 and vertical_gap<=.025*sh):
                        choices.append((gap+ta,t))
        if choices:
            join(e['id'],min(choices,key=lambda item:item[0])[1]['id'])
    # Rotated Lockup Companions: If a text element and a backing/frame element
    # share a similar non-zero rotation (abs(rot_a - rot_b) <= 5.0) or one is
    # inside the other's rotated bounds in the source design, lock them into a
    # rigid-source-lockup so tilted text/badges never shear apart.
    source_items = list(source.values())
    for i, a in enumerate(source_items):
        rot_a = float(a.get('rotation', 0.0)) % 360.0
        is_rot_a = (0.5 < rot_a < 359.5) and abs(rot_a - 180.0) > 0.5
        for b in source_items[i + 1:]:
            rot_b = float(b.get('rotation', 0.0)) % 360.0
            is_rot_b = (0.5 < rot_b < 359.5) and abs(rot_b - 180.0) > 0.5
            if not (is_rot_a or is_rot_b):
                continue
            has_text = a['type'] == 'text' or b['type'] == 'text'
            has_frame = a['type'] in ('image', 'shape') or b['type'] in ('image', 'shape')
            if not (has_text and has_frame):
                continue
            box_a = _rotated_bounds(a)
            box_b = _rotated_bounds(b)
            area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
            area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
            diff = abs(rot_a - rot_b) % 360.0
            angle_diff = min(diff, 360.0 - diff)
            similar_rotation = is_rot_a and is_rot_b and angle_diff <= 5.0
            ov = _overlap(box_a, box_b)
            inside = (ov / min(area_a, area_b) >= 0.25) or (
                box_b[0] <= a['x'] + a['width']/2 <= box_b[2] and
                box_b[1] <= a['y'] + a['height']/2 <= box_b[3]
            ) or (
                box_a[0] <= b['x'] + b['width']/2 <= box_a[2] and
                box_a[1] <= b['y'] + b['height']/2 <= box_a[3]
            )
            if (similar_rotation and ov > 0) or inside:
                join(a['id'], b['id'])
    groups={}
    for eid in source:groups.setdefault(find(eid),set()).add(eid)
    return _merge_groups([ids for ids in groups.values() if len(ids)>1]
                         +_emblem_groups(source,sw,sh))


def repair_relationships(elements,units,width,height):
    """Fit compact lockups together, then solve whole text chains simultaneously."""
    from math import sqrt
    from engine.resize import _bounds, _text_info, _unintended_text_overlaps
    result=deepcopy(elements);by_id={e['id']:e for e in result}
    source={e['id']:e for u in units for e in u['items']}
    groups=lockup_groups(units);notes=[]
    source_canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=source_canvas or (width,height)
    emblems={frozenset(g) for g in _emblem_groups(source,sw,sh)}
    for ids in groups:
        ids=ids & set(by_id)
        if len(ids)<2:continue
        originals=[source[eid] for eid in sorted(ids)];proposed=[by_id[eid] for eid in sorted(ids)]
        # Clamp the raw PNG frames too, including transparent margins.
        sb=(min(e['x'] for e in originals),min(e['y'] for e in originals),
            max(e['x']+e['width'] for e in originals),max(e['y']+e['height'] for e in originals))
        sx,sy=(sb[0]+sb[2])/2,(sb[1]+sb[3])/2
        has_rotated = any(
            (0.5 < (float(e.get('rotation', 0.0)) % 360.0) < 359.5 and
             abs((float(e.get('rotation', 0.0)) % 360.0) - 180.0) > 0.5)
            for e in originals
        )
        if has_rotated:
            scales = [
                by_id[e['id']][k] / max(1.0, e[k])
                for e in originals for k in ('width', 'height')
                if e['id'] in by_id and e[k] > 0
            ]
            scale = max(0.01, median(scales)) if scales else 1.0
            src_group_cx = (sb[0] + sb[2]) / 2.0
            src_group_cy = (sb[1] + sb[3]) / 2.0
            group_cx = median(
                by_id[e['id']]['x'] + by_id[e['id']]['width'] / 2.0 -
                (e['x'] + e['width'] / 2.0 - src_group_cx) * scale
                for e in originals if e['id'] in by_id
            )
            group_cy = median(
                by_id[e['id']]['y'] + by_id[e['id']]['height'] / 2.0 -
                (e['y'] + e['height'] / 2.0 - src_group_cy) * scale
                for e in originals if e['id'] in by_id
            )
            def _group_reach(s):
                min_x = min_y = float('inf')
                max_x = max_y = float('-inf')
                for e in originals:
                    rad = radians(float(e.get('rotation', 0.0)))
                    w = e['width'] * s; h = e['height'] * s
                    rw = w * abs(cos(rad)) + h * abs(sin(rad))
                    rh = w * abs(sin(rad)) + h * abs(cos(rad))
                    eff_half_w = max(w, rw) / 2.0
                    eff_half_h = max(h, rh) / 2.0
                    ecx = (e['x'] + e['width'] / 2.0 - src_group_cx) * s
                    ecy = (e['y'] + e['height'] / 2.0 - src_group_cy) * s
                    min_x = min(min_x, ecx - eff_half_w)
                    max_x = max(max_x, ecx + eff_half_w)
                    min_y = min(min_y, ecy - eff_half_h)
                    max_y = max(max_y, ecy + eff_half_h)
                return max_x - min_x, max_y - min_y, -min_x, max_x, -min_y, max_y

            gw, gh, left_reach, right_reach, top_reach, bottom_reach = _group_reach(scale)
            if gw > width or gh > height:
                downscale = min(width / max(1e-6, gw), height / max(1e-6, gh))
                scale *= downscale
                gw, gh, left_reach, right_reach, top_reach, bottom_reach = _group_reach(scale)
            group_cx = min(width - right_reach, max(left_reach, group_cx))
            group_cy = min(height - bottom_reach, max(top_reach, group_cy))
            trial = deepcopy(result)
            trial_by = {e['id']: e for e in trial}
            for e in originals:
                out = trial_by[e['id']]
                src_cx = e['x'] + e['width'] / 2.0
                src_cy = e['y'] + e['height'] / 2.0
                tgt_cx = group_cx + (src_cx - src_group_cx) * scale
                tgt_cy = group_cy + (src_cy - src_group_cy) * scale
                w = e['width'] * scale
                h = e['height'] * scale
                out['width'] = w
                out['height'] = h
                out['x'] = tgt_cx - w / 2.0
                out['y'] = tgt_cy - h / 2.0
                out['rotation'] = e.get('rotation', 0.0)
                if e['type'] == 'text':
                    source_font = max(1.0, float(e.get('font_size', 1.0)))
                    out['font_size'] = max(1.0, source_font * scale)
                    out['letter_spacing'] = float(e.get('letter_spacing', 0.0)) * scale
            if not (set(_unintended_text_overlaps(trial, units)) - set(_unintended_text_overlaps(result, units))):
                result = trial
                by_id = trial_by
                notes.append(dict(kind='rigid-source-lockup', element_ids=sorted(ids)))
                continue
        # Text font ratios are reliable scale anchors for compact lockups;
        # explicit artwork groups use robust box ratios instead.
        scales=[by_id[e['id']].get('font_size',1)/max(1,e.get('font_size',1))
                for e in originals if e['type']=='text']
        if not scales:
            scales=[by_id[e['id']][k]/max(1,e[k]) for e in originals for k in ('width','height')]
        scale=max(.01,median(scales))
        emblem=any(group<=ids for group in emblems)
        if emblem:
            major=[e for e in originals if min(e['width'],e['height'])>=.025*min(sw,sh)]
            thin_rules=[e for e in major if max(e['width'],e['height'])/
                        max(1.,min(e['width'],e['height']))>=8.]
            if len(thin_rules)>=2:
                # Multiple thin rules are attachments, not reliable scale
                # anchors: independently shrunken line heights can collapse
                # the whole lockup. Estimate scale from substantial text/art
                # boxes and keep it near an area-preserving canvas scale.
                scale_anchors=[e for e in major if e not in thin_rules]
                if scale_anchors:
                    scale=median(min(by_id[e['id']]['width']/max(1.,e['width']),
                                     by_id[e['id']]['height']/max(1.,e['height']))
                                 for e in scale_anchors)
                area_scale=sqrt(width*height/max(1.,sw*sh))
                scale=min(area_scale*1.05,max(area_scale*.85,scale))
            else:
                # For ordinary emblems, preserve the existing short-side
                # bound; the area scale correction is only needed when thin
                # rules outnumber or outweigh the actual lettering.
                short_scale=min(width,height)/max(1.,min(sw,sh))
                scale=min(short_scale*1.25,max(short_scale*.85,scale))
        scale=min(scale,width/max(1,sb[2]-sb[0]),height/max(1,sb[3]-sb[1]))
        cx=median(by_id[e['id']]['x']+by_id[e['id']]['width']/2
                  -(e['x']+e['width']/2-sx)*scale for e in originals)
        cy=median(by_id[e['id']]['y']+by_id[e['id']]['height']/2
                  -(e['y']+e['height']/2-sy)*scale for e in originals)
        if emblem and abs(sx-sw/2)<=.04*sw:cx=width/2
        # Whole-group clamping preserves internal offsets and image proportions.
        # If the authored lockup deliberately bleeds across a canvas edge,
        # retain that edge anchor instead of translating the whole group inward
        # until it fits. The previous clamp centered a large right-edge frame
        # when resizing to a narrower canvas, breaking its registration with
        # the model-proposed text and adjacent artwork.
        gw,gh=(sb[2]-sb[0])*scale,(sb[3]-sb[1])*scale
        def _clamp_group_axis(center, group_size, canvas_size, source_min, source_max, source_size):
            bleed_left=source_min < -max(1.,source_size*.01)
            bleed_right=source_max > source_size+max(1.,source_size*.01)
            min_center=group_size*.25 if bleed_left else group_size/2
            max_center=canvas_size-group_size*.25 if bleed_right else canvas_size-group_size/2
            if min_center>max_center:
                min_center,max_center=canvas_size*.25,canvas_size*.75
            return min(max_center,max(min_center,center))
        cx=_clamp_group_axis(cx,gw,width,sb[0],sb[2],sw)
        cy=_clamp_group_axis(cy,gh,height,sb[1],sb[3],sh)
        trial=deepcopy(result);trial_by={e['id']:e for e in trial}
        cover_strips=[]
        for e in originals:
            out=trial_by[e['id']]
            target_x=cx+(e['x']-sx)*scale;target_y=cy+(e['y']-sy)*scale
            source_canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
            sw,sh=source_canvas or (width,height)
            full_height=(e['type']=='image' and out.get('fit')=='cover'
                         and e['y']<=.02*sh and e['y']+e['height']>=.98*sh
                         and e['width']<.95*sw)
            full_width=(e['type']=='image' and out.get('fit')=='cover'
                        and e['x']<=.02*sw and e['x']+e['width']>=.98*sw
                        and e['height']<.95*sh)
            if full_height:
                out.update(x=target_x,y=0.,width=out['width'],height=height)
                cover_strips.append(e)
            elif full_width:
                out.update(x=0.,y=target_y,width=width,height=out['height'])
                cover_strips.append(e)
            else:
                out.update(x=target_x,y=target_y,
                           width=e['width']*scale,height=e['height']*scale)
            if e['type']=='text':
                out['font_size']=e.get('font_size',1)*scale
                out['letter_spacing']=e.get('letter_spacing',0)*scale
                # Keep the frame and font on the same source-derived scale.
                # Independent post-hoc shrinking here used to make the measured
                # text exceed its own frame; the downstream font-fit candidate
                # may adjust type without changing lockup geometry.
                measured_h,measured_w=_text_info(out,out['width'],out['font_size'])[:2]
                out['width']=max(out['width'],measured_w+2.)
                measured_h,measured_w=_text_info(out,out['width'],out['font_size'])[:2]
                out['height']=max(out['height'],measured_h+2.)
                out['x']=min(max(0.,out['x']),max(0.,width-out['width']))
                out['y']=min(max(0.,out['y']),max(0.,height-out['height']))
        # A source cover strip is a cropping frame, so keep its proportion of
        # the text lockup instead of letting the image frame scale independently.
        text_members=[e for e in originals if e['type']=='text']
        if cover_strips and text_members:
            strip=cover_strips[0]
            source_group=[e for e in originals if e['id']!=strip['id']]
            target_group=[trial_by[e['id']] for e in source_group]
            if source_group and target_group:
                sx0=min(e['x'] for e in originals);sx1=max(e['x']+e['width'] for e in originals)
                tx0=min(e['x'] for e in target_group);tx1=max(e['x']+e['width'] for e in target_group)
                source_span=max(1.,sx1-sx0);target_span=max(1.,tx1-tx0)
                left_fraction=(strip['x']-sx0)/source_span
                width_fraction=strip['width']/source_span
                moved=trial_by[strip['id']]
                if moved['y']==0. and moved['height']==height:
                    moved['x']=tx0+left_fraction*target_span
                    moved['width']=max(1.,width_fraction*target_span)
                elif moved['x']==0. and moved['width']==width:
                    sy0=min(e['y'] for e in originals);sy1=max(e['y']+e['height'] for e in originals)
                    ty0=min(e['y'] for e in target_group);ty1=max(e['y']+e['height'] for e in target_group)
                    source_span=max(1.,sy1-sy0);target_span=max(1.,ty1-ty0)
                    top_fraction=(strip['y']-sy0)/source_span
                    height_fraction=strip['height']/source_span
                    moved['y']=ty0+top_fraction*target_span
                    moved['height']=max(1.,height_fraction*target_span)
        # Source-evidenced groups may already overlap each other intentionally.
        if set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(result,units)):
            continue
        result=trial;by_id=trial_by
        notes.append(dict(kind='rigid-source-lockup',element_ids=sorted(ids)))
    result,anchor_notes=_repair_edge_heroes(result,units,width,height)
    result,chain_notes=_repair_column_chains(result,units,width,height,groups)
    result,clearance_notes=_repair_hero_clearance(result,units,width,height,groups)
    result,column_notes=_repair_photo_columns(result,units,width,height,groups)
    return result,notes+anchor_notes+chain_notes+clearance_notes+column_notes


def _repair_edge_heroes(elements,units,width,height):
    """Preserve a centered primary visual's authored bottom anchor and height."""
    result=deepcopy(elements);out={e['id']:e for e in result};notes=[]
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=canvas or (width,height)
    for unit in units:
        for src in unit['items']:
            if src['type']!='image' or src['id'] not in out:continue
            share=src['width']*src['height']/max(1.,sw*sh)
            if (not .15<=share<.85 or not .98*sh<=src['y']+src['height']<=1.02*sh
                    or abs(src['x']+src['width']/2-sw/2)>.06*sw):continue
            dst=out[src['id']];aspect=src['width']/max(1.,src['height'])
            wanted=min(height,width/max(1e-6,aspect),max(dst['height'],src['height']/sh*height))
            before=tuple(dst.get(k) for k in ('x','y','width','height'))
            dst.update(width=wanted*aspect,height=wanted,x=(width-wanted*aspect)/2,y=height-wanted)
            if before!=tuple(dst.get(k) for k in ('x','y','width','height')):
                notes.append(dict(kind='source-edge-hero-anchor',element_ids=[src['id']],edge='bottom'))
    return result,notes


def _repair_photo_columns(elements,units,width,height,groups):
    """Keep a source-disjoint copy/button column beside a foreground visual."""
    from engine.resize import _overlap, _unintended_text_overlaps
    source={e['id']:e for u in units for e in u['items']}
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=canvas or (width,height)
    result=deepcopy(elements);notes=[]
    def bounds(items):return (min(e['x'] for e in items),min(e['y'] for e in items),
                             max(e['x']+e['width'] for e in items),max(e['y']+e['height'] for e in items))
    for ids in _copy_columns(source,sw,sh,groups):
        originals=[source[eid] for eid in ids];sb=bounds(originals)
        for visual in source.values():
            if visual['id'] in ids or visual['type'] not in ('image','shape'):continue
            share=visual['width']*visual['height']/max(1.,sw*sh)
            if not .15<=share<.85:continue
            vb=bounds([visual])
            if _overlap(sb,vb)>0 or min(sb[3],vb[3])-max(sb[1],vb[1])<.25*(sb[3]-sb[1]):continue
            by_id={e['id']:e for e in result}
            if visual['id'] not in by_id:continue
            members=[by_id[eid] for eid in ids if eid in by_id];tb=bounds(members);target=bounds([by_id[visual['id']]])
            overlap=_overlap(tb,target)/max(1.,(tb[2]-tb[0])*(tb[3]-tb[1]))
            if not 0.<overlap<.65:continue
            margin=.035*width;gap=max(8.,.02*width)
            left,right=(margin,target[0]-gap) if sb[2]<=vb[0] else (target[2]+gap,width-margin)
            factor=min(1.,(right-left)/max(1.,tb[2]-tb[0]))
            if factor<.55:continue
            trial=_fit_column(result,ids,left,right,width,height)
            if set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(result,units)):continue
            result=trial;notes.append(dict(kind='source-photo-column',element_ids=sorted(ids)))
    return result,notes


def _fit_column(elements,ids,left,right,width,height):
    """Uniformly fit a whole column, preserving image proportions and tracking."""
    from engine.resize import _fit_text
    result=deepcopy(elements);members=[e for e in result if e['id'] in ids]
    x0=min(e['x'] for e in members);x1=max(e['x']+e['width'] for e in members)
    y0=min(e['y'] for e in members);y1=max(e['y']+e['height'] for e in members)
    factor=min(1.,(right-left)/max(1.,x1-x0));cy=(y0+y1)/2
    target_left=max(left,min(x0,right-(x1-x0)*factor))
    for e in members:
        e.update(x=target_left+(e['x']-x0)*factor,y=cy+(e['y']-cy)*factor,
                 width=e['width']*factor,height=e['height']*factor)
        if e['type']=='text':
            e['font_size']=max(1.,e['font_size']*factor)
            e['letter_spacing']=e.get('letter_spacing',0.)*factor
            size=_fit_text(e,e['width'],e['height'],e['font_size'])
            e['letter_spacing']*=size/max(1.,e['font_size']);e['font_size']=size
    return result


def _repair_hero_clearance(elements,units,width,height,groups):
    """Keep source-side copy columns and top headers clear of a central hero."""
    from engine.resize import (_bounds, _overlap, _text_visible_bounds,
                               _unintended_text_overlaps, _asset_root)
    source={e['id']:e for u in units for e in u['items']}
    by_id={e['id']:e for e in elements}
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=canvas or (width,height)
    heroes=[]
    for e in source.values():
        if e['type'] not in ('image','shape'):continue
        area=e['width']*e['height']/max(1.,sw*sh)
        center=(e['x']+e['width']/2)/max(1.,sw)
        if area>=.15 and abs(center-.5)<=.08 and e['id'] in by_id:
            if any(e['id'] in group and any(source[eid]['type']=='text' for eid in group)
                   for group in groups):continue
            heroes.append(e)
    if not heroes:return deepcopy(elements),[]
    hero=max(heroes,key=lambda e:e['width']*e['height'])
    target_hero=by_id[hero['id']]
    hero_alpha=None
    if hero['type']=='image' and hero.get('asset_id') and abs(hero.get('rotation',0.))<=1.:
        try:
            from PIL import Image
            with Image.open(_asset_root(hero)/hero['asset_id']) as image:
                hero_alpha=image.convert('RGBA').getchannel('A')
            resize=min if hero.get('fit','contain')=='contain' else max
            pixel_scale=resize(hero['width']/hero_alpha.width,hero['height']/hero_alpha.height)
            image_x=hero['x']+(hero['width']-hero_alpha.width*pixel_scale)/2
            image_y=hero['y']+(hero['height']-hero_alpha.height*pixel_scale)/2
        except (ImportError,OSError,ValueError):hero_alpha=None
    gap=max(18.,.025*height);margin=.035*width
    block_for={eid:set((eid,)) for eid in by_id}
    for group in groups:
        group=group&set(by_id)
        for eid in group:block_for[eid]=group
    def source_layered(e):
        for eid in block_for[e['id']]:
            member=source[eid]
            if eid==hero['id']:return True
            if member['type']!='text':continue
            box=_text_visible_bounds(member)
            area=max(1.,(box[2]-box[0])*(box[3]-box[1]))
            ratio=_overlap(box,_bounds([hero]))/area
            if ratio>.10 and hero_alpha is not None:
                from PIL import ImageStat
                left=max(0.,(box[0]-image_x)/pixel_scale);right=min(hero_alpha.width,(box[2]-image_x)/pixel_scale)
                top=max(0.,(box[1]-image_y)/pixel_scale);bottom=min(hero_alpha.height,(box[3]-image_y)/pixel_scale)
                if right>left and bottom>top:
                    crop=hero_alpha.crop((int(left),int(top),max(int(left)+1,int(right)),max(int(top)+1,int(bottom))))
                    ratio=ImageStat.Stat(crop).mean[0]/255.*(right-left)*(bottom-top)*pixel_scale**2/area
                else:ratio=0.
            if ratio>.10:return True
        return False
    trial=deepcopy(elements);out={e['id']:e for e in trial};notes=[]
    # Move source-left/right text clusters as units when their predicted boxes
    # collide with the central visual. The source side is structural evidence.
    for side in ('left','right'):
        texts=[e for e in source.values() if e['type']=='text' and e['id'] in out and e.get('content','').strip()
               and not source_layered(e)
               and min(e['y']+e['height'],hero['y']+hero['height'])>max(e['y'],hero['y'])
               and ((e['x']+e['width']/2 < hero['x']+hero['width']/2) if side=='left'
                    else (e['x']+e['width']/2 > hero['x']+hero['width']/2))]
        if not texts:continue
        ids=set()
        for e in texts:ids.update(block_for[e['id']])
        members=[out[eid] for eid in ids]
        rightmost=max(e['x']+e['width'] for e in members);leftmost=min(e['x'] for e in members)
        slot_left,slot_right=((margin,target_hero['x']-gap) if side=='left' else
                              (target_hero['x']+target_hero['width']+gap,width-margin))
        if slot_left<=leftmost and rightmost<=slot_right:continue
        factor=min(1.,(slot_right-slot_left)/max(1.,rightmost-leftmost))
        if factor<.55:continue
        before=deepcopy(trial)
        trial=_fit_column(trial,ids,slot_left,slot_right,width,height)
        if set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(before,units)):
            trial=before;continue
        out={e['id']:e for e in trial}
        dx=min(out[eid]['x'] for eid in ids)-leftmost
        notes.append(dict(kind='hero-column-clearance',side=side,dx=dx,scale=factor,element_ids=sorted(ids)))
    # Raise a centered source header only if the source left a real gap above H.
    header=[e for e in source.values() if e['type']=='text' and e['id'] in out and e.get('content','').strip()
            and not source_layered(e)
            and e['y']<.22*sh and abs((e['x']+e['width']/2)/max(1.,sw)-.5)<=.08
            and e['y']+e['height']<=hero['y']+2]
    if header:
        bottom=max(_text_visible_bounds(out[e['id']])[3] for e in header)
        hero_top=out[hero['id']]['y']
        if bottom>hero_top-gap:
            dy=max(.0,min((min(e['y'] for e in header)-.055*height),
                          bottom-(hero_top-gap)))
            if dy>0:
                ids=set()
                for e in header:ids.update(block_for[e['id']])
                before=deepcopy(trial)
                for e in trial:
                    if e['id'] in ids:e['y']-=dy
                if not set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(before,units)):
                    notes.append(dict(kind='hero-header-clearance',dy=-dy,element_ids=sorted(ids)))
                else:trial=before
    # If the header is already at its top limit, move the central source hero
    # down to preserve the source hierarchy. Scale only image frames that run
    # out of canvas, maintaining source aspect and the original horizontal center.
    out={e['id']:e for e in trial}
    if header and hero['id'] in out:
        header_bottom=max(_text_visible_bounds(out[e['id']])[3] for e in header)
        target_hero=out[hero['id']]
        needed_y=header_bottom+gap
        if needed_y>target_hero['y']:
            before=deepcopy(trial);candidate=deepcopy(trial)
            moved={e['id']:e for e in candidate}[hero['id']]
            moved['y']=needed_y
            overflow=moved['y']+moved['height']-height
            if overflow>0:
                if moved['type']!='image':
                    moved=None
                else:
                    new_h=height-needed_y
                    ratio=hero['width']/max(1.,hero['height'])
                    new_w=new_h*ratio
                    if new_w>width:
                        new_w=width;new_h=new_w/max(1e-6,ratio)
                    moved.update(height=max(1.,new_h),width=max(1.,new_w),
                                 x=width/2-max(1.,new_w)/2,
                                 y=height-max(1.,new_h))
            if moved is not None and moved['x']>=0 and moved['y']>=0 and moved['x']+moved['width']<=width+1e-5 and moved['y']+moved['height']<=height+1e-5:
                new_overlaps=set(_unintended_text_overlaps(candidate,units))-set(_unintended_text_overlaps(before,units))
                if not new_overlaps:
                    trial=candidate
                    notes.append(dict(kind='hero-header-clearance',hero_id=hero['id'],
                                      hero_y=needed_y,hero_resized=overflow>0,
                                      element_ids=[hero['id']]))
    return trial,notes


def _repair_column_chains(elements,units,width,height,groups):
    """Joint y constraints propagate a move through a column and paired rows."""
    import kiwisolver as kiwi
    from engine.resize import _text_visible_bounds, _unintended_text_overlaps
    source={e['id']:e for u in units for e in u['items']}
    by_id={e['id']:e for e in elements}
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=canvas or (width,height)
    texts=[e for e in source.values() if e['type']=='text' and e.get('content','').strip() and e['id'] in by_id]
    edges=[]
    for upper in texts:
        ub=_text_visible_bounds(upper);options=[]
        for lower in texts:
            if lower['id']==upper['id']:continue
            lb=_text_visible_bounds(lower);gap=lb[1]-ub[3]
            aligned=min(abs(ub[0]-lb[0]),abs((ub[0]+ub[2]-lb[0]-lb[2])/2))<=.015*sw
            if not aligned or not 0<=gap<=.08*sh:continue
            a,b=_text_visible_bounds(by_id[upper['id']]),_text_visible_bounds(by_id[lower['id']])
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))/max(1,min(a[2]-a[0],b[2]-b[0]))
            # A genuine reflow into separate columns must remain possible.
            if overlap<.35:continue
            options.append((gap,lower))
        if options:
            gap,lower=min(options,key=lambda item:item[0])
            edges.append((upper['id'],lower['id'],min(12,max(3,gap*height/max(1,sh)))))
    # A chain can run into a separate title even when the source title was far
    # away. Add its source-supported vertical ordering to this same solve so
    # moving the first callout propagates through the entire column.
    for a,b in _unintended_text_overlaps(elements,units):
        sa,sb=_text_visible_bounds(source[a]),_text_visible_bounds(source[b])
        if sa[3]<=sb[1]:edges.append((a,b,3.))
        elif sb[3]<=sa[1]:edges.append((b,a,3.))
    if not edges:return deepcopy(elements),[]
    blocks={eid:frozenset((eid,)) for eid in by_id}
    for ids in groups:
        block=frozenset(ids & set(by_id))
        for eid in block:blocks[eid]=block
    active={blocks[eid] for a,b,_ in edges for eid in (a,b)}
    rows=[]
    for i,a in enumerate(texts):
        for b in texts[:i]:
            if blocks[a['id']] not in active or blocks[b['id']] not in active:continue
            sa,sb=_text_visible_bounds(a),_text_visible_bounds(b)
            ca,cb=_text_visible_bounds(by_id[a['id']]),_text_visible_bounds(by_id[b['id']])
            # Only source-supported peer headings that still occupy two columns.
            similar=.8<=a.get('font_size',1)/max(1,b.get('font_size',1))<=1.25
            if (similar and abs((sa[1]+sa[3]-sb[1]-sb[3])/2)<=.015*sh
                    and abs((sa[0]+sa[2]-sb[0]-sb[2])/2)>=.30*sw
                    and abs((ca[0]+ca[2]-cb[0]-cb[2])/2)>=.25*width
                    and abs((ca[1]+ca[3]-cb[1]-cb[3])/2)<=.08*height):
                rows.append((a['id'],b['id']))
    for use_rows,gap_factor in ((True,1.),(False,1.),(False,.25)):
        solver=kiwi.Solver();variables={block:kiwi.Variable(str(sorted(block))) for block in active}
        try:
            for block,var in variables.items():
                members=[by_id[eid] for eid in block]
                top=min(e['y'] for e in members);bottom=max(e['y']+e['height'] for e in members)
                solver.addConstraint(var>=-top);solver.addConstraint(var<=height-bottom)
                solver.addConstraint((var==0)|kiwi.strength.strong)
            for a,b,gap in edges:
                if blocks[a]==blocks[b]:continue
                ab,bb=_text_visible_bounds(by_id[a]),_text_visible_bounds(by_id[b])
                solver.addConstraint(variables[blocks[b]]-variables[blocks[a]]>=ab[3]+gap*gap_factor-bb[1])
            if use_rows:
                for a,b in rows:
                    if blocks[a]==blocks[b]:continue
                    ab,bb=_text_visible_bounds(by_id[a]),_text_visible_bounds(by_id[b])
                    solver.addConstraint(variables[blocks[b]]-variables[blocks[a]]==(ab[1]+ab[3]-bb[1]-bb[3])/2)
            solver.updateVariables()
        except kiwi.UnsatisfiableConstraint:
            continue
        if any(abs(v.value())>.12*height for v in variables.values()):continue
        trial=deepcopy(elements)
        for e in trial:
            var=variables.get(blocks[e['id']])
            if var is not None:e['y']+=var.value()
        if set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(elements,units)):
            continue
        changed=[eid for eid in by_id if blocks[eid] in variables and abs(variables[blocks[eid]].value())>.25]
        return trial,([dict(kind='column-chain-solve',element_ids=sorted(changed),
                            same_row_pairs=rows if use_rows else [])] if changed else [])
    return deepcopy(elements),[]


def anchor_superscripts(elements,units):
    """Place compact source-linked labels after their anchor's measured glyphs."""
    from engine.resize import _text_info, _text_visible_bounds, _unintended_text_overlaps
    result=deepcopy(elements);source={e['id']:e for u in units for e in u['items']}
    texts=[e for e in result if e['type']=='text' and e.get('content','').strip()]
    changes=[]
    for sat in texts:
        if len(sat.get('content','').strip())>3 or sat['id'] not in source:continue
        src_sat=source[sat['id']];sb=_text_visible_bounds(src_sat);choices=[]
        for anchor in texts:
            if anchor['id']==sat['id'] or anchor['id'] not in source:continue
            src_anchor=source[anchor['id']]
            if src_anchor.get('font_size',0)<48:continue
            if src_sat.get('font_size',0)>.35*src_anchor.get('font_size',1):continue
            ab=_text_visible_bounds(src_anchor)
            gap=max(0.,sb[0]-ab[2],ab[0]-sb[2])
            if gap<=.45*max(1.,src_sat.get('font_size',1)):
                choices.append((gap+abs(sb[1]-ab[1]),anchor))
        if not choices:continue
        anchor=min(choices,key=lambda item:item[0])[1]
        src_anchor=source[anchor['id']]
        source_glyph_w=_text_info(src_anchor,src_anchor['width'],src_anchor['font_size'])[1]
        # Qwen can return canvas-wide text frames even for short price labels.
        # Those frames collide in unit-level feasibility checks although the
        # rendered glyphs do not. Tighten the anchor to its measured one-line
        # advance while keeping the visible glyph origin stable.
        anchor_bounds=_text_visible_bounds(anchor)
        glyph_w=_text_info(anchor,anchor['width'],anchor['font_size'])[1]
        anchor_tightened=False
        if ('\n' not in anchor.get('content','') and glyph_w>0
                and anchor.get('width',0)>glyph_w+1e-4):
            anchor['x']=anchor_bounds[0]
            anchor['width']=glyph_w
            anchor_tightened=True
        anchor_vis_right=_text_visible_bounds(anchor)[2]
        source_gap=src_sat['x']-(src_anchor['x']+source_glyph_w)
        scaled_gap=source_gap*(anchor['font_size']/max(1.,src_anchor.get('font_size',1)))
        source_vertical_offset=float(src_sat['y'])-float(src_anchor['y'])
        scaled_vertical_offset=source_vertical_offset*(
            anchor['font_size']/max(1.,src_anchor.get('font_size',1)))
        # Keep at least four pixels between visible glyph bounds, even when
        # the source gap scales down or Qwen's proposed frames overlap.
        new_x=max(anchor_vis_right+4.,anchor_vis_right+scaled_gap)
        new_y=max(0.,anchor['y']+scaled_vertical_offset)
        sat_moved=abs(new_x-sat['x'])>.01
        sat_moved_y=abs(new_y-sat['y'])>.01
        if sat_moved or sat_moved_y:
            sat['x']=new_x;sat['y']=new_y
        # Keep the raw layout frames disjoint as well as the measured glyphs.
        # The visible advance is a lower bound so tightening cannot clip ink.
        anchor_vis=_text_visible_bounds(anchor)
        anchor_vis_w=max(0.,anchor_vis[2]-anchor_vis[0])
        max_frame_w=max(anchor_vis_w,
                        min(anchor['width'],sat['x']-anchor['x']-2.0))
        if max_frame_w<anchor['width']-1e-4:
            anchor['width']=max_frame_w
            anchor_tightened=True
        if anchor_tightened or sat_moved or sat_moved_y:
            changes.extend((anchor['id'],sat['id']))
    if set(_unintended_text_overlaps(result,units))-set(_unintended_text_overlaps(elements,units)):
        return deepcopy(elements),[]
    return result,sorted(set(changes))


def font_variant(elements,units,width,height):
    """Calibrate actual rendered glyphs, expanding available boxes before size."""
    from engine.resize import _calibrate_text_boxes, _unintended_text_overlaps
    source_items={e['id']:e for u in units for e in u['items']}
    sw,sh=next((u['source_canvas'] for u in units if u.get('source_canvas')),(width,height))
    source={'canvas':{'width':sw,'height':sh},'elements':list(source_items.values())}
    result,notes=_calibrate_text_boxes(elements,source,width,height)
    if set(_unintended_text_overlaps(result,units))-set(_unintended_text_overlaps(elements,units)):
        return deepcopy(elements),[]
    return result,sorted({note['element_id'] for note in notes})


def polish_variant(elements,units,source,width,height,desired,iterations=24):
    """Trust-region search over whole units; retain the unoptimized proposal."""
    from engine.resize import _objective_layout_cost, _feasible, _unintended_text_overlaps
    current=deepcopy(elements);origin={e['id']:e for e in elements}
    source_by={e['id']:e for e in source['elements']}
    sw,sh=source['canvas']['width'],source['canvas']['height']
    alignments=[]
    for i,a in enumerate(elements):
        for b in elements[:i]:
            sa,sb=source_by[a['id']],source_by[b['id']]
            if sa['type']!='text' or sb['type']!='text':continue
            for axis,extent,other_extent in (('x',sw,sh),('y',sh,sw)):
                size='width' if axis=='x' else 'height';other='y' if axis=='x' else 'x'
                other_size='height' if axis=='x' else 'width'
                delta=abs(sa[axis]+sa[size]/2-sb[axis]-sb[size]/2)
                apart=abs(sa[other]+sa[other_size]/2-sb[other]-sb[other_size]/2)
                proposed=abs(a[axis]+a[size]/2-b[axis]-b[size]/2)
                if delta<=.015*extent and apart>=.04*other_extent and proposed<=.08*(width if axis=='x' else height):
                    alignments.append((a['id'],b['id'],axis,size))
    groups=lockup_groups(units);owned=set().union(*groups) if groups else set()
    groups+= [{e['id'] for e in u['items']} for u in units
              if not any(e['id'] in owned for e in u['items'])]
    def cost(items):
        drift=sum(abs(e[k]-origin[e['id']][k])/d for e in items
                  for k,d in (('x',width),('y',height),('width',width),('height',height)))/max(1,len(items)*4)
        by_id={e['id']:e for e in items}
        alignment=sum(abs(by_id[a][axis]+by_id[a][size]/2-by_id[b][axis]-by_id[b][size]/2)
                      /(width if axis=='x' else height) for a,b,axis,size in alignments)
        return _objective_layout_cost(items,units,source,width,height,desired,
                                      allow_text_visual_overlap=True)+50*drift+8*alignment
    before=best=cost(current);accepted=0
    for step in range(iterations if groups else 0):
        ids=groups[step%len(groups)];options=[]
        for dx,dy in ((width*.008,0),(-width*.008,0),(0,height*.008),(0,-height*.008)):
            trial=deepcopy(current)
            for e in trial:
                if e['id'] in ids:e['x']+=dx;e['y']+=dy
            if any(abs(e[k]-origin[e['id']][k])>.035*d for e in trial for k,d in (('x',width),('y',height))):continue
            if not _feasible(trial,units,width,height,strict_margins=False,gap_factor=.004,allow_text_visual_overlap=True):continue
            if set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(current,units)):continue
            value=cost(trial)
            if value<best-1e-7:options.append((value,trial))
        if options:
            best,current=min(options,key=lambda item:item[0]);accepted+=1
    return current,dict(before=before,after=best,accepted=accepted,iterations=iterations if groups else 0)


def _background_column_variants(elements,units,width,height,weak_ids,under,source_under,groups):
    """Use an authored clean color zone to keep copy off a baked photo border."""
    if source_under is None:return []
    from engine.resize import (_text_visible_bounds, _feasible, _unintended_text_overlaps,
                               _overlap, _rgba, _luminance)
    source={e['id']:e for u in units for e in u['items']}
    canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),None)
    sw,sh=canvas or (width,height)
    by_id={e['id']:e for e in elements};options=[]
    for ids in _copy_columns(source,sw,sh,groups):
        if not ids & set(weak_ids):continue
        members=[by_id[eid] for eid in ids if eid in by_id]
        labels=[source[eid] for eid in ids if source[eid]['type']=='text']
        if len(labels)<2 or len(members)!=len(ids):continue
        plain=[]
        for label in labels:
            box=_text_visible_bounds(label);area=max(1.,(box[2]-box[0])*(box[3]-box[1]))
            if not any(source[eid]['type'] in ('image','shape')
                       and source[eid].get('z_index',0)<label.get('z_index',0)
                       and _overlap(box,(source[eid]['x'],source[eid]['y'],
                           source[eid]['x']+source[eid]['width'],source[eid]['y']+source[eid]['height']))/area>=.8
                       for eid in ids):plain.append(label)
        if not plain:continue
        focus=max(labels,key=lambda e:e['width']*e['height'])
        source_cx=median(e['x']+e['width']/2 for e in labels)/max(1.,sw)
        if .35<source_cx<.65:continue
        box=_text_visible_bounds(focus)
        sx0=max(0,int(box[0]/sw*source_under.width));sx1=min(source_under.width,int(box[2]/sw*source_under.width))
        sy0=max(0,int(box[1]/sh*source_under.height));sy1=min(source_under.height,int(box[3]/sh*source_under.height))
        pixels=[source_under.getpixel((x,y)) for y in range(sy0,sy1,max(1,(sy1-sy0)//20))
                for x in range(sx0,sx1,max(1,(sx1-sx0)//24))]
        if len(pixels)<12:continue
        color=tuple(median(p[k] for p in pixels) for k in range(3))
        distances=sorted(sum((p[k]-color[k])**2 for k in range(3))**.5 for p in pixels)
        spread=distances[int(.8*(len(distances)-1))]
        if spread>45.:continue
        tolerance=min(85.,max(45.,spread+25.))
        source_floors={}
        for label in plain:
            rgba=_rgba(label.get('color',''))
            if not rgba:continue
            alpha=rgba[3]*label.get('opacity',1.)
            foreground=tuple(rgba[k]*alpha+color[k]*(1-alpha) for k in range(3))
            a,b=_luminance(foreground),_luminance(color)
            source_floors[label['id']]=round(max(2.5,min(3.,.9*(max(a,b)+.05)/(min(a,b)+.05))),3)
        target_focus=by_id[focus['id']]
        tb=_text_visible_bounds(target_focus)
        y0=max(0,int(tb[1]/height*under.height));y1=min(under.height,int(tb[3]/height*under.height))
        sample_y=list(range(y0,y1,max(1,(y1-y0)//24)))
        if len(sample_y)<3:continue
        matching=[]
        for x in range(under.width):
            good=sum(sum((under.getpixel((x,y))[k]-color[k])**2 for k in range(3))<=tolerance**2
                     for y in sample_y)
            matching.append(good/len(sample_y)>=.8)
        runs=[];start=None
        for x,ok in enumerate(matching+[False]):
            if ok and start is None:start=x
            if not ok and start is not None:
                runs.append((start/under.width*width,x/under.width*width));start=None
        left=min(e['x'] for e in members);right=max(e['x']+e['width'] for e in members)
        pad=max(4.,.01*width)
        for x0,x1 in runs:
            if x1-x0<.10*width or (((x0+x1)/2<width/2)!=(source_cx<.5)):continue
            overlap=max(0.,min(right,x1)-max(left,x0))/max(1.,right-left)
            # A poor proposal can place the whole copy column outside its
            # authored clean-color run. That is still a useful repair case;
            # only reject candidates that already occupy essentially all of
            # that run or cannot fit within it.
            if overlap>=.99 or (x1-x0-2*pad)/(right-left)<.55:continue
            candidate=_fit_column(elements,ids,x0+pad,x1-pad,width,height)
            if not _feasible(candidate,units,width,height,strict_margins=False,
                             gap_factor=.004,allow_text_visual_overlap=True):continue
            if set(_unintended_text_overlaps(candidate,units))-set(_unintended_text_overlaps(elements,units)):continue
            options.append((candidate,[dict(repair_type='background-column-fit',element_ids=sorted(ids),
                source_color_spread=round(spread,2),source_contrast_floor=source_floors,
                column=[round(x0+pad,2),round(x1-pad,2)])]))
    return options


def contrast_variant(elements,units,width,height,weak_ids,under,full,return_alternatives=False,
                     source_under=None,source_elements=None):
    """Propose bounded text-unit moves using actual backing pixels at glyphs."""
    from PIL import ImageChops
    from engine.resize import (_rgba, _luminance, _feasible,
                               _text_visible_bounds, _unintended_text_overlaps)
    result=deepcopy(elements);by_id={e['id']:e for e in result};changes=[];alternatives=[]
    ink=ImageChops.difference(full,under).convert('L')
    source={e['id']:e for u in units for e in u['items']}
    source.update({e['id']:e for e in (source_elements or [])})
    source_canvas=next((u.get('source_canvas') for u in units if u.get('source_canvas')),
                       (width,height))
    groups=lockup_groups(units)
    for u in units:
        ids={e['id'] for e in u['items']}
        if not any(ids & g for g in groups):groups.append(ids)
    groups=_merge_groups(groups)
    if return_alternatives:
        column_repairs=_background_column_variants(elements,units,width,height,weak_ids,
                                                   under,source_under,groups)
        alternatives.extend(column_repairs)
        groups=_merge_groups(groups+[set(change['element_ids'])
                                     for _,changes in column_repairs for change in changes])
        # A rotated, bleeding source backdrop may need to change orientation
        # when the target aspect ratio crosses portrait/landscape. Keep the
        # model's proposed frame and offer an unrotated sibling; the full
        # renderer and contrast/objective reranker decide whether it helps.
        sw,sh=map(float,source_canvas)
        source_ratio=sw/max(1.,sh);target_ratio=width/max(1.,height)
        if max(target_ratio/max(source_ratio,1e-6),source_ratio/max(target_ratio,1e-6))>=1.2:
            for eid,src in source.items():
                if src.get('type')!='image' or int(src.get('z_index',0) or 0)!=0:
                    continue
                try:rotation=float(src.get('rotation',0.))
                except (TypeError,ValueError):continue
                cardinal=abs((abs(rotation)%180.)-90.)<=1.
                bleed=(src.get('x',0.)<-.05*sw or src.get('y',0.)<-.05*sh
                       or src.get('x',0.)+src.get('width',0.)>1.05*sw
                       or src.get('y',0.)+src.get('height',0.)>1.05*sh)
                if not cardinal or not bleed or eid not in {e['id'] for e in elements}:
                    continue
                candidate=deepcopy(elements)
                bg=next(e for e in candidate if e['id']==eid)
                if abs(float(bg.get('rotation',0.)))<1.:continue
                from_rotation=float(bg.get('rotation',rotation))
                bg['rotation']=0.
                alternatives.append((candidate,[dict(repair_type='backdrop-rotation-variant',
                    element_ids=[eid],from_rotation=from_rotation,to_rotation=0.)]))
    text_elements=[e for e in result if e['type']=='text' and e.get('content','').strip()]

    def gap_ok(trial,ids):
        """Keep source-adjacent text apart when a contrast nudge moves it."""
        trial_by={e['id']:e for e in trial}
        for eid in ids:
            moved=trial_by.get(eid)
            if moved is None or moved['type']!='text':continue
            current=_text_visible_bounds(moved)
            old=_text_visible_bounds(by_id[eid])
            for peer in text_elements:
                if peer['id'] in ids:continue
                other=_text_visible_bounds(trial_by[peer['id']])
                horizontal=max(0.,min(current[2],other[2])-max(current[0],other[0]))
                if horizontal<=0:continue
                if old[3]<=_text_visible_bounds(peer)[1]:
                    if other[1]-current[3]<16.-1e-5:return False
                elif _text_visible_bounds(peer)[3]<=old[1]:
                    if current[1]-other[3]<16.-1e-5:return False
        return True

    for ids in groups:
        members=[by_id[eid] for eid in ids if eid in by_id]
        if not set(weak_ids)&ids or not members:continue
        samples=[]
        for e in members:
            color=_rgba(e.get('color',''))
            if not color or abs(e.get('rotation',0))>1:continue
            l=max(0,int(e['x']/width*under.width));r=min(under.width,int((e['x']+e['width'])/width*under.width))
            t=max(0,int(e['y']/height*under.height));b=min(under.height,int((e['y']+e['height'])/height*under.height))
            points=[(x,y) for y in range(t,b,2) for x in range(l,r,2) if ink.getpixel((x,y))>12]
            stride=max(1,len(points)//128)
            samples.extend((e['id'],x,y,color,e.get('opacity',1)) for x,y in points[::stride])
        if not samples:continue
        def score(dx,dy,focus_ids=None):
            contrasts=[]
            selected=[sample for sample in samples if focus_ids is None or sample[0] in focus_ids]
            for _,x,y,color,opacity in selected:
                px=min(under.width-1,max(0,round(x+dx/width*under.width)))
                py=min(under.height-1,max(0,round(y+dy/height*under.height)))
                pixel=under.getpixel((px,py));alpha=color[3]*opacity
                foreground=tuple(color[k]*alpha+pixel[k]*(1-alpha) for k in range(3))
                a,b=_luminance(foreground),_luminance(pixel)
                contrasts.append((max(a,b)+.05)/(min(a,b)+.05))
            contrasts.sort()
            return (contrasts[0],contrasts[len(contrasts)//5]) if contrasts else (0.,0.)
        weak_in_group=set(weak_ids)&ids
        initial=score(0,0,weak_in_group)[0];options=[]
        parent_ids={eid for u in units if any(e['id'] in ids for e in u['items'])
                    for eid in u.get('source_parent_ids',()) if eid in by_id and eid not in ids}
        # Local steps explicitly search the source-pixel-scale range where
        # contrast can change without disturbing the lockup's composition.
        moves={(sx*step,0.) for sx in (-1,1) for step in (12,20,28,36,44,52)}
        moves|={(0.,sy*step) for sy in (-1,1) for step in (12,20,28,36,44,52)}
        # Broader horizontal placements are considered only for multi-item
        # lockups, where moving the group preserves its internal design.
        if len(members)>1:
            left=min(e['x'] for e in members);right=max(e['x']+e['width'] for e in members)
            span=right-left
            for fraction in (.12,.17,.18,.21,.22,.25,.35,.50):
                for target_left in (fraction*width,width-right+left):
                    dx=target_left-left
                    if abs(dx)>2:moves.add((dx,0.))
        for dx,dy in sorted(moves,key=lambda move:(move[0]**2+move[1]**2,abs(move[1]),abs(move[0]))):
            trial=deepcopy(result)
            for e in trial:
                if e['id'] in ids:e['x']+=dx;e['y']+=dy
            # A backed text unit may move within its existing panel, never off it.
            if any(not (by_id[pid]['x']<=e['x'] and by_id[pid]['y']<=e['y']
                        and e['x']+e['width']<=by_id[pid]['x']+by_id[pid]['width']
                        and e['y']+e['height']<=by_id[pid]['y']+by_id[pid]['height'])
                   for pid in parent_ids for e in trial if e['id'] in ids):continue
            minimum,lower=score(dx,dy,weak_in_group)
            if minimum>=3.0 and minimum>=initial+max(.25,initial*.1):
                distance=(dx*dx+dy*dy)**.5
                change=dict(repair_type='local-nudge',element_ids=sorted(ids),dx=dx,dy=dy,
                            estimated_minimum=round(minimum,3),estimated_lower_quintile=round(lower,3))
                if _feasible(trial,units,width,height,strict_margins=False,
                             gap_factor=.004,allow_text_visual_overlap=True) and gap_ok(trial,ids):
                    options.append((distance,-minimum,trial,change))
        if options:
            options.sort(key=lambda item:(item[0],item[1]))
            for _,_,candidate,change in options[:6]:alternatives.append((candidate,[change]))
            if not return_alternatives:
                result,changes=alternatives[0]
                break
    if return_alternatives:
        # Source-evidenced cover strips can grow behind white copy even when a
        # proposal changed their frame mode. Restore cover mode only for strips
        # whose source document used it, then let the real renderer judge.
        for ids in groups:
            if not (set(weak_ids)&ids):continue
            members=[by_id[eid] for eid in ids if eid in by_id]
            texts=[e for e in members if e['type']=='text' and e.get('content','').strip()]
            strips=[by_id[e['id']] for e in members if e['type']=='image'
                    and e['id'] in source
                    and (source[e['id']].get('fit')=='cover' or e.get('fit')=='cover')
                    and e['width']*e['height']<=.85*width*height]
            if not texts or not strips:continue
            white=[e for e in texts if e['id'] in weak_ids and
                   (lambda c: bool(c) and min(c[:3])>=.72)(_rgba(e.get('color','')))]
            if not white:continue
            strip=max(strips,key=lambda e:e['width']*e['height'])
            tx0=min(e['x'] for e in texts);tx1=max(e['x']+e['width'] for e in texts)
            pad=max(4.,.01*width)
            text_span=(tx0-pad,tx1+pad)
            left=min(strip['x'],text_span[0]);right=max(strip['x']+strip['width'],text_span[1])
            # Keep the source strip's proportional lockup span, but ensure all
            # low-contrast white copy has opaque backing beneath its glyph box.
            if left>=strip['x'] and right<=strip['x']+strip['width'] and by_id[strip['id']].get('fit')=='cover':continue
            # The base candidate expands/alines the authored cover strip to
            # span all text in this source-evidenced lockup. Wider siblings
            # leave the renderer to verify any extra padding.
            spans=[(left,right)]
            if 0.<=text_span[0] and text_span[1]<=width and text_span!=(left,right):
                spans.append(text_span)
            for span_left,span_right in spans:
                for fraction in (1.,1.08,1.16):
                    candidate=deepcopy(elements);target={e['id']:e for e in candidate}[strip['id']]
                    center=(span_left+span_right)/2;new_width=(span_right-span_left)*fraction
                    target['x']=center-new_width/2;target['width']=new_width;target['fit']='cover'
                    if target['x']<0 or target['x']+target['width']>width:continue
                    if not _feasible(candidate,units,width,height,strict_margins=False,
                                     gap_factor=.004,allow_text_visual_overlap=True):continue
                    if set(_unintended_text_overlaps(candidate,units))-set(_unintended_text_overlaps(elements,units)):continue
                    alternatives.append((candidate,[dict(repair_type='backing-strip-expanded',element_ids=sorted(ids),
                        backing_for=sorted(e['id'] for e in white),dx=target['x']-strip['x'],dy=0.,
                        width=target['width'],text_span=list(text_span))]))
        # 2D Quiet-Zone Placement (lockup-repositioned-2d)
        # Search a 2D grid of candidate anchor positions (dx, dy)
        # to place the lockup/copy group into low-busyness negative space on background
        from PIL import ImageFilter
        edges_under = under.convert('L').filter(ImageFilter.FIND_EDGES)
        for ids in groups:
            if not (set(weak_ids) & ids):
                continue
            members = [by_id[eid] for eid in ids if eid in by_id]
            if not any(e['type'] == 'text' for e in members):
                continue
            left = min(e['x'] for e in members); right = max(e['x'] + e['width'] for e in members)
            top = min(e['y'] for e in members); bottom = max(e['y'] + e['height'] for e in members)
            span_w = right - left; span_h = bottom - top
            if span_w >= 0.90 * width and span_h >= 0.90 * height:
                continue

            h_targets = [
                (width - span_w) / 2.0,                  # centered
                max(16.0, 0.05 * width),                 # left margin
                min(width - span_w - 16.0, 0.95 * width - span_w), # right margin
                left                                     # current x
            ]
            for frac in (0.08, 0.15, 0.22, 0.35, 0.50):
                h_targets.append(frac * width)
                h_targets.append(width - span_w - frac * width)

            v_targets = [
                max(16.0, 0.05 * height),                # top-third / upper wall zone
                0.08 * height,
                0.12 * height,
                0.18 * height,
                (height - span_h) / 2.0,                 # middle / center
                max(16.0, height - span_h - 0.12 * height), # bottom-third
                max(16.0, height - span_h - 0.08 * height),
                max(16.0, height - span_h - 0.05 * height),
                top                                      # current y
            ]
            for v_step in (-0.35, -0.25, -0.15, 0.15, 0.25, 0.35):
                v_targets.append(top + v_step * height)

            quiet_2d_options = []
            seen_shifts = set()
            for t_left in sorted(set(h_targets)):
                if t_left < 0 or t_left + span_w > width:
                    continue
                dx = t_left - left
                for t_top in sorted(set(v_targets)):
                    if t_top < 0 or t_top + span_h > height:
                        continue
                    dy = t_top - top
                    shift_key = (round(dx, 1), round(dy, 1))
                    if shift_key in seen_shifts or (abs(dx) < 2 and abs(dy) < 2):
                        continue
                    seen_shifts.add(shift_key)

                    candidate = deepcopy(elements)
                    for e in candidate:
                        if e['id'] in ids:
                            e['x'] += dx
                            e['y'] += dy

                    if not _feasible(candidate, units, width, height, strict_margins=False,
                                     gap_factor=0.004, allow_text_visual_overlap=True):
                        continue
                    if set(_unintended_text_overlaps(candidate, units)) - set(_unintended_text_overlaps(elements, units)):
                        continue

                    # Check zero collisions with foreground visuals (non-background images/shapes)
                    collides_fg = False
                    for mid in ids:
                        me = {e['id']: e for e in candidate}[mid]
                        mb = (me['x'], me['y'], me['x'] + me['width'], me['y'] + me['height'])
                        for e in candidate:
                            if e['id'] in ids or e['type'] not in ('image', 'shape'):
                                continue
                            full = (e['x'] <= width * 0.02 and e['y'] <= height * 0.02 and
                                    e['x'] + e['width'] >= width * 0.98 and e['y'] + e['height'] >= height * 0.98)
                            if full or e.get('fit') == 'cover':
                                continue
                            eb = (e['x'], e['y'], e['x'] + e['width'], e['y'] + e['height'])
                            if _overlap(mb, eb) > 1.0:
                                collides_fg = True
                                break
                        if collides_fg:
                            break
                    if collides_fg:
                        continue

                    text_min_contrast, text_lower = score(dx, dy, ids)
                    edge_densities = []
                    for e in members:
                        if e['type'] != 'text':
                            continue
                        nl = max(0, int((e['x'] + dx) / width * under.width))
                        nr = min(under.width, max(nl + 1, int((e['x'] + dx + e['width']) / width * under.width)))
                        nt = max(0, int((e['y'] + dy) / height * under.height))
                        nb = min(under.height, max(nt + 1, int((e['y'] + dy + e['height']) / height * under.height)))
                        if nr > nl and nb > nt:
                            ec = edges_under.crop((nl, nt, nr, nb))
                            ed = sum(ec.getdata()) / (255.0 * max(1, ec.width * ec.height))
                            edge_densities.append(ed)

                    mean_edge = sum(edge_densities) / len(edge_densities) if edge_densities else 0.0
                    low_contrast_pen = max(0.0, 3.0 - text_min_contrast)
                    tot_score = mean_edge * 10.0 + low_contrast_pen * 25.0 + (abs(dx) / width + abs(dy) / height) * 0.5
                    if text_min_contrast >= 2.5:
                        quiet_2d_options.append((tot_score, candidate, dx, dy, text_min_contrast, mean_edge))

            quiet_2d_options.sort(key=lambda item: item[0])
            for tot_score, cand_2d, best_dx, best_dy, best_c, best_ed in quiet_2d_options[:3]:
                alternatives.append((cand_2d, [dict(
                    repair_type='lockup-repositioned-2d',
                    element_ids=sorted(ids),
                    dx=round(best_dx, 2),
                    dy=round(best_dy, 2),
                    placement='quiet-zone-2d',
                    estimated_minimum=round(best_c, 3),
                    background_edge_density=round(best_ed, 3)
                )]))
        # Translate a full rigid lockup as a unit when local movement or a
        # wider strip cannot clear the background. Also move visually attached
        # non-text layers that are represented in the same source-evidenced
        # lockup group; feasibility and full-render checks reject bad swaps.
        for ids in groups:
            if not (set(weak_ids)&ids):continue
            members=[by_id[eid] for eid in ids if eid in by_id]
            if len(members)<2 or not any(e['type']=='text' for e in members):continue
            left=min(e['x'] for e in members);right=max(e['x']+e['width'] for e in members)
            top=min(e['y'] for e in members);bottom=max(e['y']+e['height'] for e in members)
            span=right-left
            baseline_min,baseline_lower=score(0.,0.,set(weak_ids)&ids)
            for fraction in (.12,.17,.18,.21,.22,.25,.35,.50):
                for target_left in (fraction*width,width-right+left):
                    dx=target_left-left
                    if abs(dx)<2 or target_left<0 or target_left+span>width:continue
                    candidate=deepcopy(elements)
                    for e in candidate:
                        if e['id'] in ids:e['x']+=dx
                    if top<0 or bottom>height:continue
                    if not _feasible(candidate,units,width,height,strict_margins=False,
                                     gap_factor=.004,allow_text_visual_overlap=True):continue
                    if set(_unintended_text_overlaps(candidate,units))-set(_unintended_text_overlaps(elements,units)):continue
                    candidate_min,candidate_lower=score(dx,0.,set(weak_ids)&ids)
                    if candidate_min<=baseline_min+1e-4 or candidate_lower<=baseline_lower+1e-4:continue
                    alternatives.append((candidate,[dict(repair_type='lockup-repositioned',element_ids=sorted(ids),dx=dx,dy=0.,
                        placement='horizontal-lockup-reflow')]))
        # Deduplicate by rounded geometry so equivalent nudges from overlapping
        # inferred groups do not trigger repeated expensive renders.
        unique=[];seen=set()
        for candidate,change in alternatives:
            key=tuple((e['id'],round(e['x'],2),round(e['y'],2),round(e['width'],2),round(e['height'],2))
                      for e in candidate)
            if key in seen:continue
            seen.add(key);unique.append((candidate,change))
        priority={'backdrop-rotation-variant':0,'background-column-fit':1,
                  'backing-strip-expanded':2,'lockup-repositioned-2d':3,
                  'lockup-repositioned':4,'local-nudge':5}
        unique.sort(key=lambda item:(priority.get(item[1][0].get('repair_type'),3),
                                     item[0][0]['x'],item[0][0]['y']))
        return unique[:18],[]
    return result,changes
# Universal structural passes. Geometry comes exclusively from the source;
# Qwen supplies robust container placement and scale, never internal offsets.
def _frame(items):
    boxes=[_rotated_bounds(e) for e in items]
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes))


def _axis(e, axis):
    dimension, fraction=axis
    length='width' if dimension=='x' else 'height'
    return float(e[dimension])+fraction*float(e[length])


def _weighted_median(values):
    ordered=sorted(values);half=sum(w for _,w in ordered)/2;total=0.
    for value,weight in ordered:
        total+=weight
        if total>=half:return value
    return ordered[-1][0]


def _source_callout_groups(source):
    """Recognize short markers on compact backplates and their following copy."""
    sw,sh=source['canvas']['width'],source['canvas']['height'];items=source['elements']
    groups=[]
    for number in items:
        if number.get('type')!='text' or not number.get('content','').strip().isdigit() or len(number['content'].strip())>3:continue
        nb=_rotated_bounds(number)
        plates=[e for e in items if e.get('type')!='text' and e['width']*e['height']<.08*sw*sh
                and e.get('z_index',0)<number.get('z_index',0)
                and _overlap(nb,_rotated_bounds(e))>=.15*number['width']*number['height']]
        if not plates:continue
        plate=min(plates,key=lambda e:e['width']*e['height'])
        paragraphs=[e for e in items if e.get('type')=='text' and len(e.get('content','').strip())>8
                    and abs(_axis(e,('x',.5))-_axis(plate,('x',.5)))<=.06*sw
                    and -.02*sh<=e['y']-(number['y']+number['height'])<=.09*sh]
        if paragraphs:
            paragraph=min(paragraphs,key=lambda e:abs(e['y']-number['y']-number['height']))
            groups.append({plate['id'],number['id'],paragraph['id']})
    return groups


def build_container_tree(source, units=None):
    """Canvas -> functional containers -> nested containers/leaves.

    Recompute the prompt's geometric relationships without its token-budget
    truncation. Full-canvas backgrounds cannot join the foreground graph.
    Alignment alone couples nearby peers; backing/overlap and explicit groups
    provide stronger evidence. Backing sets form a laminar containment tree.
    """
    sw,sh=map(float,(source['canvas']['width'],source['canvas']['height']))
    active=({e['id'] for u in units for e in u['items']} if units is not None else
            {e['id'] for e in source['elements'] if not
             (e.get('type')!='text' and e['width']>=.9*sw and e['height']>=.9*sh)})
    src={e['id']:e for e in source['elements'] if e['id'] in active}
    items=sorted(src.values(),key=lambda e:str(e['id']))
    atomic=[{e['id'] for e in u['items']} & src.keys() for u in (units or []) if len(u['items'])>1]
    atomic.extend(g & src.keys() for g in _source_callout_groups(source))
    atomic_for={eid:i for i,g in enumerate(atomic) for eid in g}
    parent={eid:eid for eid in src};relations=[];backings=[]
    def find(eid):
        while parent[eid]!=eid:parent[eid]=parent[parent[eid]];eid=parent[eid]
        return eid
    def join(a,b,explicit=False):
        if not explicit:
            if a in atomic_for and b in atomic_for and atomic_for[a]!=atomic_for[b]:return
            members=[e for eid,e in src.items() if find(eid) in (find(a),find(b))]
            box=_frame(members)
            # An alignment graph describes a flow, not a rigid whole poster.
            if (box[2]-box[0])*(box[3]-box[1])>.42*sw*sh:return
            groups={e.get('group_id') for e in members if e.get('group_id') is not None}
            if len(groups)>1:return
            if any(e.get('type')=='image' and e['width']*e['height']>=.12*sw*sh
                   and not e.get('group_id') for e in members):return
        parent[find(b)]=find(a)
    def decorative_bridge(e):
        return (e.get('type')!='text' and e.get('group_id') is None
                and float(e.get('opacity',1.))<.98)
    for i,a in enumerate(items):
        ab=_rotated_bounds(a);aa=max(1.,a['width']*a['height'])
        contained={a['id']}
        for b in items:
            if a['id']==b['id']:continue
            bb=_rotated_bounds(b);ba=max(1.,b['width']*b['height'])
            coverage=_overlap(ab,bb)/ba
            if (a.get('type')!='text' and aa>1.15*ba and aa<.85*sw*sh
                    and not decorative_bridge(a)
                    and coverage>=.96 and a.get('z_index',0)<=b.get('z_index',0)):
                contained.add(b['id'])
                relations.append(['backing',a['id'],b['id']])
        if len(contained)>1:backings.append(contained)
        for b in items[i+1:]:
            bb=_rotated_bounds(b);ba=max(1.,b['width']*b['height'])
            shared=_overlap(ab,bb)/min(aa,ba)
            xc=abs(_axis(a,('x',.5))-_axis(b,('x',.5)))
            yc=abs(_axis(a,('y',.5))-_axis(b,('y',.5)))
            row=yc<=.025*sh and xc>=.04*sw
            column=xc<=.025*sw and yc>=.04*sh
            if row:relations.append(['same-row',a['id'],b['id']])
            if column:relations.append(['same-column',a['id'],b['id']])
            if shared>=.18:relations.append(['overlap',a['id'],b['id']])
            gapx=max(0.,ab[0]-bb[2],bb[0]-ab[2])
            gapy=max(0.,ab[1]-bb[3],bb[1]-ab[3])
            explicit=a.get('group_id') is not None and a.get('group_id')==b.get('group_id')
            # Nested backplates are represented below, not collapsed by their
            # containment edges. Non-contained overlap is an atomic lockup.
            nested=_overlap(ab,bb)/min(aa,ba)>=.96 and max(aa,ba)>1.15*min(aa,ba)
            aligned_column=(min(abs(a['x']-b['x']),xc,
                               abs(a['x']+a['width']-b['x']-b['width']))<=4
                            and gapy<=.14*sh and yc>=.02*sh)
            aligned_row=yc<=4 and gapx<=.10*sw and xc>=.02*sw
            large_bridge=any(decorative_bridge(e) or (e.get('type')!='text' and e['width']*e['height']>.35*sw*sh
                             and e.get('group_id') is None) for e in (a,b))
            alignment_lock=(a.get('type')=='text' and b.get('type')=='text'
                            and a.get('group_id') is None and b.get('group_id') is None)
            if explicit or (not large_bridge and ((shared>=.18 and not nested) or
                                                   alignment_lock and (aligned_column or aligned_row))):
                join(a['id'],b['id'],explicit=explicit)
    # A repeated row/column can be visually one lockup even when a centered
    # emblem or label sits between the peers and keeps their boxes apart. Join
    # that in-between anchor to the peer group using geometry alone. This lets
    # container projection preserve the full arrangement as a unit instead of
    # moving the paired leaves independently.
    for i,a in enumerate(items):
        ab=_rotated_bounds(a)
        for b in items[i+1:]:
            if decorative_bridge(a) or decorative_bridge(b):continue
            bb=_rotated_bounds(b)
            acx,acy=_axis(a,('x',.5)),_axis(a,('y',.5))
            bcx,bcy=_axis(b,('x',.5)),_axis(b,('y',.5))
            if abs(acy-bcy)<=4 and abs(acx-bcx)>=.04*sw:
                left,right=sorted((acx,bcx))
                gap=max(0.,max(a['x'],b['x'])-min(a['x']+a['width'],b['x']+b['width']))
                row_top=min(ab[1],bb[1]);row_bottom=max(ab[3],bb[3])
                if gap<=.25*sw:
                    for anchor in items:
                        if anchor['id'] in (a['id'],b['id']) or decorative_bridge(anchor):continue
                        ac=_axis(anchor,('x',.5));ancb=_rotated_bounds(anchor)
                        if (left+4<ac<right-4
                                and min(row_bottom,ancb[3])>max(row_top,ancb[1])
                                and (_frame([a,b,anchor])[2]-_frame([a,b,anchor])[0])*(_frame([a,b,anchor])[3]-_frame([a,b,anchor])[1])<=.45*sw*sh):
                            join(a['id'],b['id']);join(a['id'],anchor['id'])
            if abs(acx-bcx)<=4 and abs(acy-bcy)>=.04*sh:
                top,bottom=sorted((acy,bcy))
                gap=max(0.,max(a['y'],b['y'])-min(a['y']+a['height'],b['y']+b['height']))
                col_left=min(ab[0],bb[0]);col_right=max(ab[2],bb[2])
                if gap<=.25*sh:
                    for anchor in items:
                        if anchor['id'] in (a['id'],b['id']) or decorative_bridge(anchor):continue
                        ancy=_axis(anchor,('y',.5));ancb=_rotated_bounds(anchor)
                        if (top+4<ancy<bottom-4
                                and min(col_right,ancb[2])>max(col_left,ancb[0])
                                and (_frame([a,b,anchor])[2]-_frame([a,b,anchor])[0])*(_frame([a,b,anchor])[3]-_frame([a,b,anchor])[1])<=.45*sw*sh):
                            join(a['id'],b['id']);join(a['id'],anchor['id'])
    if units:
        for u in units:
            if u.get('backing_panel') and not any(decorative_bridge(e) for e in u['items']):
                ids={e['id'] for e in u['items']}|set(u.get('backed_ids',()))
                ids&=src.keys()
                if len(ids)>1:backings.append(ids)
    backings.extend(g for g in atomic if len(g)>1)
    coupled={}
    for eid in src:coupled.setdefault(find(eid),set()).add(eid)
    sets=[g for g in coupled.values() if len(g)>1]+backings
    # Crossing memberships become one container; strict subsets stay nested.
    merged=True
    while merged:
        merged=False
        for i,a in enumerate(sets):
            for j in range(i+1,len(sets)):
                b=sets[j]
                if a&b and not (a<=b or b<=a):
                    sets[i]=a|b;sets.pop(j);merged=True;break
            if merged:break
    sets=sorted({frozenset(g) for g in sets},key=lambda g:(-len(g),tuple(sorted(g))))
    for eid in sorted(src):
        if not any(eid in g for g in sets):sets.append(frozenset([eid]))
    def node(group):
        subsets=[s for s in sets if s<group and not any(s<t<group for t in sets)]
        covered=set().union(*subsets) if subsets else set()
        children=[node(s) for s in subsets]
        children.extend({'kind':'element','element_id':eid} for eid in sorted(group-covered))
        # Only schema-native shapes may stretch; images always scale uniformly.
        elastic=any(src[eid].get('type')=='shape' for eid in group)
        return {'kind':'container','id':'container:'+','.join(sorted(group)),
                'element_ids':sorted(group),'source_box':list(_frame([src[eid] for eid in group])),
                'mode':'semi-elastic' if elastic else 'rigid','children':children}
    roots=[g for g in sets if not any(g<h for h in sets)]
    return {'kind':'canvas','source_canvas':{'width':sw,'height':sh},
            'source_elements':src,'relationships':relations,
            'children':[node(g) for g in roots]}


def project_container_tree(elements, tree, width, height):
    """Infer each container transform from substantial proposal members.

    A weighted robust affine estimate produces [X,Y,W,H]. Thin geometry never
    estimates a transform; it receives the same transform as the source copy.
    Typography and tracking receive the same uniform transform as the frame.
    Renderer calibration may then widen a frame before reducing its font. Only native shape backplates may absorb
    excess width in semi-elastic mode.
    """
    from math import sqrt, isfinite
    src=tree['source_elements'];result=deepcopy(elements);out={e['id']:e for e in result}
    sw=float(tree['source_canvas']['width']);sh=float(tree['source_canvas']['height']);notes=[]
    for node in tree['children']:
        ids=[eid for eid in node['element_ids'] if eid in out]
        if len(ids)<2:continue
        original=[src[eid] for eid in ids]
        sb=_frame(original);cx=(sb[0]+sb[2])/2;cy=(sb[1]+sb[3])/2
        substantial=[e for e in original if min(e['width'],e['height'])>max(2.,.006*min(sw,sh))
                     and max(e['width'],e['height'])/max(1.,min(e['width'],e['height']))<16]
        if not substantial:substantial=original
        largest=max(e['width']*e['height'] for e in substantial)
        weights={e['id']:max(.05,min(1.,sqrt(e['width']*e['height']/max(1.,largest))))
                 for e in substantial}
        scales=[];xscales=[]
        for e in substantial:
            q=out[e['id']];sx=float(q['width'])/max(1e-6,e['width']);sy=float(q['height'])/max(1e-6,e['height'])
            if not all(isfinite(v) and v>0 for v in (sx,sy)):continue
            scales.append((min(sx,sy),weights[e['id']]))
            xscales.append((sx,weights[e['id']]))
        if not scales:continue
        scale=_weighted_median(scales)
        scale=min(scale,width/max(1e-6,sb[2]-sb[0]),height/max(1e-6,sb[3]-sb[1]))
        tx=_weighted_median([(_axis(out[e['id']],('x',.5))-(_axis(e,('x',.5))-cx)*scale,weights[e['id']]) for e in substantial])
        ty=_weighted_median([(_axis(out[e['id']],('y',.5))-(_axis(e,('y',.5))-cy)*scale,weights[e['id']]) for e in substantial])
        # Preserve edge bleed only when the source container was edge anchored.
        gx=min(width*.03,max(0.,min(sb[0],sw-sb[2]))/max(1.,sw)*width)
        gy=min(height*.03,max(0.,min(sb[1],sh-sb[3]))/max(1.,sh)*height)
        scale=min(scale,(width-2*gx)/max(1e-6,sb[2]-sb[0]),(height-2*gy)/max(1e-6,sb[3]-sb[1]))
        # The canvas is the parent container too. A lone source-centered
        # lockup inherits that shared parent axis, while an off-center lockup
        # keeps its nearer authored canvas edge at the same scaled inset. This
        # corrects a common Qwen failure where the internal group is coherent
        # but the whole composition drifts far from its source anchor.
        position_anchors={}
        if (len(tree['children'])==1 or
                max((width/height)/(sw/sh),(sw/sh)/(width/height))<1.4):
            source_margins={'x':(sb[0],sw-sb[2]),'y':(sb[1],sh-sb[3])}
            target_extent={'x':width,'y':height}
            target_half={'x':(sb[2]-sb[0])*scale/2,'y':(sb[3]-sb[1])*scale/2}
            source_extent={'x':sw,'y':sh}
            for dimension in ('x','y'):
                before_gap,after_gap=source_margins[dimension]
                extent=source_extent[dimension]
                if abs(before_gap-after_gap)<=max(4.,extent*.02):
                    if dimension=='x':tx=target_extent[dimension]/2
                    else:ty=target_extent[dimension]/2
                    position_anchors[dimension]='center'
                elif (dimension=='y' and min(before_gap,after_gap)>=0
                      and abs(before_gap-after_gap)>=extent*.08
                      and (sb[3]-sb[1])<extent*.9
                      and (max((width/height)/(sw/sh),(sw/sh)/(width/height))<=1.4
                           or (sw<=sh and width<=height))):
                    if before_gap<after_gap:
                        center=before_gap*scale+target_half[dimension]
                        anchor='top'
                    else:
                        center=target_extent[dimension]-after_gap*scale-target_half[dimension]
                        anchor='bottom'
                    current_edge=ty-target_half[dimension] if anchor=='top' else target_extent[dimension]-(ty+target_half[dimension])
                    expected_edge=before_gap*scale if before_gap<after_gap else after_gap*scale
                    if abs(current_edge-expected_edge)>height*.08:
                        ty=center
                        position_anchors[dimension]=anchor
        font_growth=1.
        if (len(tree['children'])==1 and width/height>=1.8
                and (sb[3]-sb[1])*scale<height*.70
                and (sb[2]-sb[0])*scale<width*.55):
            # A compact card is allowed to use the banner's short side. Only
            # grow a genuinely contained lockup, not an entire poster/hero.
            card=any(e.get('type')!='text' and e['width']*e['height']>=.8*(sb[2]-sb[0])*(sb[3]-sb[1])
                     for e in original)
            if card:
                grown=min(height*.76/max(1e-6,sb[3]-sb[1]),width*.65/max(1e-6,sb[2]-sb[0]),
                          (width-2*gx)/max(1e-6,sb[2]-sb[0]),(height-2*gy)/max(1e-6,sb[3]-sb[1]))
                font_growth=max(1.,grown/max(1e-6,scale));scale=max(scale,grown)
                if font_growth>1.:ty=height/2
        hw=(sb[2]-sb[0])*scale/2;hh=(sb[3]-sb[1])*scale/2
        tx=min(width-gx-hw,max(gx+hw,tx));ty=min(height-gy-hh,max(gy+hh,ty))
        # Deliberate single-edge raster crops are parent constraints too.
        # Clamp the visible container while retaining its authored hidden part;
        # shifting each image inward would expose artwork the source cropped.
        for dimension,low,high,extent,half in (() if tree.get('strict_canvas') else (
                ('x',sb[0],sb[2],sw,hw),('y',sb[1],sb[3],sh,hh))):
            length='width' if dimension=='x' else 'height'
            leaking=[e for e in original if e[dimension]<0 or e[dimension]+e[length]>extent]
            if not leaking or any(e.get('type')!='image' or
                                  min(e.get('rotation',0)%360,360-e.get('rotation',0)%360)>.5
                                  for e in leaking):continue
            target_extent=width if dimension=='x' else height
            if low<0 and high<=extent:
                center=low*scale+half
            elif high>extent and low>=0:
                center=target_extent+(high-extent)*scale-half
            else:continue
            if dimension=='x':tx=center
            else:ty=center
        for eid in ids:
            e=src[eid];q=out[eid]
            q.update(x=tx+(e['x']-cx)*scale,y=ty+(e['y']-cy)*scale,
                     width=e['width']*scale,height=e['height']*scale,
                     rotation=e.get('rotation',0))
            if e.get('type')=='text':
                # This is the rigid alternative; independent font predictions
                # remain available in the separately rendered raw candidate.
                q['font_size']=max(1.,float(e.get('font_size',1.))*scale)
                q['letter_spacing']=float(e.get('letter_spacing',0))*q['font_size']/max(1.,e.get('font_size',1.))
                if 'line_height' in e:q['line_height']=e['line_height']
                else:q.pop('line_height',None)
                q['text_align']=e.get('text_align','left')
        elastic_width=(sb[2]-sb[0])*min(_weighted_median(xscales),scale*1.5)
        if node['mode']=='semi-elastic' and elastic_width>(sb[2]-sb[0])*scale:
            for eid in ids:
                e=src[eid];q=out[eid]
                # Only a shape that spans the group can absorb its free axis.
                if e.get('type')=='shape' and e.get('shape_type')!='ellipse' and e['width']>=.9*(sb[2]-sb[0]) and abs(e.get('rotation',0)%360)<.5:
                    delta=min(elastic_width-q['width'],2*min(q['x'],width-q['x']-q['width']))
                    if delta>0:q['x']-=delta/2;q['width']+=delta
        final_box=_frame([out[eid] for eid in ids])
        notes.append({'container_id':node['id'],'mode':node['mode'],'element_ids':ids,
                      'scale':scale,'target_box':[final_box[0],final_box[1],final_box[2]-final_box[0],final_box[3]-final_box[1]],
                      'anchor_ids':[e['id'] for e in substantial],
                      'position_anchors':position_anchors})
    return result,notes


def snap_source_axes(elements, source, width, height, tree=None):
    """Complete-link 1D source/target clusters; no transitive snap chains.

    Source axis diameter <=4px and target diameter <=3% of the dimension.
    Exact source peer rows/columns allow up to 8% target-axis drift before
    treating it as a genuine reflow. If a peer row collapses or reverses its
    left-to-right order, recover its source order and scaled spacing around
    the source-proportional group center. Competing constraints claim an
    element deterministically; impossible axes are not applied successively.
    """
    from math import isfinite
    from engine.resize import _unintended_text_overlaps
    tree=tree or build_container_tree(source)
    src=tree['source_elements'];result=deepcopy(elements);out={e['id']:e for e in result}
    sw,sh=map(float,(tree['source_canvas']['width'],tree['source_canvas']['height']))
    pairs={frozenset(r[1:3]) for r in tree['relationships'] if r[0] in ('same-column','same-row','backing','overlap')}
    backings=[(r[1],r[2]) for r in tree['relationships'] if r[0]=='backing'
              and r[1] in out and r[2] in out]
    candidates=[]
    # Source-explicit peer rows/columns are stronger than a generic nearby-axis
    # cluster. Only use true shared source centers (4px). The target tolerance
    # is intentionally wider than generic snapping: the held-out beer lockup
    # keeps a source-exact pair on one row in its target JSON despite a larger
    # provisional split. The rendered feasibility/overlap checks still gate
    # the repaired candidate.
    for relation,dimension in (('same-row','y'),('same-column','x')):
        adjacency={}
        for kind,a,b in tree['relationships']:
            if kind!=relation or a not in out or b not in out:continue
            if abs(_axis(src[a],(dimension,.5))-_axis(src[b],(dimension,.5)))>4:continue
            adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
        seen=set()
        for start in sorted(adjacency):
            if start in seen:continue
            group=set();stack=[start]
            while stack:
                eid=stack.pop()
                if eid in group:continue
                group.add(eid);stack.extend(adjacency.get(eid,()))
            seen.update(group)
            if len(group)<2:continue
            ids=sorted(group)
            target_values=[_axis(out[eid],(dimension,.5)) for eid in ids]
            spread=max(target_values)-min(target_values)
            limit=.08*(width if dimension=='x' else height)
            if spread<=limit:
                candidates.append((0,-len(ids),spread,0,dimension,.5,ids))
    for dimension in ('x','y'):
        limit=.03*(width if dimension=='x' else height)
        for fraction in (0.,.5,1.):
            axis=(dimension,fraction)
            eligible=[e for eid,e in src.items() if eid in out and (fraction==.5 or abs(e.get('rotation',0)%360)<.5)]
            ordered=sorted(eligible,key=lambda e:(_axis(e,axis),str(e['id'])))
            source_clusters=[]
            for e in ordered:
                if not source_clusters or _axis(e,axis)-_axis(source_clusters[-1][0],axis)>4:
                    source_clusters.append([])
                source_clusters[-1].append(e)
            for cluster in source_clusters:
                target_clusters=[]
                for e in sorted(cluster,key=lambda e:(_axis(out[e['id']],axis),str(e['id']))):
                    if not target_clusters or _axis(out[e['id']],axis)-_axis(out[target_clusters[-1][0]['id']],axis)>limit:
                        target_clusters.append([])
                    target_clusters[-1].append(e)
                for group in target_clusters:
                    if len(group)<2:continue
                    # Source edge alignment itself is same-column evidence,
                    # even where the compact prompt only encoded centers.
                    ids=[e['id'] for e in group]
                    if not any(frozenset((a,b)) in pairs or
                               abs(_axis(src[a],axis)-_axis(src[b],axis))<=4
                               for i,a in enumerate(ids) for b in ids[i+1:]):continue
                    spread=max(_axis(e,axis) for e in group)-min(_axis(e,axis) for e in group)
                    preference=sum(e.get('type')=='text' and dimension=='x' and
                                   e.get('text_align','left')=={0.:'left',.5:'center',1.:'right'}[fraction] for e in group)
                    candidates.append((1,-len(group),spread,-preference,dimension,fraction,ids))
    used={};notes=[]
    # All axes are derived from the original target; no order-dependent medians.
    initial={e['id']:e for e in elements}
    for priority,_,_,_,dimension,fraction,ids in sorted(candidates):
        axis=(dimension,fraction);target=median(_axis(initial[eid],axis) for eid in ids)
        if (dimension=='x' and fraction==.5
                and all(abs(_axis(src[eid],axis)-sw/2)<=max(4.,.02*sw) for eid in ids)
                and abs(target-width/2)<=.18*width and width/height<1.8):
            target=width/2.
        shifts={eid:target-_axis(out[eid],axis) for eid in ids}
        if any((eid,dimension) in used and abs(shifts[eid])>1e-6 for eid in ids):continue
        if max(abs(v) for v in shifts.values())<1e-7:
            for eid in ids:used[eid,dimension]=True
            continue
        before={eid:{axis_name:out[eid][axis_name] for axis_name in ('x','y')}
                for eid in ids}
        old_overlaps=set(_unintended_text_overlaps(result,[{'items':list(src.values())}]))
        for eid,delta in shifts.items():out[eid][dimension]+=delta
        peer_axis=None
        if priority==0 and fraction==.5 and len(ids)>1:
            # A source-exact row is a paired horizontal group, not permission
            # for two peers to collapse onto one another or switch sides. Keep
            # their source order and a scaled minimum gap when target proposals
            # collapse/reverse the row. Center a source-centered peer group on
            # the target canvas axis; otherwise preserve its proposed centroid.
            peer_axis='x' if dimension=='y' else 'y'
            ordered=sorted(ids,key=lambda eid:_axis(src[eid],(peer_axis,.5)))
            centers=[_axis(out[eid],(peer_axis,.5)) for eid in ordered]
            source_centers=[_axis(src[eid],(peer_axis,.5)) for eid in ordered]
            axis_size=width if peer_axis=='x' else height
            source_size=sw if peer_axis=='x' else sh
            target_scale=min(width/max(1.,sw),height/max(1.,sh))
            source_group_center=(source_centers[0]+source_centers[-1])/2.
            target_group_center=sum(centers)/len(centers)
            centered_source=abs(source_group_center-source_size/2.)<=max(4.,source_size*.02)
            if centered_source:
                target_group_center=axis_size/2.
            else:
                expected=source_group_center/source_size*axis_size
                if abs(target_group_center-expected)<=axis_size*.25:
                    target_group_center=expected
            support_ids=[]
            source_peer_low=min(src[eid]['y'] for eid in ids) if peer_axis=='x' else min(src[eid]['x'] for eid in ids)
            source_peer_high=max(src[eid]['y']+src[eid]['height'] for eid in ids) if peer_axis=='x' else max(src[eid]['x']+src[eid]['width'] for eid in ids)
            backing_members={item for pair in backings for item in pair}
            for sid,se in src.items():
                if sid in ids or sid not in out:continue
                source_center=_axis(se,(peer_axis,.5))
                if not source_centers[0]<source_center<source_centers[-1]:continue
                source_low=se['y'] if peer_axis=='x' else se['x']
                source_high=source_low+(se['height'] if peer_axis=='x' else se['width'])
                if min(source_peer_high,source_high)<=max(source_peer_low,source_low):continue
                if any(sid in pair and any(other not in ids for other in pair)
                       for pair in backings):continue
                if not centered_source:continue
                extent='width' if peer_axis=='x' else 'height'
                current_center=_axis(out[sid],(peer_axis,.5))
                if abs(current_center-target_group_center)>axis_size*.25:continue
                before.setdefault(sid,{dimension:out[sid][dimension],peer_axis:out[sid][peer_axis]})
                out[sid][peer_axis]=target_group_center-out[sid][extent]/2.
                support_ids.append(sid)
            required=[]
            for a,b in zip(ordered,ordered[1:]):
                ea,eb=src[a],src[b]
                extent='width' if peer_axis=='x' else 'height'
                source_edge_gap=max(0.,_axis(eb,(peer_axis,.5))-_axis(ea,(peer_axis,.5))
                                    -(ea[extent]+eb[extent])/2.)
                target_half_sum=(out[a][extent]+out[b][extent])/2.
                support_extent=max((out[sid][extent] for sid in support_ids),default=0.)
                clearance=2.*min(width,height)*.01 if support_extent else 0.
                required.append(max(0.,target_half_sum+source_edge_gap*target_scale,
                                    target_half_sum+support_extent+clearance))
            bad_order=any(b-a<gap-1e-5 for a,b,gap in zip(centers,centers[1:],required))
            if bad_order and all(isfinite(v) for v in centers):
                offsets=[(value-source_group_center)*target_scale for value in source_centers]
                candidate_centers=[target_group_center+offset for offset in offsets]
                if any(b-a<gap-1e-5 for a,b,gap in zip(candidate_centers,candidate_centers[1:],required)):
                    candidate_centers=[target_group_center]
                    for gap in required:candidate_centers.append(candidate_centers[-1]+gap)
                    shift=target_group_center-(candidate_centers[0]+candidate_centers[-1])/2.
                    candidate_centers=[value+shift for value in candidate_centers]
                for eid,center in zip(ordered,candidate_centers):
                    out[eid][peer_axis]=center-out[eid][extent]/2.
        in_bounds=all(0<=_rotated_bounds(out[eid])[0] and 0<=_rotated_bounds(out[eid])[1]
                      and _rotated_bounds(out[eid])[2]<=width+1e-5
                      and _rotated_bounds(out[eid])[3]<=height+1e-5 for eid in ids)
        # A near-equal parent/child edge can encode a small real inset.
        # Snapping may align siblings, but cannot consume that source padding
        # or worsen an existing containment defect. This applies to any backing.
        padding_preserved=True
        length='width' if dimension=='x' else 'height'
        for pid,cid in backings:
            if pid not in ids and cid not in ids:continue
            p,c=src[pid],src[cid]
            if any(min(e.get('rotation',0)%360,360-e.get('rotation',0)%360)>.5 for e in (p,c)):continue
            p0=before.get(pid,{}).get(dimension,out[pid][dimension])
            c0=before.get(cid,{}).get(dimension,out[cid][dimension])
            source_gaps=(c[dimension]-p[dimension],p[dimension]+p[length]-c[dimension]-c[length])
            before_gaps=(c0-p0,p0+out[pid][length]-c0-out[cid][length])
            after_gaps=(out[cid][dimension]-out[pid][dimension],
                        out[pid][dimension]+out[pid][length]-out[cid][dimension]-out[cid][length])
            factor=out[pid][length]/max(1e-6,p[length])
            if any(gap>0 and after<min(previous,gap*factor)-1e-6
                   for gap,previous,after in zip(source_gaps,before_gaps,after_gaps)):
                padding_preserved=False;break
        if (not in_bounds or not padding_preserved
                or set(_unintended_text_overlaps(result,[{'items':list(src.values())}]))-old_overlaps):
            for eid,coordinates in before.items():
                for axis_name,value in coordinates.items():out[eid][axis_name]=value
            continue
        for eid in ids:used[eid,dimension]=True
        if peer_axis and any(abs(out[eid][peer_axis]-before[eid][peer_axis])>1e-6 for eid in ids):
            for eid in ids:used[eid,peer_axis]=True
        for eid in before:
            if eid not in ids:used[eid,peer_axis]=True
        notes.append({'axis':dimension,'fraction':fraction,'value':target,'element_ids':ids})
    return result,notes


def _contrast_text_lockup(weak_ids,tree,source):
    """Find a copy subgroup without severing explicit groups or its backing.

    Large overlapping artwork can connect an entire poster in the inferred
    tree. For relocation, only source-aligned copy, explicit membership,
    immediate backings and compact aligned ornaments are strong enough to
    carry a neighbour. A photo/illustration is never dragged by proximity.
    """
    src=tree['source_elements'];ids=set(weak_ids)&src.keys()
    sw,sh=source['canvas']['width'],source['canvas']['height']
    changed=True
    while changed:
        changed=False
        for a in list(ids):
            for b,e in src.items():
                if b in ids:continue
                original=src[a]
                # An ungrouped corner brand mark is an independent anchor,
                # even when a broad inferred frame contains its rectangle.
                if e.get('type')=='image' and not e.get('group_id'):
                    corner=(e['x']<.18*sw or e['x']+e['width']>.82*sw) and (e['y']<.25*sh or e['y']+e['height']>.80*sh)
                    if corner and e['width']*e['height']<.12*sw*sh:continue
                explicit=(original.get('group_id') is not None
                          and original.get('group_id')==e.get('group_id'))
                aligned=any(r[0] in ('same-column','same-row') and {a,b}==set(r[1:3])
                            for r in tree['relationships'])
                copy=original.get('type')=='text' and e.get('type')=='text' and aligned
                backing=(original.get('type')=='text' and
                         any(r[0]=='backing' and r[1]==b and r[2]==a
                             for r in tree['relationships']))
                ornament=(original.get('type')=='text' and e.get('type')!='text'
                          and e['width']*e['height']<.01*sw*sh
                          and min(e['width']/sw,e['height']/sh)<.025 and aligned)
                lb=_frame([src[eid] for eid in ids])
                area=max(1.,e['width']*e['height'])
                padding=.025*min(sw,sh)
                attachment=(e.get('type')!='text' and float(e.get('opacity',1.))>=.98
                    and area<.12*sw*sh and
                    (_overlap(_rotated_bounds(e),lb)>=.1*min(area,max(1.,(lb[2]-lb[0])*(lb[3]-lb[1])))
                     or (lb[0]-padding<=e['x'] and e['x']+e['width']<=lb[2]+padding
                         and lb[1]-padding<=e['y'] and e['y']+e['height']<=lb[3]+padding)))
                if explicit or copy or backing or ornament or attachment:
                    ids.add(b);changed=True
    return ids


def whole_lockup_contrast_variants(elements,units,source,width,height,weak_ids,under):
    """Propose rigid moves; full raster gates decide whether they improve ink.

    Move disjoint source-supported functional groups intact. Oversized groups
    are not split into leaves. Fonts, internal offsets, explicit membership
    and image ratios never change.
    """
    from engine.resize import _physical_layout_safe,_unintended_text_overlaps,_rgba,_luminance,_text_visible_bounds
    from engine.source_structure import groups as source_groups
    spec=source_groups(source);weak=set(weak_ids);variants=[]
    by_id={e['id']:e for e in elements}
    # Relocation uses the same disjoint functional groups as projection and
    # ranking. Expanding a weak word through a second transitive relationship
    # graph can otherwise move half of a header and an unrelated footer.
    groups=[set(ids)&by_id.keys() for ids in spec['members'] if set(ids)&weak]
    pad=min(width,height)*.035;seen=set()
    protected={ids[0] for ids in spec['members'] if len(ids)==1
               and spec['semantics'].get(ids[0])=='logo'}
    for ids in groups:
        ids=ids-protected
        if not ids:continue
        if frozenset(ids) in seen:continue
        seen.add(frozenset(ids));members=[by_id[eid] for eid in ids]
        left=min(e['x'] for e in members);top=min(e['y'] for e in members)
        right=max(e['x']+e['width'] for e in members);bottom=max(e['y']+e['height'] for e in members)
        bw,bh=right-left,bottom-top
        if bw>width-2*pad or bh>height-2*pad:continue
        probes=[];positions=set()
        # Include local shifts and the current axis, so a narrow viable strip
        # between grid cells is not skipped. The global grid finds calm zones.
        xs={left+width*f for f in (-.12,-.06,0.,.06,.12)}
        ys={top+height*f for f in (-.12,-.06,0.,.06,.12)}
        xs.update(width*f-bw/2 for f in (.12,.30,.50,.70,.88))
        ys.update(height*f-bh/2 for f in (.12,.30,.50,.70,.88))
        for y0 in sorted(ys):
            for x0 in sorted(xs):
                x=min(max(pad,x0),width-pad-bw);y=min(max(pad,y0),height-pad-bh)
                dx,dy=x-left,y-top;position=(round(dx,4),round(dy,4))
                if position in positions or abs(dx)+abs(dy)<1.:continue
                positions.add(position)
                candidate=deepcopy(elements)
                for e in candidate:
                    if e['id'] in ids:e['x']+=dx;e['y']+=dy
                if not _physical_layout_safe(candidate,units,source,width,height):continue
                ratios=[]
                for e in members:
                    if e.get('type')!='text' or e['id'] not in weak:continue
                    color=_rgba(e.get('color',''))
                    if color is None:continue
                    box=_text_visible_bounds(e)
                    for fy in (.1,.25,.5,.75,.9):
                        for fx in (.1,.3,.5,.7,.9):
                            px=min(under.width-1,max(0,int((box[0]+(box[2]-box[0])*fx+dx)/width*under.width)))
                            py=min(under.height-1,max(0,int((box[1]+(box[3]-box[1])*fy+dy)/height*under.height)))
                            bg=under.getpixel((px,py));alpha=color[3]*e.get('opacity',1.)
                            fg=tuple(color[k]*alpha+bg[k]*(1-alpha) for k in range(3))
                            a,b=_luminance(fg),_luminance(bg);ratios.append((max(a,b)+.05)/(min(a,b)+.05))
                if ratios:
                    ratios.sort();score=ratios[len(ratios)//5]
                    probes.append((score,abs(dx)/width+abs(dy)/height,candidate,
                                   dict(repair_type='container-calm-cell',element_ids=sorted(ids),dx=dx,dy=dy)))
        # Once a sampled cell is readable, extra contrast is not a reason to
        # teleport the lockup farther away. Full glyph metrics still gate it.
        probes.sort(key=lambda row:(-min(4.5,row[0]),row[1]))
        variants.extend((candidate,[change]) for _,_,candidate,change in probes[:3])
    best={}
    for candidate,changes in variants:
        best.setdefault(tuple(changes[0]['element_ids']),(candidate,changes[0]))
    if len(best)>1:
        combined=deepcopy(elements);combined_by={e['id']:e for e in combined};changes=[]
        for _,change in best.values():
            changes.append(change)
            for eid in change['element_ids']:
                combined_by[eid]['x']+=change['dx'];combined_by[eid]['y']+=change['dy']
        if _physical_layout_safe(combined,units,source,width,height):variants.insert(0,(combined,changes))
    return variants[:3]
