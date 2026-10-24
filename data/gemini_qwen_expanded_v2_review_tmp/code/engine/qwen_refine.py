"""Small, source-evidenced alternatives to a model's layout proposal.

These helpers never consult a target design or change content/asset pixels.
The original proposal remains available to the candidate ranker.
"""
from copy import deepcopy
from statistics import median


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
    groups={}
    for eid in source:groups.setdefault(find(eid),set()).add(eid)
    return [ids for ids in groups.values() if len(ids)>1]


def repair_relationships(elements,units,width,height):
    """Fit compact lockups together, then solve whole text chains simultaneously."""
    from engine.resize import _bounds, _text_info, _unintended_text_overlaps
    result=deepcopy(elements);by_id={e['id']:e for e in result}
    source={e['id']:e for u in units for e in u['items']}
    groups=lockup_groups(units);notes=[]
    for ids in groups:
        ids=ids & set(by_id)
        if len(ids)<2:continue
        originals=[source[eid] for eid in ids];proposed=[by_id[eid] for eid in ids]
        # Clamp the raw PNG frames too, including transparent margins.
        sb=(min(e['x'] for e in originals),min(e['y'] for e in originals),
            max(e['x']+e['width'] for e in originals),max(e['y']+e['height'] for e in originals))
        sx,sy=(sb[0]+sb[2])/2,(sb[1]+sb[3])/2
        # Text font ratios are reliable scale anchors for compact lockups;
        # explicit artwork groups use robust box ratios instead.
        scales=[by_id[e['id']].get('font_size',1)/max(1,e.get('font_size',1))
                for e in originals if e['type']=='text']
        if not scales:
            scales=[by_id[e['id']][k]/max(1,e[k]) for e in originals for k in ('width','height')]
        scale=max(.01,median(scales))
        scale=min(scale,width/max(1,sb[2]-sb[0]),height/max(1,sb[3]-sb[1]))
        cx=median(by_id[e['id']]['x']+by_id[e['id']]['width']/2
                  -(e['x']+e['width']/2-sx)*scale for e in originals)
        cy=median(by_id[e['id']]['y']+by_id[e['id']]['height']/2
                  -(e['y']+e['height']/2-sy)*scale for e in originals)
        # Whole-group clamping preserves internal offsets and image proportions.
        gw,gh=(sb[2]-sb[0])*scale,(sb[3]-sb[1])*scale
        cx=min(width-gw/2,max(gw/2,cx));cy=min(height-gh/2,max(gh/2,cy))
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
    result,chain_notes=_repair_column_chains(result,units,width,height,groups)
    result,clearance_notes=_repair_hero_clearance(result,units,width,height,groups)
    return result,notes+chain_notes+clearance_notes


def _repair_hero_clearance(elements,units,width,height,groups):
    """Keep source-side copy columns and top headers clear of a central hero."""
    from engine.resize import _text_visible_bounds, _unintended_text_overlaps
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
            heroes.append(e)
    if not heroes:return deepcopy(elements),[]
    hero=max(heroes,key=lambda e:e['width']*e['height'])
    target_hero=by_id[hero['id']]
    gap=max(18.,.025*height);margin=.035*width
    block_for={eid:set((eid,)) for eid in by_id}
    for group in groups:
        group=group&set(by_id)
        for eid in group:block_for[eid]=group
    trial=deepcopy(elements);out={e['id']:e for e in trial};notes=[]
    # Move source-left/right text clusters as units when their predicted boxes
    # collide with the central visual. The source side is structural evidence.
    for side in ('left','right'):
        texts=[e for e in source.values() if e['type']=='text' and e['id'] in out and e.get('content','').strip()
               and ((e['x']+e['width']/2 < hero['x']+hero['width']/2) if side=='left'
                    else (e['x']+e['width']/2 > hero['x']+hero['width']/2))]
        if not texts:continue
        rightmost=max(out[e['id']]['x']+out[e['id']]['width'] for e in texts)
        leftmost=min(out[e['id']]['x'] for e in texts)
        dx=0.
        if side=='left' and rightmost>target_hero['x']-gap:
            dx=target_hero['x']-gap-rightmost
            if leftmost+dx<margin:continue
        if side=='right' and leftmost<target_hero['x']+target_hero['width']+gap:
            dx=target_hero['x']+target_hero['width']+gap-leftmost
            if rightmost+dx>width-margin:continue
        if abs(dx)<1.:continue
        ids=set()
        for e in texts:ids.update(block_for[e['id']])
        before=deepcopy(trial)
        for e in trial:
            if e['id'] in ids:e['x']+=dx
        if set(_unintended_text_overlaps(trial,units))-set(_unintended_text_overlaps(before,units)):
            trial=before;continue
        out={e['id']:e for e in trial}
        notes.append(dict(kind='hero-column-clearance',side=side,dx=dx,element_ids=sorted(ids)))
    # Raise a centered source header only if the source left a real gap above H.
    header=[e for e in source.values() if e['type']=='text' and e['id'] in out and e.get('content','').strip()
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
        # Keep at least four pixels between visible glyph bounds, even when
        # the source gap scales down or Qwen's proposed frames overlap.
        new_x=max(anchor_vis_right+4.,anchor_vis_right+scaled_gap)
        sat_moved=abs(new_x-sat['x'])>.01
        if sat_moved:
            sat['x']=new_x
        # Keep the raw layout frames disjoint as well as the measured glyphs.
        # The visible advance is a lower bound so tightening cannot clip ink.
        anchor_vis=_text_visible_bounds(anchor)
        anchor_vis_w=max(0.,anchor_vis[2]-anchor_vis[0])
        max_frame_w=max(anchor_vis_w,
                        min(anchor['width'],sat['x']-anchor['x']-2.0))
        if max_frame_w<anchor['width']-1e-4:
            anchor['width']=max_frame_w
            anchor_tightened=True
        if anchor_tightened or sat_moved:
            changes.extend((anchor['id'],sat['id']))
    if set(_unintended_text_overlaps(result,units))-set(_unintended_text_overlaps(elements,units)):
        return deepcopy(elements),[]
    return result,sorted(set(changes))


def font_variant(elements,units,width,height):
    """Measure font alternatives in fixed model boxes; never rewrite target labels."""
    from engine.resize import _text, _text_info, _text_visible_bounds, _unintended_text_overlaps
    result=deepcopy(elements);source={e['id']:e for u in units for e in u['items']};changes=[]
    for e in result:
        if e['type']!='text' or not e.get('content','').strip() or e['id'] not in source:continue
        original=source[e['id']];base=max(1,float(e.get('font_size',1)))
        source_font=max(1,float(original.get('font_size',1)))
        source_lines=max(1,_text(original)[2])
        narrowed=(e['width']/max(1,e['height']))<.65*(original['width']/max(1,original['height']))
        max_lines=min(e.get('max_lines') or 1000,source_lines if not narrowed else max(source_lines,_text(e)[2]))
        # Preserve source type hierarchy through relative font ratios within
        # each actual lockup, rather than a global headline/body/tag ordering.
        low,high=max(1,base*.70),base*1.45
        def fits(size):
            probe=deepcopy(e);probe['letter_spacing']=original.get('letter_spacing',0)*base/source_font
            h,w,lines=_text(probe,e['width']*.96,size)
            # Single-line text that fits horizontally should keep its source-
            # scaled size. Increase the frame height after calibration instead
            # of shrinking a headline just to satisfy a conservative metric.
            single_line=(lines==1 and '\n' not in e.get('content','')
                         and w<=e['width'])
            return (single_line or (h<=e['height']*.95 and w<=e['width']*.96)) and lines<=max_lines
        if not fits(low):continue  # no feasible calibration in the bounded range
        for _ in range(20):
            mid=(low+high)/2
            if fits(mid):low=mid
            else:high=mid
        if abs(low-base)<base*.03:continue
        e['font_size']=low;e['letter_spacing']=original.get('letter_spacing',0)*low/source_font
        measured=_text_info(e,e['width'],low)
        measured_h,measured_w,measured_lines=measured[0],measured[1],measured[2]
        if measured_lines==1 and '\n' not in e.get('content','') and measured_w<=e['width']:
            e['height']=max(e['height'],measured_h)
            if e['y']+e['height']>height:
                e['y']=max(0.,height-e['height'])
        changes.append(e['id'])
    # Enforce a clear source-supported headline/copy hierarchy only where the
    # bounded font ranges permit it. Do not invent a universal tag size rule.
    roles={e['id']:u.get('semantic') for u in units for e in u['items']}
    incoming={e['id']:e for e in elements}
    for headline in result:
        if roles.get(headline['id'])!='headline' or headline['type']!='text':continue
        for body in result:
            if roles.get(body['id']) not in ('subheading','body') or body['type']!='text':continue
            if source[headline['id']].get('font_size',1)<1.35*source[body['id']].get('font_size',1):continue
            ceiling=headline['font_size']/1.35
            if body['font_size']>ceiling>=incoming[body['id']]['font_size']*.70:
                body['font_size']=ceiling
                body['letter_spacing']=source[body['id']].get('letter_spacing',0)*ceiling/max(1,source[body['id']].get('font_size',1))
                changes.append(body['id'])
    # Re-anchor compact superscripts after calibration, using measured glyph
    # advance instead of the usually oversized text frame.
    result,anchor_changes=anchor_superscripts(result,units)
    changes.extend(anchor_changes)
    if set(_unintended_text_overlaps(result,units))-set(_unintended_text_overlaps(elements,units)):
        return deepcopy(elements),[]
    return result,sorted(set(changes))


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


def contrast_variant(elements,units,width,height,weak_ids,under,full,return_alternatives=False):
    """Propose bounded text-unit moves using actual backing pixels at glyphs."""
    from PIL import ImageChops
    from engine.resize import (_rgba, _luminance, _feasible,
                               _text_visible_bounds, _unintended_text_overlaps)
    result=deepcopy(elements);by_id={e['id']:e for e in result};changes=[];alternatives=[]
    ink=ImageChops.difference(full,under).convert('L')
    source={e['id']:e for u in units for e in u['items']}
    groups=lockup_groups(units)
    for u in units:
        ids={e['id'] for e in u['items']}
        if not any(ids & g for g in groups):groups.append(ids)
    groups=list({frozenset(g):set(g) for g in groups}.values())
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
        priority={'backing-strip-expanded':0,'lockup-repositioned':1,'local-nudge':2}
        unique.sort(key=lambda item:(priority.get(item[1][0].get('repair_type'),3),
                                     item[0][0]['x'],item[0][0]['y']))
        return unique[:18],[]
    return result,changes
