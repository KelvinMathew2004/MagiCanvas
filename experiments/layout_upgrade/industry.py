"""Independent, bounded typography and intact-group neighborhood experiments.

No production imports this module. Source IDs are used only as structural keys.
Reference target coordinates never enter proposal generation or selection.
"""
from copy import deepcopy
from math import hypot


def rag(info):
    """Squared line-width coefficient of variation, measured by the renderer."""
    widths = info[3]
    if len(widths) < 2 or min(widths) <= 0:
        return 0.
    average = sum(widths) / len(widths)
    return sum((w - average) ** 2 for w in widths) / len(widths) / average ** 2


def paragraph_balance(elements, source, width, height, minimum_width_ratio=.70,
                      width_steps=24, minimum_rag_gain=.25, maximum_lines=6):
    from engine.resize import _text_info, _render_text_ink
    from engine.layout_contracts import controls
    original = {e['id']: e for e in source['elements']}
    control_ids = {t for _, t in controls(source)}
    result = deepcopy(elements); changes = []; skipped = {}
    for e in result:
        if e['type'] != 'text':
            continue
        old = original[e['id']]
        # No rewrite of authored breaks, rotated ribbons, controls, tracking,
        # style runs, font, height, line count, or alignment mode.
        reason = ('authored-break' if '\n' in e.get('content', '') or '\n' in old.get('content', '') else
                  'rotation' if abs(e.get('rotation', 0)) > .1 else
                  'control' if e['id'] in control_ids else None)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1; continue
        before = _text_info(e)
        if not 2 <= before[2] <= maximum_lines or before[0] > e['height'] + 1:
            continue
        score = rag(before)
        if score <= .005:
            continue
        best = (score, e['width'], before)
        for step in range(1, width_steps + 1):
            w = e['width'] * (1 - (1 - minimum_width_ratio) * step / width_steps)
            info = _text_info(e, w)
            if info[2] != before[2] or info[0] > e['height'] + 1 or info[1] > w + .5:
                continue
            candidate = (rag(info), -w)
            if candidate < (best[0], -best[1]):
                best = (candidate[0], w, info)
        if best[0] >= score * (1 - minimum_rag_gain):
            continue
        test = deepcopy(e); test['width'] = best[1]
        anchor = e.get('text_align', 'left')
        shift = (e['width'] - best[1]) * ({'center': .5, 'right': 1.}.get(anchor, 0.))
        test['x'] += shift
        ink = _render_text_ink(test)
        if ink is None or ink[2] > round(test['width']) or ink[3] > round(test['height']):
            continue
        changes.append(dict(id=e['id'],before_width=e['width'],after_width=best[1],
                            before_rag=score,after_rag=best[0],lines=before[2],anchor=anchor))
        e.update(test)
    return (result if changes else None), dict(changes=changes,skipped=skipped,
        reason='balanced measured wraps' if changes else 'no eligible material paragraph improvement')


def group_neighbors(elements, source, width, height, maximum_proposals=12,
                    maximum_groups=16, split_step=.08, minimum_quality_gain=2.):
    """Deterministic adjacent swaps and two-sided split reallocation.

    Inspired by the public tailsgit search move vocabulary, without its missing
    critic, random restarts or leaf transforms. Current source-only quality
    scores rank proposals; actual ink/contrast checks still decide eligibility.
    """
    from engine.macro_layout import composites,bounds,quality
    from engine.source_structure import _transform
    spec = composites(source, tight=True); by = {e['id']: e for e in elements}
    blocks = [[by[e['id']] for e in group] for group in spec['blocks']
              if all(e['id'] in by for e in group)]
    if not 2 <= len(blocks) <= maximum_groups:
        return [],dict(reason='unsupported group count',groups=len(blocks),generated=0)
    boxes = [bounds(b) for b in blocks]
    # Only neighboring groups participate. Source-owned multi-element blocks
    # stay intact. Ambient art, edges and regional backplates are not variables.
    pairs = sorted(((hypot(((a[0]+a[2])-(b[0]+b[2]))/width,
                            ((a[1]+a[3])-(b[1]+b[3]))/height),i,j)
                    for i,a in enumerate(boxes) for j,b in enumerate(boxes[:i])))
    candidates=[]; baseline=quality(elements,source,width,height)['cost']
    for _,i,j in pairs:
        a,b=boxes[i],boxes[j]
        aw,ah=a[2]-a[0],a[3]-a[1];bw,bh=b[2]-b[0],b[3]-b[1]
        if min(aw,ah,bw,bh)<=0 or max(aw*ah,bw*bh)/min(aw*ah,bw*bh)>4:
            continue
        # Overlapping blocks may encode an attachment missed by the inferred
        # tree. Never separate them in this experiment.
        if min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1]):
            continue
        variants=[]
        delta=((b[0]+b[2]-a[0]-a[2])/2,(b[1]+b[3]-a[1]-a[3])/2)
        variants.append(('adjacent-swap',[(i,1.,a[0]+delta[0],a[1]+delta[1]),
                                          (j,1.,b[0]-delta[0],b[1]-delta[1])]))
        # Adjust a two-sided allocation while preserving each block's aspect.
        # This differs from the already-tested single shared growth factor.
        wide=width>=height; axis=0 if wide else 1
        first,second=(i,j) if a[axis]<b[axis] else (j,i)
        for direction in (-1,1):
            transforms=[]
            for k,sign in ((first,direction),(second,-direction)):
                c=boxes[k];cw,ch=c[2]-c[0],c[3]-c[1];s=1+sign*split_step
                left=c[0] if wide and k==first else c[2]-cw*s if wide else (c[0]+c[2]-cw*s)/2
                top=c[1] if not wide and k==first else c[3]-ch*s if not wide else (c[1]+c[3]-ch*s)/2
                transforms.append((k,s,left,top))
            variants.append(('split-reallocation',transforms))
        for kind,transforms in variants:
            result=deepcopy(elements);mapped={e['id']:e for e in result}
            for k,s,left,top in transforms:
                for e in _transform(blocks[k],s,left,top,boxes[k]):mapped[e['id']].update(e)
            cost=quality(result,source,width,height)['cost']
            candidates.append(dict(elements=result,kind=kind,cost=cost,groups=[i,j]))
            if len(candidates)>=maximum_proposals:break
        if len(candidates)>=maximum_proposals:break
    return candidates,dict(reason='bounded intact-group neighborhood',groups=len(blocks),
        generated=len(candidates),before_cost=baseline,minimum_quality_gain=minimum_quality_gain)
