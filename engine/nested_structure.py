"""Distinguish regional paint from atoms and infer repeated unbacked rows.

Only source geometry and explicit ownership are evidence. A regional surface
may contain several independent cards without making them one rigid object.
"""
from collections import defaultdict


def control_captions(source):
    """Nearest same-width caption just outside a real control, not legal copy."""
    from engine.layout_contracts import controls
    by={e['id']:e for e in source['elements']};result=[]
    for pid,tid in controls(source):
        p,t=by[pid],by[tid]
        choices=[e for e in source['elements'] if e['type']=='text' and e['id']!=tid
                 and str(e.get('content','')).strip() and float(e.get('opacity',1))>0
                 and abs(e.get('rotation',0))<=.1 and .8*p['width']<=e['width']<=1.2*p['width']
                 and abs(e['x']+e['width']/2-p['x']-p['width']/2)<=4
                 and 0<=e['y']-p['y']-p['height']<=max(4.,e['font_size'])
                 and e['font_size']<t['font_size']*.85
                 and (e.get('group_id') is None or e.get('group_id')==p.get('group_id'))]
        if choices:result.append((pid,tid,min(choices,key=lambda e:e['y'])['id']))
    return result


def photo_caption_panels(source):
    """Full-width native backplates attached immediately below a photo."""
    photos=[e for e in source['elements'] if e['type']=='image' and e['width']>0]
    return {p['id'] for p in source['elements'] if p['type']=='shape'
            and any(abs(p['x']-photo['x'])<=.04*photo['width']
                    and abs(p['width']/photo['width']-1)<=.08
                    and abs(p['y']-photo['y']-photo['height'])<=max(2.,.02*source['canvas']['height'])
                    and p['height']<=.25*photo['height'] for photo in photos)}


def unpainted_ids(source):
    """Exact no-paint evidence, never a small-font or thin-stroke threshold."""
    w,h=source['canvas']['width'],source['canvas']['height']
    return {e['id'] for e in source['elements'] if e['width']<=0 or e['height']<=0
            or float(e.get('opacity',1))<=0
            or abs(e.get('rotation',0))<=.1 and
                (e['x']>=w or e['y']>=h or e['x']+e['width']<=0 or e['y']+e['height']<=0)
            or e['type']=='text' and not str(e.get('content','')).strip()}


def regional_surfaces(source):
    w, h = source['canvas']['width'], source['canvas']['height']
    result = {}
    for p in source['elements']:
        if (p['type'] != 'shape' or p.get('group_id') is not None
                or p.get('shape_type', 'rectangle') != 'rectangle'
                or abs(p.get('rotation', 0)) > .1 or p['width'] * p['height'] < .12*w*h):
            continue
        children = [e for e in source['elements'] if e is not p
                    and e.get('z_index', 0) > p.get('z_index', 0)
                    and e['x'] >= p['x']-1 and e['y'] >= p['y']-1
                    and e['x']+e['width'] <= p['x']+p['width']+1
                    and e['y']+e['height'] <= p['y']+p['height']+1]
        groups = {e['group_id'] for e in children if e.get('group_id') is not None}
        if len(groups) >= 2:
            result[p['id']] = [e['id'] for e in children]
    return result


def repeated_text_rows(elements, canvas):
    """At least three disjoint baseline pairs must support the same two axes.

    Different explicit groups never become a row. Font hierarchy and column
    widths must repeat too; a random headline beside a caption is not a row.
    """
    texts = [e for e in elements if e['type'] == 'text' and e['width'] > 1
             and e['height'] > 1 and str(e.get('content', '')).strip()
             and float(e.get('opacity', 1)) > 0 and abs(e.get('rotation', 0)) <= .1]
    candidates = []
    for a in texts:
        peers = [b for b in texts if b['x'] >= a['x']+a['width']-1
                 and min(abs(a['y']-b['y']),abs((a['y']+a['height']/2)-(b['y']+b['height']/2))) <= 4
                 and max(a.get('font_size', 1), b.get('font_size', 1))
                     <= 1.15*min(a.get('font_size', 1), b.get('font_size', 1))
                 and a.get('group_id') == b.get('group_id')]
        if peers:
            b = min(peers, key=lambda e:e['x'])
            # A table with three/four fields needs a real table model. Do not
            # silently reinterpret adjacent columns as independent two-field
            # rows (which can detach quantities from their descriptions).
            third=any(t is not a and t is not b and t.get('group_id')==a.get('group_id')
                      and min(abs(t['y']-a['y']),abs((t['y']+t['height']/2)-(a['y']+a['height']/2)))<=4
                      for t in texts)
            if not third:candidates.append((a, b))
    families = []
    for pair in sorted(candidates, key=lambda p:p[0]['y']):
        a, b = pair
        family = next((f for f in families if all(
            abs(e['x']-old['x']) <= 4 and abs(e['width']-old['width']) <= 4
            and max(e.get('font_size', 1), old.get('font_size', 1))
                <= 1.15*min(e.get('font_size', 1), old.get('font_size', 1))
            for e, old in zip(pair, f[0]))), None)
        if family is None:
            families.append([pair])
        else:
            family.append(pair)
    result = []
    for family in families:
        if len(family) < 3:
            continue
        # Separate rows cannot overlap; their pitch must be reasonably regular.
        ys = [a['y'] for a,b in family]
        gaps = [b-a for a,b in zip(ys,ys[1:])]
        if min(gaps) <= 0 or max(gaps) > 1.5*min(gaps):
            continue
        ids = [e['id'] for pair in family for e in pair]
        if len(set(ids)) == len(ids):
            result.extend((a['id'], b['id']) for a,b in family)
    return result
