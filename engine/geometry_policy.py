"""Aspect constraints shared by candidate generation, rendering and auditing.

Artwork keeps its authored proportions. Only rectangular layout surfaces
(text backplates, backgrounds, rules and edge strips) have an elastic axis.
Cover images may change their crop window; their pixels still scale uniformly.
"""
from copy import deepcopy
from functools import lru_cache
import json
from math import cos, sin, radians


def ambient_shape(element, source=None):
    """Ungrouped translucent paint is not evidence of an atomic attachment.

    Opacity can be authored either on the layer or in its fill color. Explicit
    groups remain stronger evidence and are not classified as ambient here.
    """
    if element.get('type')!='shape' or element.get('group_id') is not None:
        return False
    from engine.resize import _rgba
    color=_rgba(element.get('fill_color',''))
    alpha=float(element.get('opacity',1.))*(color[3] if color else 1.)
    if (alpha<.4 and source is not None
            and element.get('shape_type','rectangle') in ('rectangle','rounded_rectangle')):
        # Translucency alone does not make an enclosing copy panel ambient.
        # Strict containment is stronger evidence than incidental overlap.
        children=[t for t in source['elements'] if t.get('type')=='text'
                  and t.get('z_index',0)>element.get('z_index',0)
                  and t['x']>=element['x']-1 and t['y']>=element['y']-1
                  and t['x']+t['width']<=element['x']+element['width']+1
                  and t['y']+t['height']<=element['y']+element['height']+1]
        compact_copy=(len(children)==1 and element['width']*element['height']<.35*source['canvas']['width']*source['canvas']['height']
                      and children[0]['width']*children[0]['height']>.45*element['width']*element['height'])
        if len(children)>=2 or compact_copy:return False
    return alpha<.4


def elastic_shape_ids(source):
    return _elastic_shapes(json.dumps(source, sort_keys=True))


@lru_cache(maxsize=256)
def _elastic_shapes(payload):
    source = json.loads(payload)
    sw, sh = source['canvas']['width'], source['canvas']['height']
    texts = [e for e in source['elements'] if e['type'] == 'text']
    from engine.nested_structure import regional_surfaces
    elastic = set(regional_surfaces(source))
    for e in source['elements']:
        if e['type'] != 'shape' or e.get('shape_type', 'rectangle') not in ('rectangle', 'rounded_rectangle', 'line'):
            continue
        w, h = e['width'], e['height']
        if min(w, h) <= 0:
            continue
        # A divider can absorb length, but an ellipse/star/illustration cannot.
        if e.get('shape_type') == 'line' or min(w, h) <= max(4., .008 * min(sw, sh)) and max(w / h, h / w) >= 8:
            elastic.add(e['id'])
        if w >= .98 * sw and h >= .98 * sh:
            elastic.add(e['id'])
        if abs(e.get('rotation', 0)) < .1 and ((w <= .06 * sw and h >= .85 * sh) or (h <= .06 * sh and w >= .85 * sw)):
            elastic.add(e['id'])
        # Source-evidenced backplate: paint is behind, and contains a label.
        # Translucent ambient decorations must not become elastic via overlap.
        if not ambient_shape(e,source) and float(e.get('opacity', 1)) >= .9 and any(
            t.get('z_index', 0) > e.get('z_index', 0)
            and t['x'] >= e['x'] - 1 and t['y'] >= e['y'] - 1
            and t['x'] + t['width'] <= e['x'] + w + 1
            and t['y'] + t['height'] <= e['y'] + h + 1
            for t in texts
        ):
            elastic.add(e['id'])
    return frozenset(elastic)


def rigid(element, original, elastic):
    if original['type'] == 'shape':
        return original['id'] not in elastic
    return original['type'] == 'image' and element.get('fit', 'contain') != 'cover'


def preserve_aspects(elements, source, width, height):
    """Fit rigid art inside its proposed box, retaining its center and rotation."""
    originals = {e['id']: e for e in source['elements']}
    elastic = elastic_shape_ids(source)
    result = deepcopy(elements)
    for e in result:
        if e.get('type') == 'image' and e.get('fit') == 'fill':
            e['fit'] = 'contain'
        old = originals.get(e['id'])
        if old is None or min(old['width'], old['height'], e['width'], e['height']) <= 0 or not rigid(e, old, elastic):
            continue
        scale = min(e['width'] / old['width'], e['height'] / old['height'])
        a = radians(e.get('rotation', 0))
        rw = abs(old['width'] * cos(a)) + abs(old['height'] * sin(a))
        rh = abs(old['width'] * sin(a)) + abs(old['height'] * cos(a))
        scale = min(scale, width / max(rw, 1e-9), height / max(rh, 1e-9))
        w, h = old['width'] * scale, old['height'] * scale
        if abs(e['width'] - w) < 1e-8 and abs(e['height'] - h) < 1e-8:
            continue
        cx, cy = e['x'] + e['width'] / 2, e['y'] + e['height'] / 2
        dx, dy = rw * scale / 2, rh * scale / 2
        cx, cy = min(width - dx, max(dx, cx)), min(height - dy, max(dy, cy))
        e.update(x=cx - w / 2, y=cy - h / 2, width=w, height=h)
    return result


def aspect_issues(elements, source):
    originals = {e['id']: e for e in source['elements']}
    elastic = elastic_shape_ids(source)
    result = []
    for e in elements:
        old = originals.get(e['id'])
        if old is None or min(old['width'], old['height'], e['width'], e['height']) <= 0 or not rigid(e, old, elastic):
            continue
        ratio = (e['width'] / e['height']) / (old['width'] / old['height'])
        if abs(ratio - 1) > .001:
            result.append(dict(kind='art_aspect_distortion', ids=[e['id']], ratio=ratio))
    return result
