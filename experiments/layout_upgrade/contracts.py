"""Exact inventory schema and strict decoding for the normalized v2 contract."""
from copy import deepcopy
import json
import math


def proposal_schema(source):
    coordinate = dict(type='number', minimum=0, maximum=1000)
    positive = dict(type='number', exclusiveMinimum=0, maximum=1000)
    items = []
    for i, element in enumerate(source['elements']):
        properties = dict(id=dict(const=f'e{i:03d}'),
                          box=dict(type='array', prefixItems=[coordinate, coordinate, positive, positive],
                                   minItems=4, maxItems=4))
        if element['type'] == 'text':
            properties['font_scale'] = positive
        items.append(dict(type='object', properties=properties, required=list(properties), additionalProperties=False))
    return dict(type='object', properties=dict(elements=dict(type='array', prefixItems=items,
                 minItems=len(items), maxItems=len(items))), required=['elements'], additionalProperties=False)


def decode(response, source, target):
    """No delimiter repair or ID recovery: report the literal generated payload."""
    from engine.qwen_prompt import decode_v2_completion
    diagnostic = dict(json_complete=False, exact_ids=False, inventory_exact=False,
                      unknown_ids=[], duplicate_ids=[], missing_ids=[], unusable_ids=[], invalid_geometry_ids=[])
    try:
        payload = json.loads(response.strip())
        rows = payload['elements']
        if not isinstance(rows, list):
            raise ValueError('elements must be an array')
        diagnostic['json_complete'] = True
    except (ValueError, KeyError, TypeError) as exc:
        diagnostic['parse_error'] = str(exc)
        return [], diagnostic
    expected = {f'e{i:03d}' for i in range(len(source['elements']))}
    seen = set(); clean = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get('id')
        if not isinstance(key, str) or key not in expected:
            diagnostic['unknown_ids'].append(key); continue
        if key in seen:
            diagnostic['duplicate_ids'].append(key); continue
        seen.add(key)
        try:
            box = row['box']
            if not isinstance(box, list) or len(box) != 4 or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box) or min(box[2:]) <= 0:
                raise ValueError('Invalid geometry')
            original = source['elements'][int(key[1:])]
            if original['type'] == 'text' and not (isinstance(row['font_scale'], (int, float))
                    and not isinstance(row['font_scale'], bool) and math.isfinite(row['font_scale']) and row['font_scale'] > 0):
                raise ValueError('Invalid type scale')
            clean.append(row)
        except (ValueError, TypeError, KeyError):
            diagnostic['invalid_geometry_ids'].append(key)
            continue
    valid = {r['id'] for r in clean}
    diagnostic['missing_ids'] = sorted(expected - seen)
    diagnostic['unusable_ids'] = sorted(expected - valid)
    diagnostic['inventory_exact'] = (len(rows) == len(expected) and seen == expected
                               and not diagnostic['unknown_ids'] and not diagnostic['duplicate_ids'])
    diagnostic['exact_ids'] = diagnostic['inventory_exact'] and valid == expected
    return decode_v2_completion({'elements': clean}, source, target)['elements'], diagnostic


def raw_document(source, target, elements):
    doc = deepcopy(source); doc['canvas'] = deepcopy(target)
    predicted = {e['id']: e for e in elements}
    # Missing IDs remain missing in the raw view; no recovery credit.
    doc['elements'] = [{**deepcopy(e), **predicted[e['id']]} for e in source['elements'] if e['id'] in predicted]
    return doc
