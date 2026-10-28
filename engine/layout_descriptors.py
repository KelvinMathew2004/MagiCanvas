"""Shared geometry-only macro descriptors for retrieval and audits."""
from collections import Counter


def describe(source,blocks):
    from engine.resize import _analyze
    from engine.macro_layout import bounds
    sw,sh=(source['canvas'][k] for k in ('width','height'));units=_analyze(source)[1]
    semantics={e['id']:u.get('semantic',u.get('role','art')) for u in units for e in u['items']}
    mapping={'headline':'header','kicker':'header','logo':'header','hero':'visual','product':'visual','cta':'action','badge':'tag','callout':'tag','footer':'footer'}
    result=[]
    for block in blocks:
        box=bounds(block);texts=[e for e in block if e['type']=='text'];photos=[e for e in block if e['type']=='image' and e['width']*e['height']>.025*sw*sh]
        votes=Counter()
        for e in block:
            role=mapping.get(semantics.get(e['id']),'copy' if e['type']=='text' else 'art')
            votes[role]+=e.get('font_size',0)**2 if e['type']=='text' else e['width']*e['height']*.02
        role=max(votes,key=lambda k:(votes[k],k)) if votes else 'art'
        if photos and any(semantics.get(e['id'])=='hero' for e in photos):role='visual'
        kind='media' if photos else 'copy' if all(e['type']=='text' or min(e['width'],e['height'])<=6 for e in block) else 'control' if role in ('tag','action') else 'card'
        result.append(dict(ids=[e['id'] for e in block],box=box,role=role,kind=kind,
            area=max(1e-9,(box[2]-box[0])*(box[3]-box[1])/(sw*sh)),aspect=max(1e-6,(box[2]-box[0])/max(1.,box[3]-box[1]))))
    return result
