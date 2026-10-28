"""Versioned ten-feature pairwise ranker. Hard safety gates remain external.

No default production weights: offline human validation and a separate engine
ablation are required before wiring a fitted artifact into selection.
"""
import math
from statistics import mean
FEATURES=('width_use','height_use','headline_font_ratio','min_font_floor','hero_area_share',
          'rag_score','stack_balance','peer_gap_asymmetry','contrast_margin','siglip_zero_shot')


def features(source,design,contrast_margin,siglip_zero_shot):
    from engine.macro_layout import quality,bounds
    from engine.resize import _rag,_analyze
    w,h=(design['canvas'][k] for k in ('width','height'));q=quality(design['elements'],source,w,h)
    original={e['id']:e for e in source['elements']};texts=[e for e in design['elements'] if e['type']=='text' and e.get('content','').strip()]
    ref=math.sqrt(w*h/(source['canvas']['width']*source['canvas']['height']))
    major=max((e for e in source['elements'] if e['type']=='text'),key=lambda e:e['font_size'],default=None)
    ratio=q.get('headline_font',0)/max(1.,major['font_size']*ref) if major else 0.
    floor=min(16.,max(5.,min(w,h)*.024));minratio=min((e['font_size']/floor for e in texts),default=0.)
    out={e['id']:e for e in design['elements']};rows={}
    for u in _analyze(source)[1]:
        peer=u.get('peer_ids')
        if peer:rows.setdefault(tuple(sorted(peer)),[]).append(u)
    gaps=[]
    for row in rows.values():
        boxes=[bounds([out[e['id']] for e in u['items'] if e['id'] in out]) for u in row if all(e['id'] in out for e in u['items'])]
        if len(boxes)<3:continue
        boxes.sort();ds=[b[0]-a[2] for a,b in zip(boxes,boxes[1:])];height=mean(b[3]-b[1] for b in boxes)
        gaps.append((max(ds)-min(ds))/max(1.,height))
    values=[q['width_use'],q['height_use'],ratio,minratio,q.get('hero_area_share',0.),
            mean(_rag(e) for e in texts) if texts else 0.,q.get('margin_imbalance',0.),
            mean(gaps) if gaps else 0.,contrast_margin,siglip_zero_shot]
    if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite structural features')
    return [float(v) for v in values]


def fit(pairs,l2=.1):
    """Train-only standardization and fixed L2 Bradley-Terry logistic fit."""
    import numpy as np
    x=np.asarray(pairs,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,len(FEATURES)):raise ValueError('Expected N x 2 x 10 feature pairs')
    center=x.mean(axis=(0,1));scale=np.maximum(x.std(axis=(0,1)),1e-3);d=(x[:,0]-x[:,1])/scale
    weights=np.zeros(len(FEATURES))
    def objective(w):return float(np.logaddexp(0.,-d@w).mean()+.5*l2*(w@w))
    for _ in range(80):
        z=np.clip(d@weights,-60,60);p=1/(1+np.exp(z));grad=-(d.T@p)/len(d)+l2*weights
        hessian=(d.T*(p*(1-p)))@d/len(d)+l2*np.eye(len(FEATURES));step=np.linalg.solve(hessian,grad)
        alpha=1.
        while alpha>1e-8 and objective(weights-alpha*step)>objective(weights):alpha*=.5
        weights-=alpha*step
        if np.linalg.norm(alpha*step)<1e-9:break
    return dict(version=1,features=list(FEATURES),center=center.tolist(),scale=scale.tolist(),weights=weights.tolist(),l2=l2)


def score(vector,model):
    if model.get('version')!=1 or tuple(model['features'])!=FEATURES:raise ValueError('Feature contract mismatch')
    return sum((v-m)/s*w for v,m,s,w in zip(vector,model['center'],model['scale'],model['weights']))
