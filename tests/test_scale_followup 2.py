"""Controlled experiments preserve ownership and exact raster statistics."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
deps=Path(__file__).resolve().parents[1]/'data/upgrade_exploration/deps'
if deps.is_dir():sys.path.insert(0,str(deps))
from experiments.layout_upgrade.scale_followup import partition_statistics,container_growth,quota_funnel


@pytest.mark.parametrize('size',[1,2,7,19,20,21,99,500,5000,100000])
@pytest.mark.parametrize('opacity',[0.,.5,1.])
def test_partition_exactly_matches_full_sort(size,opacity):
    from engine.raster_metrics import glyph_statistics
    rng=np.random.default_rng(size);pixels=rng.integers(0,256,(1,size,3),dtype=np.uint8)
    mask=rng.integers(0,256,(1,size),dtype=np.uint8)
    assert partition_statistics(pixels,mask,(180,20,70,.6),opacity)==glyph_statistics(pixels,mask,(180,20,70,.6),opacity)
    assert partition_statistics(pixels,np.zeros_like(mask),(0,0,0,1)) is None


def test_growth_preserves_card_children_and_tracking():
    pytest.importorskip('scipy')
    panel=dict(id='p',type='shape',shape_type='rectangle',x=200.,y=200.,width=200.,height=120.,fill_color='#fff',group_id='g',z_index=1)
    label=dict(id='t',type='text',content='Example',font_family='Solstice Sans',x=220.,y=230.,width=160.,height=40.,font_size=20.,letter_spacing=2.,group_id='g',z_index=2)
    source=dict(canvas=dict(width=600,height=600),elements=[panel,label]);original=deepcopy(source)
    es,diagnostic=container_growth(deepcopy(source['elements']),source,600,600)
    assert source==original and es is not None
    scale=diagnostic['scale'];assert scale==pytest.approx(1.25)
    assert es[1]['x']-es[0]['x']==pytest.approx((label['x']-panel['x'])*scale)
    assert es[1]['font_size']==pytest.approx(20*scale) and es[1]['letter_spacing']==pytest.approx(2*scale)
    assert es[0]['width']/es[0]['height']==pytest.approx(panel['width']/panel['height'])


def test_growth_has_no_headroom_when_content_already_fills_canvas():
    pytest.importorskip('scipy')
    image=dict(id='i',type='image',asset_id='a.png',x=0.,y=0.,width=600.,height=600.,z_index=1)
    # It is a background and cannot be treated as spare foreground space.
    source=dict(canvas=dict(width=600,height=600),elements=[image])
    es,diagnostic=container_growth([image],source,600,600)
    assert es is None


def test_funnel_preserves_independent_paths_and_small_pools(monkeypatch):
    import importlib
    resize=importlib.import_module('engine.resize')
    monkeypatch.setattr(resize,'_objective_layout_cost',lambda *a,**kw:0.)
    names=['det-'+str(i) for i in range(30)]+['qwen-original','qwen-font-fitted','qwen-pre-container-font-fitted',
        'macro-band-reflow-0','source-attached-proportional','semantic-stack-0','editorial-split-0','proportional']
    cs=[dict(name=name,elements=[]) for name in names]
    kept,note=quota_funnel(cs,{},[],100,100,1.)
    assert len(kept)==12 and note['pruned']==len(cs)-12
    assert set(names[-8:])<={c['name'] for c in kept}
    assert quota_funnel(cs[:3],{},[],100,100,1.)[0]==cs[:3]
