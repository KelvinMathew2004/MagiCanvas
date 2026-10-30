"""New solver proposals preserve rigid source ownership before raster checks."""
from copy import deepcopy
import pytest
from experiments.layout_upgrade.training_free import micro_polish,translated


def test_rigid_polish_improves_shared_source_axis_without_stretch():
    source=dict(canvas=dict(width=600,height=600),elements=[dict(id=f'v{i}',type='image',asset_id='x.png',x=100.+i*180,y=200.,width=80.,height=80.,z_index=i) for i in range(3)])
    elements=deepcopy(source['elements']);elements[1]['y']+=10.;original=deepcopy(elements)
    candidates=micro_polish(elements,source,600,600)
    assert elements==original and candidates
    result=candidates[0]['elements'];assert max(e['y'] for e in result)-min(e['y'] for e in result)<1.
    assert candidates[0]['optimization']['after']<candidates[0]['optimization']['before']
    for a,b in zip(elements,result):
        for k in ('id','type','width','height','asset_id','z_index'):assert a[k]==b[k]
        assert abs(a['x']-b['x'])<=.08*600 and abs(a['y']-b['y'])<=.08*600


def test_translation_keeps_owned_children_and_source_immutable():
    elements=[dict(id='back',x=5.,y=10.,width=100.,height=30.),dict(id='type',x=15.,y=15.,width=80.,height=20.,font_size=12.,letter_spacing=.5)]
    original=deepcopy(elements);result=translated(elements,{'back':(20.,-3.),'type':(20.,-3.)})
    assert elements==original
    assert result[1]['x']-result[0]['x']==10 and result[1]['y']-result[0]['y']==5
    assert result[1]['font_size']==12 and result[1]['letter_spacing']==.5


def test_single_atomic_group_cannot_be_split_by_continuous_polish():
    panel=dict(id='p',type='shape',shape_type='rectangle',x=50.,y=100.,width=120.,height=40.,fill_color='#333333',group_id='g',z_index=1)
    label=dict(id='t',type='text',content='Learn more',font_family='Solstice Sans',x=60.,y=110.,width=100.,height=20.,font_size=12.,group_id='g',z_index=2)
    source=dict(canvas=dict(width=600,height=600),elements=[panel,label])
    assert micro_polish(deepcopy(source['elements']),source,600,600)==[]
