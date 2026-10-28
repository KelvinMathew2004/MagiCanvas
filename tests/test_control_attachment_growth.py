"""Measured control growth must not consume an authored media contact edge."""
from copy import deepcopy
import pytest
from engine.layout_contracts import _attached_growth,_grow_controls
from engine.resize import _render_text_ink


@pytest.mark.parametrize('axis,after',[('y',True),('y',False),('x',True),('x',False)])
def test_growth_keeps_each_authored_contact_edge(axis,after):
    photo=dict(id='photo',type='image',x=200.,y=200.,width=200.,height=200.)
    original=dict(id='panel',type='shape',x=200.,y=400. if after else 180.,width=200.,height=20.)
    if axis=='x':original.update(x=400. if after else 180.,y=200.,width=20.,height=200.)
    source=dict(canvas=dict(width=800,height=800),elements=[photo,original])
    proposed=dict(x=original['x']-10,y=original['y']-10,width=original['width']+20,height=original['height']+20)
    result=_attached_growth(original,original,proposed,source,{'photo':photo},800,800,10)
    extent='height' if axis=='y' else 'width'
    assert result is not None
    assert result[axis]==pytest.approx(photo[axis]+photo[extent] if after else photo[axis]-result[extent])


def test_detached_or_authored_overlapping_control_is_not_snapped_back_to_photo():
    photo=dict(id='photo',type='image',x=100.,y=100.,width=200.,height=200.)
    original=dict(id='panel',type='shape',x=100.,y=300.,width=200.,height=20.)
    source=dict(canvas=dict(width=600,height=600),elements=[photo,original])
    panel=dict(original,y=400.)
    proposed=dict(x=100.,y=395.,width=200.,height=30.)
    assert _attached_growth(panel,original,proposed,source,{'photo':photo},600,600,10)==proposed
    overlap=dict(original,y=280.);source['elements'][1]=overlap
    assert _attached_growth(overlap,overlap,proposed,source,{'photo':photo},600,600,10)==proposed


def test_growth_at_canvas_boundary_is_refused_instead_of_moving_into_media():
    photo=dict(id='photo',type='image',x=100.,y=100.,width=200.,height=200.)
    original=dict(id='panel',type='shape',x=100.,y=300.,width=200.,height=20.)
    source=dict(canvas=dict(width=600,height=600),elements=[photo,original])
    proposed=dict(x=100.,y=285.,width=200.,height=50.)
    assert _attached_growth(original,original,proposed,source,{'photo':photo},600,330,5) is None


def test_actual_font_and_backing_growth_remain_below_photo_and_center_label():
    photo=dict(id='photo',type='image',x=40.,y=80.,width=400.,height=200.,asset_id='missing.png',z_index=1)
    panel=dict(id='panel',type='shape',shape_type='rectangle',fill_color='#ffaa55',x=40.,y=280.,width=400.,height=40.,group_id='caption',z_index=2)
    label=dict(id='label',type='text',x=60.,y=290.,width=360.,height=20.,font_size=16.,font_family='Solstice Sans',color='#000000',content='Meet the people behind the collection.',text_align='center',group_id='caption',z_index=3)
    source=dict(canvas=dict(width=600,height=600),elements=[photo,panel,label]);target=deepcopy(source['elements'])
    for e in target:
        e.update(x=100+(e['x']-40)*.15,y=20+(e['y']-80)*.15,width=e['width']*.15,height=e['height']*.15)
        if e['type']=='text':e['font_size']*=.15
    before_height=target[1]['height'];_grow_controls(target,source,220,100)
    assert target[1]['height']>before_height
    assert target[1]['y']==pytest.approx(target[0]['y']+target[0]['height'])
    ink=_render_text_ink(target[2]);center=target[2]['y']+(ink[1]+ink[3])/2
    assert center==pytest.approx(target[1]['y']+target[1]['height']/2,abs=1.)
    once=deepcopy(target);_grow_controls(target,source,220,100)
    assert target==once
