"""Owned label fitting uses local parent capacity, including rotated controls."""
from copy import deepcopy
import pytest
from engine.resize import _calibrate_text_boxes,_render_text_ink
from engine.raster_metrics import glyph_overflow
from engine.layout_contracts import controls,issues


def fixture(angle):
    panel=dict(id='panel',type='shape',shape_type='rounded_rectangle',x=50.,y=40.,width=180.,height=32.,
               fill_color='#773311',rotation=angle,group_id='tag',z_index=1)
    label=dict(id='label',type='text',content='OPEN STUDIO',font_family='Solstice Sans',font_size=12.,
        letter_spacing=.4,text_align='center',color='#ffffff',x=60.,y=48.,width=160.,height=16.,
        rotation=angle,group_id='tag',z_index=2)
    return dict(canvas=dict(width=600,height=600),elements=[panel,label])


@pytest.mark.parametrize('angle',[-17.,7.,30.])
def test_noisy_rotated_label_fits_parent_without_stretching_panel(angle):
    source=fixture(angle);assert controls(source)==(('panel','label'),)
    proposal=deepcopy(source['elements']);proposal[0].update(x=300.,y=220.,width=120.,height=32.)
    proposal[1].update(x=310.,y=222.,width=90.,height=18.,font_size=55.,letter_spacing=55*.4/12)
    before=deepcopy(proposal)
    result,notes=_calibrate_text_boxes(proposal,source,600,600)
    assert proposal==before and result[0]==before[0]
    label=result[1];assert 12<=label['font_size']<55
    assert label['letter_spacing']/label['font_size']==pytest.approx(.4/12)
    assert not glyph_overflow(source,result,600,600)
    assert not [d for d in issues(result,source,600,600) if d['kind'] in ('control_label_axis','control_padding')]
    assert any(n.get('rotated_control_local_fit') for n in notes)


def test_unfittable_parent_preserves_visible_type_and_reports_failure():
    source=fixture(12);proposal=deepcopy(source['elements']);proposal[0].update(width=4.,height=4.)
    result,notes=_calibrate_text_boxes(proposal,source,600,600)
    assert result[1]['font_size']==proposal[1]['font_size']
    assert result[1].get('opacity',1)>0
    assert any(n.get('unfittable') for n in notes)


def test_clean_rotated_control_does_not_change_font_or_frame():
    source=fixture(-9);before=deepcopy(source['elements'])
    result,notes=_calibrate_text_boxes(before,source,600,600)
    for key in ('width','height','font_size','letter_spacing'):assert result[1][key]==before[1][key]
    assert not glyph_overflow(source,result,600,600)


def test_available_local_width_is_used_before_shrinking_font():
    source=fixture(-11);proposal=deepcopy(source['elements'])
    proposal[1]['width']=45.
    result,notes=_calibrate_text_boxes(proposal,source,600,600)
    assert result[0]==proposal[0]
    assert result[1]['font_size']==proposal[1]['font_size']
    assert result[1]['width']>proposal[1]['width']
    assert not glyph_overflow(source,result,600,600)
