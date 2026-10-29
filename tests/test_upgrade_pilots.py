"""Experimental completion metrics and packing must not hide violations."""
import json
import sys
from pathlib import Path

import pytest

from experiments.layout_upgrade.contracts import decode, proposal_schema


@pytest.fixture
def source():
    return dict(canvas=dict(width=400,height=400),elements=[
        dict(id='title',type='text',content='Open studio',x=10,y=10,width=200,height=40,font_size=24),
        dict(id='visual',type='image',x=10,y=80,width=160,height=80)])


def payload():
    return {'elements':[{'id':'e000','box':[10,10,500,80],'font_scale':40},
                        {'id':'e001','box':[10,100,400,200]}]}


def test_exact_inventory_is_distinct_from_numeric_validity(source):
    p=payload();p['elements'][0]['font_scale']=0
    decoded,report=decode(json.dumps(p),source,dict(width=800,height=200))
    assert report['inventory_exact'] and not report['exact_ids']
    assert report['missing_ids']==[] and report['unusable_ids']==['e000']
    assert report['invalid_geometry_ids']==['e000']
    assert [e['id'] for e in decoded]==['visual']


def test_literal_inventory_rejects_duplicates_unknown_and_truncation(source):
    p=payload();p['elements'][1]['id']='e000'
    _,report=decode(json.dumps(p),source,source['canvas'])
    assert report['duplicate_ids']==['e000'] and not report['inventory_exact']
    p=payload();p['elements'][0]['id']=['unhashable']
    _,report=decode(json.dumps(p),source,source['canvas'])
    assert not report['inventory_exact']
    _,report=decode(json.dumps(payload())[:-2],source,source['canvas'])
    assert not report['json_complete']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),'15',True,-1])
def test_invalid_dimensions_are_not_silently_repaired(source,value):
    p=payload();p['elements'][0]['box'][2]=value
    _,report=decode(json.dumps(p),source,source['canvas'])
    assert report['inventory_exact'] and not report['exact_ids']


def test_schema_has_fixed_inventory_but_no_claim_of_nonoverlap(source):
    schema=proposal_schema(source)['properties']['elements']
    assert schema['minItems']==schema['maxItems']==2
    assert [i['properties']['id']['const'] for i in schema['prefixItems']]==['e000','e001']


def test_flaw_report_counts_instances_instead_of_warning_categories():
    from scripts.build_upgrade_pilot_review import flaws
    report=flaws(dict(image_ratio_violations=['a','b'],visual_contracts=[dict(kind='primary_axis'),dict(kind='control_label_axis')],
        unintended_text_overlaps=[{},{}],control_collisions=[{}],text_hero_occlusion=[{}],
        structural_errors=["element a is off-canvas (320x50)","element id mismatch"],glyph_overflow={'a':3,'b':4}))
    assert report==dict(stretch=2,alignment=2,overlap=4,out_of_bounds=1,text_fit=2)


def test_composite_model_and_tokenizer_eos_are_both_termination_tokens():
    from types import SimpleNamespace as N
    from scripts.run_upgrade_model_pilot import generation_stop_ids
    model=N(generation_config=N(eos_token_id=42),config=N(text_config=N(eos_token_id=[42,43])))
    assert generation_stop_ids(N(eos_token_id=41),model)==[41,42,43]


def test_duplicate_aspect_reports_count_one_foreground_asset_and_exclude_background(source):
    from scripts.build_upgrade_pilot_review import flaws
    source['elements'].append(dict(id='bg',type='image',x=0,y=0,width=400,height=400,z_index=-1))
    report=dict(image_ratio_violations=['visual','bg'],visual_contracts=[
        dict(kind='art_aspect_distortion',ids=['visual']),dict(kind='art_aspect_distortion',ids=['bg'])])
    assert flaws(report,source)['stretch']==1


def test_optional_solver_keeps_atomic_shapes_in_bounds_without_stretch():
    deps=Path(__file__).resolve().parents[1]/'data/upgrade_exploration/deps'
    if deps.exists():sys.path.insert(0,str(deps))
    pytest.importorskip('ortools.sat.python.cp_model')
    from experiments.layout_upgrade.packing import candidates
    source=dict(canvas=dict(width=400,height=400),elements=[dict(id=f'a{i}',type='shape',shape_type='ellipse',
        x=x,y=100,width=40,height=40,color='#ee8833',z_index=i) for i,x in enumerate([20,150,280])])
    result=candidates(source,320,120,seconds=1.)
    assert result
    es=result[0]['elements'];assert {e['id'] for e in es}=={'a0','a1','a2'}
    for e in es:
        assert e['width']==e['height']
        assert 0<=e['x']<=320-e['width'] and 0<=e['y']<=120-e['height']
    for i,a in enumerate(es):
        for b in es[i+1:]:
            assert a['x']+a['width']<=b['x'] or b['x']+b['width']<=a['x'] or a['y']+a['height']<=b['y'] or b['y']+b['height']<=a['y']
