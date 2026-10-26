"""Source-evidenced invariants across crops, orientations, and paint order."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
from PIL import Image
from engine.clean_zones import project_rectangle,source_zones
from engine.resize import _analyze,_physical_layout_safe
from engine.layout_contracts import art_occlusions,_separate_art,image_collisions


def text(x=20,y=50):
    return dict(id='copy',type='text',x=x,y=y,width=140,height=45,font_size=28,font_family='Solstice Sans',content='SALE',text_align='left',z_index=2,color='#ffffff')


def test_cover_projection_uses_intrinsic_pixels_and_source_offsets():
    old=dict(x=-200,y=-200,width=1600,height=1000,fit='contain')
    new=dict(x=0,y=0,width=900,height=900,fit='cover')
    assert np.allclose(project_rectangle((0,0,300,600),old,new,(1600,1000)),(-90,180,180,720))


def test_clean_zone_does_not_enclose_subject_on_other_side_of_seam():
    im=Image.new('RGB',(320,180),(180,25,80));im.paste('white',(180,0,320,180))
    t=text();bg=dict(id='bg',type='image',x=0,y=0,width=320,height=180,z_index=0,asset_id='photo.png',fit='cover')
    source=dict(canvas=dict(width=320,height=180),elements=[bg,t]);units=[dict(role='copy',items=[t])]
    zones=source_zones(source,im,{'copy':{'p05':4}},units)
    assert zones and zones[0]['box'][2]<=181


def test_foreground_cutout_occlusion_is_repaired_without_moving_text():
    with TemporaryDirectory() as folder:
        path=Path(folder)/'cutout.png';Image.new('RGBA',(80,80),'red').save(path)
        art=dict(id='art',type='image',x=350,y=200,width=80,height=80,z_index=5,asset_id=str(path))
        source=dict(canvas=dict(width=500,height=350),elements=[text(),art]);items=deepcopy(source['elements']);items[1].update(x=20,y=50)
        assert art_occlusions(items,source)
        _separate_art(items,source,500,350)
        assert not art_occlusions(items,source)
        assert items[0]==source['elements'][0]


def test_reverse_fixtures_have_orientation_independent_roles():
    for path in (Path(__file__).parent/'fixtures/level3_reverse').glob('*.json'):
        source=json.loads(path.read_text())['source'];units=_analyze(source)[1];roles={e['id']:u for u in units for e in u['items']}
        assert roles['logo']['semantic']=='logo'
        assert roles['holiday_watermark']['role']=='footer'
        assert roles['legal_text']['role']=='footer'
        assert roles['ribbon_bg']['corner_callout'] and roles['ribbon_bg']['role']=='decoration'


def test_disjoint_foreground_images_are_a_hard_layout_defect():
    a=dict(id='a',type='image',x=20,y=20,width=70,height=70,z_index=1)
    b=dict(a,id='b',x=200,y=180)
    source=dict(canvas=dict(width=400,height=400),elements=[a,b]);items=[a,dict(b,x=30,y=30)]
    assert image_collisions(items,source)
    assert not _physical_layout_safe(items,_analyze(source)[1],source,400,400)


def test_mild_authored_contrast_dip_has_tolerance_but_not_invisibility():
    from engine.resize import _contrast_floor_deficit
    assert _contrast_floor_deficit(2.8,2.45)==0
    assert _contrast_floor_deficit(2.8,1.7)>0
    assert _contrast_floor_deficit(4.5,2.0)>0


def test_identity_source_rigid_legalization_is_reported_as_inherited():
    from scripts.layout_defects import assess_engine_layout
    panel=dict(id='panel',type='shape',shape_type='rectangle',x=80,y=30,width=220,height=70,z_index=1,group_id='control')
    t=text(100,35);t.update(group_id='control',font_size=20,width=170,height=30)
    source=dict(canvas=dict(width=400,height=300),elements=[panel,t]);shifted=deepcopy(source)
    for e in shifted['elements']:e['x']+=3;e['y']+=4
    result=assess_engine_layout(source,shifted,{})
    assert result['source_authored_issues']
    assert 'control_label_axis' not in result['defect_flags']
    shifted['elements'][1]['y']+=25
    changed=assess_engine_layout(source,shifted,{})
    assert 'control_label_axis' in changed['defect_flags']


def test_output_budget_reserves_all_ids_and_respects_remaining_context():
    from engine.qwen_seed import _output_budget
    class Tokenizer:
        def __call__(self,value,**kwargs):return {'input_ids':list(range(len(value)//4))}
    source={'elements':[dict(id=str(i),type='text') for i in range(17)]}
    assert _output_budget(Tokenizer(),source,2,3000)>=17*96+256
    assert _output_budget(Tokenizer(),source,2,8000)<=192


def test_early_stop_requires_every_expected_alias_to_be_closed():
    from engine.qwen_seed import QwenEarlyStoppingCriteria
    expected={f'e{i:03d}' for i in range(17)}
    class Tokenizer:
        value=''
        def decode(self,*args,**kwargs):return self.value
    tok=Tokenizer();criterion=QwenEarlyStoppingCriteria(tok,0,expected);ids=np.array([[0]*4])
    tok.value=json.dumps({'elements':[{'id':f'e{i:03d}','box':[1,2,3,4]} for i in range(16)]})
    criterion._counter=15
    assert not criterion(ids)
    tok.value=json.dumps({'elements':[{'id':f'e{i:03d}','box':[1,2,3,4]} for i in range(17)]})
    criterion._counter=15
    assert criterion(ids)


def test_final_font_fit_restores_painted_axis_for_unbacked_copy():
    from engine.resize import _finish_candidate_text,_render_text_ink,_text_visible_bounds
    originals=[]
    for i,y in enumerate((80,240)):
        e=text(0,y);e.update(id=str(i),width=260,height=90,text_align='center',content='TITLE    ',font_size=40)
        ink=_render_text_ink(e);e['x']=250-(ink[0]+ink[2])/2;originals.append(e)
    source=dict(canvas=dict(width=500,height=600),elements=originals)
    units=[dict(role='copy',semantic='headline' if i==0 else 'body',items=[e],bounds=_text_visible_bounds(e)) for i,e in enumerate(originals)]
    target=deepcopy(originals)
    for e in target:e.update(x=120,font_size=23,width=260)
    result,_=_finish_candidate_text(target,source,units,500,600)
    for e in result:
        b=_text_visible_bounds(e);assert abs((b[0]+b[2])/2-250)<.01
