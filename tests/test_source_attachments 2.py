from copy import deepcopy
from engine.source_structure import groups,seed,relationship_cost


def text(eid,x,y,w,h,content,size=16,**kwargs):
    return dict(id=eid,type='text',x=x,y=y,width=w,height=h,content=content,font_size=size,font_family='Solstice Sans',color='#111111',z_index=3,**kwargs)


def test_translucent_parent_and_its_text_are_one_attachment_group():
    source=dict(canvas=dict(width=400,height=400),elements=[
        dict(id='card',type='shape',shape_type='rectangle',x=60,y=80,width=280,height=200,opacity=.3,color='#eeeeee',z_index=1),
        text('title',100,110,180,30,'A headline',24),text('body',100,160,170,40,'Supporting copy')])
    assert set(groups(source)['members'][0])=={'card','title','body'}
    result=seed(source,[],800,500)
    assert {e['id'] for e in result}=={'card','title','body'}
    assert relationship_cost(result,source,800,500)<.01
    broken=deepcopy(result);next(e for e in broken if e['id']=='body')['x']+=200
    assert relationship_cost(broken,source,800,500)>relationship_cost(result,source,800,500)+.02


def test_clipped_side_art_stays_at_the_side_without_invalid_frames():
    source=dict(canvas=dict(width=600,height=200),elements=[
        dict(id='left',type='image',asset_id='missing_edge.png',x=-80,y=0,width=180,height=200,z_index=1),
        dict(id='right',type='image',asset_id='missing_edge.png',x=500,y=0,width=180,height=200,z_index=1),
        text('title',200,80,200,40,'Workshop',24)])
    result={e['id']:e for e in seed(source,[],1600,400)}
    assert result['left']['x']==0
    assert abs(result['right']['x']+result['right']['width']-1600)<1e-6
    assert all(e['x']>=0 and e['y']>=0 and e['x']+e['width']<=1600+1e-6 and e['y']+e['height']<=400+1e-6 for e in result.values())


def test_raw_model_outliers_do_not_control_source_group_membership():
    source=dict(canvas=dict(width=600,height=600),elements=[
        text('headline',180,150,240,40,'Title',30),
        dict(id='rule',type='image',asset_id='missing_rule.png',x=255,y=163,width=70,height=6,z_index=1),
        text('separate',20,480,120,40,'Footer')])
    spec=groups(source)
    assert any(set(g)=={'headline','rule'} for g in spec['members'])
    assert ('separate',) in spec['members']


def test_editorial_button_follows_the_copy_axis_but_inline_column_does_not():
    from engine.source_structure import coherent_copy_axes
    p=dict(id='button',type='shape',shape_type='rectangle',x=150,y=300,width=180,height=50,z_index=1,group_id='action')
    t=text('label',150,310,180,30,'Buy now',18,group_id='action',text_align='center')
    body=text('body',80,240,330,30,'Supporting text',18,text_align='left')
    source=dict(canvas=dict(width=500,height=500),elements=[p,t,body])
    units=[dict(items=[body],semantic='body'),dict(items=[p,t],semantic='cta',role='action')]
    items=deepcopy(source['elements']);items[0]['x']=200;items[1]['x']=200
    coherent_copy_axes(items,source,units,500,500)
    from engine.resize import _text_visible_bounds
    assert abs(items[0]['x']-_text_visible_bounds(items[2])[0])<1.
    items=deepcopy(source['elements']);items[0]['y']=230;items[1]['y']=240
    original=deepcopy(items)
    coherent_copy_axes(items,source,units,500,500)
    assert items==original


def test_final_rotated_bounds_are_legalized_as_a_group():
    from engine.source_structure import fit_groups_to_canvas
    from engine.resize import _rotated_bounds
    panel=dict(id='ribbon',type='shape',shape_type='rectangle',x=100,y=20,width=180,height=35,rotation=-12,group_id='r',z_index=1)
    label=text('label',100,23,180,28,'A ribbon',20,rotation=-12,group_id='r')
    source=dict(canvas=dict(width=400,height=400),elements=[panel,label])
    items=deepcopy(source['elements'])
    for e in items:e['y']-=24
    gap=items[1]['y']-items[0]['y']
    fit_groups_to_canvas(items,source,400,400)
    assert items[1]['y']-items[0]['y']==gap
    assert all(_rotated_bounds(e)[1]>=-1e-6 for e in items)


def test_source_parent_contains_copy_without_an_explicit_group_id():
    from engine.source_structure import containment_issues
    panel=dict(id='panel',type='shape',shape_type='rectangle',x=60,y=80,width=280,height=200,z_index=1)
    label=text('label',100,140,180,40,'Contained title',24)
    source=dict(canvas=dict(width=400,height=400),elements=[panel,label])
    escaped=[deepcopy(panel),dict(label,x=330)]
    assert containment_issues(escaped,source)[0]['kind']=='source_parent_containment'


def test_badge_row_is_not_moved_onto_the_cta_copy_axis():
    from engine.source_structure import coherent_copy_axes
    p=dict(id='badge',type='shape',shape_type='rectangle',x=150,y=300,width=180,height=50,z_index=1,group_id='tag')
    label=text('label',150,310,180,30,'Small batch',18,group_id='tag',text_align='center')
    body=text('body',80,240,330,30,'Supporting text',18,text_align='left')
    source=dict(canvas=dict(width=500,height=500),elements=[p,label,body])
    units=[dict(items=[body],semantic='body'),dict(items=[p,label],semantic='badge',role='tag')]
    items=deepcopy(source['elements']);before=deepcopy(items)
    coherent_copy_axes(items,source,units,500,500)
    assert items==before


def test_cta_below_headline_keeps_inline_slot_beside_supporting_paragraph():
    from engine.source_structure import coherent_copy_axes
    p=dict(id='button',type='shape',shape_type='rectangle',x=330,y=130,width=140,height=45,z_index=1,group_id='a')
    label=text('label',330,140,140,25,'Buy now',18,group_id='a',text_align='center')
    head=text('head',50,60,400,40,'Headline',24,text_align='left')
    body=text('body',50,135,260,35,'Supporting paragraph',18,text_align='left')
    source=dict(canvas=dict(width=500,height=500),elements=[p,label,head,body])
    units=[dict(items=[p,label],semantic='cta',role='action'),dict(items=[head],semantic='headline'),dict(items=[body],semantic='body')]
    items=deepcopy(source['elements']);before=deepcopy(items)
    coherent_copy_axes(items,source,units,500,500)
    assert items==before


def test_contrast_move_uses_one_functional_group_without_dragging_other_copy():
    from PIL import Image
    from unittest.mock import patch
    from engine.qwen_refine import whole_lockup_contrast_variants
    from engine.resize import _analyze
    panel=dict(id='panel',type='shape',shape_type='rectangle',x=85,y=400,width=330,height=160,z_index=1,color='#eeeeee')
    header=text('header',150,70,200,40,'Independent heading',24)
    body=text('body',110,420,240,35,'Body copy',20)
    footer=text('footer',110,480,240,30,'Supporting information',16)
    source=dict(canvas=dict(width=500,height=700),elements=[panel,header,body,footer])
    with patch('engine.resize._physical_layout_safe',return_value=True):
        variants=whole_lockup_contrast_variants(source['elements'],_analyze(source)[1],source,500,700,['body'],Image.new('RGB',(500,700),'white'))
    assert variants
    for elements,notes in variants:
        changed=notes[0]['element_ids'];assert set(changed)=={'panel','body','footer'}
        assert next(e for e in elements if e['id']=='header')==header
        offsets={(round(e['x']-next(s['x'] for s in source['elements'] if s['id']==e['id']),6),round(e['y']-next(s['y'] for s in source['elements'] if s['id']==e['id']),6)) for e in elements if e['id'] in changed}
        assert len(offsets)==1


def test_single_compact_lockup_uses_the_available_short_side():
    source=dict(canvas=dict(width=400,height=800),elements=[
        dict(id='card',type='shape',shape_type='rectangle',x=100,y=300,width=200,height=200,z_index=1),
        text('head',120,330,160,35,'Headline',24),text('body',120,390,160,40,'Body text',16)])
    es=seed(source,[],1200,400)
    assert max(e['y']+e['height'] for e in es)-min(e['y'] for e in es)>.75*400


def test_vertical_caption_calibration_keeps_its_painted_anchor_and_wrapping():
    from engine.resize import _calibrate_text_boxes,_text_visible_bounds,_text_info
    t=text('vertical',200,180,95,40,'SUNDAY, SEP 8 5 PM',12,rotation=270,text_align='right')
    source=dict(canvas=dict(width=500,height=500),elements=[t])
    before=_text_visible_bounds(t);lines=_text_info(t)[2]
    result,notes=_calibrate_text_boxes([t],source,500,500)
    after=_text_visible_bounds(result[0])
    assert abs(before[0]+before[2]-after[0]-after[2])<1e-6
    assert abs(before[1]+before[3]-after[1]-after[3])<1e-6
    assert _text_info(result[0])[2]==lines
    assert result[0]['font_size']==t['font_size'] and result[0]['rotation']==270
