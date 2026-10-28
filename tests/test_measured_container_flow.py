"""Measured containers must preserve inventory, readable ink and attachments."""
from copy import deepcopy
from math import isclose
from engine.container_flow import measured_block, measured_columns
from engine.macro_layout import bounds, measured_candidates
from engine.layout_contracts import controls, issues
from engine.resize import _analyze, _unintended_text_overlaps


def label(i,x,y,w,h,content,size=18,**kw):
    return dict(id=i,type='text',x=x,y=y,width=w,height=h,font_size=size,
        font_family='Solstice Sans',content=content,z_index=3,**kw)


def test_photo_caption_changes_orientation_without_distortion_or_new_overlap():
    photo=dict(id='photo',type='image',x=30,y=100,width=600,height=300,z_index=1)
    panel=dict(id='caption',type='shape',shape_type='rectangle',x=30,y=400,width=600,height=50,z_index=2,group_id='c')
    text=label('text',45,415,570,25,'A caption that needs room to remain readable',size=18,group_id='c')
    source=dict(canvas=dict(width=700,height=1000),elements=[photo,panel,text])
    result=measured_block(source['elements'],190,65,source,.4,(728,90))
    assert result and {e['id'] for e in result}=={'photo','caption','text'}
    by={e['id']:e for e in result};p,c=by['photo'],by['caption']
    assert isclose(p['width']/p['height'],2.)
    assert c['x']>=p['x']+p['width'] or c['y']>=p['y']+p['height']
    assert by['text']['font_size']>=7.2-1e-6
    assert bounds(result)[2]<=190 and bounds(result)[3]<=65
    placed=deepcopy(result)
    for e in placed:e['x']+=5;e['y']+=5
    assert not issues(placed,source,728,90,_analyze(source)[1])
    assert source['elements'][0]==photo  # Caller input is never mutated.


def test_wrapped_copy_uses_measured_height_and_keeps_tracking_ratio():
    t=label('a',20,20,550,200,'A reasonably long heading which can wrap into a narrow column',size=30,letter_spacing=2.)
    source=dict(canvas=dict(width=600,height=900),elements=[t])
    result=measured_block([t],180,100,source,.5,(728,90))
    assert result and result[0]['height']<100
    assert result[0]['font_size']==15 and result[0]['letter_spacing']==1
    assert result[0]['content']==t['content']
    assert measured_block([t],180,10,source,.5,(728,90)) is None


def test_peer_controls_remain_whole_when_row_wraps():
    elements=[]
    for i,x in enumerate((20,300,580)):
        elements += [dict(id=f'p{i}',type='shape',shape_type='rounded_rectangle',x=x,y=100,width=260,height=45,z_index=1,group_id=str(i)),
            label(f't{i}',x+12,112,236,22,'Choose your appointment',size=16,group_id=str(i))]
    source=dict(canvas=dict(width=900,height=900),elements=elements)
    result=measured_block(elements,110,80,source,.4,(728,90))
    assert result and len(result)==6
    placed=deepcopy(result)
    for e in placed:e['x']+=5;e['y']+=5
    assert not issues(placed,source,728,90,_analyze(source)[1])
    by={e['id']:e for e in result}
    assert len({round(by[f'p{i}']['y'],2) for i in range(3)})>1
    assert all(by[f't{i}']['font_size']>=6.4 for i in range(3))


def test_partition_reflows_a_stack_and_preserves_ids_and_ink_separation():
    blocks=[[label(str(i),20,50+i*130,500,45,'Copy with enough information to wrap',size=24)] for i in range(6)]
    source=dict(canvas=dict(width=600,height=1000),elements=sum(blocks,[]))
    frozen=deepcopy(source)
    result=measured_columns(blocks,(10,5,700,80),source,.4,(728,90),3)
    assert result and {e['id'] for e in result}=={str(i) for i in range(6)}
    assert not _unintended_text_overlaps(result,_analyze(source)[1])
    assert bounds(result)[2]<=710 and bounds(result)[3]<=85
    assert all(isclose(e['font_size'],9.6) for e in result)
    assert source==frozen


def test_model_drives_macro_membership_without_its_leaf_font_or_sizes():
    blocks=[[label(str(i),20,80+i*150,500,40,'Group heading',size=24)] for i in range(4)]
    source=dict(canvas=dict(width=600,height=900),elements=sum(blocks,[]))
    # Two clear model columns, reversed relative to source order.
    proposal=[dict(id=str(i),x=500 if i<2 else 20,y=(i%2)*30,width=30,height=10,font_size=1000) for i in range(4)]
    result=[c for c in measured_candidates(source,blocks,(10,5,700,80),728,90,proposal)
            if c['name'].startswith('qwen-') and c['macro_realization']['columns']==2]
    assert result
    c=result[0];by={e['id']:e for e in c['elements']}
    assert by['2']['x']<by['0']['x'] and by['3']['x']<by['1']['x']
    assert all(e['font_size']<30 for e in by.values())
    assert c['learned_macro'] and c['macro_locked']
    assert not [c for c in measured_candidates(source,blocks,(10,5,700,80),728,90,proposal[:1]) if c['name'].startswith('qwen-')]


def test_tall_model_bands_keep_macro_order_as_well_as_inventory():
    blocks=[[label(str(i),20+i*180,20,160,40,'A heading',size=24)] for i in range(4)]
    source=dict(canvas=dict(width=800,height=200),elements=sum(blocks,[]))
    proposal=[dict(id=str(i),x=20,y=800 if i<2 else 20,width=100,height=40) for i in range(4)]
    result=[c for c in measured_candidates(source,blocks,(20,20,360,960),400,1000,proposal)
            if c['name'].startswith('qwen-macro-band-flow-rows-2')]
    assert result
    by={e['id']:e for e in result[0]['elements']}
    assert by['2']['y']<by['0']['y']
    assert set(by)=={'0','1','2','3'}


def test_readability_fallback_cannot_bypass_hard_defects():
    from unittest.mock import patch
    from engine.macro_layout import upgrade
    source=dict(canvas=dict(width=500,height=500),elements=[label(str(i),20,30+i*25,100,20,'Text') for i in range(14)])
    old=dict(name='old',elements=deepcopy(source['elements']))
    new=dict(name='macro-test',elements=deepcopy(source['elements']),macro_locked=True)
    def rank(cs,*args):return cs[0],{'used':True}
    for flags in (['control_readability'],['text_on_text'],['glyph_geometry'],['contrast_below_source_floor'],['structural_check'],['control_collision']):
        def check(src,doc,units,visual,name):
            return dict(flagged=True,defect_flags=['control_readability'] if name=='old' else flags)
        with patch('engine.macro_layout.local_variants',return_value=[]),patch('engine.macro_layout.candidates',return_value=[new]),patch('engine.macro_layout.checks',side_effect=check),patch('engine.macro_layout.quality',side_effect=[dict(cost=50,width_use=.3,height_use=.8),dict(cost=10,width_use=.9,height_use=.8)]):
            winner,_,report=upgrade(old,{},[],[],source,728,90,1.,rank)
        assert (winner['name']=='macro-test') == (flags==['control_readability'])
        if flags==['control_readability']:
            assert report['remaining_flags']==['control_readability']


def test_unattached_decorations_use_free_space_without_a_copy_column():
    blocks=[[label(str(i),100,80+i*80,400,40,'Heading',size=24)] for i in range(4)]
    blocks.append([dict(id='loose',type='image',x=10,y=100,width=25,height=25,z_index=1)])
    source=dict(canvas=dict(width=600,height=900),elements=sum(blocks,[]))
    candidates=measured_candidates(source,blocks,(10,5,700,80),728,90)
    assert candidates
    from engine.resize import _overlap
    for c in candidates:
        art=next(e for e in c['elements'] if e['id']=='loose')
        assert all(_overlap(bounds([art]),bounds([e]))==0 for e in c['elements'] if e['type']=='text')
        assert all('loose' not in ids for ids in c['macro_realization']['order'])


def test_atomic_tracked_lettering_keeps_source_line_count_after_scaling():
    from engine.resize import _text_info
    t=label('letters',50,50,450,80,'S P A C E D  T I T L E',size=30,letter_spacing=2.)
    art=dict(id='mark',type='image',x=20,y=50,width=20,height=20,z_index=1)
    source=dict(canvas=dict(width=600,height=500),elements=[t,art])
    result=measured_block(source['elements'],150,45,source,.4,(728,90))
    assert result
    target=next(e for e in result if e['id']=='letters')
    assert _text_info(target)[2]<=_text_info(t)[2]
    assert isclose(target['letter_spacing']/target['font_size'],2/30)


def test_readability_warning_does_not_hide_simultaneous_glyph_overflow():
    from unittest.mock import patch
    from engine.macro_layout import checks
    t=label('t',10,10,100,3,'Tall glyphs',size=20)
    source=dict(canvas=dict(width=500,height=500),elements=[deepcopy(t)])
    with patch('engine.layout_contracts.issues',return_value=[dict(kind='control_readability',ids=['p','t'],amount=2.)]):
        report=checks(source,source,[],{},'probe')
    assert 'control_readability' in report['defect_flags']
    assert 'glyph_overflow' in report['defect_flags']
