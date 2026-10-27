from copy import deepcopy
from pathlib import Path
from PIL import Image,ImageDraw
from engine.macro_layout import composites,clamp_edge_strips,subpanels,quality,upgrade
from engine.layout_contracts import controls
from engine.resize import _analyze,_unintended_control_overlaps


def text(i,x,y,w,h,content,size=18,**kw):
 return dict(id=i,type='text',x=x,y=y,width=w,height=h,font_size=size,font_family='Solstice Sans',content=content,z_index=4,**kw)


def test_multiline_body_prevents_card_from_becoming_single_heading_control():
 card=dict(id='card',type='shape',x=20,y=20,width=350,height=150,group_id='g',z_index=1)
 source=dict(canvas=dict(width=500,height=500),elements=[card,text('title',40,40,300,30,'A heading',group_id='g'),text('copy',40,90,300,60,'Detailed supporting information. '*5,group_id='g')])
 assert not controls(source)
 assert any({e['id'] for e in block}=={'card','title','copy'} for block in composites(source)['blocks'])


def test_photo_caption_and_corner_badge_are_one_atomic_media_block():
 photo=dict(id='p',type='image',x=50,y=50,width=300,height=200,z_index=1)
 caption=dict(id='c',type='shape',x=50,y=250,width=300,height=40,z_index=2,group_id='cap')
 badge=dict(id='b',type='shape',x=230,y=60,width=100,height=30,z_index=2,group_id='tag')
 source=dict(canvas=dict(width=500,height=500),elements=[photo,caption,badge,text('ct',60,257,280,28,'Photo caption',group_id='cap'),text('bt',237,65,85,20,'New',group_id='tag')])
 assert len(composites(source)['blocks'])==1
 assert len(composites(source)['blocks'][0])==5


def test_source_nested_control_label_is_not_a_new_collision():
 panel=dict(id='card',type='shape',x=20,y=200,width=400,height=100,z_index=1,group_id='c')
 button=dict(id='button',type='shape',x=260,y=225,width=130,height=50,z_index=2,group_id='a')
 label=text('label',270,237,110,25,'Book now',group_id='a')
 source=dict(canvas=dict(width=600,height=600),elements=[panel,button,label])
 assert not _unintended_control_overlaps(source['elements'],source)
 # Another source-disjoint heading entering the card is still a collision.
 heading=text('heading',20,20,220,30,'Important heading')
 source['elements'].append(heading);changed=deepcopy(source['elements']);changed[-1]['y']=240
 assert _unintended_control_overlaps(changed,source)


def test_edge_strip_thickness_is_not_free_background_geometry():
 strip=dict(id='accent',type='shape',x=0,y=0,width=12,height=600,z_index=0)
 source=dict(canvas=dict(width=300,height=600),elements=[strip]);items=[dict(strip,width=1000,height=1000)]
 clamp_edge_strips(items,source,1000,250)
 assert items[0]['width']==40 and items[0]['height']==250 and items[0]['x']==0


def test_occupancy_excludes_full_canvas_and_edge_bands():
 bg=dict(id='bg',type='shape',x=0,y=0,width=500,height=500,z_index=0)
 edge=dict(id='edge',type='shape',x=0,y=0,width=10,height=500,z_index=1)
 title=text('title',200,200,100,40,'Main title',size=24)
 source=dict(canvas=dict(width=500,height=500),elements=[bg,edge,title])
 q=quality(source['elements'],source,500,500)
 assert q['width_use']<.4 and q['height_use']<.3


def test_transparent_frame_cells_restore_local_text_axes(tmp_path):
 path=tmp_path/'frame.png';im=Image.new('RGBA',(400,120));d=ImageDraw.Draw(im)
 d.rectangle((0,0,399,119),outline='white');d.line((120,0,120,119),fill='white');d.line((120,90,399,90),fill='white');im.save(path)
 panel=dict(id='grid',type='image',asset_id=str(path),x=0,y=0,width=400,height=120,z_index=1)
 label=text('copy',145,93,240,20,'Live fully',size=14)
 source=dict(canvas=dict(width=500,height=300),elements=[panel,label]);changed=[dict(panel,width=800,height=240),dict(label,x=200,y=150,width=480,height=40,font_size=28)]
 before=subpanels(changed,source);assert before>.1
 subpanels(changed,source,True)
 assert subpanels(changed,source)<.001


def test_no_upgrade_keeps_clean_incumbent_byte_identical():
 source=dict(canvas=dict(width=400,height=400),elements=[text('a',120,100,160,35,'Heading'),text('b',120,200,160,35,'Body')])
 units=_analyze(source)[1];current=dict(name='clean-existing',elements=deepcopy(source['elements']))
 frozen=deepcopy(current)
 def forbidden(*args):raise AssertionError('No render sweep needed for clean familiar geometry')
 winner,_,diag=upgrade(current,{},[],units,source,400,400,1.,forbidden)
 assert winner==frozen and current==frozen and not diag['considered']


def test_zero_size_authored_rule_does_not_crash_quality_diagnostics():
 source=dict(canvas=dict(width=500,height=500),elements=[text('a',100,100,200,35,'Heading'),dict(id='zero',type='shape',shape_type='line',x=100,y=140,width=0,height=0)])
 result=deepcopy(source['elements']);result[1].update(width=1,height=1)
 assert quality(result,source,500,500)['cost']>=0


def test_existing_compact_source_lockup_is_atomic_even_without_a_backplate():
 from unittest.mock import patch
 source=dict(canvas=dict(width=500,height=500),elements=[text('price',180,150,90,50,'$49',size=40),text('suffix',275,150,25,25,'99',size=12),text('caption',160,230,180,35,'Drinkware')])
 with patch('engine.source_structure.groups',return_value={'members':[('price','suffix','caption')]}):
  assert len(composites(source)['blocks'])==1


def test_source_supported_intentional_contact_does_not_gain_a_new_gap():
 from engine.macro_layout import clearance
 art=dict(id='art',type='image',x=50,y=100,width=100,height=100,z_index=1)
 label=text('caption',150,100,150,30,'Intentional contact')
 source=dict(canvas=dict(width=400,height=400),elements=[art,label]);items=deepcopy(source['elements'])
 assert clearance(items,source)==0
 clearance(items,source,True)
 assert items==source['elements']
