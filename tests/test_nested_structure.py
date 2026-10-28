from copy import deepcopy
from engine.nested_structure import regional_surfaces,repeated_text_rows
from engine.macro_layout import composites,_fit_regional_surfaces,bounds
from engine.container_flow import measured_block
from engine.resize import _render_text_ink


def text(i,x,y,w=120,h=30,**kw):
 return dict(id=i,type='text',x=x,y=y,width=w,height=h,font_size=18,font_family='Solstice Sans',content='An ordinary label',z_index=3,**kw)


def test_regional_paint_does_not_weld_independent_nested_cards():
 es=[dict(id='wash',type='shape',x=0,y=50,width=600,height=400,z_index=0,opacity=.6)]
 for i,x in enumerate((20,210,400)):
  es += [dict(id=f'p{i}',type='shape',x=x,y=100,width=170,height=200,z_index=1,group_id=str(i)),text(f't{i}',x+10,130,140,group_id=str(i))]
 source=dict(canvas=dict(width=600,height=600),elements=es)
 assert set(regional_surfaces(source))=={'wash'}
 blocks=composites(source)['blocks']
 assert sorted(map(len,blocks))==[2,2,2]
 items=deepcopy(es)
 for e in items[1:]:e['y']+=100
 _fit_regional_surfaces(items,source,600,600)
 assert items[0]['y']<=200 and items[0]['y']+items[0]['height']>=400
 assert {e['id'] for e in items}=={e['id'] for e in es}


def test_one_card_can_own_nested_action_without_becoming_scope_paint():
 es=[dict(id='panel',type='shape',x=20,y=20,width=500,height=400,z_index=0),
     text('title',50,50,300),dict(id='button',type='shape',x=50,y=200,width=180,height=50,z_index=1,group_id='action'),
     text('label',60,210,160,group_id='action')]
 source=dict(canvas=dict(width=600,height=600),elements=es)
 assert not regional_surfaces(source)
 assert len(composites(source)['blocks'])==1


def test_three_repeated_two_field_rows_gain_ownership_but_not_other_columns():
 es=[]
 for i in range(4):es += [text(f'n{i}',20,100+70*i,350),text(f'p{i}',400,100+70*i,80)]
 es += [text('headline',20,20,350,h=60),text('aside',400,30,100,h=60)]
 source=dict(canvas=dict(width=600,height=600),elements=es)
 assert set(repeated_text_rows(es,source['canvas']))=={(f'n{i}',f'p{i}') for i in range(4)}
 blocks=composites(source)['blocks']
 row=next(es for es in blocks if any(e['id']=='n0' for e in es))
 fitted=measured_block(row,155,120,source,.6,(160,600))
 assert fitted and len(fitted)==2
 a,b=fitted
 assert a['x']+a['width']<b['x'] and b['x']+b['width']<=155
 assert all(_render_text_ink(e)[3]<=e['height'] for e in fitted)


def test_control_allocation_is_a_limit_not_a_growth_target():
 p=dict(id='button',type='shape',shape_type='rounded_rectangle',x=20,y=20,width=180,height=50,z_index=1)
 t=text('label',35,30,150);source=dict(canvas=dict(width=600,height=600),elements=[p,t])
 out=measured_block([p,t],500,100,source,1.,(600,600))
 assert out and out[0]['width']<300 and out[1]['x']>0


def test_unpainted_source_layers_do_not_hide_visible_glyph_overflow():
 from engine.raster_metrics import glyph_overflow
 from engine.nested_structure import unpainted_ids
 a=text('visible',10,10,100,1);b=text('hidden',10,10,100,1,opacity=0)
 c=text('zero',10,20,0,20);d=text('thin',10,20,1,20)
 source=dict(canvas=dict(width=600,height=600),elements=[a,b,c,d])
 assert unpainted_ids(source)=={'hidden','zero'}
 issues=glyph_overflow(source,[a,b,dict(c,width=1),d],600,600)
 assert 'visible' in issues and 'thin' in issues and 'hidden' not in issues and 'zero' not in issues


def test_measured_card_grid_does_not_reward_thumbnail_columns():
 es=[]
 for i in range(6):
  x=10+i*180;g=str(i)
  es += [dict(id=f'panel{i}',type='shape',shape_type='rectangle',x=x,y=100,width=170,height=250,z_index=1,group_id=g),
         dict(id=f'photo{i}',type='image',x=x+5,y=105,width=160,height=140,z_index=2,group_id=g),
         text(f'name{i}',x+5,250,160,40,group_id=g)]
 source=dict(canvas=dict(width=1100,height=700),elements=es)
 out=measured_block(es,150,580,source,.35,(160,600))
 assert out
 photos=[e for e in out if e['type']=='image']
 assert min(e['width'] for e in photos)>40
 assert bounds(out)[2]<=150 and bounds(out)[3]<=580
 assert len({e['id'] for e in out})==len(es)


def test_hierarchy_cap_does_not_narrow_a_photo_caption_backplate():
 from engine.macro_layout import _cap_control_growth
 photo=dict(id='photo',type='image',x=20,y=100,width=300,height=200,z_index=1)
 panel=dict(id='caption',type='shape',x=20,y=300,width=300,height=40,z_index=2,group_id='caption')
 caption=text('caption_label',30,310,280,25,group_id='caption')
 heading=text('heading',20,30,300,40);heading['font_size']=30
 source=dict(canvas=dict(width=600,height=600),elements=[photo,panel,caption,heading])
 items=deepcopy(source['elements']);items[2]['font_size']=40;items[3]['font_size']=20
 before=deepcopy(items[:3]);_cap_control_growth(items,source,600,600)
 assert items[:3]==before


def test_multicolumn_table_is_not_split_into_overlapping_two_field_rows():
 es=[text(f'{row}_{col}',20+col*150,100+row*60,100) for row in range(4) for col in range(4)]
 assert not repeated_text_rows(es,dict(width=700,height=500))


def test_rotated_box_outside_raw_viewport_can_still_paint_inside():
 from engine.nested_structure import unpainted_ids
 e=dict(id='rotated',type='shape',x=610,y=50,width=100,height=400,rotation=45)
 assert not unpainted_ids(dict(canvas=dict(width=600,height=600),elements=[e]))
