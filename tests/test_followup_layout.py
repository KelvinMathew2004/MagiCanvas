from copy import deepcopy
from engine.container_flow import ordered_peer_rows,measured_block
from engine.nested_structure import control_captions
from engine.qwen_refine import snap_source_axes


def label(i,x,y,w,h=25,size=18,**kw):
    return dict(id=i,type='text',x=x,y=y,width=w,height=h,font_size=size,
                font_family='Solstice Sans',content='Label',z_index=2,**kw)


def test_ordered_breaks_keep_unequal_intrinsic_widths_and_order():
    widths=[90,35,75,40,85,30,55]
    breaks=ordered_peer_rows(widths,220,6,3)
    assert breaks is not None
    assert [i for start,end in breaks for i in range(start,end)]==list(range(7))
    assert all(sum(widths[a:b])+6*(b-a-1)<=220 for a,b in breaks)
    assert ordered_peer_rows([221,20],220,6,2) is None


def test_seven_controls_transpose_without_clipping_or_equal_width_stretch():
    es=[]
    for n,word in enumerate(('ONE','LONG LABEL','THREE','FOUR','FIVE','SIX','SEVEN')):
        x=n*140;dy=2*(n%2)
        es += [dict(id=f'p{n}',type='shape',shape_type='rounded_rectangle',x=x,y=20+dy,width=130,height=40,z_index=1,group_id=str(n)),
               dict(label(f't{n}',x+5,28+dy,120,size=14,group_id=str(n),text_align='center'),content=word)]
    s=dict(canvas=dict(width=1000,height=300),elements=es)
    out=measured_block(es,150,570,s,.8,(160,600))
    assert out and len(out)==14
    panels=sorted((e for e in out if e['type']=='shape'),key=lambda e:(e['y'],e['x']))
    assert [p['id'] for p in panels]==[f'p{i}' for i in range(7)]
    for i,a in enumerate(panels):
        assert a['x']+a['width']<=150.001 and a['y']+a['height']<=570.001
        for b in panels[i+1:]:
            assert a['x']+a['width']+2.9<=b['x'] or b['x']+b['width']+2.9<=a['x'] or a['y']+a['height']+2.9<=b['y']


def test_control_caption_remains_outside_backplate():
    p=dict(id='panel',type='shape',x=40,y=40,width=240,height=60,z_index=1,group_id='action')
    t=label('primary',40,55,240,30,22,group_id='action',text_align='center')
    c=label('note',40,110,240,20,12,text_align='center')
    s=dict(canvas=dict(width=400,height=300),elements=[p,t,c])
    assert control_captions(s)==[('panel','primary','note')]
    out=measured_block(s['elements'],220,150,s,.8,(400,300));assert out
    by={e['id']:e for e in out}
    assert by['note']['y']>=by['panel']['y']+by['panel']['height']+3
    assert by['primary']['font_size']>by['note']['font_size']


def test_two_contained_labels_preserve_inline_and_stacked_relationships():
    p=dict(id='panel',type='shape',x=10,y=10,width=260,height=100,z_index=1)
    a=label('a',20,25,100,25,18);b=label('b',150,25,100,25,16)
    s=dict(canvas=dict(width=400,height=300),elements=[p,a,b])
    inline=measured_block(s['elements'],260,120,s,.8,(400,300));assert inline
    _,x,y=inline
    assert x['x']+x['width']<y['x'] and abs(x['y']+x['height']/2-y['y']-y['height']/2)<.01
    s['elements'][2].update(x=20,y=65)
    stacked=measured_block(s['elements'],260,150,s,.8,(400,300));assert stacked
    _,x,y=stacked
    assert x['y']+x['height']<y['y']


def test_source_center_snap_cannot_undo_deliberate_target_left_alignment():
    a=label('heading',150,50,300,40,28,text_align='center')
    b=label('support',160,150,280,30,18,text_align='center')
    source=dict(canvas=dict(width=600,height=400),elements=[a,b])
    target=deepcopy(source['elements'])
    for e in target:e.update(x=100,text_align='left')
    out,notes=snap_source_axes(target,source,600,400)
    assert all(e['x']==100 for e in out)
    assert not any(n['axis']=='x' and n['fraction']==.5 for n in notes)


def test_translucent_tight_copy_panel_is_not_ambient_decoration():
    from engine.geometry_policy import ambient_shape
    from engine.macro_layout import composites
    panel=dict(id='panel',type='shape',shape_type='rectangle',x=20,y=100,width=400,height=100,z_index=1,fill_color='rgba(0,0,0,0.35)')
    copy=label('copy',30,110,380,80,18);copy['content']='A descriptive paragraph with several words that should wrap inside the native panel.'
    s=dict(canvas=dict(width=600,height=600),elements=[panel,copy])
    assert not ambient_shape(panel,s)
    assert len(composites(s)['blocks'])==1
    out=measured_block(s['elements'],180,220,s,1.,(200,600));assert out
    p,t=out
    assert t['x']>=p['x'] and t['y']>=p['y']
    assert t['x']+t['width']<=p['width'] and t['y']+t['height']<=p['height']
    large=dict(panel,width=600,height=600,x=0,y=0)
    assert ambient_shape(large,dict(canvas=s['canvas'],elements=[large,copy]))


def test_only_blocked_primary_axis_translation_trims_spare_frame():
    from engine.layout_contracts import _primary_axis
    from engine.resize import _text_visible_bounds
    a=label('a',0,100,100,30,18,group_id='a')
    b=label('b',0,200,100,30,18,group_id='b')
    for t in (a,b):
        ink=_text_visible_bounds(t);t['x']+=300-(ink[0]+ink[2])/2
    wash=dict(id='wash',type='shape',shape_type='rectangle',x=0,y=0,width=600,height=600,z_index=0,opacity=.3)
    s=dict(canvas=dict(width=600,height=600),elements=[wash,a,b])
    items=deepcopy([a,b])
    for t in items:t.update(x=5,width=150,font_size=10)
    units=[dict(items=[t],semantic='body') for t in (a,b)]
    _primary_axis(items,s,units,160,600,apply=True)
    for t in items:
        ink=_text_visible_bounds(t)
        assert abs((ink[0]+ink[2])/2-80)<.01
        assert t['x']>=0 and t['x']+t['width']<=160
        assert t['width']<150


def test_last_resort_grid_fits_raw_frames_not_only_painted_ink():
    import json
    from pathlib import Path
    from engine.resize import _analyze,_safe_grid,_raw_bounds,_assemble
    from engine.main import self_check
    s=json.loads((Path(__file__).parent/'fixtures/user_level3_sources/u03.json').read_text())['source']
    bg,units=_analyze(s)
    out=_safe_grid(units,320,50,(320*50/(1200*628))**.5)
    assert {e['id'] for e in out}=={e['id'] for u in units for e in u['items']}
    for e in out:
        x,y,r,b=_raw_bounds([e])
        assert x>=-1e-6 and y>=-1e-6 and r<=320+1e-6 and b<=50+1e-6
    assert not self_check(s,_assemble(out,bg,s,320,50))
