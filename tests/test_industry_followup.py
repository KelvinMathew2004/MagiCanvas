"""Verify research invariants rather than promising aesthetic correctness."""
from copy import deepcopy
import pytest
from experiments.layout_upgrade.industry import paragraph_balance,group_neighbors,rag


def text(**kwargs):
    return dict(id='t',type='text',content='A small example headline with balanced lines',x=20.,y=40.,
                width=240.,height=120.,font_size=24.,font_family='Solstice Sans',text_align='center',
                letter_spacing=1.2,**kwargs)


def test_rag_is_scale_independent():
    assert rag((0,0,2,(100.,50.)))==pytest.approx(rag((0,0,2,(200.,100.))))
    assert rag((0,0,2,(100.,100.)))==0.


@pytest.mark.parametrize('alignment',['left','center','right'])
def test_balancing_preserves_anchor_font_content_and_linecount(monkeypatch,alignment):
    import importlib
    resize=importlib.import_module('engine.resize');e=text();e['text_align']=alignment
    original=deepcopy(e);source=dict(canvas=dict(width=500,height=500),elements=[e])
    def measure(e,width=None):
        w=e['width'] if width is None else width
        return (60.,w,2,(w,100. if w>205 else w*.95),(4,4))
    monkeypatch.setattr(resize,'_text_info',measure)
    monkeypatch.setattr(resize,'_render_text_ink',lambda e:(0,0,e['width']-1,55))
    out,note=paragraph_balance([e],source,500,500)
    assert out and e==original and note['changes']
    after=out[0];fraction={'left':0.,'center':.5,'right':1.}[alignment]
    assert after['x']+fraction*after['width']==pytest.approx(e['x']+fraction*e['width'])
    assert all(after[k]==e[k] for k in ('content','height','y','font_size','letter_spacing','text_align'))


def test_balancing_preserves_authored_break_and_rotation():
    e=text();e['content']='Authored\nline break';source=dict(canvas=dict(width=500,height=500),elements=[e])
    out,note=paragraph_balance([e],source,500,500)
    assert out is None and note['skipped']['authored-break']==1
    e['content']='A long rotated headline';e['rotation']=15
    out,note=paragraph_balance([e],source,500,500)
    assert out is None and note['skipped']['rotation']==1


def test_group_neighbors_preserve_ids_rigid_children_and_tracking(monkeypatch):
    import engine.macro_layout as macro
    a=text();a['group_id']='one';a['x']=40.;a['width']=160.
    panel=dict(id='p',type='shape',x=30.,y=30.,width=190.,height=150.,group_id='one',shape_type='rectangle')
    b=deepcopy(a);b.update(id='b',x=350.,y=80.,group_id='two')
    elements=[panel,a,b];source=dict(canvas=dict(width=700,height=500),elements=deepcopy(elements))
    monkeypatch.setattr(macro,'composites',lambda *a,**k:dict(blocks=[[panel,source['elements'][1]],[b]]))
    monkeypatch.setattr(macro,'quality',lambda *a:dict(cost=10.))
    before=deepcopy(elements);proposals,note=group_neighbors(elements,source,700,500)
    assert len(proposals)==3 and elements==before
    for p in proposals:
        out={e['id']:e for e in p['elements']};assert set(out)=={'p','t','b'}
        s=out['p']['width']/panel['width']
        assert out['t']['x']-out['p']['x']==pytest.approx((a['x']-panel['x'])*s)
        assert out['t']['font_size']==pytest.approx(a['font_size']*s)
        assert out['t']['letter_spacing']==pytest.approx(a['letter_spacing']*s)


def test_group_neighbors_never_split_overlapping_composites(monkeypatch):
    import engine.macro_layout as macro
    a=text();b=deepcopy(a);b['id']='b'
    monkeypatch.setattr(macro,'composites',lambda *a,**k:dict(blocks=[[a0],[b]]))
    a0=a
    monkeypatch.setattr(macro,'quality',lambda *a:dict(cost=10.))
    source=dict(canvas=dict(width=700,height=500),elements=[a,b])
    candidates,note=group_neighbors([a,b],source,700,500)
    assert candidates==[]
