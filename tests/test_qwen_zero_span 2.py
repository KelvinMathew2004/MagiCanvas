"""A decoded-candidate fault differs from failing to load a required adapter."""
from copy import deepcopy
import pytest


def source():
    return dict(canvas=dict(width=600,height=600),elements=[
        dict(id='title',type='text',content='A new collection',font_family='Solstice Sans',font_size=32.,
            color='rgba(0,0,0,1)',x=60.,y=100.,width=480.,height=50.,z_index=1),
        dict(id='detail',type='text',content='Explore the new season',font_family='Solstice Sans',font_size=20.,
            color='rgba(0,0,0,1)',x=60.,y=220.,width=480.,height=35.,z_index=1)])


def setup(monkeypatch):
    monkeypatch.setenv('MAGICANVAS_QWEN_ENABLED','1')
    monkeypatch.setenv('MAGICANVAS_REQUIRE_QWEN_ADAPTER','1')
    monkeypatch.setenv('MAGICANVAS_DATASET_RENDERING','0')


def test_required_adapter_loading_or_generation_failure_still_raises(monkeypatch):
    setup(monkeypatch)
    def fail(*a,**kw):raise RuntimeError('required adapter cannot load')
    monkeypatch.setattr('engine.qwen_seed.generate_seed',fail)
    from engine.resize import _adapt_with_diagnostics
    with pytest.raises(RuntimeError,match='required adapter'):
        _adapt_with_diagnostics(source(),dict(width=728,height=90))


def test_zero_span_realization_fault_preserves_literal_prediction_and_continues(monkeypatch):
    setup(monkeypatch)
    def generate(src,target,base,diagnostics=None):
        proposal=deepcopy(base)
        diagnostics.update(json_complete=True,missing_ids=[],model_elements={e['id']:dict(e) for e in proposal})
        for value in diagnostics['model_elements'].values():value.pop('id')
        return proposal,'decoded test proposal'
    def fail(*a,**kw):raise ZeroDivisionError('zero source span')
    monkeypatch.setattr('engine.qwen_seed.generate_seed',generate)
    monkeypatch.setattr('engine.resize._bounded_qwen_elements',fail)
    monkeypatch.setattr('engine.resize._siglip_rerank',lambda *a,**kw:dict(used=False,reason='unit test'))
    from engine.resize import _adapt_with_diagnostics
    result,report=_adapt_with_diagnostics(source(),dict(width=728,height=90))
    assert {e['id'] for e in result['elements']}=={'title','detail'}
    d=report['qwen_diagnostics']
    assert d['proposal_received'] is True
    assert d['realization_error']==dict(type='ZeroDivisionError',message='zero source span')
    assert {e['id'] for e in d['model_elements']}=={'title','detail'}
    assert d['generation']['json_complete'] is True
    assert not report['winner'].startswith('qwen-')
