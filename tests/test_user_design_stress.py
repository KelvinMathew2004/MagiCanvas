from pathlib import Path
from copy import deepcopy
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
FILES=sorted((ROOT/'tests/fixtures/user_level3_sources').glob('u*.json'))


def test_asset_backed_copies_preserve_supplied_content_and_geometry():
    from viewer.server import validate_task
    assert len(FILES)==4
    for path in FILES:
        f=json.loads(path.read_text());old=f['original_task']['source'];new=f['source']
        assert new['canvas']==old['canvas'] and len(new['elements'])==len(old['elements'])
        assert len({e['id'] for e in new['elements']})==len(new['elements'])
        for before,after in zip(old['elements'],new['elements']):
            for k,v in before.items():
                if k not in ('id','font_family','asset_id'):assert after[k]==v,(path,k)
        assert validate_task(dict(source=new,target_canvas=f['targets'][1]))
        assert f['targets'][1]==dict(name='skyscraper',width=160,height=600)


def test_supplied_duplicate_is_rejected_without_silent_engine_repair():
    from viewer.server import validate_task
    from engine.resize import _adapt_with_diagnostics
    task=json.loads((ROOT/'tests/fixtures/user_level3_sources/u05.json').read_text())['original_task']
    before=deepcopy(task)
    with pytest.raises(ValueError,match='unique'):validate_task(task)
    with pytest.raises(ValueError,match='unique'):_adapt_with_diagnostics(task['source'],task['target_canvas'])
    assert task==before


def test_portrait_stress_selection_is_dense_and_has_no_target_leakage():
    if not (ROOT/'data/crello_validation/manifest.jsonl').exists():pytest.skip('Local dataset excluded from Git')
    from scripts.evaluate_skyscraper_stress import sources_and_records
    sources,records=sources_and_records()
    assert len(records)==13
    dataset=[r for r in records if r['source_split']=='validation']
    assert len(dataset)==4 and len({r['family_id'] for r in dataset})==4
    for r in dataset:
        s=sources[r['family_id']];assert s['canvas']['height']>s['canvas']['width']
        assert len(s['elements'])>=18 and r['target_canvas']==dict(width=160,height=600)
        assert 'messages' not in r


def test_failed_auxiliary_font_measurement_uses_renderer_metrics():
    from unittest.mock import Mock, patch
    from engine.resize import _advances
    bad=Mock();bad.getlength.side_effect=OSError('execution context too long')
    _advances.cache_clear()
    with patch('engine.resize._font',return_value=bad), patch('engine.resize._skia_advances',return_value=(420.,)) as raster:
        assert _advances('Z','bad-font',False,False,((0,1,True,None),)) == (.42,)
        raster.assert_called_once_with('Z','bad-font',True,False,1000)
    _advances.cache_clear()


def test_preview_zero_area_layers_preserve_audit_input_and_warn():
    from scripts.build_focus_reflow_viewer import _preview_nonempty
    doc={'elements':[{'id':'zero','width':0,'height':20},{'id':'visible','width':30,'height':20}]}
    before=deepcopy(doc);calls=[]
    def render(preview,path):
        calls.append(preview)
        return ['renderer warning']
    warnings=_preview_nonempty(render,doc,Path('unused'))
    assert doc==before and [e['id'] for e in calls[0]['elements']]==['visible']
    assert 'zero' in warnings[0] and warnings[1]=='renderer warning'
