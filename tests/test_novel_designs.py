"""Verify test-source independence and reproducible, legal authoring."""
from pathlib import Path
import json
from scripts.build_novel_level3_designs import make_sources
from renderer.converter.colorutil import parse_rgba

ROOT=Path(__file__).resolve().parents[1]


def test_new_designs_have_distinct_content_assets_and_neutral_ids():
    original=json.loads((ROOT/'input/chameleon_sample_level3_advanced.json').read_text())['source']
    original_text={e.get('content') for e in original['elements'] if e['type']=='text'}
    original_assets={e.get('asset_id') for e in original['elements'] if e['type']=='image'}
    sources=make_sources()
    assert len(sources)==4
    photograph_ids=[]
    for s,title,targets in sources:
        es=s.doc['elements'];w=s.doc['canvas']['width'];h=s.doc['canvas']['height']
        assert len(es)>=20 and len(targets)==2
        assert len({e['id'] for e in es})==len(es)
        assert all(e['id'].startswith(s.key+'_') and e['id'].split('_')[1].isdigit() for e in es)
        assert all(e['x']>=0 and e['y']>=0 and e['x']+e['width']<=w and e['y']+e['height']<=h for e in es)
        texts={e['content'] for e in es if e['type']=='text'}
        assert not (texts & original_text)
        assets={e['asset_id'] for e in es if e['type']=='image'}
        assert not (assets & original_assets)
        photograph_ids.extend(assets)
        if (ROOT/'data/crello_train').exists():
            assert all((ROOT/'data/crello_train'/p).is_file() for p in assets)
        # Shape renderer only accepts rgb()/rgba(), not hex color strings.
        assert all(parse_rgba(e['fill_color'])[3]==1 for e in es if e['type']=='shape')
    assert len(set(photograph_ids))==4


def test_committed_fixtures_reproduce_the_pre_inference_authored_sources():
    for s,title,targets in make_sources():
        fixture=json.loads((ROOT/'tests/fixtures/novel_level3_sources'/f'{s.key}.json').read_text())
        assert fixture['source']==s.doc
        assert fixture['title']==title
        assert fixture['targets']==[dict(name=n,width=w,height=h) for n,w,h in targets]
