from pathlib import Path
from scripts.build_fresh_review_designs import build
from engine.main import self_check


def test_fresh_sources_are_distinct_complex_complete_and_inside_their_canvas():
    sources=build();assert len(sources)==4
    all_text=[]
    for s,title,targets in sources:
        doc=s.doc;assert len(doc['elements'])>=20 and len(targets)==2
        assert not self_check(doc,doc)
        assert len({e['id'] for e in doc['elements']})==len(doc['elements'])
        assert any(e['type']=='image' for e in doc['elements'])
        assert len({e.get('group_id') for e in doc['elements'] if e.get('group_id')})>=4
        all_text.append(' '.join(e.get('content','') for e in doc['elements']))
    assert len(set(all_text))==4
    assert len({s.doc['canvas']['width']/s.doc['canvas']['height'] for s,_,_ in sources})==4


def test_authored_type_fits_before_testing_reflow_when_local_fonts_exist():
    import pytest
    if not Path('data/crello_fonts.pickle').exists():pytest.skip('Local Crello fonts are not redistributed')
    from engine.resize import _text
    for s,_,_ in build():
        for e in s.doc['elements']:
            if e['type']=='text':
                h,w,_=_text(e);assert h<=e['height']+.1 and w<=e['width']+.1,(s.key,e['id'],w,h)
