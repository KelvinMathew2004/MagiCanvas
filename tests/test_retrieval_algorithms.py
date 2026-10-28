import itertools,random,json
from copy import deepcopy
import pytest
from engine.free_space import free_rectangles,place
from engine.retrieve import assign
from scripts.build_layout_library import allowed_training


def test_hungarian_matches_bruteforce_rectangular_negative_tied_costs():
    rng=random.Random(191)
    for n,m in ((1,3),(2,4),(3,3),(4,5)):
        for _ in range(30):
            costs=[[rng.randint(-4,7)/3 for j in range(m)] for i in range(n)]
            got=assign(costs)
            expected=min(sum(costs[i][j] for i,j in enumerate(p)) for p in itertools.permutations(range(m),n))
            assert sum(costs[i][j] for i,j in got)==pytest.approx(expected)
            assert len({j for _,j in got})==n
    assert assign([])==[]
    for c in ([[1],[2]],[[float('nan')]],[[1,2],[3]]):
        with pytest.raises(ValueError):assign(c)


def test_grid_returns_every_maximal_empty_rectangle_against_exhaustive_oracle():
    rng=random.Random(231)
    for n in (3,4,5):
        for _ in range(15):
            occupied={(x,y) for y in range(n) for x in range(n) if rng.random()<.3}
            empties=[]
            for l,r in itertools.combinations(range(n+1),2):
                for t,b in itertools.combinations(range(n+1),2):
                    cells={(x,y) for x in range(l,r) for y in range(t,b)}
                    if not cells&occupied:empties.append(((l,t,r,b),cells))
            expected={rect for rect,cells in empties if not any(cells<other for _,other in empties)}
            got=set(free_rectangles(n,n,[(x,y,x+1,y+1) for x,y in occupied],cells=n))
            assert got==expected


def test_grid_is_conservative_at_fractional_boundaries_and_padding():
    rects=free_rectangles(127,83,[(33.2,20.5,61.4,59.8)],margin=3,padding=2)
    assert rects
    for l,t,r,b in rects:
        assert min(l,t)>=3 and r<=124 and b<=80
        assert min(r,63.4)<=max(l,31.2) or min(b,61.8)<=max(t,18.5)
    assert free_rectangles(10,10,[(0,0,10,10)])==[]
    assert place((3,4),[],(0,0),(10,10)) is None
    p=place((10,10),[(0,0,20,20),(80,80,100,100)],(90,90),(100,100),scale_cap=2)
    assert p['x']==80 and p['y']==80 and p['scale']==2


def test_library_excludes_transitive_model_and_preference_family_bridges():
    rows=[dict(source_id='a',target_id='b',model_split='train'),
          dict(source_id='b',target_id='c',model_split='test'),
          dict(source_id='d',target_id='e',model_split='train',family_id='shared'),
          dict(source_id='f',target_id='g',model_split='train')]
    preference=[dict(source_id='x',target_id='y',family_id='shared')]
    allowed=allowed_training(rows,preference)
    assert [r['source_id'] for r,_ in allowed]==['f']


def test_dual_granularity_preserves_native_photo_caption_badge():
    from engine.macro_layout import composites
    from tests.test_macro_layout import text
    photo=dict(id='p',type='image',x=50,y=50,width=300,height=200,z_index=1)
    badge=dict(id='b',type='shape',x=230,y=60,width=100,height=30,z_index=2)
    source=dict(canvas=dict(width=500,height=500),elements=[photo,badge,text('label',237,65,85,20,'New')])
    for tight in (True,False):assert len(composites(source,tight=tight)['blocks'])==1


def test_library_contains_only_geometry_and_no_leaking_ids():
    from pathlib import Path
    path=Path('engine/data/layout_library.json')
    if not path.exists():pytest.skip('Optional training-derived library stays local; build with scripts/build_layout_library.py')
    library=json.loads(path.read_text())
    assert library['layouts']
    for entry in library['layouts']:
        assert set(entry)=={'aspect','key','zones','roles'}
        for zone in entry['zones']:assert set(zone)=={'box','role','kind'}


def test_tight_supplement_uses_hierarchy_without_changing_legacy_partition():
    from unittest.mock import patch
    from engine.macro_layout import composites
    from tests.test_macro_layout import text
    source=dict(canvas=dict(width=500,height=500),elements=[text('title',50,40,200,40,'Headline'),text('copy',50,100,200,35,'Supporting copy'),dict(id='p',type='image',x=300,y=40,width=120,height=180,z_index=1)])
    with patch('engine.source_structure.groups',return_value={'members':[('title','copy'),('p',)]}):
        legacy=composites(source)
        tight=composites(source,tight=True)
    assert len(legacy['blocks'])==3
    assert {frozenset(e['id'] for e in es) for es in tight['blocks']}=={frozenset(('title','copy')),frozenset(('p',))}


def test_retrieved_realization_preserves_ids_and_card_children(tmp_path,monkeypatch):
    from engine.retrieve import candidates
    from engine.macro_layout import composites
    from tests.test_macro_layout import text
    source=dict(canvas=dict(width=500,height=500),elements=[text('title',30,30,200,40,'A title',size=30),dict(id='panel',type='shape',shape_type='rectangle',x=30,y=180,width=200,height=60,z_index=1,group_id='control',fill_color='#ffffff'),text('label',45,197,170,26,'Continue',group_id='control')])
    blocks=composites(source)['blocks']
    library=dict(version=1,layouts=[dict(key='geometryhash',aspect=2.,roles={'header':1,'action':1},zones=[dict(box=[.05,.1,.5,.3],role='header',kind='copy'),dict(box=[.6,.5,.3,.3],role='action',kind='control')])])
    path=tmp_path/'library.json';path.write_text(json.dumps(library));monkeypatch.setenv('MAGICANVAS_LAYOUT_LIBRARY',str(path))
    proposals=candidates(source,blocks,800,400)
    assert proposals
    out={e['id']:e for e in proposals[0]['elements']}
    assert set(out)=={'title','panel','label'}
    p,t=out['panel'],out['label']
    assert p['x']<=t['x'] and t['x']+t['width']<=p['x']+p['width']+.1
    assert p['y']<=t['y'] and t['y']+t['height']<=p['y']+p['height']+.1
    assert source['elements'][0]['font_size']==30
