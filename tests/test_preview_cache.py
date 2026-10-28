from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import pytest
from viewer.preview_cache import PreviewCache


def test_preview_cache_bounds_copies_and_does_not_cache_failures():
    cache=PreviewCache(entries=2,bytes_limit=1000);calls=[]
    def compute():calls.append(1);return {'design':{'elements':[]},'seconds':12.}
    first=cache.get_or_compute('a',compute);first['design']['elements'].append(1)
    second=cache.get_or_compute('a',compute)
    assert second['cache_hit'] and second['design']['elements']==[] and len(calls)==1
    for key in ('b','c'):cache.get_or_compute(key,compute)
    cache.get_or_compute('a',compute);assert len(calls)==4 and len(cache.items)==2
    def fail():raise ValueError('failed')
    with pytest.raises(ValueError):cache.get_or_compute('bad',fail)
    assert 'bad' not in cache.items
    tiny=PreviewCache(bytes_limit=3)
    assert not tiny.get_or_compute('big',compute)['cache_hit'] and not tiny.items


def test_identical_concurrent_previews_compute_once():
    cache=PreviewCache();calls=[]
    def compute():calls.append(1);return {'seconds':10}
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda _:cache.get_or_compute('same',compute),range(10)))
    assert len(calls)==1 and sum(r['cache_hit'] for r in results)==9


def test_preview_key_binds_source_target_assets_and_configuration(tmp_path,monkeypatch):
    from viewer.server import _preview_key
    asset=tmp_path/'p.png';asset.write_bytes(b'first')
    src={'canvas':{'width':50,'height':50},'elements':[dict(id='p',type='image',asset_id=str(asset),x=0,y=0,width=20,height=20)]}
    target=dict(width=100,height=80)
    with patch('viewer.server._asset_directory',return_value=tmp_path):
        a=_preview_key(src,target,'demo',None)
        assert a==_preview_key(src,target,'demo',None)
        asset.write_bytes(b'replaced bytes')
        assert a!=_preview_key(src,target,'demo',None)
        b=_preview_key(src,target,'demo',None);monkeypatch.setenv('MAGICANVAS_QWEN_MERGE_ADAPTER','1')
        assert b!=_preview_key(src,target,'demo',None)
        assert b!=_preview_key(src,dict(width=101,height=80),'demo',None)
