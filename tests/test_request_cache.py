import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch
from engine.request_cache import asset_lookup_request,asset_lookup_state

engine=importlib.import_module('engine.resize')


def test_new_request_rediscovers_previously_missing_asset(tmp_path,monkeypatch):
    brand=tmp_path/'brand';dataset=tmp_path/'dataset';brand.mkdir();dataset.mkdir()
    monkeypatch.setattr(engine,'_ASSETS_DIR',brand);monkeypatch.setattr(engine,'_DATASET_SPLIT_DIRS',(dataset,))
    monkeypatch.delenv('MAGICANVAS_DATASET_ASSETS_DIR',raising=False)
    @asset_lookup_request
    def request(create=False):
        first=engine._asset_root({'asset_id':'new.png'})
        if create:(dataset/'new.png').write_bytes(b'new upload')
        second=engine._asset_root({'asset_id':'new.png'})
        return (first,second),{}
    paths,report=request(create=True)
    assert paths==(brand,brand) and report['asset_lookup_cache']['roots_resolved']==1
    paths,report=request()
    assert paths==(dataset,dataset) and report['asset_lookup_cache']['hits']==1
    assert asset_lookup_state() is None


def test_parallel_requests_do_not_share_state_and_exception_resets_it():
    barrier=Barrier(2)
    @asset_lookup_request
    def request():
        state=asset_lookup_state();barrier.wait();engine._asset_root({'asset_id':'example.png'});engine._asset_root({'asset_id':'example.png'})
        return id(state),{}
    with ThreadPoolExecutor(max_workers=2) as pool:
        a,b=[f.result() for f in [pool.submit(request),pool.submit(request)]]
    assert a[0]!=b[0]
    assert a[1]['asset_lookup_cache']['roots_resolved']==b[1]['asset_lookup_cache']['roots_resolved']==1
    @asset_lookup_request
    def broken():raise ValueError('test failure')
    try:broken()
    except ValueError:pass
    assert asset_lookup_state() is None


def test_diagnostic_disable_is_exact_and_does_not_memoize(monkeypatch):
    monkeypatch.setenv('MAGICANVAS_ASSET_LOOKUP_CACHE','0')
    @asset_lookup_request
    def request():return [engine._asset_root({'asset_id':'a'}) for _ in range(3)],{}
    with patch.object(engine,'_uncached_asset_root',return_value='root') as lookup:
        result,report=request()
    assert result==['root']*3 and lookup.call_count==3
    assert not report['asset_lookup_cache']['enabled']
