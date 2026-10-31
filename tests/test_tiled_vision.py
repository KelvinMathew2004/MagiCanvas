"""Optional crop scoring cannot silently change the production default."""
from unittest.mock import patch
from PIL import Image
from engine import vision


def test_default_is_exact_global_passthrough(monkeypatch):
    monkeypatch.delenv('MAGICANVAS_SIGLIP_TILED',raising=False)
    images=[Image.new('RGB',(400,100))];reference=Image.new('RGB',(100,100));context={'bucket':'banner'}
    with patch.object(vision,'_score_layouts_global',return_value=([2.],'global')) as f:
        assert vision.score_layouts(images,reference,context)==([2.],'global')
        f.assert_called_once_with(images,reference,context)


def test_fixed_blend_matches_experiment_and_keeps_order(monkeypatch):
    monkeypatch.setenv('MAGICANVAS_SIGLIP_TILED','1')
    images=[Image.new('RGB',(400,100)),Image.new('RGB',(100,100))]
    with patch.object(vision,'_load_ranker',return_value=None),patch.object(vision,'_score_layouts_global',return_value=([10.,2.,4.,6.,20.,8.],'global')) as f:
        scores,note=vision.score_layouts(images)
        assert scores==[7.,14.]
        assert [im.size for im in f.call_args.args[0]]==[(400,100),(100,100),(100,100),(100,100),(100,100),(100,100)]
        assert 'optional fixed' in note


def test_local_global_preference_head_skips_tiling(monkeypatch):
    monkeypatch.setenv('MAGICANVAS_SIGLIP_TILED','1');images=[Image.new('RGB',(100,400))]
    with patch.object(vision,'_load_ranker',return_value=object()),patch.object(vision,'_score_layouts_global',return_value=([1.],'head')) as f:
        assert vision.score_layouts(images)==([1.],'head')
        assert f.call_args.args[0] is images or f.call_args.args[0]==images


def test_failed_optional_score_retries_global(monkeypatch):
    monkeypatch.setenv('MAGICANVAS_SIGLIP_TILED','1');images=[Image.new('RGB',(400,100))]
    with patch.object(vision,'_load_ranker',return_value=None),patch.object(vision,'_score_layouts_global',side_effect=[(None,'failed'),([3.],'global')]) as f:
        assert vision.score_layouts(images)==([3.],'global')
        assert len(f.call_args.args[0])==1
