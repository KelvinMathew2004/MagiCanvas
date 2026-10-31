from copy import deepcopy
import unittest
from unittest.mock import patch

from engine import resize


def sample(**kw):
    return dict(id='copy', type='text', content='Example', font_family='Solstice Sans',
                font_size=3, width=100, height=8, x=20, y=20, text_align='left', **kw)


class TargetFontBearingsTests(unittest.TestCase):
    def test_target_size_intrinsic_top_bearing_is_not_local_clipping(self):
        target = sample()
        source = dict(target, font_size=12)

        def ink(e):
            return (0, 1, 80, 12) if e['font_size'] == 12 else (0, -1, 20, 4)

        original = deepcopy(target)
        with patch.object(resize, '_render_text_ink', side_effect=ink), \
             patch.object(resize, '_text_info', return_value=(8, 20, 1)):
            self.assertEqual(resize._source_ink_overhang(target, source), (0., 1.))
        self.assertEqual(target, original)

    def test_centered_frame_overflow_does_not_become_intrinsic_bearing(self):
        target = dict(sample(), text_align='center', width=10)
        source = dict(target, font_size=12, width=100)

        def ink(e):
            if e['font_size'] == 12:
                return (0, 1, 80, 12)
            return (-5, 0, 15, 4) if e['text_align'] == 'center' else (0, 0, 20, 4)

        with patch.object(resize, '_render_text_ink', side_effect=ink), \
             patch.object(resize, '_text_info', return_value=(8, 20, 1)):
            self.assertEqual(resize._source_ink_overhang(target, source), (0., 0.))

    def test_canvas_overflow_is_still_rejected_for_intrinsic_bearing(self):
        from engine.raster_metrics import glyph_overflow
        target = sample()
        target['y'] = 0
        source = dict(canvas=dict(width=200, height=100), elements=[dict(target, font_size=12)])

        def ink(e):
            return (0, 1, 80, 12) if e['font_size'] == 12 else (0, -1, 20, 4)

        with patch.object(resize, '_render_text_ink', side_effect=ink), \
             patch.object(resize, '_text_info', return_value=(8, 20, 1)):
            self.assertEqual(glyph_overflow(source, [target], 200, 100), {'copy': 1.0})

    def test_new_horizontal_bearings_keep_existing_source_policy(self):
        target = sample()
        source = dict(target, font_size=12)

        def ink(e):
            return (0, 1, 80, 12) if e['font_size'] == 12 else (-1, 0, 20, 4)

        with patch.object(resize, '_render_text_ink', side_effect=ink), \
             patch.object(resize, '_text_info', return_value=(8, 20, 1)):
            self.assertEqual(resize._source_ink_overhang(target, source), (0., 0.))


if __name__ == '__main__':
    unittest.main()
