"""Viewer export regression: downloaded bytes keep native canvas dimensions."""
import base64
from io import BytesIO
import json
from pathlib import Path
import unittest

from PIL import Image
from viewer.server import adapt_preview

ROOT = Path(__file__).resolve().parents[1]


class ViewerExportTests(unittest.TestCase):
    def test_native_exports_above_and_below_renderer_preview_cap(self):
        task = json.loads((ROOT/'input/chameleon_sample_level1_warmup.json').read_text())
        for width, height in ((3000, 1000), (728, 90)):
            with self.subTest(target=(width, height)):
                task['target_canvas'] = dict(width=width, height=height)
                result = adapt_preview(task)
                raw = base64.b64decode(result['output_png'].split(',', 1)[1])
                with Image.open(BytesIO(raw)) as image:
                    self.assertEqual(image.format, 'PNG')
                    self.assertEqual(image.size, (width, height))
                self.assertEqual(result['design']['canvas'], task['target_canvas'])
                self.assertEqual(len(result['design']['elements']), len(task['source']['elements']))


if __name__ == '__main__':
    unittest.main()
