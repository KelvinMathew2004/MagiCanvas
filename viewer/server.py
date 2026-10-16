"""Development preview only. The submission container still runs engine.main."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.resize import resize, _text, _bounds
from engine.main import self_check


def validate_task(task):
    if not isinstance(task, dict) or not isinstance(task.get('source'), dict) or not isinstance(task.get('target_canvas'), dict):
        raise ValueError('Expected {"source": {"canvas": ..., "elements": [...]}, "target_canvas": ...}.')
    source, target = task['source'], task['target_canvas']
    for canvas in (source.get('canvas', {}), target):
        if any(type(canvas.get(k)) is not int or canvas[k] <= 0 or canvas[k] > 12000 for k in ('width', 'height')):
            raise ValueError('Preview dimensions must be integer pixels from 1 to 12000.')
        if canvas['width'] * canvas['height'] > 40000000:
            raise ValueError('Preview canvas exceeds 40 million pixels.')
    elements = source.get('elements')
    if not isinstance(elements, list) or not 1 <= len(elements) <= 500:
        raise ValueError('A preview needs between 1 and 500 elements.')
    import math
    ids = set()
    for e in elements:
        if not isinstance(e, dict) or not isinstance(e.get('id'), str) or e['id'] in ids:
            raise ValueError('Each element needs a unique string id.')
        ids.add(e['id'])
        for k in ('x', 'y', 'width', 'height'):
            if type(e.get(k)) not in (int, float) or not math.isfinite(e[k]):
                raise ValueError(f"{e['id']}: {k} must be finite.")
        if e['width'] < 0 or e['height'] < 0:
            raise ValueError('Element dimensions cannot be negative.')
        if e.get('type') == 'text':
            if not isinstance(e.get('content'), str) or not isinstance(e.get('font_family'), str) or type(e.get('font_size')) not in (int, float) or not math.isfinite(e['font_size']) or e['font_size'] < 1:
                raise ValueError('Text requires content, font_family and a positive font_size.')
        elif e.get('type') == 'image':
            if not isinstance(e.get('asset_id'), str):
                raise ValueError('Images require asset_id.')
        elif e.get('type') == 'shape':
            if e.get('shape_type') not in ('rectangle', 'ellipse', 'line') or not isinstance(e.get('fill_color'), str):
                raise ValueError('Shapes require shape_type and fill_color.')
        else:
            raise ValueError('Element type must be text, image or shape.')
    return source, target


def adapt_preview(task):
    source, target = validate_task(task)
    started = perf_counter()
    output = resize(source, target)
    warnings = self_check(source, output)
    for e in output['elements']:
        if e['type'] == 'text':
            h, w, lines = _text(e)
            if h > e['height'] + .1 or w > e['width'] + .1 or (e.get('max_lines') and lines > e['max_lines']):
                warnings.append(f"{e['id']}: text may not fit at this resolution.")
            if e['font_size'] < 8:
                warnings.append(f"{e['id']}: {e['font_size']:.1f}px text is very small. Copy cannot be removed or rewritten under stage-one rules.")
    # Import failures must be visible; never silently show placeholder boxes.
    from renderer.converter.convert import render_design_to_png
    images = []
    with TemporaryDirectory(prefix='magicanvas-preview-') as directory:
        for name, design in (('source', source), ('adapted', output)):
            path = Path(directory) / (name + '.png')
            warnings.extend(render_design_to_png(design, ROOT/'brand_kit'/'assets', path))
            images.append('data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode())
    return dict(design=output, source=source, source_png=images[0], output_png=images[1],
                warnings=list(dict.fromkeys(warnings)), seconds=round(perf_counter()-started, 3))


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, kind='application/json'):
        body = json.dumps(data).encode() if kind == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/samples':
            samples = [dict(name=p.stem, task=json.loads(p.read_text())) for p in sorted((ROOT/'input').glob('*.json'))]
            return self.reply(200, samples)
        files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/theme.mjs': ('theme.mjs', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path in files:
            name, kind = files[path]
            return self.reply(200, (Path(__file__).parent/name).read_bytes(), kind)
        self.reply(404, dict(error='Not found'))

    def do_POST(self):
        if urlparse(self.path).path != '/api/adapt':
            return self.reply(404, dict(error='Not found'))
        # The dev server is loopback-only; accept same-origin browser requests.
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        if origin and origin not in (f'http://{host}', f'https://{host}'):
            return self.reply(403, dict(error='Cross-origin requests are disabled.'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 4 * 1024 * 1024:
                return self.reply(413, dict(error='Task JSON must be smaller than 4 MB.'))
            task = json.loads(self.rfile.read(length))
            result = adapt_preview(task)
        except (ValueError, TypeError, KeyError) as exc:
            return self.reply(400, dict(error=str(exc)))
        except ImportError as exc:
            return self.reply(503, dict(error=f'Real renderer unavailable: {exc}. Install requirements.txt in the active environment.'))
        except Exception as exc:
            return self.reply(500, dict(error=f'Preview failed: {exc}'))
        self.reply(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5050)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'MagiCanvas preview: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
