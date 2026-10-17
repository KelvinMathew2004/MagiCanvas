"""Development preview only. The submission container still runs engine.main."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIGHTBOX_SHORT_SIDE = 1080
LIGHTBOX_MAX_PIXELS = 40_000_000
LIGHTBOX_MAX_SIDE = 24_000


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


def render_preview_data(design):
    from renderer.converter.convert import render_design_to_png

    with TemporaryDirectory(prefix='magicanvas-preview-') as directory:
        path = Path(directory) / 'preview.png'
        warnings = render_design_to_png(design, ROOT/'brand_kit'/'assets', path,
                         max_size=max(design['canvas']['width'], design['canvas']['height']))
        png_data = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    return png_data, warnings


def adapt_preview(task, source_png=None, validated=None):
    import importlib
    import renderer.converter.assets
    import engine.resize
    import engine.main
    importlib.reload(renderer.converter.assets)
    importlib.reload(engine.resize)
    importlib.reload(engine.main)
    current_adapt = engine.resize._adapt_with_diagnostics
    current_text = engine.resize._text
    current_self_check = engine.main.self_check

    source, target = validated or validate_task(task)
    started = perf_counter()
    # Render the original before starting the expensive search. The streaming
    # endpoint sends this preview to the browser immediately, then continues.
    if source_png is None:
        source_png, source_warnings = render_preview_data(source)
    else:
        source_warnings = []
    output, diagnostics = current_adapt(source, target)
    warnings = source_warnings + current_self_check(source, output)
    gate = diagnostics.get('quality_gate', {})
    if gate and not gate.get('passed', False):
        faults = ', '.join(gate.get('faults', {}).keys())
        repair_passes = sum(bool(item.get('used'))
                            for item in diagnostics.get('refinement', {}).get('rounds', []))
        warnings.append(
            f"Layout quality checks still flag {faults} after "
            f"{repair_passes} automatic repair pass(es)."
        )
    for e in output['elements']:
        if e['type'] == 'text':
            h, w, lines = current_text(e)
            if h > e['height'] + .1 or w > e['width'] + .1 or (e.get('max_lines') and lines > e['max_lines']):
                warnings.append(f"{e['id']}: text may not fit at this resolution.")
            if e['font_size'] < 8:
                warnings.append(f"{e['id']}: {e['font_size']:.1f}px text is very small. Copy cannot be removed or rewritten under stage-one rules.")
    output_png, output_warnings = render_preview_data(output)
    warnings.extend(output_warnings)
    return dict(design=output, source=source, source_png=source_png, output_png=output_png,
                warnings=list(dict.fromkeys(warnings)), seconds=round(perf_counter()-started, 3))


def _sse_message(event, data):
    payload=json.dumps(data,separators=(',',':'))
    return f'event: {event}\ndata: {payload}\n\n'.encode()


def render_lightbox(design):
    """Render a detail-view copy with a 1080px minimum short side when safe."""
    from renderer.converter.convert import render_design_to_png

    lightbox_design = deepcopy(design)
    width = design['canvas']['width']
    height = design['canvas']['height']
    longest = max(width, height)
    shortest = min(width, height)
    scale = min(
        max(1.0, LIGHTBOX_SHORT_SIDE / shortest),
        math.sqrt(LIGHTBOX_MAX_PIXELS / (width * height)),
        LIGHTBOX_MAX_SIDE / longest,
    )

    if scale > 1:
        lightbox_design['canvas']['width'] = round(width * scale)
        lightbox_design['canvas']['height'] = round(height * scale)
        for element in lightbox_design['elements']:
            for key in ('x', 'y', 'width', 'height', 'font_size', 'letter_spacing',
                        'corner_radius', 'stroke_width'):
                if isinstance(element.get(key), (int, float)):
                    element[key] *= scale

    max_size = max(lightbox_design['canvas']['width'], lightbox_design['canvas']['height'])
    with TemporaryDirectory(prefix='magicanvas-lightbox-') as directory:
        path = Path(directory) / 'detail.png'
        render_design_to_png(lightbox_design, ROOT/'brand_kit'/'assets', path, max_size=max_size)
        return path.read_bytes(), lightbox_design['canvas']


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, kind='application/json', headers=None):
        body = json.dumps(data).encode() if kind == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        for name, value in (headers or {}).items():
            self.send_header(name, str(value))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/samples':
            samples = [dict(name=p.stem, task=json.loads(p.read_text())) for p in sorted((ROOT/'input').glob('*.json'))]
            return self.reply(200, samples)
        files = {
            '/': ('index.html', 'text/html; charset=utf-8'),
            '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
            '/chameleon-loader.mjs': ('chameleon-loader.mjs', 'text/javascript; charset=utf-8'),
            '/theme.mjs': ('theme.mjs', 'text/javascript; charset=utf-8'),
            '/style.css': ('style.css', 'text/css; charset=utf-8'),
        }
        if path in files:
            name, kind = files[path]
            return self.reply(200, (Path(__file__).parent/name).read_bytes(), kind)
        self.reply(404, dict(error='Not found'))

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ('/api/adapt', '/api/adapt-stream', '/api/lightbox'):
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
            if path == '/api/lightbox':
                design = task.get('design') if isinstance(task, dict) else None
                if not isinstance(design, dict):
                    raise ValueError('Expected a design to render for the lightbox.')
                source, _ = validate_task({'source': design, 'target_canvas': design.get('canvas', {})})
                png, canvas = render_lightbox(source)
                return self.reply(200, png, 'image/png', {
                    'X-Image-Width': canvas['width'],
                    'X-Image-Height': canvas['height'],
                })
            if path == '/api/adapt-stream':
                source, target = validate_task(task)
                source_png, source_warnings = render_preview_data(source)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(_sse_message('source', dict(source_png=source_png,
                                                              source_size=source['canvas'])))
                self.wfile.flush()
                try:
                    result = adapt_preview(task, source_png=source_png,
                                           validated=(source,target))
                    result['warnings']=list(dict.fromkeys(source_warnings+result['warnings']))
                    self.wfile.write(_sse_message('adapted', result))
                    self.wfile.flush()
                except Exception as exc:
                    self.wfile.write(_sse_message('error', dict(error=f'Adaptation failed: {exc}')))
                    self.wfile.flush()
                self.close_connection=True
                return
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
