"""Development preview only. The submission container still runs engine.main."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
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


def _asset_directory(mode, sample_id=None):
    if mode == 'demo':
        return ROOT / 'brand_kit' / 'assets'
    if mode != 'test':
        raise ValueError("Asset mode must be 'demo' or 'test'.")
    if sample_id:
        sample_path = (DATA_ROOT / sample_id).resolve()
        if not sample_path.is_relative_to(DATA_ROOT.resolve()) or not sample_path.is_file():
            raise ValueError('Unknown test design.')
        if sample_path.parent.name == 'designs' and (sample_path.parent.parent / 'assets').is_dir():
            return sample_path.parent.parent
    return DATA_ROOT / 'crello_sample'


def _render_function(mode):
    if mode == 'test' and (DATA_ROOT / 'crello_fonts.pickle').is_file():
        from scripts.dataset_renderer import render_design_to_png
        return render_design_to_png, DATA_ROOT / 'crello_fonts.pickle'
    from renderer.converter.convert import render_design_to_png
    return render_design_to_png, ROOT / 'brand_kit' / 'fonts'


def _with_resolved_assets(design, assets_dir):
    """Turn relative asset ids into absolute paths for the local renderer."""
    resolved = deepcopy(design)
    for element in resolved.get('elements', []):
        asset_id = element.get('asset_id')
        if element.get('type') != 'image' or not isinstance(asset_id, str):
            continue
        path = Path(asset_id)
        candidate = path if path.is_absolute() else assets_dir / path
        if candidate.is_file():
            element['asset_id'] = str(candidate.resolve())
    return resolved


def _restore_asset_ids(design, original):
    """Keep downloaded/output JSON portable after local rendering."""
    original_ids = {e.get('id'): e.get('asset_id') for e in original.get('elements', [])}
    for element in design.get('elements', []):
        eid = element.get('id')
        if eid in original_ids and original_ids[eid] is not None:
            element['asset_id'] = original_ids[eid]
    return design


def _sample_records(mode):
    if mode == 'demo':
        files = sorted((ROOT / 'input').glob('*.json'))
        root = ROOT / 'input'
    elif mode == 'test':
        files = sorted(DATA_ROOT.rglob('*.json')) if DATA_ROOT.is_dir() else []
        root = DATA_ROOT
    else:
        raise ValueError("Mode must be 'demo' or 'test'.")

    records = []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(value.get('source'), dict) and isinstance(value.get('target_canvas'), dict):
                task = {'source': value['source'], 'target_canvas': value['target_canvas']}
            elif isinstance(value.get('canvas'), dict) and isinstance(value.get('elements'), list):
                task = {'source': value, 'target_canvas': value['canvas']}
            else:
                continue
            relative = path.relative_to(root).as_posix()
            records.append({'name': path.stem,
                            'id': relative if mode == 'test' else None,
                            'task': task})
        except (OSError, ValueError, TypeError):
            continue
    return records


def render_preview_data(design, assets_dir=None, asset_mode='demo'):
    render_design_to_png, fonts = _render_function(asset_mode)
    assets_dir = assets_dir or (ROOT / 'brand_kit' / 'assets')
    with TemporaryDirectory(prefix='magicanvas-preview-') as directory:
        path = Path(directory) / 'preview.png'
        if asset_mode == 'test' and fonts.suffix == '.pickle':
            warnings = render_design_to_png(
                design, assets_dir, path, fonts_file=fonts,
                max_size=max(design['canvas']['width'], design['canvas']['height']))
        else:
            warnings = render_design_to_png(
                design, assets_dir, path, fonts_dir=fonts,
                max_size=max(design['canvas']['width'], design['canvas']['height']))
        png_data = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    return png_data, warnings


def _adapt_preview_uncached(task, source_png=None, validated=None, asset_mode='demo', sample_id=None):
    # Keep one consistent module graph for the lifetime of the server. Partial
    # reloads mix new callers with stale dependent modules, and concurrent
    # requests can observe half-reloaded state. Restart after editing code.
    from engine.resize import _adapt_with_diagnostics, _text
    from engine.main import self_check

    source, target = validated or validate_task(task)
    original_source = deepcopy(source)
    assets_dir = _asset_directory(asset_mode, sample_id)
    source = _with_resolved_assets(source, assets_dir)
    started = perf_counter()
    # Render the original before starting the expensive search. The streaming
    # endpoint sends this preview to the browser immediately, then continues.
    if source_png is None:
        source_png, source_warnings = render_preview_data(source, assets_dir, asset_mode)
    else:
        source_warnings = []
    output, diagnostics = _adapt_with_diagnostics(source, target)
    preview_started = perf_counter()
    warnings = source_warnings + self_check(source, output)
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
            h, w, lines = _text(e)
            if h > e['height'] + .1 or w > e['width'] + .1 or (e.get('max_lines') and lines > e['max_lines']):
                warnings.append(f"{e['id']}: text may not fit at this resolution.")
            if e['font_size'] < 8:
                warnings.append(f"{e['id']}: {e['font_size']:.1f}px text is very small. Copy cannot be removed or rewritten under stage-one rules.")
    output_png, output_warnings = render_preview_data(output, assets_dir, asset_mode)
    warnings.extend(output_warnings)
    _restore_asset_ids(output, original_source)
    return dict(design=output, source=original_source, source_png=source_png, output_png=output_png,
                warnings=list(dict.fromkeys(warnings)), seconds=round(perf_counter()-started, 3),
                timings=dict(**diagnostics.get('seconds', {}), preview_render=round(perf_counter()-preview_started,3)),
                inference={k:diagnostics.get('qwen_diagnostics',{}).get('generation',{}).get(k) for k in ('device','dtype','generation_seconds','adapter_merged','runtime')})


from viewer.preview_cache import PreviewCache
_PREVIEW_CACHE=PreviewCache()


def _preview_key(source,target,asset_mode,sample_id):
    from hashlib import sha256
    from engine.resize import _asset_root
    from engine import qwen_seed
    files=[]
    def stamp(path):
        path=Path(path)
        try:
            stat=path.stat();files.append((str(path.resolve()),stat.st_mtime_ns,stat.st_size))
        except OSError:files.append((str(path),'missing'))
    assets=_asset_directory(asset_mode,sample_id)
    resolved=_with_resolved_assets(source,assets)
    for e in resolved['elements']:
        if e['type']=='image':stamp(_asset_root(e)/e['asset_id'])
    # A font/weight replacement invalidates the preview and requires a server
    # restart to reload the model. Code changes likewise require a restart.
    for base in (ROOT/'brand_kit/fonts',qwen_seed.MODEL_DIR,qwen_seed.ADAPTER_DIR):
        stamp(base)
        if base.is_dir():
            for path in sorted(base.rglob('*')):
                if path.is_file():stamp(path)
    stamp(DATA_ROOT/'crello_fonts.pickle');stamp(ROOT/'engine/data/layout_library.json')
    config={k:v for k,v in os.environ.items() if k.startswith('MAGICANVAS_')}
    return sha256(json.dumps([source,target,asset_mode,sample_id,files,config],sort_keys=True).encode()).hexdigest()


def adapt_preview(task,source_png=None,validated=None,asset_mode='demo',sample_id=None):
    source,target=validated or validate_task(task)
    compute=lambda:_adapt_preview_uncached(task,source_png,(source,target),asset_mode,sample_id)
    if os.environ.get('MAGICANVAS_PREVIEW_CACHE','1')=='0':return dict(compute(),cache_hit=False)
    started=perf_counter()
    response=_PREVIEW_CACHE.get_or_compute(_preview_key(source,target,asset_mode,sample_id),compute)
    response['computed_seconds']=response['seconds']
    response['request_seconds']=round(perf_counter()-started,3)
    if response['cache_hit']:
        response['seconds']=response['request_seconds']
        if source_png is not None:response['source_png']=source_png
    return response


def _sse_message(event, data):
    payload=json.dumps(data,separators=(',',':'))
    return f'event: {event}\ndata: {payload}\n\n'.encode()


def render_lightbox(design, assets_dir=None, asset_mode='demo'):
    """Render a detail-view copy with a 1080px minimum short side when safe."""
    render_design_to_png, fonts = _render_function(asset_mode)
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
        if asset_mode == 'test' and fonts.suffix == '.pickle':
            render_design_to_png(lightbox_design, assets_dir or ROOT/'brand_kit'/'assets', path,
                                 fonts_file=fonts, max_size=max_size)
        else:
            render_design_to_png(lightbox_design, assets_dir or ROOT/'brand_kit'/'assets', path,
                                 fonts_dir=fonts, max_size=max_size)
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
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/api/samples':
            mode = parse_qs(parsed.query).get('mode', ['demo'])[0]
            try:
                return self.reply(200, _sample_records(mode))
            except ValueError as exc:
                return self.reply(400, dict(error=str(exc)))
        files = {
            '/': ('index.html', 'text/html; charset=utf-8'),
            '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
            '/ai-image-loader.mjs': ('ai-image-loader.mjs', 'text/javascript; charset=utf-8'),
            '/chameleon-loader.mjs': ('chameleon-loader.mjs', 'text/javascript; charset=utf-8'),
            '/theme.mjs': ('theme.mjs', 'text/javascript; charset=utf-8'),
            '/style.css': ('style.css', 'text/css; charset=utf-8'),
        }
        if path in files:
            name, kind = files[path]
            return self.reply(200, (Path(__file__).parent/name).read_bytes(), kind)
        if path.endswith('.mjs') or path.endswith('.js') or path.endswith('.css'):
            target = (Path(__file__).parent / path.lstrip('/')).resolve()
            if target.is_relative_to(Path(__file__).parent.resolve()) and target.is_file():
                kind = 'text/css; charset=utf-8' if path.endswith('.css') else 'text/javascript; charset=utf-8'
                return self.reply(200, target.read_bytes(), kind)
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
                asset_mode = task.get('asset_mode', 'demo')
                assets_dir = _asset_directory(asset_mode, task.get('sample_id'))
                source = _with_resolved_assets(source, assets_dir)
                png, canvas = render_lightbox(source, assets_dir, asset_mode)
                return self.reply(200, png, 'image/png', {
                    'X-Image-Width': canvas['width'],
                    'X-Image-Height': canvas['height'],
                })
            if path == '/api/adapt-stream':
                source, target = validate_task(task)
                asset_mode = task.get('asset_mode', 'demo')
                sample_id = task.get('sample_id')
                assets_dir = _asset_directory(asset_mode, sample_id)
                resolved_source = _with_resolved_assets(source, assets_dir)
                source_png, source_warnings = render_preview_data(resolved_source, assets_dir, asset_mode)
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
                                           validated=(source,target), asset_mode=asset_mode,
                                           sample_id=sample_id)
                    result['warnings']=list(dict.fromkeys(source_warnings+result['warnings']))
                    self.wfile.write(_sse_message('adapted', result))
                    self.wfile.flush()
                except Exception as exc:
                    self.wfile.write(_sse_message('error', dict(error=f'Adaptation failed: {exc}')))
                    self.wfile.flush()
                self.close_connection=True
                return
            result = adapt_preview(task, asset_mode=task.get('asset_mode', 'demo'),
                                   sample_id=task.get('sample_id'))
        except (ValueError, TypeError, KeyError) as exc:
            return self.reply(400, dict(error=str(exc)))
        except ImportError as exc:
            return self.reply(503, dict(error=f'Real renderer unavailable: {exc}. Install requirements.txt in the active environment.'))
        except Exception as exc:
            return self.reply(500, dict(error=f'Preview failed: {exc}'))
        self.reply(200, result)


def main():
    qwen_local=Path(os.environ.get('MAGICANVAS_QWEN_DIR',ROOT/'models/qwen3-0.6b'))
    if (qwen_local/'config.json').is_file() and any(qwen_local.glob('*.safetensors')):
        os.environ.setdefault('MAGICANVAS_QWEN_ENABLED','1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5050)
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        try:
            with urlopen(f'http://127.0.0.1:{args.port}/', timeout=1) as response:
                is_preview = response.status == 200 and b'<title>MagiCanvas</title>' in response.read(8192)
        except Exception:
            is_preview = False
        if is_preview:
            print(f'MagiCanvas preview is already running: http://127.0.0.1:{args.port}', flush=True)
            return
        raise SystemExit(
            f'Port {args.port} is already in use by another service. '
            f'Close it or start the preview with --port {args.port + 1}.'
        ) from None
    print(f'MagiCanvas preview: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
