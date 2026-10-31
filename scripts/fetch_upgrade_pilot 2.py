#!/usr/bin/env python3
"""Download pinned, native-loader pilot weights; never activate a new model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/upgrade_pilot.json')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    rows = json.loads(args.config.read_text())['models']
    report = ROOT / 'data/upgrade_exploration/downloads.json'
    results = json.loads(report.read_text()) if report.exists() else {}
    for row in rows:
        key = row['model_id']; target = ROOT / row['directory']
        if results.get(key, {}).get('revision') == row['revision'] and results[key].get('complete') and target.is_dir():
            print('Already downloaded:', key, flush=True); continue
        if shutil.disk_usage(ROOT).free < 18 * 2**30:
            raise RuntimeError('Less than 18 GiB free; preserve disk headroom before another model download')
        print('Downloading pinned model:', key, row['revision'], flush=True)
        result = dict(row, complete=False, started=datetime.now(timezone.utc).isoformat())
        results[key] = result
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=2) + '\n')
        try:
            snapshot_download(repo_id=key, revision=row['revision'], local_dir=target, max_workers=2,
                              allow_patterns=['*.safetensors', '*.safetensors.index.json',
                                              'config.json', 'generation_config.json', 'tokenizer*',
                                              'vocab.json', 'merges.txt', 'special_tokens_map.json',
                                              '*processor_config.json', 'chat_template*', 'LICENSE*', 'README.md'])
            result.update(complete=True, bytes=sum(p.stat().st_size for p in target.glob('*.safetensors')),
                          completed=datetime.now(timezone.utc).isoformat())
            print('Ready:', key, result['bytes'], 'weight bytes', flush=True)
        except Exception as exc:
            result['error'] = str(exc)
            print('Download failed:', key, str(exc), flush=True)
        report.write_text(json.dumps(results, indent=2) + '\n')
    if not all(results.get(r['model_id'], {}).get('complete') for r in rows):
        sys.exit(1)


if __name__ == '__main__':
    main()
