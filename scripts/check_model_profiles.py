#!/usr/bin/env python3
"""Verify the two local model/adapter packages without loading a generator.

Reports the environment settings used by the existing loader. Does not mutate
environment, execute inference, or claim that shape checks prove base identity.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect_profile(profile, root=ROOT):
    from scripts.verify_qwen_adapter import verify
    model = root / profile['model_dir']
    adapter = root / profile['adapter_dir']
    config = json.loads((model / 'config.json').read_text())
    adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
    if config.get('model_type') != profile['model_type'] or config.get('hidden_size') != profile['hidden_size']:
        raise ValueError('Profile and local base configuration differ')
    if adapter_config.get('base_model_name_or_path') != profile['training_base_alias']:
        raise ValueError('Profile and adapter training-base metadata differ')
    result = verify(adapter, model)
    if result['prompt_version'] != profile['prompt_version']:
        raise ValueError('Profile and adapter prompt contracts differ')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if not (model / name).is_file():
            raise ValueError(f'Missing offline tokenizer artifact: {name}')
    return dict(compatible_local_shape_checks=True, verification=result,
                model_config_sha256=sha256((model / 'config.json').read_bytes()).hexdigest(),
                tokenizer_sha256={name: sha256((model / name).read_bytes()).hexdigest()
                                  for name in ('tokenizer.json', 'tokenizer_config.json')},
                release_provenance_pinned=bool(profile.get('base_revision') and profile.get('adapter_revision')),
                environment=dict(MAGICANVAS_QWEN_ENABLED='1', MAGICANVAS_REQUIRE_QWEN_ADAPTER='1',
                                 MAGICANVAS_QWEN_DIR=str(model), MAGICANVAS_QWEN_ADAPTER_DIR=str(adapter),
                                 MAGICANVAS_QWEN_PROMPT_VERSION=str(profile['prompt_version'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/layout_model_profiles.json')
    parser.add_argument('--profile', help='Inspect a single named package; default: both')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if args.profile and args.profile not in spec['profiles']:
        parser.error('Unknown profile')
    profiles = {k: v for k, v in spec['profiles'].items() if not args.profile or k == args.profile}
    result = dict(production_changed=False, fresh_inference=False,
                  config_sha256=sha256(args.config.read_bytes()).hexdigest(), profiles={},
                  limitations=['Checks family, q_proj input width, tokenizer presence and declared training base; not full checkpoint identity or every adapter tensor.',
                               'Local release revisions are not pinned; pin original base, adapter and tokenizer provenance before final packaging.',
                               'Model changes take effect in a restarted process, not a hot swap of the cached loader.'])
    for key, profile in profiles.items():
        result['profiles'][key] = inspect_profile(profile)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
