#!/usr/bin/env python3
"""Audit the frozen VLM bundle locally without changing data or training weights."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Local preparation keeps optional vision dependencies outside the production venv.
sys.path.insert(0, str(ROOT / 'data/upgrade_exploration/deps'))
sys.path.insert(0, str(ROOT))


def audit(bundle, archive, base):
    import torch
    from PIL import Image
    from training.continue_remaining import verify_uniform_bundle
    from training.providers import components, language_linear_targets
    from training.train import collate, encode
    from scripts.evaluate_qwen_adapter import read_sources

    torch.set_num_threads(2)
    manifest = verify_uniform_bundle(bundle)
    # The launcher returns the validated manifest; keep this explicit for older callers.
    if manifest is None:
        manifest = json.loads((bundle / 'manifest.json').read_text())
    spec = next(s for s in manifest['models'] if s['kind'] == 'vision_language')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        expected = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()}
        assert set(z.namelist()) == expected
        for name in expected:
            assert z.read(name) == (bundle / name).read_bytes(), name
    for name in ('train.py', 'providers.py', 'requirements.txt', 'continue_remaining.py'):
        assert (bundle / name).read_bytes() == (ROOT / 'training' / name).read_bytes(), name

    tokenizer, processor = components(spec, base, local_files_only=True)
    assert processor is not None
    assert sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() == spec['tokenizer_sha256']
    get_source = read_sources()
    images, splits, families, endpoints = set(), {}, {}, {}
    for split in ('train', 'validation'):
        rows = [json.loads(x) for x in (bundle / spec['dataset_dir'] / (split + '.jsonl')).read_text().splitlines()]
        lengths, image_tokens, sources, targets = [], [], set(), set()
        families[split] = {r['family_id'] for r in rows}
        for i, row in enumerate(rows):
            assert row['quality_weight'] == 1 and row['sampling_repeat'] == 1
            assert row['model_split'] == split and row['prompt_version'] == 2
            source = get_source(row['source_split'], row['source_template_id'])
            key = sha256(json.dumps(dict(source=source, audit=manifest['audit_environment_sha256']), sort_keys=True).encode()).hexdigest()
            assert row['source_image'] == 'images/' + key + '.png', row['pair_id']
            images.add(row['source_image'])
            with Image.open(bundle / row['source_image']) as im:
                encoded = encode(row, tokenizer, processor=processor, image=im.convert('RGB'))
            batch = collate([encoded])
            assert batch['pixel_values'].ndim == 2 and batch['pixel_values'].numel() > 0
            assert batch['image_grid_thw'].shape == (1, 3)
            tokens = encoded['input_ids']
            labels = encoded['labels']
            start = next(j for j, label in enumerate(labels) if label != -100)
            assert all(label == -100 for label in labels[:start])
            assert labels[start:] == tokens[start:]
            answer = tokenizer(row['messages'][-1]['content'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
            assert labels[start:] == answer
            positions = [j for j, token in enumerate(tokens) if token == tokenizer.convert_tokens_to_ids('<|image_pad|>')]
            assert positions and all(labels[j] == -100 for j in positions)
            grid = batch['image_grid_thw'][0]
            assert len(positions) == int(grid.prod()) // processor.image_processor.merge_size**2
            assert len(tokens) == row['token_length']
            decoded = json.loads(row['messages'][-1]['content'])
            ids = [e['id'] for e in decoded['elements']]
            assert len(ids) == len(set(ids)) == row['n_elements']
            assert set(ids) == {f'e{j:03d}' for j in range(len(source['elements']))}
            lengths.append(len(tokens)); image_tokens.append(len(positions))
            sources.add(row['source_template_id'])
            targets.add(row.get('target_template_id') or row['pair_id'].split('__to__', 1)[1])
            if (i + 1) % 100 == 0:
                print(split, i + 1, '/', len(rows), flush=True)
        endpoints[split] = sources | targets
        if split == 'train':
            assert len(targets) == len(rows)
        splits[split] = dict(rows=len(rows), unique_targets=len(targets), maximum_tokens=max(lengths),
                             minimum_image_tokens=min(image_tokens), maximum_image_tokens=max(image_tokens),
                             all_source_images_and_response_masks_verified=True)
    assert not families['train'] & families['validation']
    assert not endpoints['train'] & endpoints['validation']
    test_rows = [json.loads(x) for x in (ROOT / 'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines()]
    test_endpoints = {value for r in test_rows for value in (
        r['source_template_id'], r.get('target_template_id') or r['pair_id'].split('__to__', 1)[1])}
    assert not (endpoints['train'] | endpoints['validation']) & test_endpoints

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, attn_implementation='sdpa')
    names = language_linear_targets(model)
    assert names and all('language_model' in n and '.visual.' not in n for n in names)
    projection_counts = {suffix: sum(n.endswith('.' + suffix) for n in names)
                         for suffix in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')}
    assert all(n == config.text_config.num_hidden_layers for n in projection_counts.values())
    assert len(names) == sum(projection_counts.values())
    old = subprocess.check_output(['git', 'show', 'ae0fe91:models/qwen3-layout-lora-v2/adapter_model.safetensors'], cwd=ROOT)
    assert old == (ROOT / 'models/qwen3-layout-lora-v2/adapter_model.safetensors').read_bytes()
    return dict(archive_sha256=sha256(archive.read_bytes()).hexdigest(), splits=splits, unique_source_pngs=len(images),
                source_image_hashes_recomputed=True, train_validation_and_test_endpoint_disjoint=True,
                processor_size=processor.image_processor.size, language_lora_projection_counts=projection_counts,
                champion_matches_git_weight_sha256=sha256(old).hexdigest(), production_changed=False,
                cuda_training_tested=False, quality_gain_proven=False,
                limitations=['CPU preprocessing and architecture checks do not prove CUDA fit or layout quality.',
                             'Vision is frozen and source-image resolution is capped at 262144 pixels for the T4 pilot.',
                             'Crello-only targets do not cover all native-shape or dense official-design distributions.',
                             'The exact original champion training dataset/run remains unrecorded.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'data/training_next/uniform_v1/bundle')
    parser.add_argument('--archive', type=Path, default=ROOT/'data/training_next/layout_training_bundle_uniform.zip')
    parser.add_argument('--base', type=Path, default=ROOT/'models/qwen3-vl-2b-instruct')
    parser.add_argument('--output', type=Path, default=ROOT/'data/training_next/vlm_full_confirmation.json')
    args = parser.parse_args()
    result = audit(args.bundle, args.archive, args.base)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
