#!/usr/bin/env python3
"""Version the screened corpus with unit weights, without editing old bundles."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.continue_remaining import REMAINING,POLICY,verify_uniform_bundle


def build(source,dest,archive):
    source=source.resolve();dest=dest.resolve();archive=archive.resolve()
    if dest==source or source in dest.parents:raise ValueError('Use a separate versioned destination')
    if dest.exists() or archive.exists():raise ValueError('Versioned bundle already exists; preserve it and choose a new destination')
    original=json.loads((source/'manifest.json').read_text())
    for name,digest in original['bundle_hashes'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha256((source/p).read_bytes()).hexdigest()!=digest:
            raise ValueError('Prepared source bundle changed: '+name)
    specs={s['key']:s for s in original['models']}
    manifest=deepcopy(original);manifest['models']=[deepcopy(specs[k]) for k in REMAINING]
    manifest['training_queue']=list(REMAINING)
    manifest['queue_policy']='VL-2B first, then 0.8B; one independent CUDA job at a time; completed 0.6B excluded'
    manifest['data_policy']=dict(name=POLICY,quality_weight=1.,sampling_repeat=1,
        screening='Retain the exact screened train/validation inventory; no excluded target restored',
        source_manifest_sha256=sha256((source/'manifest.json').read_bytes()).hexdigest())
    manifest['recipe'].update(dense_repeat=1,quality_weight=1.,sampling_repeat=1,
        recovery_checkpoint_steps=10,data_policy=POLICY)
    manifest['provider_preflight']={k:original['provider_preflight'][k] for k in REMAINING}
    copied_images=set();common_ids={};dest.mkdir(parents=True)
    for spec in manifest['models']:
        folder=dest/spec['dataset_dir'];folder.mkdir(parents=True)
        for split in ('train','validation'):
            rows=[json.loads(line) for line in (source/spec['dataset_dir']/(split+'.jsonl')).read_text().splitlines() if line.strip()]
            if len(rows)!=original['splits'][split]['rows']:raise ValueError('Screened inventory changed')
            ids=[r['pair_id'] for r in rows]
            if split in common_ids and ids!=common_ids[split]:raise ValueError('Provider inventories differ')
            common_ids[split]=ids
            for row in rows:
                row['quality_weight']=1.;row['sampling_repeat']=1
                image=row.get('source_image')
                if image:
                    if image not in original['bundle_hashes']:raise ValueError('Source image was not hashed')
                    path=dest/image;path.parent.mkdir(parents=True,exist_ok=True)
                    if image not in copied_images:shutil.copy2(source/image,path);copied_images.add(image)
            (folder/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for split,stats in manifest['splits'].items():
        stats.update(epoch_visits=stats['rows'],downweighted=0)
    manifest['source_pngs']=len(copied_images)
    # Preserve the modern, already-audited trainer and architecture providers.
    for name in ('train.py','providers.py','requirements.txt'):shutil.copy2(source/name,dest/name)
    shutil.copy2(ROOT/'training/continue_remaining.py',dest/'continue_remaining.py')
    manifest['bundle_hashes']={str(p.relative_to(dest)):sha256(p.read_bytes()).hexdigest()
        for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify_uniform_bundle(dest)
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest))
    return dict(bundle=str(archive),sha256=sha256(archive.read_bytes()).hexdigest(),
        bytes=archive.stat().st_size,models=list(REMAINING),splits=manifest['splits'],source_pngs=len(copied_images))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/training_next/bundle')
    p.add_argument('--dest',type=Path,default=ROOT/'data/training_next/uniform_v1/bundle')
    p.add_argument('--archive',type=Path,default=ROOT/'data/training_next/layout_training_bundle_uniform.zip')
    a=p.parse_args();report=build(a.source,a.dest,a.archive)
    (a.dest.parent/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
