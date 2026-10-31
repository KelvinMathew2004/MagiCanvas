"""Run the two challengers from a versioned uniform-data bundle."""
from pathlib import Path
from hashlib import sha256
import argparse,importlib.util,json,sys
from transformers import TrainerCallback

REMAINING=('qwen3-vl-2b-instruct','qwen3.5-0.8b')
POLICY='audited-uniform-v1'


def verify_uniform_bundle(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest.get('data_policy',{}).get('name')!=POLICY:
        raise ValueError('Use layout_training_bundle_uniform.zip; weighted bundles cannot use this launcher')
    for name,digest in manifest['bundle_hashes'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha256((bundle/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Uniform bundle changed: '+name)
    specs={s['key']:s for s in manifest['models']}
    for model in REMAINING:
        spec=specs[model]
        for split in ('train','validation'):
            name=str(Path(spec['dataset_dir'])/(split+'.jsonl'))
            if name not in manifest['bundle_hashes']:raise ValueError('Unhashed training data: '+name)
            rows=[json.loads(line) for line in (bundle/name).read_text().splitlines() if line.strip()]
            if len(rows)!=manifest['splits'][split]['rows']:raise ValueError('Uniform split row count differs')
            if any(r.get('sampling_repeat')!=1 or r.get('quality_weight')!=1. for r in rows):
                raise ValueError('Expected uniform unit weights and one visit per row')
    return manifest


def claim_output(bundle,output,names):
    """Refuse unlabelled/weighted checkpoints before any GPU work starts."""
    identity=dict(data_policy=POLICY,bundle_sha256=sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
        launcher_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    marker=output/'uniform_run_identity.json'
    if marker.is_file():
        if json.loads(marker.read_text())!=identity:raise ValueError('Output belongs to another bundle/launcher; use a new output folder')
    else:
        if any((output/name).exists() for name in REMAINING):
            raise ValueError('Existing model folders have no uniform identity; never resume a weighted checkpoint here')
        output.mkdir(parents=True,exist_ok=True)
        with marker.open('x') as stream:stream.write(json.dumps(identity,indent=2)+'\n')
    for name in names:
        final=output/name/'final/training_manifest.json'
        if final.is_file() and json.loads(final.read_text()).get('bundle_sha256')!=identity['bundle_sha256']:
            raise ValueError('Completed adapter belongs to another bundle')
    return identity


class RecoveryCheckpoint(TrainerCallback):
    """Add restart checkpoints without changing epoch validation/best selection."""
    def __init__(self,every=10):
        if every<1:raise ValueError('Checkpoint interval must be positive')
        self.every=every

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step>0 and state.global_step%self.every==0:
            control.should_save=True
        return control


def configure(module,names=REMAINING,every=10):
    """Patch this child process only; never rewrite the hashed bundle."""
    selected=module.selected_specs;trainer_class=module.ResponseTrainer
    wanted=tuple(names)
    if not wanted or len(set(wanted))!=len(wanted) or set(wanted)-set(REMAINING):
        raise ValueError('Choose one or both remaining challengers; the completed 0.6B is excluded')

    def remaining_specs(manifest):
        available={s['key']:s for s in selected(manifest)}
        if set(wanted)-available.keys():raise ValueError('Requested challenger is absent from this bundle queue')
        return [available[k] for k in wanted]

    class RestartSafeTrainer(trainer_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.add_callback(RecoveryCheckpoint(every))

    module.selected_specs=remaining_specs
    module.ResponseTrainer=RestartSafeTrainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('/content/layout_training_uniform'))
    p.add_argument('--output',type=Path,default=Path('/content/drive/MyDrive/MagiCanvas/uniform_v2_candidates'))
    p.add_argument('--models',nargs='+',choices=REMAINING,default=list(REMAINING))
    p.add_argument('--save-every',type=int,default=10)
    a=p.parse_args();bundle=a.bundle.resolve()
    if a.save_every<1:p.error('--save-every must be positive')
    if not (bundle/'manifest.json').is_file() or not (bundle/'train.py').is_file():
        p.error('Restore layout_training_bundle_uniform.zip to --bundle before continuing')
    verify_uniform_bundle(bundle)
    claim_output(bundle,a.output.resolve(),a.models)
    sys.path.insert(0,str(bundle))
    spec=importlib.util.spec_from_file_location('magicanvas_original_training',bundle/'train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    configure(module,a.models,a.save_every)
    print('Uniform challengers:',a.models,'recovery checkpoint every',a.save_every,'optimizer steps; epoch validation unchanged',flush=True)
    module.train(bundle,a.output.resolve())


if __name__=='__main__':main()
