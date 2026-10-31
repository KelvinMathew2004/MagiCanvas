from types import SimpleNamespace
import pytest
from transformers import TrainerControl,TrainerState,TrainingArguments
from training.continue_remaining import RecoveryCheckpoint,configure,REMAINING


def test_uniform_output_identity_blocks_unlabelled_or_changed_checkpoints(tmp_path):
    import json
    from training.continue_remaining import claim_output
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'manifest.json').write_text('{}')
    output=tmp_path/'output';(output/REMAINING[0]/'checkpoints').mkdir(parents=True)
    with pytest.raises(ValueError,match='no uniform identity'):claim_output(bundle,output,REMAINING)
    fresh=tmp_path/'fresh';identity=claim_output(bundle,fresh,[REMAINING[0]])
    assert claim_output(bundle,fresh,[REMAINING[1]])==identity
    (bundle/'manifest.json').write_text('{"different":true}')
    with pytest.raises(ValueError,match='another bundle'):claim_output(bundle,fresh,REMAINING)


def test_uniform_export_changes_only_sampling_policy_and_retains_image_conditioning(tmp_path):
    import json
    from hashlib import sha256
    from zipfile import ZipFile
    from scripts.prepare_uniform_training_bundle import build
    from training.continue_remaining import verify_uniform_bundle,POLICY
    source=tmp_path/'source';source.mkdir();models=[]
    for name in REMAINING:
        folder=source/'datasets'/name;folder.mkdir(parents=True)
        models.append(dict(key=name,dataset_dir='datasets/'+name))
        for split in ('train','validation'):
            row=dict(pair_id=split,messages=[dict(role='assistant',content='{"elements":[]}')],
                sampling_repeat=3,quality_weight=.5)
            if name==REMAINING[0]:row['source_image']='images/source.png'
            (folder/(split+'.jsonl')).write_text(json.dumps(row)+'\n')
    (source/'images').mkdir();(source/'images/source.png').write_bytes(b'fixture-source-image')
    for name in ('train.py','providers.py','requirements.txt'):(source/name).write_text('unchanged-'+name)
    manifest=dict(models=models,splits={s:dict(rows=1,excluded_confirmed_invisible=79 if s=='train' else 0) for s in ('train','validation')},
        recipe=dict(dense_repeat=3),provider_preflight={name:dict(response_mask_verified=True) for name in REMAINING})
    manifest['bundle_hashes']={str(p.relative_to(source)):sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    (source/'manifest.json').write_text(json.dumps(manifest))
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    dest=tmp_path/'uniform';archive=tmp_path/'uniform.zip';report=build(source,dest,archive)
    assert {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}==before
    current=verify_uniform_bundle(dest)
    assert current['data_policy']['name']==POLICY and current['training_queue']==list(REMAINING)
    assert current['splits']['train']['epoch_visits']==1 and current['splits']['train']['excluded_confirmed_invisible']==79
    for spec in current['models']:
        row=json.loads((dest/spec['dataset_dir']/'train.jsonl').read_text())
        assert row['quality_weight']==1 and row['sampling_repeat']==1
        assert row['messages'][0]['content']=='{"elements":[]}'
    assert (dest/'train.py').read_bytes()==before['train.py']
    with ZipFile(archive) as z:assert z.read('images/source.png')==b'fixture-source-image'
    with pytest.raises(ValueError,match='already exists'):build(source,dest,archive)
    (dest/'datasets'/REMAINING[0]/'train.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):verify_uniform_bundle(dest)


def test_save_callback_adds_recovery_without_requesting_validation(tmp_path):
    callback=RecoveryCheckpoint(10);args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,report_to='none')
    for step in (0,1,9,10,11,20):
        control=callback.on_step_end(args,TrainerState(global_step=step),TrainerControl())
        assert control.should_save==(step>0 and step%10==0)
        assert not control.should_evaluate
    control=TrainerControl(should_save=True,should_evaluate=True)
    assert callback.on_step_end(args,TrainerState(global_step=11),control).should_save
    assert control.should_evaluate
    with pytest.raises(ValueError):RecoveryCheckpoint(0)


def test_queue_excludes_completed_model_and_preserves_bundle():
    class Trainer:
        def __init__(self):self.callbacks=[]
        def add_callback(self,cb):self.callbacks.append(cb)
    manifest=dict(training_queue=['qwen3-0.6b',*REMAINING],models=[dict(key=k) for k in ('qwen3-0.6b',*REMAINING)])
    before=repr(manifest)
    module=SimpleNamespace(selected_specs=lambda m:m['models'],ResponseTrainer=Trainer)
    configure(module)
    assert [s['key'] for s in module.selected_specs(manifest)]==list(REMAINING)
    assert repr(manifest)==before
    assert isinstance(module.ResponseTrainer().callbacks[0],RecoveryCheckpoint)


def test_continuation_can_select_one_challenger_and_fails_closed():
    module=SimpleNamespace(selected_specs=lambda m:m['models'],ResponseTrainer=type('Trainer',(),{}))
    with pytest.raises(ValueError):configure(module,['qwen3-0.6b'])
    configure(module,[REMAINING[1]])
    assert module.selected_specs({'models':[{'key':REMAINING[1]}]})==[{'key':REMAINING[1]}]
    with pytest.raises(ValueError):module.selected_specs({'models':[]})


def test_real_trainer_keeps_epoch_best_selection_and_resumes_recovery_checkpoint(tmp_path):
    import json,torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from training.train import ResponseTrainer,collate
    torch.set_num_threads(1)
    class Rows(torch.utils.data.Dataset):
        repeats=[1]*4
        def __len__(self):return 4
        def __getitem__(self,i):
            return dict(input_ids=[1,2,3,4,5],attention_mask=[1]*5,
                        labels=[-100,-100,3,4,5],quality_weight=1.)
    def model():
        torch.manual_seed(3407)
        return Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32))
    module=SimpleNamespace(selected_specs=lambda m:m['models'],ResponseTrainer=ResponseTrainer)
    configure(module,every=3)
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,report_to='none',
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=1,
        num_train_epochs=2,save_strategy='epoch',eval_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=10,
        remove_unused_columns=False,disable_tqdm=True,seed=3407)
    trainer=module.ResponseTrainer(model=model(),args=args,train_dataset=Rows(),eval_dataset=Rows(),data_collator=collate)
    trainer.train()
    for step in (3,4,6,8):
        cp=tmp_path/f'checkpoint-{step}'
        assert (cp/'optimizer.pt').is_file() and (cp/'rng_state.pth').is_file()
    assert [r['step'] for r in trainer.state.log_history if 'eval_loss' in r]==[4,8]
    assert trainer.state.best_model_checkpoint and trainer.state.best_global_step in (4,8)
    resume=module.ResponseTrainer(model=model(),args=args,train_dataset=Rows(),eval_dataset=Rows(),data_collator=collate)
    resume.train(resume_from_checkpoint=str(tmp_path/'checkpoint-6'))
    assert resume.state.global_step==8
    saved=json.loads((tmp_path/'checkpoint-8/trainer_state.json').read_text())
    assert saved['global_step']==8 and saved['best_model_checkpoint']
