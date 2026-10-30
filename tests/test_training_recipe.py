"""Response-only supervision and accumulation correctness, without CUDA/downloads."""
from collections import Counter
from copy import deepcopy
import json
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer
from training.train import CoverageSampler,collate,encode,response_loss,ResponseTrainer


def tiny_model():
    torch.manual_seed(42)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        attention_dropout=0.,max_position_embeddings=32)).eval()


def item(tokens,answer=3,weight=1.):
    return dict(input_ids=tokens,attention_mask=[1]*len(tokens),
        labels=[-100]*(len(tokens)-answer)+tokens[-answer:],quality_weight=weight)


def test_response_logits_match_full_loss_and_gradients():
    a=tiny_model();b=deepcopy(a);inputs=collate([item([1,2,3,4,5,6])])
    loss,_=response_loss(a,inputs);loss.backward()
    expected=b(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=inputs['labels']).loss
    expected.backward();assert torch.allclose(loss,expected,atol=1e-6)
    for x,y in zip(a.parameters(),b.parameters()):assert torch.allclose(x.grad,y.grad,atol=2e-6)


def test_accumulation_is_normalized_once_and_quality_weight_is_explicit():
    model=tiny_model();first=collate([item([1,2,3,4,5],2,.5)]);second=collate([item([2,3,4,5,6,7],3)])
    loss1,_=response_loss(model,first,5);loss2,_=response_loss(model,second,5)
    normal1,_=response_loss(model,first);normal2,_=response_loss(model,second)
    assert torch.allclose(loss1+loss2,(normal1*2+normal2*3)/5)


def test_dense_sampler_covers_every_unique_row_and_changes_order_per_epoch():
    sampler=CoverageSampler([1,3,1,3]);a=list(sampler);assert Counter(a)=={0:1,1:3,2:1,3:3}
    sampler.set_epoch(1);assert list(sampler)!=a
    assert len(sampler)==8


def test_collator_keeps_eos_label_and_disallows_unaccounted_padding():
    row=item([1,2,3,4]);batch=collate([row]);assert batch['labels'][0,-1]==4
    with pytest.raises(ValueError):collate([row,row])


def test_local_tokenizer_exact_prefix_no_thinking_and_length_rejection():
    from pathlib import Path
    folder=Path('models/qwen3-0.6b')
    if not folder.exists():pytest.skip('Local tokenizer not installed')
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    row=dict(pair_id='test',messages=[dict(role='user',content='Return JSON.'),dict(role='assistant',content='{"elements":[]}')])
    out=encode(row,tokenizer);first=next(i for i,v in enumerate(out['labels']) if v!=-100)
    assert out['labels'][-1]==tokenizer.eos_token_id
    assert tokenizer.decode(out['input_ids'][first:]).startswith('{"elements"')
    assert all(v==-100 for v in out['labels'][:first])
    with pytest.raises(ValueError):encode(row,tokenizer,max_length=2)


def test_prepared_gold_preserves_source_styles_and_maps_aliases():
    from scripts.prepare_training_run import supervised_target
    source=dict(canvas=dict(width=100,height=100),elements=[dict(id='title',type='text',content='Keep all copy',
        font_size=20,letter_spacing=2,font_family='Solstice Sans',x=0,y=0,width=40,height=30,color='#abcdef')])
    row=dict(target_canvas=dict(width=200,height=100),messages=[dict(content=json.dumps(dict(elements=[
        dict(id='e000',box=[100,200,500,300],font_scale=150)])))])
    before=deepcopy(source);out=supervised_target(row,source)
    assert source==before and out['elements'][0]['content']=='Keep all copy'
    assert out['elements'][0]['color']=='#abcdef' and out['elements'][0]['font_size']==15
    assert out['elements'][0]['letter_spacing']==1.5
    assert out['elements'][0]['x']==20


def test_trainer_explicit_accumulation_contract(tmp_path):
    from transformers import TrainingArguments
    model=tiny_model()
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,report_to='none')
    trainer=ResponseTrainer(model=model,args=args)
    assert trainer.loss_is_scaled_for_ga is True and trainer.model_accepts_loss_kwargs is True
    assert trainer._loss_shifts_labels is True


def test_t4_can_attempt_unchanged_sequential_recipe_without_bf16():
    from training.train import cuda_profile
    profile=cuda_profile(int(14.7*2**30),False)
    assert profile['precision']=='fp16'
    assert cuda_profile(24*2**30,True)['precision']=='bf16'
    with pytest.raises(ValueError,match='14 GiB'):cuda_profile(8*2**30,False)


def test_training_adamw_avoids_tensor_list_memory_peak(tmp_path):
    from transformers import TrainingArguments
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,report_to='none',optim='adamw_torch')
    cls,kwargs=ResponseTrainer.get_optimizer_cls_and_kwargs(args)
    assert cls is torch.optim.AdamW and kwargs['foreach'] is False


def test_zero_rate_preflight_optimizer_does_not_modify_adapter():
    model=tiny_model();before=[p.detach().clone() for p in model.parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.,foreach=False)
    loss,_=response_loss(model,collate([item([1,2,3,4,5,6])]))
    loss.backward();optimizer.step()
    assert optimizer.state
    assert all(torch.equal(p,original) for p,original in zip(model.parameters(),before))


def test_colab_launcher_streams_both_channels_and_propagates_failure(tmp_path,capsys):
    import ast,re,subprocess,sys
    from pathlib import Path
    blocks=re.findall(r'```python\n(.*?)```',Path('docs/TRAINING_RUNBOOK.md').read_text(),re.S)
    launch=next(block for block in blocks if 'def stream_training' in block)
    tree=ast.parse(launch)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='stream_training')
    namespace={'subprocess':subprocess}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'colab_stream','exec'),namespace)
    log=tmp_path/'console.log'
    namespace['stream_training']([sys.executable,'-u','-c',
        'import sys; print("loading",flush=True); print("progress",file=sys.stderr,flush=True)'],log)
    assert 'loading' in capsys.readouterr().out
    assert 'loading' in log.read_text() and 'progress' in log.read_text()
    with pytest.raises(subprocess.CalledProcessError):
        namespace['stream_training']([sys.executable,'-u','-c','raise SystemExit(7)'],log)


def test_real_trainer_accumulation_matches_manual_token_normalization(tmp_path):
    from transformers import TrainingArguments
    model=tiny_model();manual=deepcopy(model)
    data=[item([1,2,3,4,5],2,.5),item([2,3,4,5,6,7],3)]
    class TinyRows(torch.utils.data.Dataset):
        repeats=[1,1]
        def __len__(self):return 2
        def __getitem__(self,i):return data[i]
    optimizer=torch.optim.SGD(manual.parameters(),lr=.01)
    for r in data:
        loss,_=response_loss(manual,collate([r]),5);loss.backward()
    optimizer.step()
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,report_to='none',
        per_device_train_batch_size=1,gradient_accumulation_steps=2,max_steps=1,
        learning_rate=.01,optim='sgd',weight_decay=0.,max_grad_norm=1000.,
        remove_unused_columns=False,save_strategy='no',disable_tqdm=True,lr_scheduler_type='constant')
    trainer=ResponseTrainer(model=model,args=args,train_dataset=TinyRows(),data_collator=collate)
    trainer.train()
    for a,b in zip(model.parameters(),manual.parameters()):assert torch.allclose(a,b,atol=2e-6)


def test_promotion_cannot_hide_a_new_failure_behind_average_improvement():
    from scripts.evaluate_trained_layouts import promotion
    def row(key):return dict(pair_id=key,source_design={'elements':[{}]*18},raw_proposal_valid=True,
        raw_geometry_mae=.1,raw_relative_font_mae=.2,geometry_mae=.05,font_mae=.2,
        exact_ids=True,valid=True,flagged=False,defect_flags=[],qwen_family_win=False)
    old=[row('reference:one'),row('review:two')];new=deepcopy(old)
    for r in new:r.update(raw_geometry_mae=.04,raw_relative_font_mae=.1,geometry_mae=.04,font_mae=.1,qwen_family_win=True)
    assert promotion(old,new)['eligible_for_human_review']
    new[1].update(flagged=True,defect_flags=['glyph_overflow'])
    result=promotion(old,new)
    assert not result['eligible_for_human_review'] and not result['checks']['no_new_flag_types']


def test_review_filter_ignores_only_numeric_roundoff_and_preserves_real_edits():
    from scripts.refresh_regression_viewer import documents_differ
    a={'elements':[{'y':1124.4861441508592,'content':'Unchanged'}]}
    b=deepcopy(a);b['elements'][0]['y']=1124.4861441508588
    assert not documents_differ(a,b)
    b['elements'][0]['y']+=.01;assert documents_differ(a,b)
    b=deepcopy(a);b['elements'][0]['content']='Changed';assert documents_differ(a,b)


def test_training_audit_snapshot_invalidates_changed_assets_and_fonts(tmp_path,monkeypatch):
    import scripts.prepare_training_run as prep
    monkeypatch.setattr(prep,'ROOT',tmp_path)
    monkeypatch.setattr(prep,'__file__',str(tmp_path/'scripts/prepare_training_run.py'))
    monkeypatch.setattr(prep,'sources',lambda:lambda split,tid:{'elements':[{'type':'image','asset_id':'asset.png'}]})
    root=tmp_path/'data/crello_train';root.mkdir(parents=True)
    asset=root/'asset.png';asset.write_bytes(b'first')
    font=tmp_path/'data/crello_fonts.pickle';font.write_bytes(b'font-one')
    rows=[dict(source_split='train',source_template_id='test')]
    first=prep.audit_environment(rows);asset.write_bytes(b'second')
    second=prep.audit_environment(rows);assert first!=second
    font.write_bytes(b'font-two');assert second!=prep.audit_environment(rows)


def test_hybrid_attention_lora_backward_and_complete_language_projection_coverage():
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    from peft import LoraConfig,get_peft_model
    from training.providers import language_linear_targets
    config=Qwen3_5TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        layer_types=['linear_attention']*3+['full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
    model=Qwen3_5ForCausalLM(config)
    targets=language_linear_targets(model)
    assert any(n.endswith('in_proj_qkv') for n in targets) and not any(n.endswith('lm_head') for n in targets)
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,task_type='CAUSAL_LM',target_modules=targets))
    loss,_=response_loss(model,collate([item([1,2,3,4,5,6])]))
    loss.backward();assert torch.isfinite(loss)
    assert any(p.grad is not None and bool((p.grad!=0).any()) for n,p in model.named_parameters() if 'lora_B' in n)


def test_multimodal_collator_preserves_patch_grid_dimensions():
    row=item([1,2,3,4]);row.update(pixel_values=torch.ones(4,24),image_grid_thw=torch.tensor([[1,2,2]]),
        mm_token_type_ids=torch.tensor([[0,1,0,0]]))
    out=collate([row]);assert out['pixel_values'].shape==(4,24)
    assert out['image_grid_thw'].shape==(1,3) and out['input_ids'].shape==(1,4)


def test_image_conditioned_lora_backward_keeps_vision_frozen():
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    from training.providers import language_linear_targets
    config=Qwen3VLConfig(text_config=dict(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'mrope_section':[1,1,2],'partial_rotary_factor':1.}),
        vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,out_hidden_size=16,
            patch_size=2,temporal_patch_size=2,spatial_merge_size=2,deepstack_visual_indexes=[]),
        image_token_id=3,vision_start_token_id=4,vision_end_token_id=5,video_token_id=9)
    model=Qwen3VLForConditionalGeneration(config);targets=language_linear_targets(model)
    assert all('.visual.' not in n for n in targets)
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,task_type='CAUSAL_LM',target_modules=targets))
    inputs=dict(input_ids=torch.tensor([[1,4,3,5,6,7,8]]),attention_mask=torch.ones(1,7,dtype=torch.long),
        labels=torch.tensor([[-100]*4+[6,7,8]]),quality_weight=torch.tensor([1.]),
        pixel_values=torch.rand(4,24),image_grid_thw=torch.tensor([[1,2,2]]),
        mm_token_type_ids=torch.tensor([[0,0,1,0,0,0,0]]))
    model.eval()
    loss,outputs=response_loss(model,inputs)
    native={k:v for k,v in inputs.items() if k!='quality_weight'}
    expected=model(**native).loss
    assert torch.allclose(loss,expected,atol=1e-6)
    # Identical text with different pixels must change the answer logits.
    alternate={k:v for k,v in inputs.items() if k not in ('quality_weight','labels')}
    alternate['pixel_values']=torch.zeros_like(alternate['pixel_values'])
    with torch.no_grad():alternate_logits=model(**alternate,logits_to_keep=4).logits
    assert not torch.equal(outputs.logits.detach(),alternate_logits)
    loss.backward();assert torch.isfinite(loss)
    assert not any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n)
    assert any(p.grad is not None and bool((p.grad!=0).any()) for n,p in model.named_parameters() if 'lora_B' in n)


def test_research_reserves_do_not_automatically_consume_training_time():
    from pathlib import Path
    from training.train import selected_specs
    manifest=json.loads(Path('config/training_candidates.json').read_text())
    assert len(manifest['models'])==6
    assert [s['key'] for s in selected_specs(manifest)]==['qwen3-0.6b','qwen3.5-0.8b','qwen3-vl-2b-instruct']
    invalid=deepcopy(manifest);invalid['training_queue'].append('unknown')
    with pytest.raises(ValueError):selected_specs(invalid)


def test_identity_bypass_is_retained_but_not_miscounted_as_a_dense_generation_failure():
    from scripts.evaluate_trained_layouts import promotion
    def row(key):return dict(pair_id=key,source_design={'elements':[{}]*18,'canvas':{'width':100,'height':100}},
        target_canvas={'width':200,'height':100},raw_proposal_valid=True,raw_geometry_mae=.1,
        raw_relative_font_mae=.2,geometry_mae=.05,font_mae=.2,exact_ids=True,valid=True,
        flagged=False,defect_flags=[],qwen_family_win=False)
    old=[row('reference:a'),row('review:b')];identity=row('review:identity')
    identity.update(target_canvas={'width':100,'height':100},raw_proposal_valid=False,
        raw_geometry_mae=None,raw_relative_font_mae=None)
    old.append(identity);new=deepcopy(old)
    for r in new[:2]:r.update(raw_geometry_mae=.04,raw_relative_font_mae=.1,geometry_mae=.04,font_mae=.1,qwen_family_win=True)
    result=promotion(old,new)
    assert result['dense_tasks']==2 and result['eligible_for_human_review']
