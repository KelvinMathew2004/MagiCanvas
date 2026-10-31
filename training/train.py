"""Train the prescribed model queue on a prepared, hashed v2 bundle.

Run on a CUDA GPU, after installing requirements.txt. Mac CPU/MPS is reserved
for preparation and evaluation. No weights are uploaded by this program.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,math,random
import torch
from torch.utils.data import Dataset,Sampler
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
try:
    from training.providers import components,prompt_text,tokenize,language_linear_targets
except ModuleNotFoundError:
    from providers import components,prompt_text,tokenize,language_linear_targets


def encode(row,tokenizer,max_length=6144,processor=None,image=None):
    prompt=prompt_text(row['messages'][:-1],tokenizer,processor,image)
    before=tokenize(prompt,tokenizer,processor,image)
    prefix=before['input_ids'][0].tolist() if processor is not None else before['input_ids']
    answer=row['messages'][-1]['content'];json.loads(answer)
    full=tokenize(prompt+answer+tokenizer.eos_token,tokenizer,processor,image)
    tokens=full['input_ids'][0].tolist() if processor is not None else full['input_ids']
    if tokens[:len(prefix)]!=prefix:raise ValueError('Chat prefix token boundary mismatch')
    if len(tokens)>max_length:raise ValueError(f"{row['pair_id']}: {len(tokens)} tokens; no truncation permitted")
    if len(tokens)<=len(prefix) or tokens[-1]!=tokenizer.eos_token_id:raise ValueError('Missing response/EOS')
    result=dict(input_ids=tokens,attention_mask=[1]*len(tokens),
        labels=[-100]*len(prefix)+tokens[len(prefix):],quality_weight=float(row.get('quality_weight',1)))
    if processor is not None:result.update({k:v for k,v in full.items() if k not in ('input_ids','attention_mask')})
    return result


class Rows(Dataset):
    def __init__(self,rows,tokenizer,processor=None,bundle=None):
        self.tokenizer=tokenizer;self.processor=processor;self.bundle=bundle
        self.rows=[encode(r,tokenizer) for r in rows] if processor is None else rows
        self.repeats=[int(r.get('sampling_repeat',1)) for r in rows]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        if self.processor is None:return row
        from PIL import Image
        with Image.open(self.bundle/row['source_image']) as im:
            return encode(row,self.tokenizer,processor=self.processor,image=im.convert('RGB'))


class CoverageSampler(Sampler):
    """Every unique row once per epoch; additional dense visits are explicit."""
    def __init__(self,repeats,seed=3407):
        self.indices=[i for i,n in enumerate(repeats) for _ in range(n)]
        self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.indices)
    def __iter__(self):
        values=list(self.indices);random.Random(self.seed+self.epoch).shuffle(values);return iter(values)


def collate(items):
    if len(items)!=1:raise ValueError('This memory-bounded recipe requires microbatch size 1')
    return {k:v if isinstance(v,torch.Tensor) else torch.tensor([v],dtype=torch.float32 if k=='quality_weight' else torch.long)
            for k,v in items[0].items()}


def response_loss(model,inputs,num_items_in_batch=None):
    """Only materialize vocabulary logits for answer positions, not the prompt.

    Trainer passes the shifted response-token count across the accumulation
    window. Dividing by it avoids both double accumulation scaling and prompt
    loss. Validation uses its own response-token count. EOS remains supervised.
    """
    inputs=dict(inputs);labels=inputs.pop('labels');weight=inputs.pop('quality_weight')
    count=int((labels[0]!=-100).sum().item())
    if count<1 or bool((labels[0,-count:]==-100).any()):raise ValueError('Response must be a contiguous suffix')
    outputs=model(**inputs,logits_to_keep=count+1)
    logits=outputs.logits[:,:-1,:].float();target=labels[:,-count:]
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='sum')
    divisor=num_items_in_batch if num_items_in_batch is not None else count
    loss=loss*weight.reshape(())/divisor
    return loss,outputs


class ResponseTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=True
        self.loss_is_scaled_for_ga=True
        self._loss_shifts_labels=True
    def _get_train_sampler(self,train_dataset=None):
        dataset=train_dataset if train_dataset is not None else self.train_dataset
        return CoverageSampler(dataset.repeats,self.args.seed)
    @staticmethod
    def get_optimizer_cls_and_kwargs(args,model=None):
        cls,kwargs=Trainer.get_optimizer_cls_and_kwargs(args,model)
        # foreach=None may choose CUDA tensor lists and increase peak memory.
        if cls is torch.optim.AdamW:kwargs['foreach']=False
        return cls,kwargs
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        loss,outputs=response_loss(model,inputs,num_items_in_batch)
        return (loss,outputs) if return_outputs else loss


def selected_specs(manifest):
    queue=manifest.get('training_queue',[s['key'] for s in manifest['models']])
    if len(queue)!=len(set(queue)) or set(queue)-{s['key'] for s in manifest['models']}:
        raise ValueError('Invalid prescribed training queue')
    return [s for key in queue for s in manifest['models'] if s['key']==key]


def cuda_profile(total_memory,bf16_supported):
    """Capacity is per sequential job, never the number of prepared models.

    A T4 is allowed to attempt the unchanged recipe. Actual fit is established
    by the model-specific forward/backward/optimizer preflight, not this floor.
    """
    if total_memory<14*2**30:
        raise ValueError('Use a CUDA GPU with at least 14 GiB visible VRAM. A Colab T4 is supported for preflight; actual model fit is checked before training.')
    return dict(precision='bf16' if bf16_supported else 'fp16',
        total_memory_bytes=total_memory,minimum_memory_gib=14)


def preflight_training(model,dataset,records,bf16):
    """Exercise long inputs and long answers with real AdamW state allocated.

    lr=0 leaves the fresh adapter unchanged. Keep the optimizer across probes
    to include its state plus accumulated-gradient storage in the memory peak.
    No graph/output tensors survive this function (important on smaller GPUs).
    """
    from time import perf_counter
    parameters=[p for p in model.parameters() if p.requires_grad]
    # Long answers can cost more vocabulary-logit memory than a longer prompt.
    longest=max(range(len(records)),key=lambda i:records[i].get('token_length')
        or len(dataset[i]['input_ids']))
    longest_answer=max(range(len(records)),key=lambda i:len(dataset.tokenizer(
        records[i]['messages'][-1]['content'],add_special_tokens=False)['input_ids']))
    indices=list(dict.fromkeys([longest,longest_answer]))
    optimizer=torch.optim.AdamW(parameters,lr=0.,weight_decay=0.,foreach=False)
    probes=[];model.train();torch.cuda.reset_peak_memory_stats()
    for index in indices:
        optimizer.zero_grad(set_to_none=True)
        # Subsequent accumulation microsteps also retain existing gradients.
        for parameter in parameters:parameter.grad=torch.zeros_like(parameter)
        batch=collate([dataset[index]]);batch={k:v.to('cuda') for k,v in batch.items()}
        print('CUDA preflight row:',records[index]['pair_id'],int(batch['input_ids'].shape[-1]),'tokens; forward/backward/AdamW',flush=True)
        started=perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16 if bf16 else torch.float16):
            loss,outputs=response_loss(model,batch)
        del outputs
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite CUDA preflight loss')
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError('Nonfinite CUDA preflight gradients; do not start this model training on the current backend/precision.')
        optimizer.step();torch.cuda.synchronize()
        probes.append(dict(pair_id=records[index]['pair_id'],tokens=int(batch['input_ids'].shape[-1]),
            response_tokens=int((batch['labels']!=-100).sum()),loss=float(loss.detach()),seconds=perf_counter()-started))
        del batch,loss
    result=dict(probes=probes,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        note='Actual CUDA forward/backward plus zero-learning-rate AdamW step on long-input/long-answer rows, with existing gradient storage. This is a feasibility check, not a guarantee for every row or a quality result.')
    optimizer.zero_grad(set_to_none=True);del optimizer
    torch.cuda.empty_cache()
    return result


def train(bundle,output):
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required. Use the Colab GPU runtime; prepare/evaluate on your Mac.')
    if torch.cuda.device_count()!=1:raise SystemExit('Use one visible CUDA GPU for the controlled recipe.')
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    manifest=json.loads((bundle/'manifest.json').read_text())
    try:profile=cuda_profile(torch.cuda.get_device_properties(0).total_memory,
        torch.cuda.is_bf16_supported(including_emulation=False))
    except ValueError as exc:raise SystemExit(str(exc)) from exc
    for name,digest in manifest['bundle_hashes'].items():
        if sha256((bundle/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Bundle changed: {name}')
    chosen=selected_specs(manifest)
    print('GPU:',torch.cuda.get_device_name(0),profile,flush=True)
    print('Prescribed training queue:',[s['key'] for s in chosen],flush=True)
    for spec in chosen:
        name=spec['key'];dest=output/name
        if (dest/'final/adapter_model.safetensors').exists():
            saved=json.loads((dest/'final/training_manifest.json').read_text())
            if saved['bundle_sha256']!=sha256((bundle/'manifest.json').read_bytes()).hexdigest():raise ValueError('Existing adapter belongs to another data bundle')
            print('Already completed:',dest,flush=True);continue
        set_seed(3407);bf16=profile['precision']=='bf16'
        print('Preparing provider:',name,flush=True)
        tokenizer,processor=components(spec)
        if sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()!=spec['tokenizer_sha256']:
            raise ValueError('Tokenizer differs from prepared local tokenizer')
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        records={s:[json.loads(x) for x in (bundle/spec.get('dataset_dir','')/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation')}
        print('Encoding training/validation rows:',{s:len(rows) for s,rows in records.items()},flush=True)
        datasets={s:Rows(rows,tokenizer,processor,bundle) for s,rows in records.items()}
        from transformers import AutoModelForImageTextToText
        factory=AutoModelForImageTextToText if spec.get('kind')=='vision_language' else AutoModelForCausalLM
        print('Loading 4-bit base:',spec['repo'],flush=True)
        model=factory.from_pretrained(spec['repo'],revision=spec['revision'],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16),
            device_map={'':0},attn_implementation='sdpa',dtype=torch.bfloat16 if bf16 else torch.float16)
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False
        targets=language_linear_targets(model)
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',
            task_type='CAUSAL_LM',target_modules=targets))
        try:preflight=preflight_training(model,datasets['train'],records['train'],bf16)
        except torch.cuda.OutOfMemoryError as exc:
            raise SystemExit(f'{name} failed its GPU memory preflight. Completed earlier adapters remain saved. Restart the runtime to release memory; this model needs a larger GPU for the unchanged full-length recipe. No rows were truncated or models silently skipped.') from exc
        dest.mkdir(parents=True,exist_ok=True);(dest/'cuda_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
        print('CUDA feasibility preflight:',preflight,flush=True)
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();set_seed(3407)
        updates=math.ceil(sum(datasets['train'].repeats)/16)*3
        args=TrainingArguments(output_dir=str(dest/'checkpoints'),num_train_epochs=3,learning_rate=1e-4,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=16,
            bf16=bf16,fp16=not bf16,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',weight_decay=.01,
            warmup_steps=math.ceil(updates*.03),lr_scheduler_type='linear',max_grad_norm=1.,
            eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
            metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=2,
            prediction_loss_only=True,logging_steps=10,disable_tqdm=False,seed=3407,data_seed=3407,
            remove_unused_columns=False,dataloader_num_workers=0,report_to='none')
        trainer=ResponseTrainer(model=model,args=args,train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],data_collator=collate,processing_class=tokenizer)
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint=get_last_checkpoint(str(dest/'checkpoints')) if (dest/'checkpoints').exists() else None
        print('Starting epochs:',name,'resume checkpoint:',checkpoint or 'fresh adapter',flush=True)
        trainer.train(resume_from_checkpoint=checkpoint)
        final=dest/'final';trainer.save_model(str(final));tokenizer.save_pretrained(final)
        if processor is not None:processor.save_pretrained(final)
        (final/'magicanvas_prompt_version.txt').write_text('2\n')
        import subprocess
        (final/'environment.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
        (final/'training_manifest.json').write_text(json.dumps(dict(base=spec,
            bundle_sha256=sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
            recipe=manifest['recipe'],cuda_preflight=preflight,lora_targets=targets,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            validation=trainer.evaluate(),gpu=torch.cuda.get_device_name(0),
            cuda=torch.version.cuda,torch=torch.__version__,runtime_profile=profile),indent=2)+'\n')
        print('Saved local adapter:',final,flush=True)
        del trainer,model,datasets
        import gc
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();train(args.bundle.resolve(),args.output.resolve())
