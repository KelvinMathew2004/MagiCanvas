#!/usr/bin/env python3
"""Fresh, opt-in generators; repair/selection is replayed in a separate process.

New base models are explicitly UNTUNED baselines, never credited with the
existing adapters' training. Public comparisons display Original/Resized only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data/upgrade_exploration/deps'))


def generation_stop_ids(tokenizer, model):
    """Align grammar termination with composite-model and tokenizer metadata."""
    result=[]
    for value in (getattr(tokenizer,'eos_token_id',None),getattr(model.generation_config,'eos_token_id',None),
                  getattr(getattr(model.config,'text_config',None),'eos_token_id',None)):
        for token in value if isinstance(value,list) else [value]:
            if isinstance(token,int) and token not in result:result.append(token)
    if not result:raise ValueError('No declared EOS token for this provider')
    return result


def freeze_cases(output, indices):
    viewer = ROOT / 'data/gemini_review_qwen_projection_fixes_2026-10-25/review/regression_suite'
    rows = json.loads((viewer / 'manifest.json').read_text())
    cases = []
    for i in indices:
        row = rows[i]; audit = json.loads((viewer / row['audit']).read_text())
        cases.append(dict(pair_id=row['pair_id'], cohort=row['cohort'], source=audit['source'],
                          target=audit['target'], source_split=audit.get('source_split') or 'train',
                          source_image=str(viewer / row['source']),
                          audit_sha256=sha256((viewer / row['audit']).read_bytes()).hexdigest()))
    payload = json.dumps(cases, sort_keys=True, indent=2) + '\n'
    path = output / 'cases.json'
    if path.exists() and path.read_text() != payload:
        previous=json.loads(path.read_text())
        # A viewer refresh changes the enclosing output audit, even when the
        # actual source/prompt/image inputs are unchanged. Retain original
        # provenance rather than mislabeling an existing frozen input snapshot.
        keys=('pair_id','cohort','source','target','source_split','source_image')
        if len(previous)!=len(cases) or any(any(a[k]!=b[k] for k in keys) for a,b in zip(previous,cases)):
            raise ValueError('Pilot inputs changed; choose another output folder')
        return previous
    path.write_text(payload)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['qwen06', 'qwen17', 'coder15', 'qwen3508', 'qwen35', 'qwen_vl'], required=True)
    parser.add_argument('--grammar', action='store_true')
    parser.add_argument('--sample-seed',type=int,help='Opt-in seeded sampling; omission keeps greedy decoding')
    parser.add_argument('--image', action='store_true', help='Source image only; never a human resized target')
    parser.add_argument('--indices', nargs='+', type=int, default=[5, 14, 18, 49, 30, 40])
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    parser.add_argument('--attention', choices=['sdpa','eager'], default='sdpa',
                        help='Native PyTorch attention; eager is a diagnostic reference only')
    parser.add_argument('--case-seconds', type=float, default=180)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/upgrade_exploration/model_pilot_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = freeze_cases(args.output, args.indices)
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['MAGICANVAS_QWEN_ENABLED'] = '1'
    os.environ['MAGICANVAS_QWEN_DEVICE'] = args.device
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
    from engine.qwen_prompt import messages_for
    from engine.runtime import select_runtime, synchronize
    from experiments.layout_upgrade.contracts import decode, proposal_schema
    runtime = select_runtime(torch, 'qwen')
    configurations = {
        'qwen06': ('qwen3-0.6b', 'qwen3-layout-lora-v2', 'text'),
        'qwen17': ('qwen3-1.7b', 'qwen3-layout-lora-v2-expanded-1.7b', 'text'),
        'coder15': ('qwen2.5-coder-1.5b-instruct', None, 'text'),
        'qwen3508': ('qwen3.5-0.8b', None, 'text'),
        'qwen35': ('qwen3.5-2b', None, 'vl'),
        'qwen_vl': ('qwen3-vl-2b-instruct', None, 'vl')}
    directory, adapter, kind = configurations[args.arm]
    if args.image and kind != 'vl':
        parser.error('--image requires a multimodal provider')
    arm = args.arm + ('_image' if args.image else '_text') + ('_grammar' if args.grammar else '_free')
    if args.sample_seed is not None:arm+=f'_sample{args.sample_seed}'
    checkpoint = args.output / (arm + '.jsonl')
    done = {r['pair_id'] for r in map(json.loads, checkpoint.read_text().splitlines())} if checkpoint.exists() else set()
    if done == {c['pair_id'] for c in cases}:
        print('Complete checkpoint:', arm); return
    started = perf_counter(); path = ROOT / 'models' / directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        factory = AutoModelForImageTextToText if kind == 'vl' else AutoModelForCausalLM
        model = factory.from_pretrained(path, local_files_only=True, dtype=runtime.dtype,
                                        attn_implementation=args.attention).to(runtime.device).eval()
        if adapter:
            from scripts.verify_qwen_adapter import verify
            verify(ROOT / 'models' / adapter, path)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ROOT / 'models' / adapter, local_files_only=True)
            if runtime.merge_adapter:
                model = model.merge_and_unload(safe_merge=True)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True) if args.image else None
        stop_ids=generation_stop_ids(tokenizer,model)
        load_seconds = perf_counter() - started
    except Exception as exc:
        (args.output / (arm + '.load_error.json')).write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}',
            runtime=runtime.diagnostics(), model=str(path), tuned=bool(adapter)), indent=2) + '\n')
        raise
    code = [Path(__file__), ROOT / 'experiments/layout_upgrade/contracts.py', ROOT / 'engine/qwen_prompt.py']
    config = json.loads((path / 'config.json').read_text())
    manifest = dict(arm=arm, tuned=bool(adapter), model=str(path), adapter=adapter,
                    runtime=runtime.diagnostics(), image_conditioned=args.image, grammar=args.grammar,
                    attention=args.attention,
                    eos_token_ids=stop_ids,source_image_max_pixels=512*512 if args.image else None,
                    sampling=dict(seed=args.sample_seed,temperature=.7,top_p=.9) if args.sample_seed is not None else None,
                    context_limit=8192, case_time_limit=args.case_seconds, load_seconds=load_seconds,
                    model_config_sha256=sha256((path / 'config.json').read_bytes()).hexdigest(),
                    files_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in code},
                    cases_sha256=sha256((args.output / 'cases.json').read_bytes()).hexdigest(),
                    fresh_inference=True, production_changed=False)
    marker = args.output / (arm + '.manifest.json')
    if marker.exists():
        old = json.loads(marker.read_text())
        for key in ('files_sha256', 'cases_sha256', 'model_config_sha256', 'grammar', 'image_conditioned'):
            if old[key] != manifest[key]:
                raise ValueError('Experiment configuration changed; use a new output folder')
    else:
        marker.write_text(json.dumps(manifest, indent=2) + '\n')
    compiler = None
    if args.grammar:
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        vocabulary = model.get_output_embeddings().weight.shape[0]
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocabulary,stop_token_ids=stop_ids))
    for i, case in enumerate(cases):
        if case['pair_id'] in done: continue
        source, target = case['source'], case['target']; result = dict(case, arm=arm, tuned=bool(adapter))
        try:
            def encode(messages):
                if args.image:
                    from PIL import Image
                    messages = [dict(role='user', content=[dict(type='image', image=Image.open(case['source_image']).convert('RGB')),
                                  dict(type='text', text='\n'.join(m['content'] for m in messages))])]
                    template_options={'enable_thinking':False} if 'enable_thinking' in str(processor.chat_template) else {}
                    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                              return_dict=True, return_tensors='pt', **template_options,
                              processor_kwargs={'images_kwargs': {'size':{'longest_edge':512*512,'shortest_edge':256*256}}})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                return tokenizer(prompt, return_tensors='pt', truncation=False)
            messages = messages_for(source, target, version=2)
            inputs = encode(messages)
            if inputs['input_ids'].shape[1] > 4096 and len(source['elements']) > 16:
                messages = messages_for(source, target, version=2,
                    relation_token_counter=lambda s: len(tokenizer(s, add_special_tokens=False)['input_ids']),
                    relation_token_budget=900)
                inputs = encode(messages)
            tokens = inputs['input_ids'].shape[1]
            budget = min(4096, max(512, 96*len(source['elements'])+128), 8192-tokens)
            if budget < 128: raise ValueError('Insufficient remaining context')
            options = {}
            if args.sample_seed is not None:
                seed=(args.sample_seed+int(sha256(case['pair_id'].encode()).hexdigest()[:8],16))%(2**31)
                torch.manual_seed(seed)
                if runtime.device=='mps':torch.mps.manual_seed(seed)
                options.update(temperature=.7,top_p=.9)
            if compiler:
                compiled = compiler.compile_json_schema(json.dumps(proposal_schema(source)))
                options['logits_processor'] = [LogitsProcessor(compiled)]
            inputs = {k: v.to(runtime.device) if hasattr(v, 'to') else v for k,v in inputs.items()}
            synchronize(torch, runtime.device); start = perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=budget, max_time=args.case_seconds,
                                         do_sample=args.sample_seed is not None, use_cache=True, repetition_penalty=1.,
                                         pad_token_id=tokenizer.eos_token_id,eos_token_id=stop_ids, **options)
            synchronize(torch, runtime.device)
            generated = output[0, tokens:]; response = tokenizer.decode(generated, skip_special_tokens=True)
            elements, diagnostics = decode(response, source, target)
            result.update(response=response, qwen_model_elements=elements, raw_proposal_valid=diagnostics['exact_ids'],
                          raw_json_complete=diagnostics['json_complete'], diagnostics=diagnostics,
                          prompt_tokens=tokens, output_tokens=len(generated), token_budget=budget,
                          hit_token_limit=len(generated)>=budget, generation_seconds=perf_counter()-start,
                          peak_process_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          qwen_status='Fresh experimental proposal' if elements else 'Experimental model returned no decodable proposal')
        except Exception as exc:
            result.update(error=f'{type(exc).__name__}: {exc}', qwen_model_elements=[], raw_proposal_valid=False,
                          raw_json_complete=False, qwen_status='Experimental generation failed')
        with checkpoint.open('a') as stream:
            stream.write(json.dumps(result) + '\n'); stream.flush(); os.fsync(stream.fileno())
        print(arm, i+1, len(cases), 'raw exact:', result['raw_proposal_valid'],
              'seconds:', round(result.get('generation_seconds', 0), 2), result.get('error',''), flush=True)


if __name__ == '__main__': main()
