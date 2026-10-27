#!/usr/bin/env python3
"""Fail-closed verification of a local, matching offline LoRA artifact."""
import argparse,json,re
from pathlib import Path
from hashlib import sha256


def verify(adapter,model):
    adapter=Path(adapter);model=Path(model)
    required=('adapter_config.json','adapter_model.safetensors','magicanvas_prompt_version.txt')
    for name in required:
        if not (adapter/name).is_file():raise ValueError(f'Missing required adapter artifact: {adapter/name}')
    cfg=json.loads((adapter/'adapter_config.json').read_text());base=json.loads((model/'config.json').read_text())
    if cfg.get('peft_type')!='LORA' or int(cfg.get('r',0))<=0:raise ValueError('Expected a LoRA adapter configuration')
    if (adapter/'magicanvas_prompt_version.txt').read_text().strip() not in ('1','2','3'):raise ValueError('Unsupported prompt contract')
    name=cfg.get('base_model_name_or_path','').lower();family=base.get('model_type','').lower()
    if family not in name:raise ValueError('Adapter and base model families differ')
    # LoRA matrices expose the true hidden width even when trained with an
    # Unsloth/4-bit alias rather than the deployed full-precision base name.
    from safetensors import safe_open
    checked=0
    with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        for key in f.keys():
            if key.endswith('q_proj.lora_A.weight'):
                if f.get_slice(key).get_shape()[1]!=base['hidden_size']:raise ValueError('Adapter hidden width does not match base')
                checked+=1
    if not checked:raise ValueError('No supported Qwen q_proj LoRA matrices found')
    return dict(adapter_sha256=sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),base_model_type=family,hidden_size=base['hidden_size'],prompt_version=int((adapter/'magicanvas_prompt_version.txt').read_text()),checked_q_projections=checked)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();result=verify(a.adapter,a.model)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
