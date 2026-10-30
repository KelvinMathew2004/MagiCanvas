"""Experimental package swap for controlled evaluation; production loader unchanged."""
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
from copy import deepcopy
import json,os


def verify_candidate(adapter,base,spec):
    from safetensors import safe_open
    saved=json.loads((adapter/'training_manifest.json').read_text())
    if any(saved['base'][k]!=spec[k] for k in ('key','repo','revision','kind')):
        raise ValueError('Candidate package/model provenance mismatch')
    if (adapter/'magicanvas_prompt_version.txt').read_text().strip()!='2':raise ValueError('Prompt contract mismatch')
    config=json.loads((base/'config.json').read_text());text=config.get('text_config',config)
    hidden=text['hidden_size'];checked=0
    with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as file:
        for key in file.keys():
            if key.endswith('q_proj.lora_A.weight'):
                if file.get_slice(key).get_shape()[1]!=hidden:raise ValueError('Adapter/base hidden width mismatch')
                checked+=1
    if not checked:raise ValueError('Missing expected language attention LoRA matrices')
    return dict(language_q_projections=checked,hidden_size=hidden,kind=spec['kind'])


class ImageAwareTokenizer:
    def __init__(self,tokenizer,processor):
        self.tokenizer=tokenizer;self.processor=processor;self.image=None
    def __getattr__(self,key):return getattr(self.tokenizer,key)
    def apply_chat_template(self,messages,**kwargs):
        from training.providers import prompt_text
        return prompt_text(messages,self.tokenizer,self.processor,self.image)
    def __call__(self,text,**kwargs):
        if self.processor is not None and '<|vision_start|>' in text:
            from training.providers import tokenize
            return tokenize(text,self.tokenizer,self.processor,self.image,return_tensors=kwargs.get('return_tensors'))
        return self.tokenizer(text,**kwargs)


@contextmanager
def candidate_provider(spec,base,adapter,cache):
    """Single-request-at-a-time evaluation of the normal engine with a new provider."""
    import torch
    import engine.qwen_seed as seed
    import engine.resize as resize
    from unittest.mock import patch
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText
    from peft import PeftModel
    from training.providers import components
    from engine.runtime import select_runtime
    runtime=select_runtime(torch,'qwen');tok,processor=components(spec,base,local_files_only=True)
    wrapper=ImageAwareTokenizer(tok,processor)
    @lru_cache(maxsize=1)
    def load():
        factory=AutoModelForImageTextToText if processor is not None else AutoModelForCausalLM
        model=factory.from_pretrained(base,local_files_only=True,dtype=runtime.dtype,attn_implementation='sdpa')
        model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False)
        if runtime.merge_adapter:model=model.merge_and_unload(safe_merge=True)
        model=model.to(runtime.device).eval();model._magicanvas_runtime=runtime.diagnostics()
        model._magicanvas_adapter_merged=runtime.merge_adapter
        return model,wrapper,torch,runtime.device,2
    original=resize._adapt_with_diagnostics
    def adapt(source,canvas,*args,**kwargs):
        if processor is not None:
            from PIL import Image
            cache.mkdir(parents=True,exist_ok=True)
            key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest();path=cache/(key+'.png')
            if not path.exists():
                d=deepcopy(source);d['elements']=[e for e in d['elements'] if min(e['width'],e['height'])>0]
                root=Path(__file__).resolve().parents[1]
                fonts={e.get('font_family','') for e in d['elements'] if e['type']=='text'}
                if fonts<={'Solstice Sans','Solstice Serif',''}:
                    from renderer.converter.convert import render_design_to_png
                    render_design_to_png(d,root/'brand_kit/assets',path,fonts_dir=root/'brand_kit/fonts',max_size=768)
                else:
                    from scripts.dataset_renderer import render_design_to_png
                    image=next((e for e in source['elements'] if e['type']=='image'),{})
                    render_design_to_png(d,resize._asset_root(image),path,root/'data/crello_fonts.pickle',max_size=768,adjust_tight_text_widths=False)
            with Image.open(path) as im:wrapper.image=im.convert('RGB')
        return original(source,canvas,*args,**kwargs)
    try:
        with patch.object(seed,'_load',load),patch.object(resize,'_adapt_with_diagnostics',adapt):yield
    finally:
        load.cache_clear()
        if wrapper.image is not None:wrapper.image.close()
        import gc
        gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()
        if torch.cuda.is_available():torch.cuda.empty_cache()
