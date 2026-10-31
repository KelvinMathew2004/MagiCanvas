"""Model-specific chat/image preparation behind one normalized layout contract."""
from copy import deepcopy


def components(spec,path=None,local_files_only=False):
    from transformers import AutoTokenizer,AutoProcessor
    kwargs=dict(local_files_only=local_files_only)
    if path is None:kwargs['revision']=spec['revision']
    location=path if path is not None else spec['repo']
    tokenizer=AutoTokenizer.from_pretrained(location,**kwargs)
    processor=None
    if spec.get('kind')=='vision_language':
        processor=AutoProcessor.from_pretrained(location,**kwargs)
        processor.image_processor.size={'longest_edge':262144,'shortest_edge':65536}
        tokenizer=processor.tokenizer
    return tokenizer,processor


def prompt_text(messages,tokenizer,processor=None,image=None):
    if processor is None:
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if image is None:raise ValueError('The image-conditioned provider needs the source image')
    converted=deepcopy(messages)
    user=next(m for m in converted if m['role']=='user')
    user['content']=[dict(type='image',image=image),dict(type='text',text=user['content'])]
    return processor.apply_chat_template(converted,tokenize=False,add_generation_prompt=True,enable_thinking=False)


def tokenize(prompt,tokenizer,processor=None,image=None,return_tensors=None):
    if processor is not None:
        return processor(text=prompt,images=image,return_tensors=return_tensors or 'pt')
    return tokenizer(prompt,add_special_tokens=False,return_tensors=return_tensors)


def language_linear_targets(model):
    import torch
    names=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)
           and not name.endswith('lm_head') and not any(part in name.split('.') for part in ('visual','vision_model','vision_tower','mtp'))]
    if not names:raise ValueError('No language linear projections found')
    return names
