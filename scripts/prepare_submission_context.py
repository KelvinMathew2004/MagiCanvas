#!/usr/bin/env python3
"""Prepare a private local adapter-required Docker layer; never publish weights.

First build the base image with `docker build --target runtime -t magicanvas:base .`.
This separate, ignored context contains only the selected verified adapter.
"""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.verify_qwen_adapter import verify


def prepare(adapter,model,output,base_image):
    result=verify(adapter,model);output.mkdir(parents=True,exist_ok=True)
    dest=output/'adapter';dest.mkdir(exist_ok=True)
    for name in ('adapter_config.json','adapter_model.safetensors','magicanvas_prompt_version.txt'):shutil.copy2(adapter/name,dest/name)
    (output/'adapter_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'Dockerfile').write_text(f'''FROM {base_image}
COPY adapter/ /app/models/qwen3-layout-lora/
COPY adapter_manifest.json /app/models/adapter_manifest.json
ENV MAGICANVAS_QWEN_ENABLED=1 MAGICANVAS_REQUIRE_QWEN_ADAPTER=1 MAGICANVAS_QWEN_ADAPTER_DIR=/app/models/qwen3-layout-lora
RUN python scripts/verify_qwen_adapter.py --adapter /app/models/qwen3-layout-lora --model /app/models/qwen-base
''')
    print(json.dumps(dict(context=str(output),verified=result,public_upload=False,command=f'docker build --platform linux/amd64 -t magicanvas:submission {output}')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--adapter',type=Path,default=ROOT/'models/qwen3-layout-lora-v2');p.add_argument('--model',type=Path,default=ROOT/'models/qwen3-0.6b');p.add_argument('--output',type=Path,default=ROOT/'data/submission_context');p.add_argument('--base-image',default='magicanvas:base');a=p.parse_args();prepare(a.adapter,a.model,a.output,a.base_image)
