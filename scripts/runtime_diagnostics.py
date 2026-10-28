#!/usr/bin/env python3
"""Print detected offline execution capabilities; never load/download weights."""
from pathlib import Path
import json,platform,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from engine.runtime import select_runtime
    result=dict(platform=platform.system(),architecture=platform.machine(),torch_version=torch.__version__,
                qwen=select_runtime(torch,'qwen').diagnostics(),vision=select_runtime(torch,'vision').diagnostics(),
                model_loaded=False,network_used=False)
    if platform.system()=='Darwin':
        try:result['cpu_model']=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
        except (OSError,subprocess.CalledProcessError):pass
        result['execution_note']='Native macOS can use Metal/MPS. Linux Docker on this Mac cannot expose MPS.'
    if torch.cuda.is_available():result['gpu_model']=torch.cuda.get_device_name(0)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
