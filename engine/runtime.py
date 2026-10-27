"""Offline runtime capabilities; acceleration is measured, never assumed."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Runtime:
    device: str
    dtype: object
    threads: int
    reason: str

    def diagnostics(self):
        return dict(device=self.device,dtype=str(self.dtype),cpu_threads=self.threads,reason=self.reason)


def select_runtime(torch,component='qwen'):
    """Preserve CPU vision by default; native Qwen uses available acceleration.

    CPU BF16 and accelerated vision are explicit benchmarkable overrides, not
    portable promises. Never mutate inter-op threads after models are loaded.
    """
    default='auto' if component=='qwen' else 'cpu'
    requested=os.environ.get(f'MAGICANVAS_{component.upper()}_DEVICE',default).lower()
    mps=bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()) and os.environ.get('MAGICANVAS_DISABLE_MPS')!='1'
    cuda=bool(torch.cuda.is_available())
    if requested=='auto':device='cuda' if cuda else 'mps' if mps else 'cpu'
    elif requested in ('cpu','cuda','mps'):
        if requested=='cuda' and not cuda or requested=='mps' and not mps:raise ValueError(f'{requested} unavailable')
        device=requested
    else:raise ValueError(f'Unknown runtime device {requested}')
    name=os.environ.get(f'MAGICANVAS_{component.upper()}_DTYPE','auto').lower()
    if name=='auto':
        dtype=(torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16 if device in ('cuda','mps') else torch.float32)
    elif name in ('float32','float16','bfloat16'):dtype=getattr(torch,name)
    else:raise ValueError(f'Unknown runtime dtype {name}')
    if device=='cpu' and dtype==torch.float16:raise ValueError('CPU float16 is not a supported portable default')
    configured=os.environ.get('MAGICANVAS_CPU_THREADS')
    if configured:
        threads=int(configured)
        if not 1<=threads<=max(1,os.cpu_count() or 1):raise ValueError('Invalid MAGICANVAS_CPU_THREADS')
        torch.set_num_threads(threads)
    return Runtime(device,dtype,torch.get_num_threads(),'capability-aware; CPU BF16/accelerated vision require explicit opt-in')


def synchronize(torch,device):
    if device=='mps':torch.mps.synchronize()
    elif device=='cuda':torch.cuda.synchronize()
