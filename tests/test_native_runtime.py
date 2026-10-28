from types import SimpleNamespace
import pytest
from engine.runtime import select_runtime


def fake(mps=False,cuda=False,bf16=False):
    return SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:mps)),cuda=SimpleNamespace(is_available=lambda:cuda,is_bf16_supported=lambda:bf16),float32='fp32',float16='fp16',bfloat16='bf16',get_num_threads=lambda:4)


def test_native_metal_uses_fp16_and_in_memory_merge_without_a_chip_name_rule(monkeypatch):
    monkeypatch.delenv('MAGICANVAS_QWEN_MERGE_ADAPTER',raising=False)
    runtime=select_runtime(fake(mps=True))
    assert (runtime.device,runtime.dtype,runtime.profile,runtime.merge_adapter)==('mps','fp16','apple-metal',True)
    monkeypatch.setenv('MAGICANVAS_QWEN_MERGE_ADAPTER','0')
    assert not select_runtime(fake(mps=True)).merge_adapter


def test_arm64_without_metal_stays_cpu_and_cuda_dtype_is_capability_selected(monkeypatch):
    monkeypatch.setattr('engine.runtime.platform.machine',lambda:'arm64')
    cpu=select_runtime(fake());assert cpu.device=='cpu' and cpu.dtype=='fp32' and not cpu.merge_adapter
    gpu=select_runtime(fake(cuda=True,bf16=True));assert gpu.profile=='nvidia-cuda' and gpu.dtype=='bf16' and not gpu.merge_adapter
    assert select_runtime(fake(cuda=True,bf16=False)).dtype=='fp16'
    with pytest.raises(ValueError):
        monkeypatch.setenv('MAGICANVAS_QWEN_DEVICE','mps');select_runtime(fake())
