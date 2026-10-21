"""Optional local Qwen layout proposal generator.

Qwen only proposes element boxes. The engine still sends the proposal through
its existing constraint, repair, renderer, and ranking stages. Loading is
explicitly opt-in with MAGICANVAS_QWEN_ENABLED=1.
"""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from engine.qwen_prompt import messages_for


MODEL_DIR = Path(os.environ.get(
    "MAGICANVAS_QWEN_DIR",
    Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b",
))
ADAPTER_DIR = Path(os.environ.get(
    "MAGICANVAS_QWEN_ADAPTER_DIR",
    Path(__file__).resolve().parents[1] / "models" / "qwen3-layout-lora",
))


@lru_cache(maxsize=1)
def _load():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        load_dtype = torch.float16 if device == "mps" else "auto"
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), local_files_only=True, torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        if ADAPTER_DIR.is_dir():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=False,
                                              local_files_only=True)
        model.to(device)
        model.eval()
        return model, tokenizer, torch, device
    except Exception:
        return None


def _extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None


def generate_seed(source, target_canvas, base_elements):
    """Return (element list or None, diagnostic note)."""
    loaded = _load()
    if loaded is None:
        return None, f"Qwen unavailable (local weights/dependencies missing at {MODEL_DIR})"
    model, tokenizer, torch, device = loaded
    by_id = {e["id"]: e for e in source["elements"]}
    targets = list(source["elements"])
    if not targets:
        return None, "Qwen proposal skipped (no movable elements)"
    messages = messages_for(source, target_canvas)
    try:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        max_tokens = min(1800, 120 + len(targets) * 55)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens,
                                    do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.eos_token_id)
        generated = output[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        payload = _extract_json(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
            return None, "Qwen returned invalid JSON"
        proposals = {}
        for item in payload["elements"]:
            if not isinstance(item, dict) or item.get("id") not in by_id:
                continue
            try:
                values = [float(item[k]) for k in ("x", "y", "width", "height")]
            except (KeyError, TypeError, ValueError):
                continue
            if all(map(lambda n: n == n and abs(n) < 1e8, values)) and values[2] > 0 and values[3] > 0:
                proposal=dict(zip(("x", "y", "width", "height"), values))
                if by_id[item["id"]]["type"] == "text":
                    try:
                        font_size=float(item["font_size"])
                    except (KeyError,TypeError,ValueError):
                        continue
                    if not (font_size == font_size and 0 < font_size < 1e5):
                        continue
                    proposal["font_size"]=font_size
                proposals[item["id"]] = proposal
        expected = set(by_id)
        if set(proposals) != expected:
            return None, f"Qwen proposal incomplete ({len(proposals)}/{len(expected)} elements)"
        result = deepcopy(base_elements)
        canvas_w,canvas_h=float(target_canvas["width"]),float(target_canvas["height"])
        from engine.resize import _fit_text
        for element in result:
            if element["id"] in proposals:
                proposed = proposals[element["id"]]
                original = by_id[element["id"]]
                proposed["width"] = min(canvas_w,max(1.,proposed["width"]))
                proposed["height"] = min(canvas_h,max(1.,proposed["height"]))
                proposed["x"] = min(canvas_w-proposed["width"],max(0.,proposed["x"]))
                proposed["y"] = min(canvas_h-proposed["height"],max(0.,proposed["y"]))
                if original["type"] == "image":
                    aspect=float(original["width"])/max(1e-6,float(original["height"]))
                    width=min(proposed["width"],proposed["height"]*aspect,canvas_w)
                    height=width/aspect
                    if height>canvas_h:
                        height=canvas_h;width=height*aspect
                    proposed["width"],proposed["height"]=width,height
                    proposed["x"]=min(canvas_w-width,max(0.,proposed["x"]))
                    proposed["y"]=min(canvas_h-height,max(0.,proposed["y"]))
                elif original["type"] == "text":
                    desired=max(1.,float(proposed.get("font_size",element.get("font_size",1))))
                    element.update(proposed)
                    element["font_size"]=_fit_text(element,element["width"],element["height"],desired)
                    continue
                original.update(proposed)
        adapter_note = " with local LoRA adapter" if ADAPTER_DIR.is_dir() else " (base model)"
        return result, f"Qwen proposed boxes for {len(result)} movable elements{adapter_note}; device={device}"
    except Exception as exc:
        return None, f"Qwen proposal failed: {type(exc).__name__}: {exc}"
