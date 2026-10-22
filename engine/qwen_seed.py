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
_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
_DEFAULT_ADAPTER = (_MODEL_ROOT / "qwen3-layout-lora-v2"
                    if (_MODEL_ROOT / "qwen3-layout-lora-v2").is_dir()
                    else _MODEL_ROOT / "qwen3-layout-lora")
ADAPTER_DIR = Path(os.environ.get("MAGICANVAS_QWEN_ADAPTER_DIR", _DEFAULT_ADAPTER))


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
        version_file = ADAPTER_DIR / "magicanvas_prompt_version.txt"
        try:
            if os.environ.get("MAGICANVAS_QWEN_PROMPT_VERSION"):
                prompt_version = int(os.environ["MAGICANVAS_QWEN_PROMPT_VERSION"])
            else:
                prompt_version = int(version_file.read_text().strip()) if version_file.is_file() else 1
        except (OSError, ValueError):
            prompt_version = 1
        model.to(device)
        model.eval()
        return model, tokenizer, torch, device, prompt_version
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
    model, tokenizer, torch, device, prompt_version = loaded
    by_id = {e["id"]: e for e in source["elements"]}
    targets = list(source["elements"])
    if not targets:
        return None, "Qwen proposal skipped (no movable elements)"
    messages = messages_for(source, target_canvas, version=prompt_version)
    try:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > 4096:
            return None, (f"Qwen prompt exceeds the 4096-token safety limit "
                          f"({inputs['input_ids'].shape[1]} tokens)")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        max_tokens = min(2048, 240 + len(targets) * 75)
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
        aliases = {f"e{i:03d}": e["id"] for i, e in enumerate(source["elements"])}
        canvas_w, canvas_h = float(target_canvas["width"]), float(target_canvas["height"])
        for item in payload["elements"]:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            item_id = aliases.get(raw_id) if prompt_version == 2 else raw_id
            if item_id not in by_id:
                continue
            if prompt_version == 2:
                try:
                    normalized = [float(value) for value in item["box"]]
                    if len(normalized) != 4:
                        continue
                    decoded = {"id": item_id,
                               "x": normalized[0] * canvas_w / 1000,
                               "y": normalized[1] * canvas_h / 1000,
                               "width": normalized[2] * canvas_w / 1000,
                               "height": normalized[3] * canvas_h / 1000}
                    if by_id[item_id]["type"] == "text":
                        decoded["font_size"] = float(item["font_scale"]) * min(canvas_w, canvas_h) / 1000
                    item = decoded
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
            try:
                values = [float(item[k]) for k in ("x", "y", "width", "height")]
            except (KeyError, TypeError, ValueError):
                continue
            if all(map(lambda n: n == n and abs(n) < 1e8, values)) and values[2] > 0 and values[3] > 0:
                proposal=dict(zip(("x", "y", "width", "height"), values))
                if by_id[item_id]["type"] == "text":
                    try:
                        font_size=float(item["font_size"])
                    except (KeyError,TypeError,ValueError):
                        continue
                    if not (font_size == font_size and 0 < font_size < 1e5):
                        continue
                    proposal["font_size"]=font_size
                proposals[item_id] = proposal
        expected = set(by_id)
        missing = expected-set(proposals)
        if missing:
            # Keep valid Qwen predictions and use proportional geometry only
            # for IDs it omitted, so a partial response can still be repaired.
            base_by_id={e['id']:e for e in base_elements}
            for element_id in missing:
                fallback=base_by_id.get(element_id)
                if fallback is None:
                    return None, f"Qwen proposal incomplete ({len(proposals)}/{len(expected)} elements)"
                proposal={key:fallback[key] for key in ('x','y','width','height')}
                if by_id[element_id]['type']=='text':
                    proposal['font_size']=fallback.get('font_size',by_id[element_id].get('font_size',1.))
                proposals[element_id]=proposal
        result = deepcopy(base_elements)
        from engine.resize import _fit_text, _text
        for element in result:
            if element["id"] in proposals:
                proposed = proposals[element["id"]]
                original = by_id[element["id"]]
                proposed["width"] = min(canvas_w,max(1.,proposed["width"]))
                proposed["height"] = min(canvas_h,max(1.,proposed["height"]))
                proposed["x"] = min(canvas_w-proposed["width"],max(0.,proposed["x"]))
                proposed["y"] = min(canvas_h-proposed["height"],max(0.,proposed["y"]))
                sw,sh=float(source["canvas"]["width"]),float(source["canvas"]["height"])
                full=(original["x"]<=sw*.02 and original["y"]<=sh*.02
                      and original["x"]+original["width"]>=sw*.98
                      and original["y"]+original["height"]>=sh*.98)
                behind=all(original.get("z_index",0)<=text.get("z_index",0)
                           for text in source["elements"] if text["type"]=="text")
                is_background=(full and behind and not original.get("group_id")
                               and original["type"]!="text")
                if original["type"] == "image" and not is_background:
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
                    source_font=max(1.,float(original.get("font_size",1)))
                    source_spacing=float(original.get("letter_spacing",0))
                    element["letter_spacing"]=source_spacing*desired/source_font
                    element.update(proposed)
                    _,measured_width,_=_text(element,element["width"],desired)
                    if measured_width>element["width"] and "\n" not in str(original.get("content", "")):
                        natural_width=_text(element,1e9,desired)[1]+desired*.2
                        max_width=min(canvas_w,element["width"]*1.35)
                        if natural_width<=max_width:
                            align=element.get("text_align","left")
                            center=element["x"]+element["width"]/2
                            right=element["x"]+element["width"]
                            old_x=element["x"]
                            element["width"]=natural_width
                            if align=="center":
                                element["x"]=min(canvas_w-natural_width,max(0.,center-natural_width/2))
                            elif align=="right":
                                element["x"]=max(0.,right-natural_width)
                            else:
                                element["x"]=min(canvas_w-natural_width,max(0.,old_x))
                    fitted=_fit_text(element,element["width"],element["height"],desired)
                    element["letter_spacing"]=source_spacing*fitted/source_font
                    element["font_size"]=fitted
                    measured_height=_text(element)[0]
                    element["height"]=min(element["height"],max(1.,measured_height))
                    continue
                element.update(proposed)
        adapter_note = (f" with local LoRA adapter (prompt v{prompt_version})"
                        if ADAPTER_DIR.is_dir() else f" (base model, prompt v{prompt_version})")
        completion_note=(f"; proportional fallback for {len(missing)} omitted element(s)"
                         if missing else "")
        return result, f"Qwen proposed boxes for {len(result)} movable elements{adapter_note}{completion_note}; device={device}"
    except Exception as exc:
        return None, f"Qwen proposal failed: {type(exc).__name__}: {exc}"
