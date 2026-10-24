"""Optional local Qwen layout proposal generator.

Qwen only proposes element boxes. The engine still sends the proposal through
its existing constraint, repair, renderer, and ranking stages. Loading is
explicitly opt-in with MAGICANVAS_QWEN_ENABLED=1.
"""
from __future__ import annotations

import json
import os
import re
from math import cos, radians, sin, sqrt
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from engine.qwen_prompt import messages_for, font_reference


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
        if prompt_version not in (1,2,3):raise ValueError('Unsupported prompt version')
        if version_file.is_file() and int(version_file.read_text().strip())!=prompt_version:
            raise ValueError('Adapter marker and runtime prompt version disagree')
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
        # A token-limited array may contain complete element objects. Recover
        # only those objects; never guess a missing number or closing value.
        match=re.search(r'"elements"\s*:\s*\[',text)
        if not match:return None
        decoder=json.JSONDecoder();items=[];position=match.end()
        while position<len(text):
            while position<len(text) and text[position] in ' \r\n\t,':position+=1
            if position>=len(text) or text[position]!='{':break
            try:item,end=decoder.raw_decode(text,position)
            except json.JSONDecodeError:break
            if not isinstance(item,dict):break
            items.append(item);position=end
        return {"elements":items} if items else None


def _rotated_bounds(element):
    """Compute visual bounds for a rotated element around its center pivot."""
    x = float(element.get('x', 0.0))
    y = float(element.get('y', 0.0))
    w = float(element.get('width', 1.0))
    h = float(element.get('height', 1.0))
    rad = radians(float(element.get('rotation', 0.0)))
    cx, cy = x + w / 2.0, y + h / 2.0
    rw = w * abs(cos(rad)) + h * abs(sin(rad))
    rh = w * abs(sin(rad)) + h * abs(cos(rad))
    return (cx - rw / 2.0, cy - rh / 2.0, cx + rw / 2.0, cy + rh / 2.0)


class QwenEarlyStoppingCriteria:
    """Stop generation when all expected element IDs are emitted and closed or loop repeats."""
    def __init__(self, tokenizer, prompt_len: int, expected_ids: set[str]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.expected_ids = set(expected_ids)
        self.N = len(self.expected_ids)
        self.stopped_early = False
        self._counter = 0

    def __call__(self, input_ids, scores=None, **kwargs) -> bool:
        self._counter += 1
        if self._counter % 4 != 0:
            return False
        cur_tokens = input_ids[0, self.prompt_len:]
        text = self.tokenizer.decode(cur_tokens, skip_special_tokens=True)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        matches = list(re.finditer(r'"id"\s*:\s*"([^"]+)"', text))
        unique_ids = []
        has_dup = False
        for m in matches:
            mid = m.group(1)
            if mid in unique_ids:
                has_dup = True
                break
            unique_ids.append(mid)

        in_str = False
        esc = False
        c_depth = 0
        s_depth = 0
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == '{': c_depth += 1
                elif ch == '}': c_depth -= 1
                elif ch == '[': s_depth += 1
                elif ch == ']': s_depth -= 1

        if self.expected_ids and len(unique_ids) >= self.N:
            if s_depth == 0 and c_depth == 0 and len(text.strip()) > 10:
                self.stopped_early = True
                return True
            if has_dup or len(matches) > self.N:
                self.stopped_early = True
                return True
            if c_depth == 1 and s_depth == 1 and len(matches) >= self.N:
                tail = text[matches[self.N - 1].start():]
                if '}' in tail:
                    after_elem = tail.split('}', 1)[1]
                    if re.search(r',\s*\{\s*"?i?d?"?', after_elem):
                        self.stopped_early = True
                        return True
        return False


def _output_budget(tokenizer,source,version,prompt_tokens,context_limit=8192):
    """Estimate the actual JSON token cost and reserve space for every ID."""
    rows=[]
    for index,e in enumerate(source['elements']):
        if version in (2,3):
            row={'id':f'e{index:03d}','box':[123.456,123.456,123.456,123.456]}
            if e['type']=='text':row['font_scale']=123.456
        else:
            row={'id':e['id'],'x':123.456,'y':123.456,'width':123.456,'height':123.456}
            if e['type']=='text':row['font_size']=123.456
        rows.append(row)
    estimated=len(tokenizer(json.dumps({'elements':rows},separators=(',',':')),
                            add_special_tokens=False)['input_ids'])
    available=max(0,context_limit-prompt_tokens)
    return min(4096,available,max(512,int(estimated*1.5)+64))


def _missing_proposal(original,source_by_id,proposals,fallback):
    """Recover omissions relative to a predicted parent/peer when evidenced."""
    choices=[]
    for eid,prediction in proposals.items():
        parent=source_by_id[eid]
        shared_group=(original.get('group_id') is not None and original.get('group_id')==parent.get('group_id'))
        iw=max(0,min(original['x']+original['width'],parent['x']+parent['width'])-max(original['x'],parent['x']))
        ih=max(0,min(original['y']+original['height'],parent['y']+parent['height'])-max(original['y'],parent['y']))
        contained=iw*ih/max(1,original['width']*original['height'])>=.8 and parent.get('z_index',0)<original.get('z_index',0)
        dx=original['x']+original['width']/2-parent['x']-parent['width']/2
        dy=original['y']+original['height']/2-parent['y']-parent['height']/2
        nearby=(abs(dx)<=.15*max(original['width'],parent['width']) and
                abs(dy)<=1.5*(original['height']+parent['height']))
        if not (shared_group or contained or nearby):continue
        choices.append((0 if shared_group else 1 if contained else 2,dx*dx+dy*dy,eid))
    if not choices:
        result={key:fallback[key] for key in ('x','y','width','height')}
        if original['type']=='text':result['font_size']=fallback.get('font_size',original.get('font_size',1))
        return result,None
    _,_,eid=min(choices);parent=source_by_id[eid];anchor=proposals[eid]
    scale=sqrt(anchor['width']*anchor['height']/max(1,parent['width']*parent['height']))
    if original['type']=='text' and parent['type']=='text':
        scale=anchor.get('font_size',parent.get('font_size',1))/max(1,parent.get('font_size',1))
    result=dict(x=anchor['x']+anchor['width']/2+(original['x']-parent['x']-parent['width']/2)*scale,
                y=anchor['y']+anchor['height']/2+(original['y']-parent['y']-parent['height']/2)*scale,
                width=original['width']*scale,height=original['height']*scale)
    if original['type']=='text':result['font_size']=original.get('font_size',1)*scale
    return result,eid


def _corner_anchor(original, proposed, sw, sh, canvas_w, canvas_h):
    """Correct an aspect-clamped compact source ornament back to its corner.

    The model can mistake a thin corner frame for a long rule.  The source
    corner is reliable structural evidence, so use its normalized margin only
    when the proposed aspect is a severe outlier; ordinary elements stay fully
    model controlled.
    """
    sx, sy = float(original['x']), float(original['y'])
    sw0, sh0 = float(original['width']), float(original['height'])
    source_area = sw0 * sh0 / max(1., sw * sh)
    left, right = sx <= .14*sw, sx+sw0 >= .86*sw
    top, bottom = sy <= .14*sh, sy+sh0 >= .86*sh
    if source_area > .06 or not (left or right) or not (top or bottom):
        return None
    aspect = sw0 / max(1e-6, sh0)
    pw, ph = proposed['width'], proposed['height']
    source_skew=max(aspect,1/max(1e-6,aspect))
    model_skew=max(pw/max(1e-6,ph),ph/max(1e-6,pw))
    if model_skew <= 1.8*source_skew:
        return None
    h=min(canvas_h, max(1., min(pw/aspect, ph)))
    w=min(canvas_w, h*aspect)
    h=w/aspect
    # Scale the authored corner margin with the canvas, rather than centering
    # inside the hallucinated bar-sized box.
    x=min(canvas_w-w,max(0.,sx/max(1.,sw)*canvas_w))
    y=min(canvas_h-h,max(0.,sy/max(1.,sh)*canvas_h))
    if right:
        x=min(canvas_w-w,max(0.,canvas_w-(sw-sx-sw0)/max(1.,sw)*canvas_w-w))
    if bottom:
        y=min(canvas_h-h,max(0.,canvas_h-(sh-sy-sh0)/max(1.,sh)*canvas_h-h))
    return dict(x=x,y=y,width=w,height=h)


def _contained_source_children(source, raw, clamped, proposals, canvas_w, canvas_h):
    """Affinely keep strongly evidenced panel children registered after clamp."""
    from engine.resize import _overlap
    source_by={e['id']:e for e in source['elements']}
    raw_by={e['id']:e for e in raw}
    out_by={e['id']:e for e in clamped}
    for panel in source['elements']:
        if panel['type'] not in ('image','shape') or panel.get('fit')=='cover':
            continue
        area=panel['width']*panel['height']/max(1.,source['canvas']['width']*source['canvas']['height'])
        if not .10 <= area <= .85 or panel['id'] not in raw_by or panel['id'] not in out_by:
            continue
        raw_panel, out_panel=raw_by[panel['id']],out_by[panel['id']]
        if raw_panel['width']<=0 or raw_panel['height']<=0:continue
        children=[]
        for child in source['elements']:
            if child['id']==panel['id'] or child.get('z_index',0)<=panel.get('z_index',0):continue
            inside_source=_overlap((panel['x'],panel['y'],panel['x']+panel['width'],panel['y']+panel['height']),
                                   (child['x'],child['y'],child['x']+child['width'],child['y']+child['height']))
            source_ratio=inside_source/max(1.,child['width']*child['height'])
            raw_child=raw_by.get(child['id'])
            if raw_child is None:continue
            inside_raw=_overlap((raw_panel['x'],raw_panel['y'],raw_panel['x']+raw_panel['width'],raw_panel['y']+raw_panel['height']),
                                (raw_child['x'],raw_child['y'],raw_child['x']+raw_child['width'],raw_child['y']+raw_child['height']))
            if source_ratio>=.75 and inside_raw/max(1.,raw_child['width']*raw_child['height'])>=.65:
                children.append(child)
        if len(children)<2:continue
        sx=out_panel['width']/raw_panel['width'];sy=out_panel['height']/raw_panel['height']
        if sx>.92 and sy>.92 and abs(out_panel['x']-raw_panel['x'])<4:
            continue
        for child in children:
            raw_child=raw_by[child['id']];out=out_by[child['id']]
            rx=(child['x']-panel['x'])/max(1.,panel['width'])
            ry=(child['y']-panel['y'])/max(1.,panel['height'])
            # Preserve the proposal's size, but map its anchor into the
            # correspondingly clamped source panel. This avoids splitting a
            # backing card from its actual text/divider contents.
            out['x']=out_panel['x']+rx*out_panel['width']
            out['y']=out_panel['y']+ry*out_panel['height']
            out['x']=min(canvas_w-out['width'],max(0.,out['x']))
            out['y']=min(canvas_h-out['height'],max(0.,out['y']))
            if out.get('type')=='text':
                pad=max(6.,.04*out_panel['width'])
                out['width']=max(1.,min(out['width'],out_panel['x']+out_panel['width']-out['x']-pad))
    return clamped


def generate_seed(source, target_canvas, base_elements, diagnostics=None):
    """Return (element list or None, diagnostic note)."""
    requested=os.environ.get('MAGICANVAS_QWEN_PROMPT_VERSION')
    marker=ADAPTER_DIR/'magicanvas_prompt_version.txt'
    if requested and marker.is_file() and marker.read_text().strip()!=requested:
        return None,'Qwen adapter prompt-version marker disagrees with requested format'
    loaded = _load()
    if loaded is None:
        return None, f"Qwen unavailable (local weights/dependencies missing at {MODEL_DIR})"
    model, tokenizer, torch, device, prompt_version = loaded
    diagnostics={} if diagnostics is None else diagnostics
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
        context_limit=min(8192,int(getattr(model.config,'max_position_embeddings',8192)))
        max_tokens=_output_budget(tokenizer,source,prompt_version,inputs['input_ids'].shape[1],context_limit)
        if max_tokens<128:return None,'Qwen has insufficient remaining context for JSON output'
        expected_ids = {f"e{i:03d}" for i in range(len(source["elements"]))} if prompt_version in (2, 3) else set(by_id)
        stop_criteria = QwenEarlyStoppingCriteria(tokenizer, inputs["input_ids"].shape[1], expected_ids)
        try:
            from transformers import StoppingCriteriaList
            sc_list = StoppingCriteriaList([stop_criteria])
        except ImportError:
            sc_list = [stop_criteria]
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens,
                                    do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.eos_token_id,
                                    stopping_criteria=sc_list)
        generated = output[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        # Check if response needs clean closing after early stopping
        clean_text = re.sub(r'<think>.*?</think>', '', response, flags=re.S).strip()
        found_matches = list(re.finditer(r'"id"\s*:\s*"([^"]+)"', clean_text))
        if expected_ids and len(found_matches) >= len(expected_ids):
            nth_pos = found_matches[len(expected_ids) - 1].start()
            pos = nth_pos
            c_depth = 1
            in_str = False
            esc = False
            end_pos = None
            while pos < len(clean_text):
                ch = clean_text[pos]
                if esc:
                    esc = False
                    pos += 1
                    continue
                if ch == '\\':
                    esc = True
                    pos += 1
                    continue
                if ch == '"':
                    in_str = not in_str
                    pos += 1
                    continue
                if not in_str:
                    if ch == '{':
                        c_depth += 1
                    elif ch == '}':
                        c_depth -= 1
                        if c_depth == 0:
                            end_pos = pos + 1
                            break
                pos += 1
            if end_pos is not None:
                tail = clean_text[end_pos:].strip()
                if not (tail.startswith(']') and '}' in tail):
                    clean_text = clean_text[:end_pos].rstrip() + "\n]}"
                    response = clean_text
        try:
            json.loads(re.sub(r'<think>.*?</think>','',response,flags=re.S).strip())
            json_complete=True
        except (json.JSONDecodeError,TypeError):json_complete=False
        diagnostics.update(prompt_tokens=int(inputs['input_ids'].shape[1]),
                           generated_tokens=int(generated.shape[0]),output_token_budget=max_tokens,
                           hit_token_limit=int(generated.shape[0])>=max_tokens and not stop_criteria.stopped_early,prompt_version=prompt_version,
                           json_complete=json_complete,duplicate_ids=[],unknown_ids=[])
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
            item_id = aliases.get(raw_id) if prompt_version in (2,3) else raw_id
            if item_id not in by_id:
                diagnostics['unknown_ids'].append(raw_id)
                continue
            if item_id in proposals:
                diagnostics['duplicate_ids'].append(item_id)
                continue
            if prompt_version in (2,3):
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
                        decoded["font_size"] = float(item["font_scale"]) * font_reference(target_canvas,prompt_version) / 1000
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
        diagnostics.update(model_elements=deepcopy(proposals),missing_ids=sorted(missing),fallback_anchors={})
        if len(missing)>max(2,int(len(expected)*.20)) or not proposals:
            return None,f'Qwen omitted too many IDs ({len(missing)}/{len(expected)})'
        if missing:
            base_by_id={e['id']:e for e in base_elements}
            # Use only model-supplied anchors, so one recovered omission cannot
            # propagate a speculative transform through the remaining IDs.
            anchors=deepcopy(proposals)
            for element_id in sorted(missing):
                fallback=base_by_id.get(element_id)
                if fallback is None:
                    # Background elements are reassembled independently.
                    fallback=by_id[element_id]
                proposal,anchor=_missing_proposal(by_id[element_id],by_id,anchors,fallback)
                diagnostics['fallback_anchors'][element_id]=anchor
                proposals[element_id]=proposal
        result = deepcopy(base_elements)
        raw_decoded=[]
        for base in result:
            raw_decoded.append({**deepcopy(base),**deepcopy(proposals.get(base['id'],{}))})
        from engine.resize import _fit_text, _text
        bleed_preserved=[]
        source_canvas_w=float(source['canvas']['width'])
        source_canvas_h=float(source['canvas']['height'])
        source_texts=[e for e in source['elements'] if e['type']=='text']
        # Text inside a visible backing card follows the card's own layout
        # scale. Applying a canvas-width typography floor to those labels
        # enlarges the entire card and can undo a correct compact adaptation.
        source_panel_children=set()
        canvas_area=max(1.,source_canvas_w*source_canvas_h)
        for panel in source['elements']:
            if panel.get('type')=='text' or panel.get('type')!='image':
                continue
            area=float(panel.get('width',0))*float(panel.get('height',0))
            if area/canvas_area<.025 or area/canvas_area>.82:
                continue
            px,py=float(panel.get('x',0)),float(panel.get('y',0))
            pr=px+float(panel.get('width',0));pb=py+float(panel.get('height',0))
            contained=[t for t in source_texts
                       if float(panel.get('z_index',0))<float(t.get('z_index',0))
                       and float(t.get('x',0))>=px-2
                       and float(t.get('y',0))>=py-2
                       and float(t.get('x',0))+float(t.get('width',0))<=pr+2
                       and float(t.get('y',0))+float(t.get('height',0))<=pb+2]
            if len(contained)>=2:
                source_panel_children.update(t['id'] for t in contained)
        source_background_ids={e['id'] for e in source['elements']
            if (e['type']!='text' and not e.get('group_id')
                and e['x']<=source_canvas_w*.02 and e['y']<=source_canvas_h*.02
                and e['x']+e['width']>=source_canvas_w*.98
                and e['y']+e['height']>=source_canvas_h*.98
                and all(e.get('z_index',0)<=t.get('z_index',0) for t in source_texts))}
        for element in result:
            if element["id"] in proposals:
                proposed = proposals[element["id"]]
                original = by_id[element["id"]]
                rot = float(original.get('rotation', 0.0)) % 360.0
                is_rotated = (0.5 < rot < 359.5) and abs(rot - 180.0) > 0.5
                if is_rotated:
                    orig_rot = float(original.get('rotation', 0.0))
                    rad = radians(orig_rot)
                    sw0 = float(original.get('width', 1.0))
                    sh0 = float(original.get('height', 1.0))
                    proposed_w = float(proposed.get('width', sw0))
                    proposed_h = float(proposed.get('height', sh0))
                    scale = min(proposed_w / max(1e-6, sw0), proposed_h / max(1e-6, sh0))
                    w = max(1.0, sw0 * scale)
                    h = max(1.0, sh0 * scale)
                    cx = float(proposed.get('x', 0.0)) + proposed_w / 2.0
                    cy = float(proposed.get('y', 0.0)) + proposed_h / 2.0
                    rw = w * abs(cos(rad)) + h * abs(sin(rad))
                    rh = w * abs(sin(rad)) + h * abs(cos(rad))
                    eff_w = max(w, rw)
                    eff_h = max(h, rh)
                    if eff_w > canvas_w or eff_h > canvas_h:
                        downscale = min(canvas_w / max(1e-6, eff_w), canvas_h / max(1e-6, eff_h))
                        w = max(1.0, w * downscale)
                        h = max(1.0, h * downscale)
                        scale *= downscale
                        rw = w * abs(cos(rad)) + h * abs(sin(rad))
                        rh = w * abs(sin(rad)) + h * abs(cos(rad))
                        eff_w = max(w, rw)
                        eff_h = max(h, rh)
                    # The rotated-image path used to center every clamped box,
                    # bypassing the edge-anchor handling below. Preserve a
                    # strong source bleed edge for substantial bottom-layer
                    # images so a side backdrop does not jump beneath copy.
                    src_bounds=_rotated_bounds(original)
                    src_x0,src_y0,src_x1,src_y1=src_bounds
                    visible_w=max(0.,min(source_canvas_w,src_x1)-max(0.,src_x0))
                    visible_h=max(0.,min(source_canvas_h,src_y1)-max(0.,src_y0))
                    visible_share=visible_w*visible_h/max(1.,source_canvas_w*source_canvas_h)
                    anchored_bleed=(original['type']=='image'
                        and original.get('z_index',0)==0 and visible_share>=.40)
                    left_anchor=(anchored_bleed and src_x0<=.02*source_canvas_w
                                 and src_x1<.95*source_canvas_w)
                    right_anchor=(anchored_bleed and src_x1>=.98*source_canvas_w
                                  and src_x0>.05*source_canvas_w)
                    top_anchor=(anchored_bleed and src_y0<=.02*source_canvas_h
                                and src_y1<.95*source_canvas_h)
                    bottom_anchor=(anchored_bleed and src_y1>=.98*source_canvas_h
                                   and src_y0>.05*source_canvas_h)
                    if left_anchor:
                        cx=eff_w/2.
                    elif right_anchor:
                        cx=canvas_w-eff_w/2.
                    if top_anchor:
                        cy=eff_h/2.
                    elif bottom_anchor:
                        cy=canvas_h-eff_h/2.
                    cx = min(canvas_w - eff_w / 2.0, max(eff_w / 2.0, cx))
                    cy = min(canvas_h - eff_h / 2.0, max(eff_h / 2.0, cy))
                    proposed['x'] = cx - w / 2.0
                    proposed['y'] = cy - h / 2.0
                    proposed['width'] = w
                    proposed['height'] = h
                    proposed['rotation'] = orig_rot
                    element.update(proposed)
                    if original['type'] == 'text':
                        source_font = max(1.0, float(original.get('font_size', 1.0)))
                        element['font_size'] = max(1.0, source_font * scale)
                        source_spacing = float(original.get('letter_spacing', 0.0))
                        element['letter_spacing'] = source_spacing * scale
                    continue
                proposed["width"] = min(canvas_w,max(1.,proposed["width"]))
                proposed["height"] = min(canvas_h,max(1.,proposed["height"]))
                proposed["x"] = min(canvas_w-proposed["width"],max(0.,proposed["x"]))
                proposed["y"] = min(canvas_h-proposed["height"],max(0.,proposed["y"]))
                sw,sh=float(source["canvas"]["width"]),float(source["canvas"]["height"])
                is_background=original['id'] in source_background_ids
                source_x=float(original.get('x',0));source_y=float(original.get('y',0))
                source_w=float(original.get('width',0));source_h=float(original.get('height',0))
                source_area=max(0.,min(sw,source_x+source_w)-max(0.,source_x))*max(
                    0.,min(sh,source_y+source_h)-max(0.,source_y))/(sw*sh)
                bleed_edges={
                    'left':source_x<=sw*.02,
                    'right':source_x+source_w>=sw*.98,
                    'top':source_y<=sh*.02,
                    'bottom':source_y+source_h>=sh*.98,
                }
                edge_anchored=(original['type']=='image' and source_area>=.40
                               and sum(bleed_edges.values())>=2)
                corner=_corner_anchor(original,proposed,sw,sh,canvas_w,canvas_h) \
                    if original['type']=='image' and not is_background else None
                if corner is not None:
                    proposed.update(corner)
                    element.update(proposed)
                    continue
                spans_height=(source_y<=sh*.02 and source_y+source_h>=sh*.98
                              and source_w<sw*.95)
                spans_width=(source_x<=sw*.02 and source_x+source_w>=sw*.98
                             and source_h<sh*.95)
                if (original['type']=='image' and not is_background and source_area<=.30
                        and (spans_height or spans_width)):
                    # A narrow source strip that spans one canvas axis remains
                    # a strip at that axis on the target. Cover crops the
                    # image within the frame, preserving the source's full-span
                    # structural role without distorting the asset pixels.
                    if spans_height:
                        proposed['y'],proposed['height']=0.,canvas_h
                        proposed['width']=min(canvas_w,max(1.,proposed['width']))
                        proposed['x']=min(canvas_w-proposed['width'],max(0.,proposed['x']))
                    else:
                        proposed['x'],proposed['width']=0.,canvas_w
                        proposed['height']=min(canvas_h,max(1.,proposed['height']))
                        proposed['y']=min(canvas_h-proposed['height'],max(0.,proposed['y']))
                    element['fit']='cover'
                    element.update(proposed)
                    bleed_preserved.append(element['id'])
                    continue
                if edge_anchored and not is_background and original.get('fit')=='cover':
                    # Overscanned photos and edge-anchored backdrops need a
                    # target frame that bleeds to their source edges. Clamping
                    # them to their asset aspect ratio creates an artificial
                    # seam through foreground copy; the renderer's cover fit
                    # crops the raster while retaining full frame coverage.
                    if bleed_edges['left'] and bleed_edges['right']:
                        proposed['x'],proposed['width']=0.,canvas_w
                    elif bleed_edges['left']:
                        proposed['x']=0.
                    elif bleed_edges['right']:
                        proposed['x']=canvas_w-proposed['width']
                    if bleed_edges['top'] and bleed_edges['bottom']:
                        proposed['y'],proposed['height']=0.,canvas_h
                    elif bleed_edges['top']:
                        proposed['y']=0.
                    elif bleed_edges['bottom']:
                        proposed['y']=canvas_h-proposed['height']
                    proposed['width']=min(canvas_w,max(1.,proposed['width']))
                    proposed['height']=min(canvas_h,max(1.,proposed['height']))
                    proposed['x']=min(canvas_w-proposed['width'],max(0.,proposed['x']))
                    proposed['y']=min(canvas_h-proposed['height'],max(0.,proposed['y']))
                    element['fit']='cover'
                    bleed_preserved.append(element['id'])
                elif original["type"] == "image" and not is_background:
                    aspect=float(original["width"])/max(1e-6,float(original["height"]))
                    center_x=proposed["x"]+proposed["width"]/2
                    center_y=proposed["y"]+proposed["height"]/2
                    width=min(proposed["width"],proposed["height"]*aspect,canvas_w)
                    height=width/aspect
                    if height>canvas_h:
                        height=canvas_h;width=height*aspect
                    proposed["width"],proposed["height"]=width,height
                    proposed["x"]=min(canvas_w-width,max(0.,center_x-width/2))
                    proposed["y"]=min(canvas_h-height,max(0.,center_y-height/2))
                    if edge_anchored:
                        if bleed_edges['left'] and not bleed_edges['right']:proposed['x']=0.
                        elif bleed_edges['right'] and not bleed_edges['left']:proposed['x']=canvas_w-width
                        if bleed_edges['top'] and not bleed_edges['bottom']:proposed['y']=0.
                        elif bleed_edges['bottom'] and not bleed_edges['top']:proposed['y']=canvas_h-height
                elif original["type"] == "text":
                    desired=max(1.,float(proposed.get("font_size",element.get("font_size",1))))
                    source_font=max(1.,float(original.get("font_size",1)))
                    source_spacing=float(original.get("letter_spacing",0))
                    element.update(proposed)
                    # Keep the original typographic scale as a conservative
                    # lower bound when the target gets wider. Qwen predicts
                    # font_scale from the short side, which can over-shrink
                    # every line in a wide target even when the source-to-
                    # target width ratio says the text should stay the same
                    # size or grow. The model can still enlarge text freely;
                    # it just cannot collapse it below 90% of the source
                    # font's width-proportional size.
                    width_scale=canvas_w/max(1.,source_canvas_w)
                    source_width_floor=(source_font*width_scale*.90
                                        if width_scale>1.0
                                        and element['id'] not in source_panel_children
                                        else 0.)
                    desired=max(desired,source_width_floor)
                    element["letter_spacing"]=source_spacing*desired/source_font
                    # Qwen's font prediction can be right while its normalized
                    # text frame remains near the source width. Fitting into
                    # that undersized frame silently shrinks the typography
                    # (for example, a display price on a canvas with a much
                    # larger short side). For a one-line label, first expand
                    # the frame to the measured glyph width when that width
                    # fits comfortably on the canvas; preserve its alignment.
                    measured_height,natural_width,measured_lines=_text(
                        element,1e9,desired)
                    content=str(original.get("content", ""))
                    source_lines=max(1,_text(original)[2])
                    explicit_breaks="\n" in content
                    preserves_lines=(measured_lines==source_lines if explicit_breaks
                                     else measured_lines==1)
                    natural_width+=desired*.12
                    max_frame_width=canvas_w*.94
                    if (preserves_lines and natural_width>element["width"]
                            and natural_width<=max_frame_width):
                        align=element.get("text_align","left")
                        old_x=element["x"];old_right=old_x+element["width"]
                        center=old_x+element["width"]/2
                        element["width"]=natural_width
                        if align=="center":
                            element["x"]=min(canvas_w-natural_width,
                                            max(0.,center-natural_width/2))
                        elif align=="right":
                            element["x"]=min(canvas_w-natural_width,
                                            max(0.,old_right-natural_width))
                        else:
                            element["x"]=min(canvas_w-natural_width,max(0.,old_x))
                        element["height"]=max(element["height"],measured_height)
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
                    measured_height,_,measured_lines=_text(
                        element,element["width"],desired)
                    if (preserves_lines and measured_lines==source_lines
                            and measured_height<=canvas_h):
                        # Qwen's height box can be undersized even when the
                        # predicted font and width are sound. Grow it before
                        # fitting so height alone cannot silently shrink the
                        # font back below the source-proportional floor.
                        element["height"]=max(element["height"],measured_height)
                    fitted=_fit_text(element,element["width"],element["height"],desired)
                    element["letter_spacing"]=source_spacing*fitted/source_font
                    element["font_size"]=fitted
                    continue
                element.update(proposed)
        _contained_source_children(source,raw_decoded,result,proposals,canvas_w,canvas_h)
        adapter_note = (f" with local LoRA adapter (prompt v{prompt_version})"
                        if ADAPTER_DIR.is_dir() else f" (base model, prompt v{prompt_version})")
        completion_note=(f"; recovered {len(missing)} omitted element(s), {sum(v is not None for v in diagnostics['fallback_anchors'].values())} relation-anchored"
                         if missing else "")
        bleed_note=(f"; preserved edge bleed on {len(bleed_preserved)} image(s)"
                    if bleed_preserved else "")
        return result, f"Qwen proposed boxes for {len(result)} movable elements{adapter_note}{completion_note}{bleed_note}; device={device}"
    except Exception as exc:
        return None, f"Qwen proposal failed: {type(exc).__name__}: {exc}"
