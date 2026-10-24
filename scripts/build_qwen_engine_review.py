#!/usr/bin/env python3
"""Render held-out Qwen-off/on engine candidates into a local review page."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SPLIT_DIRS={name:ROOT/"data"/f"crello_{name}" for name in ("train","validation","test")}
OUT_DIR=ROOT/"data"/"qwen_lora"/"engine_review"
EXCLUSION_FILE=ROOT/"data"/"qwen_lora"/"excluded_pairs.json"


def _exclusions():
    if not EXCLUSION_FILE.is_file():
        return set(), set()
    document=json.loads(EXCLUSION_FILE.read_text())
    rows=document.get("excluded_pairs", [])
    pairs={row["pair_id"] for row in rows}
    templates={template_id for row in rows for template_id in row.get("template_ids", [])}
    return pairs, templates


def _is_excluded_pair(pair_id, excluded_pairs, excluded_templates):
    parts=pair_id.split("__to__",1)
    if len(parts)==2 and (parts[0] in excluded_templates or parts[1] in excluded_templates):
        return True
    if pair_id in excluded_pairs:
        return True
    return len(parts)==2 and f"{parts[1]}__to__{parts[0]}" in excluded_pairs


def _index(folder):
    result={}
    for line in (folder/"manifest.jsonl").read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row.get("status")=="exported":result[row["template_id"]]=row
    return result


def _select(records,limit,unique_families=True):
    if unique_families:
        # Pair directions that share design IDs belong to one review family.
        # Keep one direction per connected component so the review measures
        # generalization across designs instead of repeating alternate sizes.
        parent={}
        def find(item):
            parent.setdefault(item,item)
            if parent[item]!=item:parent[item]=find(parent[item])
            return parent[item]
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[b]=a
        for row in records:
            source,target=row["pair_id"].split("__to__",1)
            union(source,target)
        families={}
        for row in records:
            source=row["pair_id"].split("__to__",1)[0]
            families.setdefault(find(source),[]).append(row)
        unique=[]
        for rows in families.values():
            def ranking(row):
                prompt=row["messages"][1]["content"]
                match=re.search(r"Source aspect ratio:\s*([0-9.]+)",prompt)
                source_ratio=float(match.group(1)) if match else 1.0
                target=row["target_canvas"]["width"]/row["target_canvas"]["height"]
                aspect_change=abs(math.log(max(1e-6,target)/max(1e-6,source_ratio)))
                return aspect_change,row["pair_id"]
            unique.append(max(rows,key=ranking))
        records=unique
    buckets={"banner":[],"landscape":[],"square":[],"portrait":[]}
    for row in records:
        canvas=row["target_canvas"];ratio=canvas["width"]/canvas["height"]
        key="banner" if ratio>=1.8 else "landscape" if ratio>=1.2 else "square" if ratio>=.8 else "portrait"
        buckets[key].append(row)
    for rows in buckets.values():rows.sort(key=lambda row:(-len(row["messages"][-1]["content"]),row["pair_id"]))
    selected=[]
    while len(selected)<min(limit,len(records)):
        added=False
        for rows in buckets.values():
            if rows and len(selected)<limit:
                selected.append(rows.pop(0));added=True
        if not added:break
    return selected


def _render_candidate(source,target_canvas,enabled):
    import engine.resize as resize_module
    os.environ["MAGICANVAS_QWEN_ENABLED"]="1" if enabled else "0"
    captured=[]
    captured_micro=[]
    if enabled:
        import engine.qwen_seed as qwen_seed
        original_generate=qwen_seed.generate_seed
        original_snap=resize_module._micro_snap_qwen_proposal
        def capture_seed(*args,**kwargs):
            result=original_generate(*args,**kwargs)
            if result[0] is not None:
                captured.append(result[0])
            return result
        def capture_micro(elements,*args,**kwargs):
            result=original_snap(elements,*args,**kwargs)
            captured_micro.append(result[0])
            return result
        qwen_seed.generate_seed=capture_seed
        resize_module._micro_snap_qwen_proposal=capture_micro
        try:
            design,report=resize_module._adapt_with_diagnostics(source,target_canvas)
        finally:
            qwen_seed.generate_seed=original_generate
            resize_module._micro_snap_qwen_proposal=original_snap
    else:
        design,report=resize_module._adapt_with_diagnostics(source,target_canvas)
    layer=report.get("qwen_diagnostics",{}).get("layer_preserved_elements")
    return (design,report,(captured[-1] if captured else None),
            (captured_micro[-1] if captured_micro else None),
            layer)


def _geometry_metrics(elements,target_elements,canvas):
    target_by_id={e["id"]:e for e in target_elements}
    source_by_id={e["id"]:e for e in elements}
    ids=sorted(set(source_by_id)&set(target_by_id))
    if not ids:return {"matched":0,"geometry_mae":None,"font_relative_mae":None}
    w,h=canvas["width"],canvas["height"]
    errors=[];font_errors=[]
    for element_id in ids:
        a,b=source_by_id[element_id],target_by_id[element_id]
        errors.extend((abs(float(a.get(k,0))-float(b.get(k,0)))/denom)
                      for k,denom in (("x",w),("y",h),("width",w),("height",h)))
        if "font_size" in a and "font_size" in b:
            font_errors.append(abs(float(a["font_size"])-float(b["font_size"]))
                               /max(1.,float(b["font_size"])))
    return {"matched":len(ids),"geometry_mae":round(sum(errors)/len(errors),5),
            "font_relative_mae":round(sum(font_errors)/len(font_errors),5) if font_errors else None}


def _html(rows,renderer_mode):
    data=json.dumps(rows,ensure_ascii=False).replace("</","<\\/")
    renderer_note=("starter renderer; dataset-only tight-text-width adjustment disabled"
                   if renderer_mode=='starter_renderer_unadjusted' else
                   "dataset renderer with tight-text-width adjustment enabled")
    return '''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen held-out engine comparison</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#0b1018;color:#edf2f8}
body{margin:0;padding:22px}header{display:flex;gap:12px;align-items:center;position:sticky;top:0;background:#0b1018;padding:8px 0;z-index:2}
button{font:inherit;padding:10px 18px;border:1px solid #607086;border-radius:10px;background:#1b2635;color:white;cursor:pointer}
button:disabled{opacity:.45}#count{font-variant-numeric:tabular-nums;color:#bdc9d8}#meta{color:#bdc9d8;margin:8px 0 16px;white-space:pre-wrap}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
figure{margin:0;padding:12px;background:#161f2b;border:1px solid #2a3747;border-radius:14px;min-width:0}
figcaption{font-weight:650;margin-bottom:8px}img{display:block;width:100%;height:auto;max-height:75vh;object-fit:contain;background:#10151d;border-radius:8px}
#diag{margin-top:16px;padding:14px;background:#161f2b;border:1px solid #2a3747;border-radius:12px;white-space:pre-wrap;line-height:1.55}
@media(max-width:760px){body{padding:12px}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><button id="prev">← Previous</button><button id="next">Next →</button><span id="count"></span></header><div id="meta"></div>
<main class="grid">
<figure><figcaption>Dataset preview</figcaption><img id="preview"></figure>
<figure><figcaption>Local target JSON render (unadjusted)</figcaption><img id="target"></figure>
<figure><figcaption>Engine, Qwen off</figcaption><img id="base"></figure>
<figure><figcaption>Engine, Qwen on</figcaption><img id="adapter"></figure></main>
<section id="diag"></section><p>Four views per case: the dataset reference, converted target JSON, and final engine output with Qwen off/on. Intermediate Qwen stages and geometry diagnostics remain in the manifest and candidate JSON. Use arrow keys or buttons to compare. Renderer mode: __RENDERER_NOTE__.</p>
<script>const rows=__ROWS__;let i=0;function show(){const r=rows[i];for(const key of ['preview','target','base','adapter'])document.getElementById(key).src=r[key]||'';document.getElementById('meta').textContent=`${r.pair_id} · ${r.canvas.width}×${r.canvas.height} · ${r.element_count} elements · off: ${r.off_status} · on: ${r.on_status}`;document.getElementById('diag').textContent=r.diagnostics;document.getElementById('count').textContent=`${i+1} / ${rows.length}`;document.getElementById('prev').disabled=i===0;document.getElementById('next').disabled=i===rows.length-1}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});show();</script></body></html>'''.replace('__ROWS__',data).replace('__RENDERER_NOTE__',renderer_note)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"data"/"qwen_lora"/"test.jsonl")
    parser.add_argument("--output",type=Path,default=OUT_DIR)
    parser.add_argument("--limit",type=int,default=8)
    parser.add_argument("--include-family-repeats",action="store_true",
                        help="Allow multiple resize directions from the same connected design family")
    parser.add_argument("--raw-starter-renderer",action="store_true",
                        help="Skip the dataset-only text-width tolerance for every review render")
    args=parser.parse_args()
    if args.limit<1:parser.error("--limit must be positive")
    renderer_mode=('starter_renderer_unadjusted' if args.raw_starter_renderer
                   else 'dataset_tight_text_adjusted')
    os.environ["MAGICANVAS_DATASET_RENDERING"]="1"
    from scripts.dataset_renderer import render_design_to_png
    from engine.qwen_prompt import decode_v2_completion, record_prompt_version
    indexes={split:_index(folder) for split,folder in SPLIT_DIRS.items()}
    records=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    excluded_pairs,excluded_templates=_exclusions()
    records=[row for row in records if not _is_excluded_pair(
        row.get("pair_id", ""), excluded_pairs, excluded_templates)]
    if records:
        if len({record_prompt_version(row) for row in records})!=1:raise SystemExit("Mixed prompt versions in review input")
        prompt_version=record_prompt_version(records[0])
        if prompt_version in (2,3):
            import engine.qwen_seed as qwen_seed
            marker=qwen_seed.ADAPTER_DIR/"magicanvas_prompt_version.txt"
            adapter_version=int(marker.read_text().strip()) if marker.is_file() else 1
            if adapter_version!=prompt_version:
                raise SystemExit(f"Held-out JSONL requires prompt v{prompt_version}, but installed adapter is v{adapter_version}. Train/install the v2 adapter before building this comparison.")
        os.environ["MAGICANVAS_QWEN_PROMPT_VERSION"]=str(prompt_version)
    selected=_select(records,args.limit,unique_families=not args.include_family_repeats)
    args.output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for number,row in enumerate(selected,1):
        source_id,target_id=row["pair_id"].split("__to__",1)
        source_split,target_split=row["source_split"],row["target_split"]
        source_row=indexes[source_split][source_id];target_row=indexes[target_split][target_id]
        source=json.loads((SPLIT_DIRS[source_split]/source_row["design"]).read_text())
        target=json.loads((SPLIT_DIRS[target_split]/target_row["design"]).read_text())
        target_canvas=row["target_canvas"]
        prefix=f"{number:02d}_{source_id}__{target_id}"
        preview_path=SPLIT_DIRS[target_split]/target_row["preview"]
        with Image.open(preview_path) as image:preview_size=image.size
        target_path=args.output/f"{prefix}_target.png"
        render_design_to_png(target,SPLIT_DIRS[target_split],target_path,
                             ROOT/"data"/"crello_fonts.pickle",
                             adjust_tight_text_widths=not args.raw_starter_renderer)
        with Image.open(target_path) as image:
            if image.size!=preview_size:image.convert("RGB").resize(preview_size,Image.Resampling.LANCZOS).save(target_path)
        off,off_report,_,_,_=_render_candidate(source,target_canvas,False)
        on,on_report,raw_qwen,micro_qwen,layer_qwen=_render_candidate(source,target_canvas,True)
        off_path=args.output/f"{prefix}_qwen_off.png";on_path=args.output/f"{prefix}_qwen_on.png"
        raw_path=args.output/f"{prefix}_raw_qwen.png"
        micro_path=args.output/f"{prefix}_micro_qwen.png"
        layer_path=args.output/f"{prefix}_layer_preserved_qwen.png"
        if raw_qwen is not None:
            import engine.resize as resize_module
            backgrounds,_=resize_module._analyze(source)
            raw_design=resize_module._assemble(raw_qwen,backgrounds,source,
                                               target_canvas["width"],target_canvas["height"])
            raw_warnings=render_design_to_png(raw_design,SPLIT_DIRS[source_split],raw_path,
                                              ROOT/"data"/"crello_fonts.pickle",
                                              adjust_tight_text_widths=not args.raw_starter_renderer)
            with Image.open(raw_path) as image:
                if image.size!=preview_size:image.convert("RGB").resize(preview_size,Image.Resampling.LANCZOS).save(raw_path)
        else:
            raw_warnings=["Qwen did not return a proposal"]
        if micro_qwen is not None:
            import engine.resize as resize_module
            backgrounds,_=resize_module._analyze(source)
            micro_design=resize_module._assemble(micro_qwen,backgrounds,source,
                                                target_canvas["width"],target_canvas["height"])
            micro_warnings=render_design_to_png(micro_design,SPLIT_DIRS[source_split],micro_path,
                                                ROOT/"data"/"crello_fonts.pickle",
                                                adjust_tight_text_widths=not args.raw_starter_renderer)
            with Image.open(micro_path) as image:
                if image.size!=preview_size:image.convert("RGB").resize(preview_size,Image.Resampling.LANCZOS).save(micro_path)
        else:
            micro_warnings=["No Qwen proposal to micro-repair"]
        if layer_qwen is not None:
            import engine.resize as resize_module
            backgrounds,_=resize_module._analyze(source)
            layer_design=resize_module._assemble(layer_qwen,backgrounds,source,
                                                target_canvas["width"],target_canvas["height"])
            layer_warnings=render_design_to_png(layer_design,SPLIT_DIRS[source_split],layer_path,
                                                ROOT/"data"/"crello_fonts.pickle",
                                                adjust_tight_text_widths=not args.raw_starter_renderer)
            with Image.open(layer_path) as image:
                if image.size!=preview_size:image.convert("RGB").resize(preview_size,Image.Resampling.LANCZOS).save(layer_path)
        else:
            layer_warnings=["No broken source artwork stack detected"]
        off_warnings=render_design_to_png(off,SPLIT_DIRS[source_split],off_path,
                                          ROOT/"data"/"crello_fonts.pickle",
                                          adjust_tight_text_widths=not args.raw_starter_renderer)
        on_warnings=render_design_to_png(on,SPLIT_DIRS[source_split],on_path,
                                        ROOT/"data"/"crello_fonts.pickle",
                                        adjust_tight_text_widths=not args.raw_starter_renderer)
        for path in (off_path,on_path):
            with Image.open(path) as image:
                if image.size!=preview_size:image.convert("RGB").resize(preview_size,Image.Resampling.LANCZOS).save(path)
        target_payload=json.loads(row["messages"][-1]["content"])
        target_answer=(decode_v2_completion(target_payload,source,target_canvas,record_prompt_version(row))["elements"]
                       if target_payload.get("elements") and "box" in target_payload["elements"][0]
                       else target_payload["elements"])
        import engine.resize as resize_module
        _,source_units=resize_module._analyze(source)
        source_ids={e['id'] for e in source['elements']}
        hard_checks={}
        for label,design in (("qwen_off",off),("qwen_on",on)):
            output_ids={e['id'] for e in design.get('elements',[])}
            hard_checks[label]={
                "all_source_ids_preserved":output_ids==source_ids,
                "missing_ids":sorted(source_ids-output_ids),
                "extra_ids":sorted(output_ids-source_ids),
                "unintended_text_overlaps":resize_module._unintended_text_overlaps(
                    design.get('elements',[]),source_units),
            }
        if raw_qwen is not None:
            backgrounds,_=resize_module._analyze(source)
            raw_design=resize_module._assemble(raw_qwen,backgrounds,source,
                                               target_canvas["width"],target_canvas["height"])
            hard_checks['raw_qwen']={
                "all_source_ids_preserved":{e['id'] for e in raw_design['elements']}==source_ids,
                "unintended_text_overlaps":resize_module._unintended_text_overlaps(
                    raw_design['elements'],source_units),
            }
        qdiag=on_report.get("qwen_diagnostics",{})
        pipeline_elements=qdiag.get("constrained_elements")
        raw_metrics=_geometry_metrics(raw_qwen,target_answer,target_canvas) if raw_qwen else None
        constrained_metrics=_geometry_metrics(pipeline_elements,target_answer,target_canvas) if pipeline_elements else None
        final_metrics=_geometry_metrics(on["elements"],target_answer,target_canvas)
        off_metrics=_geometry_metrics(off["elements"],target_answer,target_canvas)
        visual_report=on_report.get("visual",{})
        siglip_report=dict(visual_report.get("perceptual",{}))
        siglip_report.update({k:visual_report[k] for k in
                              ("objective_cost_components","rejected_contrast_variants","source_contrast")
                              if k in visual_report})
        from engine.qwen_seed import ADAPTER_DIR
        diagnostics=(f"Adapter: {ADAPTER_DIR.resolve()}\n"
                     f"Qwen status: {qdiag.get('status','not reported')}\n"
                     f"Proposal received: {qdiag.get('proposal_received',False)} · "
                     f"constraints accepted: {qdiag.get('constraint_feasible',False)} · "
                     f"optimized: {qdiag.get('optimized',False)}\n"
                     f"Source-stack candidate IDs: {qdiag.get('layer_stack_repaired_ids',[])}\n"
                     f"Qwen candidate energy: {qdiag.get('optimized_energy',qdiag.get('constrained_energy','n/a'))} · "
                     f"engine winner: {on_report.get('winner','unknown')} · "
                     f"Qwen win: {on_report.get('winner','').startswith('qwen-')}\n"
                     f"Hard checks: {hard_checks}\n"
                     f"Target geometry MAE (normalized): raw Qwen {raw_metrics['geometry_mae'] if raw_metrics else 'n/a'} · "
                     f"after engine constraints {constrained_metrics['geometry_mae'] if constrained_metrics else 'n/a'} · "
                     f"final Qwen-on engine {final_metrics['geometry_mae']} · "
                     f"Qwen-off engine {off_metrics['geometry_mae']}\n"
                     f"Relative font-size MAE: raw Qwen {raw_metrics['font_relative_mae'] if raw_metrics else 'n/a'} · "
                     f"final Qwen-on {final_metrics['font_relative_mae']} · Qwen-off {off_metrics['font_relative_mae']}\n"
                     f"SigLIP rerank: {siglip_report.get('note',siglip_report.get('reason','not used'))} · "
                     f"Qwen eligible: {siglip_report.get('qwen_eligible',False)} · "
                     f"safety: {siglip_report.get('qwen_safety_checks','n/a')}\n"
                     f"Interpretation: Qwen proposes geometry; the engine applies constraints, repairs, renders, and ranks candidates.")
        # Persist model and engine geometry as text evidence as well as PNGs.
        output_dir=args.output/"candidate_outputs"
        output_dir.mkdir(exist_ok=True)
        output_file=output_dir/f"{prefix}.json"
        output_file.write_text(json.dumps(dict(pair_id=row['pair_id'],source=source,
            target_canvas=target_canvas,decoded_model=qdiag.get('model_elements'),
            decoder_checked=raw_qwen,relationship_repaired=micro_qwen,
            layer_preserved=layer_qwen,qwen_off=off,qwen_on=on,
            winner=on_report.get('winner')),ensure_ascii=False,indent=2)+"\n")
        entries.append({"adapter_dir":str(ADAPTER_DIR.resolve()),
                        "candidate_output":str(output_file.relative_to(args.output)),"pair_id":row["pair_id"],"canvas":target_canvas,
                        "element_count":len(source["elements"]),
                        "renderer_mode":renderer_mode,
                        "preview":os.path.relpath(preview_path,args.output),
                        "target":target_path.name,"raw":raw_path.name if raw_qwen is not None else "",
                        "micro":micro_path.name if micro_qwen is not None else "",
                        "layer":layer_path.name if layer_qwen is not None else "",
                        "base":off_path.name,"adapter":on_path.name,
                        "off_status":"render warnings" if off_warnings else "rendered",
                        "on_status":"render warnings" if on_warnings else "rendered",
                        "off_warnings":off_warnings,"on_warnings":on_warnings,
                        "raw_warnings":raw_warnings,
                        "layer_warnings":layer_warnings,
                        "hard_checks":hard_checks,
                        "diagnostics":diagnostics,
                        "qwen_diagnostics":{k:v for k,v in qdiag.items() if not k.endswith('_elements')},
                        "siglip_diagnostics":{k:v for k,v in siglip_report.items()
                                              if k in ("used","note","reason","qwen_eligible",
                                                       "qwen_safety_checks","trained_ranker_loaded","tie_band","scores",
                                                       "objective_costs","selection_costs","objective_cost_components",
                                                       "busy_background_tolerated_names",
                                                       "qwen_family_band","eligible_qwen_names",
                                                       "contrast_repair_suppressed_names",
                                                       "source_lockup_rejected_names","source_contrast",
                                                       "rejected_contrast_variants")},
                        "visual_candidate_diagnostics":[
                            {k:c.get(k) for k in ("name","cost","layout_defect_cost","source_lockup_error","has_layout_defects","checked","low_contrast",
                                "readability_floor","readability_floor_met",
                                "invisible_text","occlusion_defects","contrast_by_text","contrast_repairs","busy_background","warnings","error")
                             if k in c}
                            for c in visual_report.get("candidates",[])],
                        "target_geometry_mae":{"raw_qwen":raw_metrics,"after_constraints":constrained_metrics,
                                                "qwen_on_final":final_metrics,"qwen_off_final":off_metrics}})
        print(f"Rendered {number}/{len(selected)}: {row['pair_id']}",flush=True)
    (args.output/"review.html").write_text(_html(entries,renderer_mode),encoding="utf-8")
    (args.output/"manifest.json").write_text(json.dumps(entries,indent=2)+"\n")
    print(json.dumps({"review_page":str((args.output/"review.html").resolve()),
                      "examples":len(entries),"output":str(args.output.resolve())},indent=2))


if __name__=="__main__":main()
