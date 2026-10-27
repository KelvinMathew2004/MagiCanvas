#!/usr/bin/env python3
"""Run both engine modes over held-out Qwen pairs and render only top defects.

The JSONL checkpoint contains measurements for every processed case. Only the
highest-ranked failing cases are rendered into the review page. The rank uses
the engine's rendered layout defect cost, rendered glyph contrast/overflow,
source lockup error, source-axis alignment, and Qwen-on vs Qwen-off geometry.
SigLIP remains inside the engine's existing coarse tie-break path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SPLIT_DIRS={name:ROOT/"data"/f"crello_{name}" for name in ("train","validation","test")}
DEFAULT_INPUTS={
    "test":ROOT/"data"/"qwen_lora_expanded_v2"/"test.jsonl",
    "validation":ROOT/"data"/"qwen_lora_expanded_v2"/"validation.jsonl",
}
DEFAULT_OUTPUT=ROOT/"data"/"qwen_lora"/"batch_defect_mining_v1"
_WORKER_SOURCE_FOR=None
_WORKER_ADAPTER=None
_WORKER_MODEL=None


def _index(folder):
    result={}
    for line in (folder/"manifest.jsonl").read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row.get("status")=="exported":result[row["template_id"]]=row
    return result


def _load_sources():
    indexes={split:_index(folder) for split,folder in SPLIT_DIRS.items()}
    cache={}
    def get(split,design_id):
        key=(split,design_id)
        if key not in cache:
            row=indexes[split][design_id]
            cache[key]=json.loads((SPLIT_DIRS[split]/row["design"]).read_text())
        return cache[key]
    return get,indexes


def _target_elements(record,source):
    from engine.qwen_prompt import decode_v2_completion,record_prompt_version
    payload=json.loads(record["messages"][-1]["content"])
    if payload.get("elements") and "box" in payload["elements"][0]:
        payload=decode_v2_completion(payload,source,record["target_canvas"],
                                     record_prompt_version(record))
    return payload.get("elements",[])


def _geometry_mae(elements,target,canvas):
    expected={row["id"]:row for row in target}
    actual={row["id"]:row for row in elements or []}
    if not expected:return None
    w,h=float(canvas["width"]),float(canvas["height"])
    errors=[]
    for eid,gold in expected.items():
        output=actual.get(eid)
        if output is None:
            errors.extend((1.,1.,1.,1.))
            continue
        errors.extend((abs(float(output.get(key,0))-float(gold.get(key,0)))/denom
                       for key,denom in (("x",w),("y",h),("width",w),("height",h))))
    return sum(errors)/len(errors)


def _winner_raster_metrics(report):
    winner=report.get("winner")
    visual=report.get("visual") or {}
    candidates=visual.get("candidates") or []
    selected=next((candidate for candidate in candidates
                   if candidate.get("name")==winner),None)
    if selected is None:
        selected=next((candidate for candidate in candidates
                       if candidate.get("name","").startswith("qwen-container-projected")),None)
    return selected or {}


def _source_axis_alignment(source,elements,tolerance=4.):
    """Measure how far source-aligned axes drifted, using bounded clusters."""
    output={e["id"]:e for e in elements or []}
    canvas=source["canvas"];items=[]
    for element in source.get("elements",[]):
        if element["id"] not in output or abs(float(element.get("rotation",0)))>1e-4:continue
        if (element["width"]>=.92*canvas["width"] and
                element["height"]>=.92*canvas["height"]):continue
        x=float(element["x"]);y=float(element["y"]);w=float(element["width"]);h=float(element["height"])
        items.append((element,{
            "x_left":(x,float(output[element["id"]]["x"]),float(canvas["width"])),
            "x_center":(x+w/2,float(output[element["id"]]["x"])+float(output[element["id"]]["width"])/2,float(canvas["width"])),
            "x_right":(x+w,float(output[element["id"]]["x"])+float(output[element["id"]]["width"]),float(canvas["width"])),
            "y_top":(y,float(output[element["id"]]["y"]),float(canvas["height"])),
            "y_center":(y+h/2,float(output[element["id"]]["y"])+float(output[element["id"]]["height"])/2,float(canvas["height"])),
            "y_bottom":(y+h,float(output[element["id"]]["y"])+float(output[element["id"]]["height"]),float(canvas["height"])),
        }))
    groups=[]
    for axis in ("x_left","x_center","x_right","y_top","y_center","y_bottom"):
        ordered=sorted(((coord[0],coord[1],coord[2],element["id"])
                        for element,coords in items for coord in (coords[axis],)),key=lambda row:row[0])
        cluster=[]
        def flush():
            if len(cluster)<2:return
            target_median=median(row[1] for row in cluster)
            errors=[abs(row[1]-target_median)/max(1.,row[2]) for row in cluster]
            groups.append({"axis":axis,"ids":[row[3] for row in cluster],
                           "target_median":round(target_median,3),
                           "mean_error":sum(errors)/len(errors),
                           "max_error":max(errors)})
        for row in ordered:
            if cluster and row[0]-cluster[0][0]>tolerance:
                flush();cluster=[]
            cluster.append(row)
        flush()
    return {"cluster_count":len(groups),
            "mean_normalized_error":(sum(group["mean_error"] for group in groups)/len(groups)
                                     if groups else 0.),
            "max_normalized_error":max((group["max_error"] for group in groups),default=0.),
            "clusters":groups}


def _hard_failure(row):
    if "defect_flags" in row:
        return bool(row.get("defect_flags") or row.get("review_flags") or row.get("engine_error"))
    return bool(row.get("engine_error") or row.get("raw_json_complete") is False
                or row.get("missing_ids") or row.get("extra_ids")
                or row.get("layout_has_defects") or row.get("low_contrast_count",0)>0
                or row.get("invisible_text_ids") or row.get("glyph_overflow",{}).get("total_px",0)>.5
                or row.get("source_lockup_error",0.)>.025
                or row.get("axial_alignment",{}).get("mean_normalized_error",0.)>.004
                or (row.get("qwen_on_geometry_mae") is not None
                    and row.get("qwen_off_geometry_mae") is not None
                    and row["qwen_on_geometry_mae"]>row["qwen_off_geometry_mae"]+.002))


def _rank(row):
    delta=(row.get("qwen_on_geometry_mae") or 0.)-(row.get("qwen_off_geometry_mae") or 0.)
    return (float(row.get("layout_defect_cost") or 0.),
            int(row.get("low_contrast_count") or 0),
            int(bool(row.get("invisible_text_ids"))),
            float(row.get("glyph_overflow",{}).get("total_px",0.)),
            float(row.get("source_lockup_error") or 0.),
            float(row.get("axial_alignment",{}).get("mean_normalized_error",0.)),
            delta)


def _atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    temporary.replace(path)


def run_case(record,source_for,adapter_dir,model_dir):
    import engine.qwen_seed as qwen_seed
    import engine.resize as resize_module
    from scripts.layout_defects import assess_engine_layout
    source_id,target_id=record["pair_id"].split("__to__",1)
    source=source_for(record["source_split"],source_id)
    target=_target_elements(record,source);canvas=record["target_canvas"]
    started=perf_counter()
    os.environ["MAGICANVAS_QWEN_ENABLED"]="0"
    off,off_report=resize_module._adapt_with_diagnostics(source,canvas)
    os.environ["MAGICANVAS_QWEN_ENABLED"]="1"
    qwen_seed.MODEL_DIR=Path(model_dir);qwen_seed.ADAPTER_DIR=Path(adapter_dir)
    on,on_report=resize_module._adapt_with_diagnostics(source,canvas)
    selected=_winner_raster_metrics(on_report)
    qdiag=on_report.get("qwen_diagnostics") or {}
    generation=qdiag.get("generation") or {}
    defects=assess_engine_layout(source,on,on_report)
    on_ids={e["id"] for e in on.get("elements",[])}
    source_ids={e["id"] for e in source.get("elements",[])}
    alignment=_source_axis_alignment(source,on.get("elements",[]))
    result={
        "pair_id":record["pair_id"],"family_id":record.get("family_id"),
        "source_template_id":source_id,"target_template_id":target_id,
        "source_split":record["source_split"],"evaluation_split":record.get("evaluation_split"),
        "target_canvas":canvas,
        "element_count":len(source_ids),"winner":on_report.get("winner"),
        "raw_json_complete":bool(generation.get("json_complete")),
        "raw_proposal_received":bool(qdiag.get("proposal_received")),
        "missing_ids":sorted(source_ids-on_ids),"extra_ids":sorted(on_ids-source_ids),
        "qwen_on_geometry_mae":_geometry_mae(on.get("elements"),target,canvas),
        "qwen_off_geometry_mae":_geometry_mae(off.get("elements"),target,canvas),
        "layout_defect_cost":selected.get("layout_defect_cost"),
        "layout_has_defects":bool(selected.get("has_layout_defects",selected.get("name") is None)),
        "low_contrast_ids":selected.get("low_contrast",[]),
        "low_contrast_count":len(selected.get("low_contrast",[])),
        "invisible_text_ids":selected.get("invisible_text",[]),
        "contrast_by_text":selected.get("contrast_by_text",{}),
        "source_lockup_error":selected.get("source_lockup_error"),
        "axial_alignment":alignment,
        "glyph_overflow":{"total_px":sum(defects["glyph_overflow"].values()),
                          "by_text_id":defects["glyph_overflow"]},
        "defect_flags":defects["defect_flags"],
        "exact_defects":defects,
        "warnings":selected.get("warnings",[]),
        "qwen_off_winner":off_report.get("winner"),
        "qwen_off_layout_defect_cost":_winner_raster_metrics(off_report).get("layout_defect_cost"),
        "siglip":(on_report.get("visual",{}).get("perceptual",{}).get("note")
                  or on_report.get("visual",{}).get("perceptual",{}).get("reason")),
        "runtime_seconds":round(perf_counter()-started,2),
        "outputs":{"qwen_off":off,"qwen_on":on},
        "diagnostics":{
            "qwen_status":qdiag.get("status"),"proposal_received":qdiag.get("proposal_received"),
            "container_tree":qdiag.get("container_tree"),"selected_structure":qdiag.get("selected_structure"),
            "visual_winner":selected,"siglip_report":on_report.get("visual",{}).get("perceptual"),
        },
    }
    result["review_flags"]=[]
    if not result["raw_json_complete"]:result["review_flags"].append("raw_json_incomplete")
    if (result["qwen_on_geometry_mae"] is not None and result["qwen_off_geometry_mae"] is not None
            and result["qwen_on_geometry_mae"]>result["qwen_off_geometry_mae"]+.002):
        result["review_flags"].append("geometry_regression_vs_qwen_off")
    if alignment.get("mean_normalized_error",0.)>.004:result["review_flags"].append("source_axis_drift")
    if (result["source_lockup_error"] or 0.)>.025:result["review_flags"].append("source_lockup_drift")
    result["failing"]=_hard_failure(result)
    result["rank"]=_rank(result)
    return result


def _initialize_worker(adapter_dir,model_dir,prompt_version):
    """Load dataset indexes once per worker; keep model loading lazy and cached."""
    global _WORKER_SOURCE_FOR,_WORKER_ADAPTER,_WORKER_MODEL
    os.environ["MAGICANVAS_DATASET_RENDERING"]="1"
    os.environ["MAGICANVAS_QWEN_PROMPT_VERSION"]=str(prompt_version)
    import engine.qwen_seed as qwen_seed
    _WORKER_ADAPTER=Path(adapter_dir)
    _WORKER_MODEL=Path(model_dir)
    qwen_seed.MODEL_DIR=_WORKER_MODEL
    qwen_seed.ADAPTER_DIR=_WORKER_ADAPTER
    qwen_seed._load.cache_clear()
    _WORKER_SOURCE_FOR,_=_load_sources()


def _run_worker(record):
    return run_case(record,_WORKER_SOURCE_FOR,_WORKER_ADAPTER,_WORKER_MODEL)


def _review_html(rows):
    payload=json.dumps(rows,ensure_ascii=False).replace("</","<\\/")
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen batch defect mining</title>
<style>:root{color-scheme:dark;font:15px system-ui;background:#0b1018;color:#edf2f8}body{margin:0;padding:18px}header{position:sticky;top:0;background:#0b1018;padding:8px 0;z-index:2;display:flex;gap:10px;align-items:center}button{font:inherit;padding:9px 14px;border:1px solid #607086;border-radius:9px;background:#1b2635;color:white}button:disabled{opacity:.45}#count,#meta{color:#c8d2df}#meta{white-space:pre-wrap;margin:10px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure,section{margin:0;padding:12px;background:#161f2b;border:1px solid #2a3747;border-radius:12px;min-width:0}figcaption{font-weight:650;margin-bottom:8px}img{display:block;width:100%;max-height:76vh;object-fit:contain;background:#10151d;border-radius:7px}section{margin-top:14px;white-space:pre-wrap;line-height:1.45}@media(max-width:760px){.grid{grid-template-columns:1fr}body{padding:10px}}</style></head><body>
<header><button id="prev">← Previous</button><button id="next">Next →</button><span id="count"></span></header><div id="meta"></div><main class="grid"><figure><figcaption>Dataset target preview</figcaption><img id="preview"></figure><figure><figcaption>Local target JSON render</figcaption><img id="target"></figure><figure><figcaption>Qwen off</figcaption><img id="off"></figure><figure><figcaption>Qwen on</figcaption><img id="on"></figure></main><section id="metrics"></section>
<script>const rows=__ROWS__;let i=0;function show(){if(!rows.length){document.getElementById('meta').textContent='No flagged outputs';document.getElementById('prev').disabled=true;document.getElementById('next').disabled=true;return}const r=rows[i];for(const k of ['preview','target','off','on'])document.getElementById(k).src=r[k]||'';document.getElementById('meta').textContent=`Rank ${i+1}/${rows.length} · ${r.pair_id} · ${r.target_canvas.width}×${r.target_canvas.height} · ${r.element_count} elements · winner ${r.winner}`;document.getElementById('metrics').textContent=JSON.stringify(r.metrics,null,2);document.getElementById('count').textContent=`Worst failures · ${i+1} / ${rows.length}`;document.getElementById('prev').disabled=i===0;document.getElementById('next').disabled=i===rows.length-1}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});show()</script></body></html>'''.replace("__ROWS__",payload)


def build_top_viewer(rows,records,source_for,indexes,output,top):
    from scripts.dataset_renderer import render_design_to_png
    output.mkdir(parents=True,exist_ok=True)
    record_by_pair={row["pair_id"]:row for row in records}
    selected=sorted(rows,key=lambda row:row["rank"],reverse=True)
    failures=[row for row in selected if row.get("failing")]
    chosen=failures[:top]
    entries=[]
    for rank,row in enumerate(chosen,1):
        record=record_by_pair[row["pair_id"]]
        source_id,target_id=row["pair_id"].split("__to__",1)
        target_split=record["target_split"]
        target_meta=indexes[target_split][target_id]
        preview_path=SPLIT_DIRS[target_split]/target_meta["preview"]
        source=source_for(record["source_split"],source_id)
        target=source_for(target_split,target_id)
        prefix=f"{rank:02d}_{source_id}__{target_id}"
        target_path=output/f"{prefix}_target.png"
        render_design_to_png(target,SPLIT_DIRS[target_split],target_path,
                             ROOT/"data"/"crello_fonts.pickle",adjust_tight_text_widths=False)
        paths={"preview":os.path.relpath(preview_path,output),
               "target":target_path.name,"target_canvas":row["target_canvas"],
               "pair_id":row["pair_id"],"element_count":row["element_count"],
               "winner":row.get("winner"),"rank":rank}
        for label in ("qwen_off","qwen_on"):
            design=row["outputs"][label]
            png=output/f"{prefix}_{label}.png"
            warnings=render_design_to_png(design,SPLIT_DIRS[record["source_split"]],png,
                                           ROOT/"data"/"crello_fonts.pickle",adjust_tight_text_widths=False)
            paths["off" if label=="qwen_off" else "on"]=png.name
            paths.setdefault("render_warnings",{})[label]=warnings
        paths["metrics"]={key:value for key,value in row.items()
                           if key not in ("outputs","diagnostics","rank")}
        entries.append(paths)
        print(f"  rendered top-{rank}: {row['pair_id']}",flush=True)
    (output/"review.html").write_text(_review_html(entries),encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(entries,indent=2)+"\n")
    return output/"review.html",len(entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test",type=Path,default=DEFAULT_INPUTS["test"])
    parser.add_argument("--validation",type=Path,default=DEFAULT_INPUTS["validation"])
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-dir",type=Path,default=None)
    parser.add_argument("--model-dir",type=Path,default=None)
    parser.add_argument("--top",type=int,default=10)
    parser.add_argument("--workers",type=int,default=1,
                        help="Independent engine workers; each loads its own cached Qwen model")
    parser.add_argument("--resume",action="store_true",help="Resume from results.jsonl in the output directory")
    parser.add_argument("--limit",type=int,default=0,help="Debug limit; 0 means all 398 cases")
    args=parser.parse_args()
    if args.top<1 or args.limit<0 or args.workers<1:
        parser.error("--top and --workers must be positive and --limit nonnegative")
    os.environ["MAGICANVAS_DATASET_RENDERING"]="1"
    from engine.qwen_prompt import record_prompt_version
    from engine import qwen_seed
    inputs=[]
    for eval_split,path in (("test",args.test),("validation",args.validation)):
        split_rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in split_rows:row["evaluation_split"]=eval_split
        inputs.extend(split_rows)
    if args.limit:inputs=inputs[:args.limit]
    if not inputs:raise SystemExit("No evaluation examples found")
    versions={record_prompt_version(row) for row in inputs}
    if len(versions)!=1:raise SystemExit("Mixed prompt versions in batch inputs")
    prompt_version=versions.pop()
    adapter=(args.adapter_dir or Path(os.environ.get("MAGICANVAS_QWEN_ADAPTER_DIR",qwen_seed.ADAPTER_DIR))).expanduser().resolve()
    model=(args.model_dir or Path(os.environ.get("MAGICANVAS_QWEN_DIR",qwen_seed.MODEL_DIR))).expanduser().resolve()
    if not adapter.is_dir():raise SystemExit(f"No Qwen adapter at {adapter}")
    if not model.is_dir():raise SystemExit(f"No Qwen base model at {model}")
    marker=adapter/"magicanvas_prompt_version.txt"
    adapter_version=int(marker.read_text().strip()) if marker.is_file() else 1
    if adapter_version!=prompt_version:
        raise SystemExit(f"Input prompt v{prompt_version} does not match adapter prompt v{adapter_version}")
    os.environ["MAGICANVAS_QWEN_PROMPT_VERSION"]=str(prompt_version)
    records_path=args.output/"results.jsonl"
    import hashlib
    tracked=[args.test,args.validation,Path(__file__),ROOT/'engine/resize.py',
             ROOT/'engine/qwen_refine.py',ROOT/'engine/qwen_seed.py',ROOT/'engine/qwen_prompt.py',
             ROOT/'scripts/layout_defects.py',adapter/'adapter_config.json',
             adapter/'adapter_model.safetensors',model/'config.json',model/'tokenizer_config.json']
    manifest=dict(metric_version=2,model_dir=str(model),adapter_dir=str(adapter),
        cases=[r['pair_id'] for r in inputs],
        base_weight_files={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(model.glob('*.safetensors'))},
        sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked if p.is_file()})
    marker=args.output/'run_manifest.json'
    if args.resume and records_path.exists():
        if not marker.exists() or json.loads(marker.read_text())!=manifest:
            raise SystemExit('Checkpoint predates these metrics or engine/data/model settings changed. Use a new output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(manifest,indent=2)+'\n')
    results=[];completed=set()
    if args.resume and records_path.is_file():
        latest_by_pair={}
        for line in records_path.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            latest_by_pair[row.get("pair_id")]=row
        results=list(latest_by_pair.values())
        for row in results:
            if not row.get("engine_error"):completed.add(row.get("pair_id"))
    pending=[row for row in inputs if row["pair_id"] not in completed]
    source_for,indexes=_load_sources()
    args.output.mkdir(parents=True,exist_ok=True)
    qwen_seed.MODEL_DIR=model;qwen_seed.ADAPTER_DIR=adapter;qwen_seed._load.cache_clear()
    print(f"Batch: {len(inputs)} cases ({sum(row['evaluation_split']=='test' for row in inputs)} test + "
          f"{sum(row['evaluation_split']=='validation' for row in inputs)} validation), {len(pending)} pending",flush=True)
    with records_path.open("a" if args.resume else "w") as checkpoint:
        def save_result(record,row,started):
            nonlocal results
            if row is None:
                row={"pair_id":record.get("pair_id"),"source_split":record.get("source_split"),
                     "target_canvas":record.get("target_canvas"),"engine_error":"Worker returned no result",
                     "failing":True,"rank":[float("inf"),0,0,0,0,0,0]}
            checkpoint.write(json.dumps(row,ensure_ascii=False)+"\n")
            checkpoint.flush();os.fsync(checkpoint.fileno())
            results=[existing for existing in results
                     if existing.get("pair_id")!=row.get("pair_id")]
            results.append(row)
            elapsed=row.get("runtime_seconds")
            if elapsed is None:elapsed=perf_counter()-started
            print(f"  {len(results)}/{len(inputs)} · {row['pair_id']} · "
                  f"defect {row.get('layout_defect_cost')} · low contrast {row.get('low_contrast_count',0)} · "
                  f"MAE on/off {row.get('qwen_on_geometry_mae')}/{row.get('qwen_off_geometry_mae')} · "
                  f"{elapsed:.1f}s",flush=True)
        def capture_error(record,exc):
            return {"pair_id":record.get("pair_id"),"source_split":record.get("source_split"),
                    "target_canvas":record.get("target_canvas"),"engine_error":f"{type(exc).__name__}: {exc}",
                    "failing":True,"rank":[float("inf"),0,0,0,0,0,0]}

        if args.workers==1:
            for record in pending:
                started=perf_counter()
                try:row=run_case(record,source_for,adapter,model)
                except Exception as exc:row=capture_error(record,exc)
                save_result(record,row,started)
                gc.collect()
        elif pending:
            started_by_pair={record["pair_id"]:perf_counter() for record in pending}
            try:
                with ProcessPoolExecutor(
                    max_workers=args.workers,initializer=_initialize_worker,
                    initargs=(str(adapter),str(model),prompt_version),
                ) as pool:
                    futures={pool.submit(_run_worker,record):record for record in pending}
                    for future in as_completed(futures):
                        record=futures[future]
                        try:row=future.result()
                        except Exception as exc:row=capture_error(record,exc)
                        save_result(record,row,started_by_pair[record["pair_id"]])
            except KeyboardInterrupt:
                print("Interrupted; completed rows remain checkpointed and can be resumed.",flush=True)
    # Keep the resumable checkpoint canonical after a retry: the append-only
    # log may contain an older failure followed by its successful replacement.
    # A compact latest-row snapshot prevents the final artifact from exposing
    # stale errors or counting a pair twice.
    compact_path=records_path.with_suffix(records_path.suffix+".tmp")
    with compact_path.open("w") as checkpoint:
        for row in results:
            checkpoint.write(json.dumps(row,ensure_ascii=False)+"\n")
        checkpoint.flush();os.fsync(checkpoint.fileno())
    compact_path.replace(records_path)
    good=[row for row in results if not row.get("engine_error")]
    for row in good:
        row["failing"]=_hard_failure(row);row["rank"]=_rank(row)
    ordered=sorted(good,key=lambda row:row["rank"],reverse=True)
    summary={
        "test_cases":sum(row.get("evaluation_split")=="test" for row in good),
        "validation_cases":sum(row.get("evaluation_split")=="validation" for row in good),
        "processed_cases":len(good),"errors":[row for row in results if row.get("engine_error")],
        "failing_cases":sum(bool(row.get("failing")) for row in good),
        "adapter_dir":str(adapter),"model_dir":str(model),
        "rank_order":["layout_defect_cost desc","low_contrast_count desc","invisible_text","glyph_overflow_px","source_lockup_error","axial_alignment_error","Qwen-on geometry MAE regression vs Qwen-off"],
        "top_20":[{k:v for k,v in row.items() if k not in ("outputs","diagnostics")} for row in ordered[:20]],
    }
    _atomic_json(args.output/"results.json",ordered)
    _atomic_json(args.output/"summary.json",summary)
    viewer,count=build_top_viewer(good,inputs,source_for,indexes,args.output/"top_failures",args.top)
    print(json.dumps({"summary":str((args.output/"summary.json").resolve()),
                      "all_case_results":str((args.output/"results.json").resolve()),
                      "review_page":str(viewer.resolve()),"review_cases":count,
                      "processed":len(good),"errors":len(results)-len(good),
                      "failing":summary["failing_cases"]},indent=2))


if __name__=="__main__":main()
