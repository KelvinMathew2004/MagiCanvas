#!/usr/bin/env python3
"""Compare the local Qwen base model and LoRA on held-out Qwen examples."""
from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

SPLIT_DIRS={
    "train":ROOT/"data"/"crello_train",
    "validation":ROOT/"data"/"crello_validation",
    "test":ROOT/"data"/"crello_test",
}


def read_sources():
    indexes={}
    for split,folder in SPLIT_DIRS.items():
        index={}
        for line in (folder/"manifest.jsonl").read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get("status")=="exported":index[row["template_id"]]=row
        indexes[split]=index
    cache={}

    def get(split,design_id):
        key=(split,design_id)
        if key not in cache:
            row=indexes[split][design_id]
            cache[key]=json.loads((SPLIT_DIRS[split]/row["design"]).read_text())
        return cache[key]
    return get


def measure(prediction,gold,canvas):
    if prediction is None:
        return {"valid":False,"geometry_mae":None,"font_mae":None,
                "out_of_canvas":None,"exact_ids":False}
    expected={row["id"]:row for row in gold.get("elements",[])}
    actual={row["id"]:row for row in prediction}
    exact_ids=set(expected)==set(actual) and len(prediction)==len(actual)
    if not exact_ids or not expected:
        return {"valid":False,"geometry_mae":None,"font_mae":None,
                "out_of_canvas":None,"exact_ids":exact_ids}
    cw,ch=float(canvas["width"]),float(canvas["height"])
    geometry=[];fonts=[];out_of_canvas=0
    for element_id,target in expected.items():
        result=actual[element_id]
        geometry.extend((abs(result["x"]-target["x"])/cw,
                         abs(result["y"]-target["y"])/ch,
                         abs(result["width"]-target["width"])/cw,
                         abs(result["height"]-target["height"])/ch))
        if "font_size" in target:
            fonts.append(abs(float(result.get("font_size",0))-float(target["font_size"]))
                         /max(1.,float(target["font_size"])))
        if (result["x"]<0 or result["y"]<0 or result["width"]<=0 or result["height"]<=0
                or result["x"]+result["width"]>cw+1e-4
                or result["y"]+result["height"]>ch+1e-4):
            out_of_canvas+=1
    return {"valid":True,"geometry_mae":sum(geometry)/len(geometry),
            "font_mae":sum(fonts)/len(fonts) if fonts else None,
            "out_of_canvas":out_of_canvas,"exact_ids":True}


def run_mode(records,source_for,adapter_dir,model_dir=None):
    import torch
    import engine.qwen_seed as qwen_seed
    from engine.qwen_prompt import decode_v2_completion, record_prompt_version
    qwen_seed._load.cache_clear()
    if model_dir is not None:qwen_seed.MODEL_DIR=Path(model_dir)
    qwen_seed.ADAPTER_DIR=adapter_dir
    results=[]
    for index,row in enumerate(records,1):
        started=perf_counter()
        source_id=row["pair_id"].split("__to__",1)[0]
        source=source_for(row["source_split"],source_id)
        diagnostics={}
        # Omission fallback needs target-scaled geometry, not original pixels.
        from engine.resize import _analyze, _proportional, _assemble
        backgrounds,units=_analyze(source)
        canvas=row['target_canvas']
        base=_proportional(units,canvas['width'],canvas['height'],source['canvas']['width'],source['canvas']['height'])
        proposal,note=qwen_seed.generate_seed(source,canvas,base,diagnostics=diagnostics)
        gold=json.loads(row["messages"][-1]["content"])
        if gold.get("elements") and "box" in gold["elements"][0]:
            gold=decode_v2_completion(gold,source,canvas,record_prompt_version(row))
        model_rows=[dict(id=eid,**values) for eid,values in diagnostics.get('model_elements',{}).items()]
        result=measure(model_rows or None,gold,canvas)
        if (not diagnostics.get('json_complete') or diagnostics.get('duplicate_ids')
                or diagnostics.get('unknown_ids')):
            result['valid']=False
            if diagnostics.get('duplicate_ids') or diagnostics.get('unknown_ids'):result['exact_ids']=False
        # Completeness is assessed before fallback, font fitting, and bounds clamping.
        result.update(pair_id=row['pair_id'],family_id=row.get('family_id'),status=note,
                      runtime=measure(_assemble(proposal,backgrounds,source,canvas["width"],canvas["height"])["elements"] if proposal is not None else None,gold,canvas),
                      generation={k:v for k,v in diagnostics.items() if k!='model_elements'})
        results.append(result)
        print(f"  {index}/{len(records)}: {'valid' if results[-1]['valid'] else 'invalid'} "
              f"({perf_counter()-started:.1f}s)",flush=True)
    qwen_seed._load.cache_clear()
    gc.collect()
    if hasattr(torch,"mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


def run_engine_mode(records,source_for,adapter_dir,model_dir=None,checkpoint_path=None):
    """Evaluate the final render with the same gates for every layout family."""
    import torch
    import engine.qwen_seed as qwen_seed
    from engine.qwen_prompt import decode_v2_completion, record_prompt_version
    from engine.resize import _adapt_with_diagnostics
    from scripts.layout_defects import assess_engine_layout

    os.environ["MAGICANVAS_QWEN_ENABLED"]="1"
    if model_dir is not None:qwen_seed.MODEL_DIR=Path(model_dir)
    qwen_seed.ADAPTER_DIR=Path(adapter_dir);qwen_seed._load.cache_clear()
    latest={}
    if checkpoint_path and checkpoint_path.is_file():
        for line in checkpoint_path.read_text().splitlines():
            if line.strip():
                row=json.loads(line);latest[row['pair_id']]=row
    results=[];old_render_mode=os.environ.get('MAGICANVAS_DATASET_RENDERING')
    if checkpoint_path:checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        for index,row in enumerate(records,1):
            if row['pair_id'] in latest and not latest[row['pair_id']].get('engine_error'):
                results.append(latest[row['pair_id']]);continue
            started=perf_counter();canvas=row['target_canvas']
            try:
                official=row.get('official_path')
                os.environ['MAGICANVAS_DATASET_RENDERING']='0' if official else '1'
                source=(json.loads(Path(official).read_text())['source'] if official else
                        source_for(row['source_split'],row['pair_id'].split('__to__',1)[0]))
                design,report=_adapt_with_diagnostics(source,canvas)
                if row.get('messages'):
                    gold=json.loads(row['messages'][-1]['content'])
                    if gold.get('elements') and 'box' in gold['elements'][0]:
                        gold=decode_v2_completion(gold,source,canvas,record_prompt_version(row))
                    measured=measure(design.get('elements'),gold,canvas)
                else:
                    gold={'elements':source['elements']}
                    measured=measure(design.get('elements'),gold,canvas)
                    measured.update(geometry_mae=None,font_mae=None)
                qdiag=report.get('qwen_diagnostics',{});generation=qdiag.get('generation',{})
                raw_measure=measure(qdiag.get('model_elements') or None,gold,canvas)
                raw_valid=bool(qdiag.get('proposal_received') and generation.get('json_complete')
                    and not generation.get('duplicate_ids') and not generation.get('unknown_ids')
                    and not generation.get('missing_ids') and raw_measure['valid'])
                defects=assess_engine_layout(source,design,report)
                measured.update(defects,pair_id=row['pair_id'],family_id=row.get('family_id'),
                    winner=report.get('winner',''),qwen_family_win=report.get('winner','').startswith('qwen-'),
                    hard_checks_passed=not defects['flagged'],raw_proposal_valid=raw_valid,
                    raw_geometry_mae=raw_measure['geometry_mae'] if not official else None,
                    raw_relative_font_mae=raw_measure['font_mae'] if not official else None,
                    raw_json_complete=bool(generation.get('json_complete')),
                    qwen_status=report.get('qwen'),source_split=row.get('source_split','official'),
                    generation=generation,
                    target_canvas=canvas,official_path=official,source_design=source,output_design=design,
                    required_families=(report.get('visual') or {}).get('required_families',{}),
                    shortlist_coverage=(report.get('visual') or {}).get('shortlist_coverage',{}),
                    rendered_candidates=(report.get('visual') or {}).get('candidates',[]),
                    qwen_model_elements=qdiag.get('model_elements',[]),
                    qwen_container_tree=qdiag.get('container_tree',{}),
                    macro_reflow=qdiag.get('macro_reflow',{}),
                    runtime=round(perf_counter()-started,2))
            except Exception as exc:
                measured=dict(pair_id=row['pair_id'],family_id=row.get('family_id'),valid=False,
                    exact_ids=False,geometry_mae=None,font_mae=None,raw_geometry_mae=None,
                    raw_relative_font_mae=None,hard_checks_passed=False,flagged=True,
                    defect_flags=['engine_error'],engine_error=f'{type(exc).__name__}: {exc}',
                    runtime=round(perf_counter()-started,2))
            results.append(measured)
            if checkpoint_path:
                with checkpoint_path.open('a') as stream:
                    stream.write(json.dumps(measured,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            print(f"  {index}/{len(records)}: {measured.get('winner',measured.get('engine_error'))} · "
                  f"flags {measured.get('defect_flags')} ({measured['runtime']:.1f}s)",flush=True)
    finally:
        if old_render_mode is None:os.environ.pop('MAGICANVAS_DATASET_RENDERING',None)
        else:os.environ['MAGICANVAS_DATASET_RENDERING']=old_render_mode
        qwen_seed._load.cache_clear();gc.collect()
        if hasattr(torch,'mps') and torch.backends.mps.is_available():torch.mps.empty_cache()
    return results


def summarize(results):
    valid=[row for row in results if row["valid"]]
    def average(key):
        values=[row[key] for row in valid if row[key] is not None]
        return sum(values)/len(values) if values else None
    families=defaultdict(list)
    for row in valid:
        families[row.get('family_id') or row.get('pair_id','unknown')].append(row)
    def family_average(key):
        values=[]
        for members in families.values():
            scores=[row[key] for row in members if row[key] is not None]
            if scores:values.append(sum(scores)/len(scores))
        return sum(values)/len(values) if values else None
    return {
        "examples":len(results),"valid_json_and_complete_ids":len(valid),
        "valid_rate":len(valid)/max(1,len(results)),
        "exact_ids_rate":sum(row["exact_ids"] for row in results)/max(1,len(results)),
        "mean_normalized_geometry_mae":average("geometry_mae"),
        "mean_relative_font_size_mae":average("font_mae"),
        "mean_out_of_canvas_elements":average("out_of_canvas"),
        "families_with_valid_predictions":len(families),
        "family_macro_geometry_mae":family_average('geometry_mae'),
        "family_macro_font_relative_mae":family_average('font_mae'),
    }


def summarize_engine(results):
    valid=[row for row in results if row["valid"]]
    raw_valid=[row for row in results if row.get("raw_proposal_valid")]
    def average(key):
        values=[row[key] for row in valid if row[key] is not None]
        return sum(values)/len(values) if values else None
    def raw_average(key):
        values=[row[key] for row in raw_valid if row.get(key) is not None]
        return sum(values)/len(values) if values else None
    families=defaultdict(list)
    for row in valid:
        families[row.get("family_id") or row.get("pair_id","unknown")].append(row)
    family_maes=[sum(r["geometry_mae"] for r in rows if r["geometry_mae"] is not None)
                 /len([r for r in rows if r["geometry_mae"] is not None])
                 for rows in families.values() if any(r["geometry_mae"] is not None for r in rows)]
    wins=sum(bool(row.get("qwen_family_win")) for row in results)
    hard_passes=sum(bool(row.get("hard_checks_passed")) for row in results)
    raw_valid_count=sum(bool(row.get("raw_proposal_valid")) for row in results)
    return {
        "examples":len(results),"valid_final_layouts":len(valid),
        "valid_rate":len(valid)/max(1,len(results)),
        "mean_post_constraint_geometry_mae":average("geometry_mae"),
        "mean_post_constraint_relative_font_mae":average("font_mae"),
        "mean_raw_geometry_mae_on_valid_proposals":raw_average("raw_geometry_mae"),
        "mean_raw_relative_font_mae_on_valid_proposals":raw_average("raw_relative_font_mae"),
        "family_macro_post_constraint_geometry_mae":sum(family_maes)/len(family_maes) if family_maes else None,
        "qwen_family_wins":wins,"qwen_family_win_rate":wins/max(1,len(results)),
        "raw_proposal_valid_count":len(raw_valid),
        "raw_proposal_valid_rate":raw_valid_count/max(1,len(results)),
        "raw_json_complete_count":sum(bool(row.get("raw_json_complete")) for row in results),
        "raw_json_complete_rate":sum(bool(row.get("raw_json_complete")) for row in results)/max(1,len(results)),
        "hard_check_passes":hard_passes,
        "hard_check_pass_rate":hard_passes/max(1,len(results)),
        "families_with_valid_layouts":len(families),
        "flagged_cases":sum(bool(r.get("flagged")) for r in results),
        "engine_errors":sum(bool(r.get("engine_error")) for r in results),
    }


def _engine_worker(args):
    records,adapter,model,checkpoint=args
    return run_engine_mode(records,read_sources(),Path(adapter),Path(model),Path(checkpoint))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"data"/"qwen_lora"/"test.jsonl")
    parser.add_argument("--output",type=Path,default=ROOT/"data"/"qwen_lora"/"evaluation.json")
    parser.add_argument("--adapter-dir",type=Path,default=None,
                        help="Adapter to evaluate; defaults to MAGICANVAS_QWEN_ADAPTER_DIR or the installed v2 adapter")
    parser.add_argument("--compare-adapter-dir",type=Path,default=None,
                        help="Evaluate a second adapter on the same examples and compare it with --adapter-dir")
    parser.add_argument("--model-dir",type=Path,default=None,
                        help="Local base model for --adapter-dir; defaults to MAGICANVAS_QWEN_DIR or the installed Qwen3 0.6B")
    parser.add_argument("--compare-model-dir",type=Path,default=None,
                        help="Local base model for --compare-adapter-dir; defaults to --model-dir")
    parser.add_argument("--engine",action="store_true",
                        help="Measure final engine-selected layouts, post-constraint geometry, and Qwen-family wins")
    parser.add_argument('--engine-only',action='store_true',help='Run the installed adapter through engine stress gates without extra raw/base passes')
    parser.add_argument('--official-samples',type=Path,help='Also stress source JSON tasks in this folder across all six formats')
    parser.add_argument('--workers',type=int,default=1,help='Independent workers for --engine-only')
    parser.add_argument('--resume',action='store_true',help='Resume --engine-only checkpoints')
    parser.add_argument('--review-top',type=int,default=10,help='Render only the worst flagged outliers')
    parser.add_argument("--limit",type=int,default=0,
                        help="Examples to evaluate (0 means the full held-out test set)")
    args=parser.parse_args()
    if args.workers<1 or args.review_top<1:parser.error('workers and review-top must be positive')
    if args.limit<0:parser.error("--limit must be nonnegative")
    if args.resume and not (args.engine_only or args.engine):
        parser.error('--resume is supported for --engine-only and paired --engine evaluation')
    records=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:records=records[:args.limit]
    if not records:raise SystemExit("No held-out examples found.")
    from engine.qwen_prompt import record_prompt_version
    versions={record_prompt_version(row) for row in records}
    if len(versions)!=1:raise SystemExit('Mixed prompt versions in evaluation file')
    prompt_version=versions.pop()
    # Legacy files lack family metadata: infer connected components from pair
    # endpoints so repeated directions are not counted as independent families.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for row in records:
        a,b=row['pair_id'].split('__to__',1);parent[find(a)]=find(b)
    for row in records:
        row.setdefault('family_id',find(row['pair_id'].split('__to__',1)[0]))
    source_for=read_sources()
    adapter_dir=args.adapter_dir or (Path(os.environ["MAGICANVAS_QWEN_ADAPTER_DIR"])
                                     if os.environ.get("MAGICANVAS_QWEN_ADAPTER_DIR") else None)
    if adapter_dir is None:
        adapter_dir=ROOT/"models"/f"qwen3-layout-lora-v{prompt_version}"
        if not adapter_dir.is_dir():adapter_dir=ROOT/"models"/"qwen3-layout-lora"
    adapter_dir=adapter_dir.expanduser().resolve()
    if not adapter_dir.is_dir():raise SystemExit(f"No adapter directory at {adapter_dir}")
    marker=adapter_dir/"magicanvas_prompt_version.txt"
    adapter_prompt_version=int(marker.read_text().strip()) if marker.is_file() else 1
    if adapter_prompt_version!=prompt_version:
        raise SystemExit(f"Dataset uses prompt v{prompt_version}, but {adapter_dir} uses prompt v{adapter_prompt_version}. Train/install the matching adapter first.")
    os.environ["MAGICANVAS_QWEN_PROMPT_VERSION"]=str(prompt_version)
    import engine.qwen_seed as qwen_seed
    model_dir=(args.model_dir or Path(os.environ.get("MAGICANVAS_QWEN_DIR",qwen_seed.MODEL_DIR))).expanduser().resolve()
    if not model_dir.is_dir():raise SystemExit(f"No base model directory at {model_dir}")
    compare_model_dir=(args.compare_model_dir or model_dir).expanduser().resolve()
    if args.compare_adapter_dir and not compare_model_dir.is_dir():
        raise SystemExit(f"No comparison base model directory at {compare_model_dir}")
    report={"test_file":str(args.input.resolve()),"adapter_dir":str(adapter_dir),
            "model_dir":str(model_dir),"examples":len(records)}
    if args.engine_only:
        if args.compare_adapter_dir:parser.error('--engine-only evaluates one installed adapter; use --engine for A/B comparison')
        from scripts.layout_defects import official_records,build_outlier_viewer
        test_count=len(records)
        if args.official_samples:records.extend(official_records(args.official_samples))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        import hashlib
        fingerprint_files=[args.input,ROOT/'engine/resize.py',ROOT/'engine/qwen_refine.py',
                           ROOT/'engine/qwen_seed.py',ROOT/'engine/qwen_prompt.py',
                           ROOT/'scripts/layout_defects.py',Path(__file__).resolve(),
                           adapter_dir/'adapter_config.json',adapter_dir/'adapter_model.safetensors',
                           model_dir/'config.json',model_dir/'tokenizer_config.json']
        if args.official_samples:fingerprint_files.extend(sorted(args.official_samples.glob('*.json')))
        manifest=dict(model_dir=str(model_dir),adapter_dir=str(adapter_dir),workers=args.workers,
            base_weight_files={path.name:[path.stat().st_size,path.stat().st_mtime_ns]
                               for path in sorted(model_dir.glob('*.safetensors'))},
            sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in fingerprint_files if path.is_file()})
        marker=args.output.with_name(args.output.stem+'.run_manifest.json')
        if args.resume and marker.is_file() and json.loads(marker.read_text())!=manifest:
            raise SystemExit('Engine, data, or run settings differ from this checkpoint. Use a new output directory.')
        if args.resume and not marker.exists() and list(args.output.parent.glob(args.output.stem+'.worker*.jsonl')):
            raise SystemExit('Existing checkpoints have no engine fingerprint. Use a new output directory.')
        marker.write_text(json.dumps(manifest,indent=2)+'\n')
        worker_args=[]
        for i in range(args.workers):
            checkpoint=args.output.with_name(args.output.stem+f'.worker{i}.jsonl')
            if not args.resume and checkpoint.exists():checkpoint.unlink()
            worker_args.append((records[i::args.workers],str(adapter_dir),str(model_dir),str(checkpoint)))
        if args.workers==1:engine_results=_engine_worker(worker_args[0])
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                engine_results=[row for chunk in pool.map(_engine_worker,worker_args) for row in chunk]
        order={r['pair_id']:i for i,r in enumerate(records)}
        engine_results.sort(key=lambda r:order[r['pair_id']])
        report.update(metric_scope='Final engine stress gates',examples=len(records),test_cases=test_count,
                      official_combinations=len(records)-test_count,adapter_engine=summarize_engine(engine_results),
                      test_engine=summarize_engine(engine_results[:test_count]),
                      official_engine=summarize_engine(engine_results[test_count:]),
                      run_manifest=manifest,engine_results=engine_results)
        report['review_page']=build_outlier_viewer(engine_results,records,args.output.parent/'outliers',args.review_top)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='engine_results'},indent=2),flush=True)
        return
    if args.compare_adapter_dir:
        compare_dir=args.compare_adapter_dir.expanduser().resolve()
        if not compare_dir.is_dir():raise SystemExit(f"No comparison adapter directory at {compare_dir}")
        for label,path in (("candidate",adapter_dir),("comparison",compare_dir)):
            marker=path/"magicanvas_prompt_version.txt"
            marker_version=int(marker.read_text().strip()) if marker.is_file() else 1
            if marker_version!=prompt_version:
                raise SystemExit(f"Dataset uses prompt v{prompt_version}, but {path} uses prompt v{marker_version}.")
        if args.engine:
            print("Evaluating candidate adapter through the full engine",flush=True)
            from scripts.layout_defects import build_ab_viewer
            import hashlib
            output_dir=args.output.parent
            candidate_checkpoint=args.output.with_name(args.output.stem+'.candidate.jsonl')
            comparison_checkpoint=args.output.with_name(args.output.stem+'.comparison.jsonl')
            manifest_path=args.output.with_name(args.output.stem+'.run_manifest.json')
            output_dir.mkdir(parents=True,exist_ok=True)
            code_paths=[ROOT/'engine/resize.py',ROOT/'engine/qwen_refine.py',ROOT/'engine/qwen_seed.py',
                        ROOT/'engine/qwen_prompt.py',ROOT/'scripts/layout_defects.py',Path(__file__).resolve()]
            def model_signature(folder):
                return {p.name:[p.stat().st_size,p.stat().st_mtime_ns]
                        for p in sorted(folder.glob('*.safetensors'))}
            def adapter_signature(folder):
                return {name:[(folder/name).stat().st_size,(folder/name).stat().st_mtime_ns]
                        for name in ('adapter_config.json','adapter_model.safetensors',
                                     'magicanvas_prompt_version.txt') if (folder/name).is_file()}
            run_manifest={'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'candidate':{'adapter':str(adapter_dir),'model':str(model_dir),
                             'adapter_files':adapter_signature(adapter_dir),'model_weights':model_signature(model_dir)},
                'comparison':{'adapter':str(compare_dir),'model':str(compare_model_dir),
                               'adapter_files':adapter_signature(compare_dir),'model_weights':model_signature(compare_model_dir)},
                'code_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in code_paths if path.is_file()},'examples':len(records)}
            if args.resume:
                if not manifest_path.is_file():
                    raise SystemExit('No paired A/B run manifest exists; start with a new output path or rerun without --resume.')
                if json.loads(manifest_path.read_text())!=run_manifest:
                    raise SystemExit('Model, adapter, data, or engine code differs from the A/B checkpoints. Use a new output path.')
            else:
                for checkpoint in (candidate_checkpoint,comparison_checkpoint):
                    if checkpoint.exists():checkpoint.unlink()
                manifest_path.write_text(json.dumps(run_manifest,indent=2)+'\n')
            candidate_results=run_engine_mode(records,source_for,adapter_dir,model_dir,candidate_checkpoint)
            print("Evaluating comparison adapter through the full engine",flush=True)
            comparison_results=run_engine_mode(records,source_for,compare_dir,compare_model_dir,comparison_checkpoint)
            candidate_summary=summarize_engine(candidate_results)
            comparison_summary=summarize_engine(comparison_results)
            candidate_mae=candidate_summary["mean_post_constraint_geometry_mae"]
            comparison_mae=comparison_summary["mean_post_constraint_geometry_mae"]
            candidate_font_mae=candidate_summary["mean_post_constraint_relative_font_mae"]
            comparison_font_mae=comparison_summary["mean_post_constraint_relative_font_mae"]
            report.update(metric_scope="Final engine-selected layouts after constraints, local repair, rendering, and ranking",
            comparison_adapter_dir=str(compare_dir),
                comparison_model_dir=str(compare_model_dir),
                candidate_engine=candidate_summary,comparison_engine=comparison_summary,
                results={"candidate":candidate_results,"comparison":comparison_results},
                run_manifest=run_manifest,
                candidate_beats_both_comparison_metrics=(
                    candidate_mae is not None and comparison_mae is not None
                    and candidate_mae<comparison_mae
                    and candidate_summary["raw_json_complete_rate"]>=comparison_summary["raw_json_complete_rate"]
                    and candidate_summary["raw_proposal_valid_rate"]>=comparison_summary["raw_proposal_valid_rate"]
                    and candidate_summary["qwen_family_win_rate"]>comparison_summary["qwen_family_win_rate"]
                    and candidate_summary["valid_rate"]==1.0
                    and candidate_summary["hard_check_pass_rate"]==1.0))
            report['review_page']=build_ab_viewer(candidate_results,comparison_results,records,
                output_dir/'ab_viewer',args.review_top,
                candidate_label=f"Candidate · {model_dir.name} + {adapter_dir.name}",
                comparison_label=f"Baseline · {compare_model_dir.name} + {compare_dir.name}")
        else:
            print("Evaluating candidate adapter on raw held-out proposals",flush=True)
            candidate_results=run_mode(records,source_for,adapter_dir,model_dir)
            print("Evaluating comparison adapter on raw held-out proposals",flush=True)
            comparison_results=run_mode(records,source_for,compare_dir,compare_model_dir)
            report.update(metric_scope="Raw decoded model JSON; no engine constraints are applied",
            comparison_adapter_dir=str(compare_dir),
                comparison_model_dir=str(compare_model_dir),
                candidate=summarize(candidate_results),comparison=summarize(comparison_results),
                results={"candidate":candidate_results,"comparison":comparison_results})
    else:
        print("Evaluating base Qwen on held-out examples",flush=True)
        base_results=run_mode(records,source_for,ROOT/"models"/"no-qwen-adapter",model_dir)
        print("Evaluating Qwen with the selected LoRA",flush=True)
        adapter_results=run_mode(records,source_for,adapter_dir,model_dir)
        base_summary=summarize(base_results)
        adapter_summary=summarize(adapter_results)
        base_mae=base_summary["mean_normalized_geometry_mae"]
        adapter_mae=adapter_summary["mean_normalized_geometry_mae"]
        report.update(base=base_summary,adapter=adapter_summary,
            metric_scope="Raw decoded model JSON; runtime recovery is reported separately per case",
            results={"base":base_results,"adapter":adapter_results},
            adapter_beats_base_geometry=(adapter_mae<base_mae
                                         if adapter_mae is not None and base_mae is not None else None))
        if args.engine:
            print("Evaluating selected adapter through the full engine",flush=True)
            engine_results=run_engine_mode(records,source_for,adapter_dir,model_dir)
            report["adapter_engine"]=summarize_engine(engine_results)
            report["engine_results"]=engine_results
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print(f"Saved: {args.output}")


if __name__=="__main__":main()
