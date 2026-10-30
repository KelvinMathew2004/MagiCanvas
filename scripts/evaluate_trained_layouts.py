#!/usr/bin/env python3
"""Fresh baseline/new-adapter inference on 89 reference pairs + the 59 review tasks.

No Qwen-off arm, no old prediction reuse, no automatic production promotion.
Checkpoints resume only when all recorded engine/data/weight hashes still match.
"""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import argparse,json,os,sys,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'data/upgrade_exploration/deps'))


def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return sum(values)/len(values) if values else None


def promotion(baseline,candidate):
    base={r['pair_id']:r for r in baseline};new={r['pair_id']:r for r in candidate}
    if base.keys()!=new.keys():raise ValueError('Evaluation inventories differ')
    ref=[k for k in base if k.startswith('reference:')];review=[k for k in base if k.startswith('review:')]
    common=[k for k in ref if base[k].get('raw_proposal_valid') and new[k].get('raw_proposal_valid')]
    def attempted_task(row):
        canvas=row.get('source_design',{}).get('canvas')
        # Exact-size identity preservation intentionally bypasses generation.
        # Keep its final output in regression gates, not raw model completeness.
        return not (canvas is not None and canvas==row.get('target_canvas'))
    dense=[k for k in base if attempted_task(base[k]) and len(base[k].get('source_design',{}).get('elements',[]))>=18]
    def valid(rows,ids):return sum(bool(rows[k].get('raw_proposal_valid')) for k in ids)
    def lower(key):
        a=mean([base[k] for k in common],key);b=mean([new[k] for k in common],key)
        return a is not None and b is not None and b<a-1e-8
    added={k:sorted(set(new[k].get('defect_flags',[]))-set(base[k].get('defect_flags',[]))) for k in base}
    added={k:v for k,v in added.items() if v}
    bg=mean([base[k] for k in ref],'geometry_mae');ng=mean([new[k] for k in ref],'geometry_mae')
    bf=mean([base[k] for k in ref],'font_mae');nf=mean([new[k] for k in ref],'font_mae')
    checks=dict(raw_reference_ids_nonregression=all(not base[k].get('raw_proposal_valid') or new[k].get('raw_proposal_valid') for k in ref),
        raw_dense_previously_valid_preserved=all(not base[k].get('raw_proposal_valid') or new[k].get('raw_proposal_valid') for k in dense),
        raw_dense_ids_improve_or_remain_complete=bool(dense) and
            (valid(new,dense)>valid(base,dense) or valid(new,dense)==len(dense)),
        raw_geometry_improves_on_common_valid=lower('raw_geometry_mae'),
        raw_font_improves_on_common_valid=lower('raw_relative_font_mae'),
        no_new_flag_types=not added,
        review_flagged_at_most_one=sum(bool(new[k].get('flagged')) for k in review)<=1,
        all_final_ids_valid=all(r.get('exact_ids') and r.get('valid') for r in new.values()),
        no_engine_errors=not any(r.get('engine_error') for r in new.values()),
        final_reference_geometry_improves=ng is not None and bg is not None and ng<bg-1e-8,
        historical_geometry_reference_beaten=ng is not None and ng<.04477334347,
        final_reference_font_nonregression=nf is not None and bf is not None and nf<=bf+1e-8,
        learned_wins_increase=sum(bool(r.get('qwen_family_win')) for r in new.values())>
                              sum(bool(r.get('qwen_family_win')) for r in base.values()))
    return dict(eligible_for_human_review=all(checks.values()),checks=checks,new_flags=added,
        common_valid_reference_pairs=len(common),reference_pairs=len(ref),review_tasks=len(review),
        dense_tasks=len(dense),dense_raw_valid_before=valid(base,dense),dense_raw_valid_after=valid(new,dense),
        final_reference_geometry_before=bg,final_reference_geometry_after=ng,
        note='Development/reference gates, not an unseen test. No adapter is installed automatically. Historical .04477334347 belongs to its recorded selector replay; the fresh baseline is also required.')


def verify_recovered(adapter,model):
    """A recovered checkpoint is evaluable, without inventing training provenance."""
    from scripts.verify_qwen_adapter import verify
    result=verify(adapter,model)
    recovered=json.loads((adapter/'recovery_manifest.json').read_text())
    for name,digest in recovered['files_sha256'].items():
        if Path(name).name!=name or sha256((adapter/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Recovered artifact changed: '+name)
    if recovered['global_step']!=recovered['max_steps'] or recovered['completed_epochs']!=3:
        raise ValueError('This comparison expects the completed three-epoch recovery')
    cfg=json.loads((adapter/'adapter_config.json').read_text())
    if cfg['base_model_name_or_path']!='Qwen/Qwen3-0.6B':raise ValueError('Recovered single-candidate path requires matching Qwen3-0.6B')
    return dict(verification=result,recovery=recovered,original_training_manifest_present=(adapter/'training_manifest.json').is_file(),
                limitation='Recovered weights are verified for evaluation; absent original bundle/environment provenance is not reconstructed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--trained',type=Path,help='Root containing all three finished training packages')
    mode.add_argument('--candidate-adapter',type=Path,help='One recovered completed Qwen3-0.6B final export')
    p.add_argument('--reuse-inference-from',type=Path,help='Replay matching saved predictions after an engine-only correction; never reuse final layouts')
    args=p.parse_args()
    from scripts.refresh_regression_viewer import PACK,fingerprint
    from scripts.evaluate_qwen_adapter import run_engine_mode,read_sources,summarize_engine
    import engine.qwen_seed as qwen_seed
    manifest=json.loads((PACK/'review/regression_suite/run_manifest.json').read_text())
    current=fingerprint(ROOT/'config/regression_review.json')
    for k in ('files_sha256','model','model_weights','adapter','layout_experiments'):
        if current.get(k)!=manifest['fingerprint'].get(k):raise SystemExit('Refresh current regression viewer before fresh model evaluation')
    signature=sha256(json.dumps(manifest['fingerprint'],sort_keys=True).encode()).hexdigest()[:16]
    run=ROOT/'data/qwen_lora/regression_suite'/signature
    if not (run/'summary.json').exists():raise ValueError('Current completed replay snapshot is unavailable')
    originals=read_sources();source_map={};records=[]
    for r in map(json.loads,(ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines()):
        r=deepcopy(r);old=r['pair_id'];key='reference:'+old.split('__to__')[0]
        source_map[(r['source_split'],key)]=originals(r['source_split'],old.split('__to__')[0])
        r['pair_id']='reference:'+old;records.append(r)
    for file in sorted(run.glob('[0-9][0-9].json')):
        saved=json.loads(file.read_text());r=deepcopy(saved['record']);after=saved['after']
        if r.get('official_path') and json.loads(Path(r['official_path']).read_text())['source']!=after['source_design']:
            raise ValueError('Official source changed since the current review snapshot')
        key='review:'+r['pair_id'].split('__to__')[0]
        source_map[(r['source_split'],key)]=after['source_design']
        r['pair_id']='review:'+r['pair_id'];records.append(r)
    if len(records)!=148:raise ValueError(f'Expected frozen 89+59, found {len(records)}')
    if args.candidate_adapter:
        # Start with current review tasks for early visual triage; every one of
        # the 148 tasks is still required before reporting promotion gates.
        records.sort(key=lambda r:not r['pair_id'].startswith('review:'))
    candidate_config=json.loads((ROOT/'config/training_candidates.json').read_text())
    candidates=[] if args.candidate_adapter else [s for key in candidate_config['training_queue'] for s in candidate_config['models'] if s['key']==key]
    provider_specs={s['key']:s for s in candidates}
    specs=[('baseline',ROOT/'models/qwen3-0.6b',ROOT/'models/qwen3-layout-lora-v2')]
    specs += [(s['key'],ROOT/'models'/s['key'],args.trained/s['key']/'final') for s in candidates]
    provenance={}
    if args.candidate_adapter:
        specs.append(('audited-0.6b-recovered',ROOT/'models/qwen3-0.6b',args.candidate_adapter.resolve()))
    from scripts.verify_qwen_adapter import verify
    from training.inference import verify_candidate,candidate_provider
    for name,model,adapter in specs:
        if name=='baseline':verify(adapter,model)
        elif args.candidate_adapter:provenance[name]=verify_recovered(adapter,model)
        else:verify_candidate(adapter,model,provider_specs[name])
    from engine.resize import _asset_root
    assets={_asset_root(e)/e['asset_id'] for source in source_map.values() for e in source['elements']
            if e.get('type')=='image' and e.get('asset_id')}
    signature=dict(engine={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'engine').glob('*.py'))},
        regression_fingerprint=current,
        assets={str(p):sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING' for p in sorted(assets)},
        evaluator=sha256(Path(__file__).read_bytes()).hexdigest(),
        core_evaluator=sha256((ROOT/'scripts/evaluate_qwen_adapter.py').read_bytes()).hexdigest(),
        inference_reuse_origin=str(args.reuse_inference_from.resolve()) if args.reuse_inference_from else None,
        inference_reuse_code_sha256=sha256((ROOT/'scripts/reuse_adapter_inference.py').read_bytes()).hexdigest() if args.reuse_inference_from else None,
        inputs=sha256(json.dumps(dict(records=records,sources={str(k):v for k,v in source_map.items()}),sort_keys=True).encode()).hexdigest(),
        runtime_environment={k:v for k,v in os.environ.items() if k.startswith('MAGICANVAS_')},
        adapters={name:{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(adapter.iterdir())
            if p.suffix in ('.json','.txt','.safetensors')} for name,_,adapter in specs},
        candidates=candidates,recovered_candidate_provenance=provenance,models={name:{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(model.iterdir())
            if p.suffix in ('.json','.txt','.safetensors')} for name,model,_ in specs})
    folder=ROOT/'data/training_next/evaluation'/sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    folder.mkdir(parents=True,exist_ok=True);(folder/'manifest.json').write_text(json.dumps(signature,indent=2)+'\n')
    print('Fresh paired evaluation folder:',folder,flush=True)
    os.environ['MAGICANVAS_REQUIRE_QWEN_ADAPTER']='1';os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    results={}
    for name,model,adapter in specs:
        print('Starting arm:',name,'adapter:',adapter,flush=True)
        if args.reuse_inference_from:
            from scripts.reuse_adapter_inference import restore_predictions
            restore_predictions(args.reuse_inference_from,folder,signature,name,records,source_map,adapter,model)
        if name=='baseline' or args.candidate_adapter:
            results[name]=run_engine_mode(records,lambda split,sid:source_map[(split,sid)],adapter,model,folder/f'{name}.jsonl')
        else:
            with candidate_provider(provider_specs[name],model,adapter,folder/'source_images'):
                results[name]=run_engine_mode(records,lambda split,sid:source_map[(split,sid)],adapter,model,folder/f'{name}.jsonl')
        if any(r.get('engine_error') for r in results[name]):raise RuntimeError(f'{name}: engine errors; inspect checkpoint, then resume')
    report=dict(summaries={k:summarize_engine(v) for k,v in results.items()},provenance=provenance,
        qwen_enabled=True,fresh_inference=True,production_changed=False,
        gates={k:promotion(results['baseline'],results[k]) for k in results if k!='baseline'})
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    publish(folder,results,report,PACK/'review/trained_adapters')
    print(json.dumps(dict(report=str(folder/'report.json'),viewer='http://127.0.0.1:8766/review/trained_adapters/index.html',gates=report['gates']),indent=2))


def publish(folder,results,report,dest):
    from scripts.dataset_renderer import render_design_to_png as dataset_render
    from renderer.converter.convert import render_design_to_png as brand_render
    dest.mkdir(parents=True,exist_ok=True);images=dest/'images';images.mkdir(exist_ok=True)
    cards=[]
    published=[]
    for i,base in enumerate(results['baseline']):
        source=base['source_design'];split=base.get('source_split','train');official=base.get('official_path')
        def render(doc,path):
            # Inert source spacer/zero-area elements remain in audit JSON but
            # cannot be passed as nonpositive raster dimensions.
            doc=deepcopy(doc);doc['elements']=[e for e in doc['elements'] if min(e['width'],e['height'])>0]
            if official:brand_render(doc,ROOT/'brand_kit/assets',path,fonts_dir=ROOT/'brand_kit/fonts')
            else:dataset_render(doc,ROOT/f'data/crello_{split}',path,ROOT/'data/crello_fonts.pickle',adjust_tight_text_widths=False)
        render(source,images/f'{i}-original.png')
        options=[];audits={}
        for name,rows in results.items():
            r=rows[i];render(r['output_design'],images/f'{i}-{name}.png');audits[name]=r
            options.append(f'<option value="images/{i}-{name}.png">{html.escape(name)}</option>')
        published.append(dict(index=i,pair_id=base['pair_id'],source=f'images/{i}-original.png',
            outputs={name:f'images/{i}-{name}.png' for name in results},audit=f'{i}-audit.json'))
        (dest/f'{i}-audit.json').write_text(json.dumps(audits,ensure_ascii=False)+'\n')
        cards.append(f'<article><h2>{html.escape(base["pair_id"])}</h2><label>Resized model <select onchange="document.getElementById(\'r{i}\').src=this.value">'+''.join(options)+f'</select></label><div class="pair"><figure><figcaption>Original</figcaption><img src="images/{i}-original.png"></figure><figure><figcaption>Resized</figcaption><img id="r{i}" src="images/{i}-baseline.png"></figure></div><details><summary>Audit</summary><a href="{i}-audit.json">Raw and final geometry / flags</a></details></article>')
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (dest/'manifest.json').write_text(json.dumps(published,indent=2)+'\n')
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fresh trained-adapter comparison</title><style>body{font:16px system-ui;background:#101820;color:#eee;margin:24px}article{padding:20px;border:1px solid #435160;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr}img{max-width:100%;max-height:680px;object-fit:contain}a{color:#acd5ff}</style><h1>Fresh inference: baseline and audited adapters</h1><p>All 148 development/reference tasks. Qwen enabled. Model selector changes only the Resized panel. Automatic gates do not certify aesthetic quality. <a href="report.json">Promotion report</a></p>'+''.join(cards))


if __name__=='__main__':main()
