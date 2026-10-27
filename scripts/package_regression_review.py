#!/usr/bin/env python3
"""Refresh Gemini's code/evidence packet and small current-image upload ZIP.

Run only after refresh_regression_viewer.py; refuse stale snapshots. No model
weights, secrets, full datasets or font archives are copied or uploaded.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.refresh_regression_viewer import PACK,fingerprint


def digest_manifest(folder):
    (folder/'MANIFEST.sha256').write_text('\n'.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder).as_posix()}' for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.sha256','.DS_Store'))+'\n')


def package(packet=PACK,test_log=None):
    from PIL import Image,ImageDraw
    viewer=packet/'review/regression_suite'
    manifest=json.loads((viewer/'run_manifest.json').read_text())
    fp=fingerprint(ROOT/'config/regression_review.json')
    if fp['files_sha256']!=manifest['fingerprint']['files_sha256']:
        raise SystemExit('Viewer is stale: replay the fixed cohort before packaging.')
    summary=json.loads((viewer/'summary.json').read_text());rows=json.loads((viewer/'manifest.json').read_text())
    changed=[r for r in summary['results'] if r['max_geometry_change']>1e-4]
    flips=[r for r in summary['results'] if r['before_winner'] is not None and r['before_winner']!=r['winner']]
    added=[r for r in summary['results'] if r['before_flags'] is not None and set(r['flags'])-set(r['before_flags'])]
    flagged=[r for r in summary['results'] if r['flags']]
    test_result='No test log supplied; no new test count claimed.'
    if test_log:
        lines=Path(test_log).read_text().splitlines();test_result=next((line for line in reversed(lines) if 'passed' in line),lines[-1] if lines else 'Empty test log')
    latest=f'''# Current unified regression review — October 27, 2026

Open `review/regression_suite/index.html` (the main-engine URL points here).
All {len(rows)} conversions share one engine snapshot: 18 official, eight
independently authored alternate conversions, eight complex held-out and four
complex validation dataset cases. Only Original and Resized are displayed.
Intermediates, actual raw validity and selected-candidate diagnostics are in
per-case audit JSON. Zero automated flags is not proof of aesthetic perfection.

## Current changes

Read docs/ENGINE_UPGRADE_CHECKLIST.md and docs/ENGINE_UPGRADE_RESULTS.md first.
Implemented bounded missing-ID completion, gated atomic sibling grid/row/column
candidates, exact full-document/backing render reuse, equivalent vectorized
glyph statistics, capability-aware runtimes and fail-closed adapter-required
submission preparation. Existing final peer equalization was verified rather
than duplicated. No design IDs or content strings drive the engine.

## Verification scope

Both arms keep Qwen enabled and use the same cached raw predictions and adapter.
The baseline is the frozen prior engine; the after arm uses current code.
This 38-case comparison is not Qwen-off or fresh model generation.
{len(changed)} layouts changed materially; {len(flips)} winners flipped;
{len(added)} cases acquired new automated flag types. {len(flagged)} current
outputs still have automated flags. Inspect those outliers and winner changes
in `evidence/unified_regression_summary.json`. Tests: {test_result}.

This fixed development cohort does not replace the full 89/309 release checks;
neither full split is newly claimed here. The alternate designs and official
samples have no human resized target; their MAE is deliberately null. Dataset
cases were chosen by complexity/family diversity, not selected for good outputs.

## Next steps

Fresh, separate 20-case A/B evidence is in evidence/engine_upgrade_dense_ab_summary.json
and review/dense_ab/index.html in the full local packet. Raw exact IDs improve
12/20 to 20/20 with 1.7B, but final learned-family wins stay 5/20 in both arms.
Final dataset geometry MAE improves ~6%; final font MAE worsens ~15% and generation
takes ~1.8 times longer. The default remains 0.6B. No Qwen retraining is indicated
by the missing-ID experiment alone; the larger model still loses seven/eight
authored layouts to deterministic candidates despite complete inventories.

The user supplied 60 genuine occupancy preferences, all hash-verified; 46 pass
fidelity gates. Their valid criticism was that the original batch taught only
scale. A randomized actual-engine-candidate batch now replaces that shortcut.
No labels or head weights were invented and no ranker was trained/promoted from
the narrow batch. Public adapter publication remains pending; local artifacts
are verified, but no Docker build is claimed because the daemon is unavailable.

Performance is measured, not promised: exact render caching reused 9/16 requests;
two-process rendering was slower than serial on that microbenchmark, so it
stays opt-in. The cold replay profile points to geometry/asset lookups as well
as imports and vision, not rendering alone. Read the measured scope in results.

The compact sibling ZIP contains all 38 current original/resized contact sheets,
selected output/raw/source JSON, important current code and evidence. It excludes
weights, full assets and fonts and is a review packet, not a runnable submission.
'''
    (packet/'LATEST_UPDATE.md').write_text(latest)
    prompt='''# Gemini review request

Read LATEST_UPDATE.md, docs/ENGINE_UPGRADE_RESULTS.md and the checklist first.
Review the 38 current original/resized comparisons. Distinguish model proposals
from engine-selected layouts. Identify repeated root causes, avoid design-ID
patches, and propose bounded changes with regression checks. Do not assume that
zero flags means clean alignment, that 1.7B is automatically better, or that
SigLIP2 needs full-backbone fine-tuning. Audit bounded completion and atomic
sibling candidate safety, exact raster-cache keys, source/target glyph metric
equivalence, weak preference-label distribution and extreme-aspect failures.
Historical documents have their own earlier scope; current evidence is named
unified_regression_*. No adapter or full assets are distributed in this packet.
'''
    (packet/'GEMINI_REVIEW_PROMPT.md').write_text(prompt)
    (packet/'README_FOR_GEMINI.md').write_text('Current review: read LATEST_UPDATE.md and GEMINI_REVIEW_PROMPT.md first. Open review/regression_suite/index.html for all 38 current Original/Resized comparisons. Important code and evidence are refreshed by package_regression_review.py. Other review folders and audit documents are historical checkpoints with their own scope. The sibling compact ZIP is the small current upload; neither packet includes model weights or full assets/fonts.\n')
    for dirname in ('code/engine','code/scripts','code/tests','docs','evidence','config'):(packet/dirname).mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'engine').glob('*.py'))+[ROOT/p for p in (
        'scripts/refresh_regression_viewer.py','scripts/package_regression_review.py','scripts/replay_qwen_engine.py',
        'scripts/build_focus_reflow_viewer.py','scripts/layout_defects.py','scripts/evaluate_qwen_adapter.py',
        'scripts/train_siglip_ranker.py','scripts/fetch_siglip.py','scripts/refresh_review_viewer.py',
        'scripts/build_upgrade_review.py','scripts/evaluate_dense_qwen_ab.py','scripts/evaluate_dense_engine_ab.py',
        'scripts/prepare_siglip_preferences.py','scripts/prepare_engine_preferences.py','scripts/prepare_dense_continuation.py',
        'scripts/benchmark_render_cache.py','scripts/profile_frozen_engine.py','scripts/verify_qwen_adapter.py',
        'scripts/prepare_submission_context.py',
        'tests/test_copy_axis_contracts.py','tests/test_layout_mechanics.py','tests/test_macro_layout.py',
        'tests/test_upgrade_primitives.py','tests/test_preference_labels.py','tests/test_qwen_pipeline.py','tests/test_source_evidence.py','Dockerfile')]
    for p in sources:
        target=packet/'code'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('ENGINE_UPGRADE_CHECKLIST.md','ENGINE_UPGRADE_RESULTS.md','REGRESSION_REVIEW_AND_NEXT_STEPS.md','PEER_ROW_ALIGNMENT.md','UNIVERSAL_REFLOW_RESEARCH.md','DAY4_5_AUDIT.md'):
        shutil.copy2(ROOT/'docs'/name,packet/'docs'/name)
    shutil.copy2(ROOT/'config/regression_review.json',packet/'config/regression_review.json')
    shutil.copy2(ROOT/'README.md',packet/'CURRENT_PROJECT_README.md')
    for src,name in [(viewer/'summary.json','unified_regression_summary.json'),(viewer/'run_manifest.json','unified_regression_manifest.json')]:
        shutil.copy2(src,packet/'evidence'/name)
    trace=ROOT/'data/qwen_lora/shared_column_audit'
    for p in trace.glob('*.json'):shutil.copy2(p,packet/'evidence'/('copy_axis_'+p.name))
    for p in trace.glob('*_comparison.jpg'):shutil.copy2(p,packet/'evidence'/('copy_axis_'+p.name))
    upgrade=ROOT/'data/qwen_lora/engine_upgrade'
    for name in ('dense_ab_summary.json','regression_pixels.json','peer_gaps.json','render_benchmark.json','human_labels_received.json','preferences_v2_summary.json'):
        if (upgrade/name).is_file():shutil.copy2(upgrade/name,packet/'evidence'/('engine_upgrade_'+name))
    if (upgrade/'profile/profile.json').is_file():shutil.copy2(upgrade/'profile/profile.json',packet/'evidence/engine_upgrade_profile.json')
    for arm in ('0.6b','1.7b'):
        raw=ROOT/'data/qwen_lora/dense_ab_fresh_v1'/f'{arm}.manifest.json'
        if raw.is_file():shutil.copy2(raw,packet/'evidence'/f'engine_upgrade_raw_{arm}_manifest.json')
    # The private adapter context contains weights and is deliberately never copied.
    overview=packet/'review/index.html'
    overview.write_text('<!doctype html><meta charset="utf-8"><title>MagiCanvas review</title><h1>Current engine review</h1><p><a href="regression_suite/index.html">Unified 38-case Original/Resized viewer</a></p><p><a href="dense_ab/index.html">Fresh 20-case model A/B (experimental)</a></p><p><a href="preferences/review.html">Randomized actual-engine preference review</a></p><p>Read LATEST_UPDATE.md and ENGINE_UPGRADE_RESULTS.md. No model-off arm, public upload or automatically generated approvals.</p>')
    page=viewer/'index.html'
    nav='<p id="upgrade-links"><a href="../dense_ab/index.html">Fresh model A/B experiment</a> · <a href="../preferences/review.html">Human preference review</a></p>'
    body=page.read_text()
    if 'id="upgrade-links"' not in body:page.write_text(body.replace('</header>',nav+'</header>'))
    if test_log:shutil.copy2(test_log,packet/'evidence/unified_regression_tests.log')
    small=ROOT/'data/gemini_review_current_compact';stage=small.with_name(small.name+'.next')
    if stage.exists():shutil.rmtree(stage)
    for name in ('images','cases','docs','evidence','code/engine','code/scripts','code/tests','config'):(stage/name).mkdir(parents=True,exist_ok=True)
    for name in ('LATEST_UPDATE.md','GEMINI_REVIEW_PROMPT.md','CURRENT_PROJECT_README.md'):shutil.copy2(packet/name,stage/name)
    for p in sources:
        dest=stage/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(ROOT/'config/regression_review.json',stage/'config/regression_review.json')
    for p in (packet/'docs').glob('*.md'):
        if p.name in ('ENGINE_UPGRADE_CHECKLIST.md','ENGINE_UPGRADE_RESULTS.md','REGRESSION_REVIEW_AND_NEXT_STEPS.md','PEER_ROW_ALIGNMENT.md','DAY4_5_AUDIT.md'):shutil.copy2(p,stage/'docs'/p.name)
    for p in (packet/'evidence').glob('*'):
        if p.name.startswith(('unified_regression_','copy_axis_','engine_upgrade_')) and p.is_file():shutil.copy2(p,stage/'evidence'/p.name)
    cards=[]
    for i,row in enumerate(rows):
        sheet=Image.new('RGB',(1200,700),'#182532');draw=ImageDraw.Draw(sheet)
        for col,(label,key) in enumerate((('Original','source'),('Resized','image'))):
            with Image.open(viewer/row[key]) as im:
                im=im.convert('RGB');im.thumbnail((580,650));sheet.paste(im,(col*600+(600-im.width)//2,38+(650-im.height)//2))
            draw.text((col*600+10,12),label,fill='white')
        image=f'images/{i:02d}.jpg';sheet.save(stage/image,quality=82,optimize=True)
        audit=json.loads((viewer/row['audit']).read_text())
        (stage/'cases'/f'{i:02d}_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        shutil.copy2(viewer/row['json'],stage/'cases'/f'{i:02d}_output.json')
        cards.append(f'<article><h2>{html.escape(row["case"])}</h2><img loading="lazy" src="{image}" alt="Original and resized"><p>{html.escape(row["cohort"])} · {html.escape(row["winner"])} · flags {html.escape(str(row["defect_flags"]))}</p><a href="cases/{i:02d}_audit.json">Audit JSON</a></article>')
    (stage/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current Gemini review</title><style>body{background:#101820;color:white;font:16px system-ui;margin:24px}img{width:100%;max-width:1200px}article{margin-bottom:40px}</style><h1>All 38 current original/resized comparisons</h1><p>Read LATEST_UPDATE.md. Qwen enabled; saved model predictions replayed through the current engine. This compact packet excludes runtime weights and full assets.</p>'+''.join(cards))
    (stage/'README_FOR_GEMINI.md').write_text('Read LATEST_UPDATE.md, then GEMINI_REVIEW_PROMPT.md. Open index.html for all 38 comparisons. Current engine code and per-case raw/source/output geometry are included. Historical archives, model weights, full assets and fonts are excluded. This packet is not runnable standalone.\n')
    digest_manifest(stage)
    backup=small.with_name(small.name+'.previous')
    if backup.exists():shutil.rmtree(backup)
    if small.exists():small.rename(backup)
    stage.rename(small)
    dest=small.with_suffix('.zip');temporary=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(small.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(small.parent))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip():raise RuntimeError('ZIP verification failed')
    temporary.replace(dest);digest_manifest(packet)
    print(json.dumps(dict(cases=len(rows),images=len(rows),compact_zip=str(dest),zip_bytes=dest.stat().st_size,changed=len(changed),winner_flips=len(flips),new_flag_cases=len(added),flagged=len(flagged))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,default=PACK);parser.add_argument('--test-log',type=Path)
    args=parser.parse_args();package(args.packet,args.test_log)
