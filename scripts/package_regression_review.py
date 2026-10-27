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

## Diagnosis and change

The reported Level 2 leaderboard started with an aligned copy column. Independent
optimization moved its supporting text; a partial hierarchy repair did not move
the attached kicker, and a later source-center snap split the reflowed left
edges by 17.7595px. Candidate-owned column-axis contracts now preserve an
already-declared left/center/right axis through those operations. No template
IDs or content strings enter the rule; unsafe moves are rejected. The final
reported headline/body frame lefts now agree exactly; kicker/frame discrepancy
is under 0.5px. Separate inline action columns remain independent.

## Verification scope

Both arms keep Qwen enabled and use the same cached raw predictions, adapter,
selector and current engine. The baseline disables only this change's new
copy-axis contracts. This is not Qwen-off or fresh model generation.
{len(changed)} layouts changed materially; {len(flips)} winners flipped;
{len(added)} cases acquired new automated flag types. {len(flagged)} current
outputs still have automated flags. Inspect those outliers and winner changes
in `evidence/unified_regression_summary.json`. Tests: {test_result}.

This fixed development cohort does not replace the full 89/309 release checks;
neither full split is newly claimed here. The alternate designs and official
samples have no human resized target; their MAE is deliberately null. Dataset
cases were chosen by complexity/family diversity, not selected for good outputs.

## Next steps

Read `docs/REGRESSION_REVIEW_AND_NEXT_STEPS.md`. Keep the current 0.6B adapter
until a fresh paired A/B justifies promotion; the previous 1.7B run did not.
The dense alternate cohort still has seven/eight saved proposals missing IDs;
fresh paired 0.6B/1.7B inference on that cohort is the next model experiment.
The project already uses SigLIP2. Collect same-renderer human preference labels
before training its existing small ranking head. NaFlex is an optional offline
encoder A/B for extreme aspects, not a substitute for exact geometry checks.
No training, model promotion, weight download or external upload occurred.

The compact sibling ZIP contains all 38 current original/resized contact sheets,
selected output/raw/source JSON, important current code and evidence. It excludes
weights, full assets and fonts and is a review packet, not a runnable submission.
'''
    (packet/'LATEST_UPDATE.md').write_text(latest)
    prompt='''# Gemini review request

Read LATEST_UPDATE.md and docs/REGRESSION_REVIEW_AND_NEXT_STEPS.md first.
Review the 38 current original/resized comparisons. Distinguish model proposals
from engine-selected layouts. Identify repeated root causes, avoid design-ID
patches, and propose bounded changes with regression checks. Do not assume that
zero flags means clean alignment, that 1.7B is automatically better, or that
SigLIP2 needs full-backbone fine-tuning. Audit candidate-axis contract safety,
winner flips, remaining attachment/occupancy defects and extreme-aspect failures.
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
        'tests/test_copy_axis_contracts.py','tests/test_layout_mechanics.py','tests/test_macro_layout.py')]
    for p in sources:
        target=packet/'code'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('REGRESSION_REVIEW_AND_NEXT_STEPS.md','PEER_ROW_ALIGNMENT.md','UNIVERSAL_REFLOW_RESEARCH.md','DAY4_5_AUDIT.md'):
        shutil.copy2(ROOT/'docs'/name,packet/'docs'/name)
    shutil.copy2(ROOT/'config/regression_review.json',packet/'config/regression_review.json')
    shutil.copy2(ROOT/'README.md',packet/'CURRENT_PROJECT_README.md')
    for src,name in [(viewer/'summary.json','unified_regression_summary.json'),(viewer/'run_manifest.json','unified_regression_manifest.json')]:
        shutil.copy2(src,packet/'evidence'/name)
    trace=ROOT/'data/qwen_lora/shared_column_audit'
    for p in trace.glob('*.json'):shutil.copy2(p,packet/'evidence'/('copy_axis_'+p.name))
    for p in trace.glob('*_comparison.jpg'):shutil.copy2(p,packet/'evidence'/('copy_axis_'+p.name))
    if test_log:shutil.copy2(test_log,packet/'evidence/unified_regression_tests.log')
    small=ROOT/'data/gemini_review_current_compact';stage=small.with_name(small.name+'.next')
    if stage.exists():shutil.rmtree(stage)
    for name in ('images','cases','docs','evidence','code/engine','code/scripts','code/tests','config'):(stage/name).mkdir(parents=True,exist_ok=True)
    for name in ('LATEST_UPDATE.md','GEMINI_REVIEW_PROMPT.md','CURRENT_PROJECT_README.md'):shutil.copy2(packet/name,stage/name)
    for p in sources:
        dest=stage/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(ROOT/'config/regression_review.json',stage/'config/regression_review.json')
    for p in (packet/'docs').glob('*.md'):
        if p.name in ('REGRESSION_REVIEW_AND_NEXT_STEPS.md','PEER_ROW_ALIGNMENT.md','DAY4_5_AUDIT.md'):shutil.copy2(p,stage/'docs'/p.name)
    for p in (packet/'evidence').glob('*'):
        if p.name.startswith(('unified_regression_','copy_axis_')) and p.is_file():shutil.copy2(p,stage/'evidence'/p.name)
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
