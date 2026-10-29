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


def doc_path(name):
    current=ROOT/'docs'/name
    return current if current.is_file() else ROOT/'docs/archive'/name


def digest_manifest(folder):
    (folder/'MANIFEST.sha256').write_text('\n'.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder).as_posix()}' for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.sha256','.DS_Store'))+'\n')


def package(packet=PACK,test_log=None):
    from PIL import Image,ImageDraw
    viewer=packet/'review/regression_suite'
    manifest=json.loads((viewer/'run_manifest.json').read_text())
    fp=fingerprint(ROOT/'config/regression_review.json')
    if any(fp.get(k)!=manifest['fingerprint'].get(k) for k in ('files_sha256','model','model_weights','adapter','layout_experiments')):
        raise SystemExit('Viewer is stale: replay the fixed cohort before packaging.')
    summary=json.loads((viewer/'summary.json').read_text());rows=json.loads((viewer/'manifest.json').read_text())
    changed=[r for r in summary['results'] if r['max_geometry_change']>1e-4]
    flips=[r for r in summary['results'] if r['before_winner'] is not None and r['before_winner']!=r['winner']]
    added=[r for r in summary['results'] if r['before_flags'] is not None and set(r['flags'])-set(r['before_flags'])]
    flagged=[r for r in summary['results'] if r['flags']]
    test_result='No test log supplied; no new test count claimed.'
    if test_log:
        lines=Path(test_log).read_text().splitlines();test_result=next((line for line in reversed(lines) if 'passed' in line),lines[-1] if lines else 'Empty test log')
    flow_path=ROOT/'data/qwen_lora/engine_upgrade/dense_reflow_summary.json'
    flow=json.loads(flow_path.read_text()) if flow_path.is_file() else None
    flow_note='No additional dense-packing replay summary supplied.'
    if flow:
        hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
        if flow.get('engine_sha256')!=hashes:
            flow=None
            flow_note='Older dense reflow evidence belongs to its recorded earlier engine snapshot. It is historical and excluded from this current compact packet; the fixed 38 below were freshly replayed.'
    if flow:
        flow_note=(f"Measured text/backing packing and Qwen-driven macro bands are now available. "
            f"The four attachment tasks have {sum(bool(r['flags']) for r in flow['attachments'])} flagged outputs; "
            f"both reviewed leaderboards replace their tiny proportional fallback. "
            f"The 23 neither-good comparison pairs reduce to {flow['reviewed']['tasks']} unique tasks "
            f"({flow['reviewed']['formats'].get('leaderboard',0)} leaderboards), replayed separately; "
            f"{flow['reviewed']['changed']} materially changed and {flow['reviewed']['flagged']} have automated flags. "
            "Many human-rejected outputs remain unchanged. See docs/archive/DENSE_REFLOW_AUDIT.md, "
            "evidence/dense_reflow_summary.json and the additional image sheets. "
            "No claim that every rejected design is fixed or that zero flags means visually perfect.")
    latest=f'''# Current unified regression review — October 29, 2026

Open `review/regression_suite/index.html` (the main-engine URL points here).
All {len(rows)} conversions share one engine snapshot: the fixed 38
(18 official, eight original alternates and 12 complex dataset cases) plus
{summary.get('supplemental_cases',0)} supplemental conversions, including fresh independent designs and supplied/skyscraper stress cases. Only Original and Resized are displayed.
Intermediates, actual raw validity and selected-candidate diagnostics are in
per-case audit JSON. Zero automated flags is not proof of aesthetic perfection.

## Current changes

Read docs/archive/NESTED_REFLOW_AUDIT.md for the latest source-scope, repeated-row, card-grid and measured-type changes, then docs/archive/ENGINE_UPGRADE_CHECKLIST.md and docs/archive/ENGINE_UPGRADE_RESULTS.md.
Implemented bounded missing-ID completion, gated atomic sibling grid/row/column
candidates, exact full-document/backing render reuse, equivalent vectorized
glyph statistics, capability-aware runtimes and fail-closed adapter-required
submission preparation. Existing final peer equalization was verified rather
than duplicated. No design IDs or content strings drive the engine.

## Verification scope

Both arms keep Qwen enabled and use the same cached raw predictions and adapter.
The baseline is the frozen prior engine; the after arm uses current code.
The fixed 38-case comparison is not Qwen-off or fresh model generation. Supplemental cases use recorded Qwen-enabled outcomes, including explicit no-proposal refusals, then frozen replay; inference provenance is recorded separately.
{len(changed)} output geometries changed; {len(flips)} winners flipped;
{len(added)} cases acquired new automated flag types. {len(flagged)} current
outputs still have automated flags. Inspect those outliers and winner changes
in `evidence/unified_regression_summary.json`. Tests: {test_result}.

This fixed development cohort does not replace the full 89/309 release checks;
neither full split is newly claimed here. The alternate designs and official
samples have no human resized target; their MAE is deliberately null. Dataset
cases were chosen by complexity/family diversity, not selected for good outputs.

## Next steps

Historical fresh generation and its earlier selector replay are documented in evidence/engine_upgrade_dense_ab_summary.json
and review/dense_ab/index.html in the full local packet. Generation seconds
belong to the earlier inference run. Raw exact IDs improve
12/20 to 20/20 with 1.7B, but final learned-family wins stay 5/20 in both arms.
Final dataset geometry MAE improves ~6%; final font MAE worsens ~15% and generation
takes ~1.8 times longer. The default remains 0.6B. No Qwen retraining is indicated
by the missing-ID experiment alone; the larger model still loses seven/eight
authored layouts to deterministic candidates despite complete inventories.

The completed human campaign has 178 eligible pairs, with repeats/quarantines
preserved. The earlier 768-dimensional image head tied at 9/17 validation and
was not deployed. A subsequent fixed ten-feature structural/raster pilot matched
14/17 validation and 31/37 verified test preferences (versus 9/17 and 26/37).
Five exact-pixel reconstruction failures were excluded. The actual campaign
validation subset only tied at 5/8, so the new head also is not deployed. The
37 test outcomes have now been consumed; do not call them sealed or tune to them.
Read docs/archive/CHAMELEON_ALGORITHM_UPGRADES.md and evidence/chameleon_structural_ranker_report.json.

A generic source-contact growth fix prevents a backing from expanding into
adjacent media. Measured container packing now addresses short-slot capacity;
unresolved decorations retain their arrangement and atomic lockups retain their
source line-count limit. Macro gates and batch audits share exact glyph overflow,
which cannot be hidden by another warning. Only a readability-only compromise
can bypass a prior defective result, with its flags retained and all other
hard gates intact. No family-name veto or new unsupported photo overlay.

{flow_note}
Open review/attachment_growth/index.html and review/reviewed_outliers/index.html
in the full packet. The fixed regression result is reported above. Raw/intermediate
geometry remains in audit JSON; public comparisons show Original and Resized.

The completed queue has 160 distinct comparisons plus eight repeats: 56 from
twelve newly authored Level 3-style sources, 104 dataset comparisons, all six
target formats; its generation snapshot is recorded separately from
the current regression engine. Open review/preferences_campaign/review.html
in the full local packet. Public adapter publication remains pending; no Docker
build is claimed because the daemon is unavailable.

Safe performance recommendations are implemented, not all speculative ones:
exact caching/vectorization, runtime controls and bounded token stopping.
Two-process rendering was slower on the diagnostic and stays opt-in. A new
request-owned asset cache reduces repeated discovery without caching geometry.
One complete replay reduced filesystem discovery from 54,437 calls to 307
(four roots inside the bounded request, 303 discoveries outside). The earlier cache-only replay kept all 38
rendered pixel buffers identical; the later contact fix has the scope above. Timings were noisy under concurrent
work; no portable speedup or 6–12 second runtime is claimed.

The compact sibling ZIP contains all current original/resized contact sheets,
selected output/raw/source JSON, important current code and evidence. It excludes
weights, full assets and fonts and is a review packet, not a runnable submission.
'''
    (packet/'LATEST_UPDATE.md').write_text(latest)
    prompt='''# Gemini review request

Read LATEST_UPDATE.md, docs/archive/ENGINE_UPGRADE_RESULTS.md and the checklist first.
Review the current unified original/resized comparisons (the fixed 38 plus the supplemental cohorts in config/fresh_design_review.json). Distinguish model proposals
from engine-selected layouts. Identify repeated root causes, avoid design-ID
patches, and propose bounded changes with regression checks. Do not assume that
zero flags means clean alignment, that 1.7B is automatically better, or that
SigLIP2 needs full-backbone fine-tuning. Audit bounded completion and atomic
sibling candidate safety, exact raster-cache keys, source/target glyph metric
equivalence, weak preference-label distribution and extreme-aspect failures. Read the
seven human-preferred constraint conflicts and DENSE_REFLOW_AUDIT.md; distinguish photo/caption
realization defects from selector preference before removing safety gates. Check the new glyph-overflow gate, preserved atomic
line counts, unsupported-decoration guard and why Qwen macro bands still lose.
Historical documents have their own earlier scope; current evidence is named
unified_regression_*. No adapter or full assets are distributed in this packet.
'''
    (packet/'GEMINI_REVIEW_PROMPT.md').write_text(prompt)
    (packet/'README_FOR_GEMINI.md').write_text('Current review: read LATEST_UPDATE.md and GEMINI_REVIEW_PROMPT.md first. Open review/regression_suite/index.html for all current Original/Resized comparisons. Important code and evidence are refreshed by package_regression_review.py. Other review folders and audit documents are historical checkpoints with their own scope. The sibling compact ZIP is the small current upload; neither packet includes model weights or full assets/fonts.\n')
    for dirname in ('code/engine','code/scripts','code/tests','docs','evidence','config'):(packet/dirname).mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'engine').glob('*.py'))+[ROOT/p for p in (
        'scripts/refresh_regression_viewer.py','scripts/package_regression_review.py','scripts/replay_qwen_engine.py',
        'scripts/build_focus_reflow_viewer.py','scripts/layout_defects.py','scripts/evaluate_qwen_adapter.py',
        'scripts/train_siglip_ranker.py','scripts/fetch_siglip.py','scripts/refresh_review_viewer.py',
        'scripts/build_upgrade_review.py','scripts/evaluate_dense_qwen_ab.py','scripts/evaluate_dense_engine_ab.py',
        'scripts/prepare_siglip_preferences.py','scripts/prepare_engine_preferences.py','scripts/prepare_dense_continuation.py',
        'scripts/benchmark_render_cache.py','scripts/profile_frozen_engine.py','scripts/verify_qwen_adapter.py',
        'scripts/prepare_submission_context.py','scripts/import_preference_labels.py',
        'scripts/prepare_preference_campaign.py','scripts/preference_campaign_sources.py',
        'scripts/preference_campaign_view.py','scripts/benchmark_asset_lookup.py',
        'scripts/audit_attachment_growth.py','scripts/replay_campaign_attachments.py',
        'scripts/audit_chameleon_import.py','scripts/audit_competitor_repositories.py','scripts/build_layout_library.py',
        'scripts/build_fresh_review_designs.py','scripts/build_novel_level3_designs.py','scripts/evaluate_fresh_review_designs.py','scripts/runtime_diagnostics.py',
        'tests/test_native_runtime.py','tests/test_fresh_review_designs.py','tests/test_user_design_stress.py','tests/test_replay_pairs.py',
        'scripts/audit_nested_reflow.py','tests/test_nested_structure.py','tests/test_followup_layout.py','docs/archive/GEMINI_FOLLOWUP_DECISIONS.md','scripts/prepare_user_design_stress.py','scripts/evaluate_skyscraper_stress.py','scripts/compare_capacity_models.py','scripts/publish_capacity_comparison.py','tests/test_prompt_capacity.py',
        'scripts/benchmark_layout_retrieval.py','scripts/calibrate_structural_ranker.py','scripts/profile_viewer_request.py',
        'tests/test_retrieval_algorithms.py','tests/test_structural_ranker.py','tests/test_preview_cache.py',
        'viewer/server.py','viewer/preview_cache.py','viewer/app.js','engine/data/README.md',
        'scripts/replay_reviewed_outliers.py','scripts/summarize_dense_reflow.py',
        'tests/test_copy_axis_contracts.py','tests/test_layout_mechanics.py','tests/test_macro_layout.py',
        'tests/test_measured_container_flow.py','tests/test_upgrade_primitives.py','tests/test_request_cache.py','tests/test_control_attachment_growth.py','tests/test_preference_labels.py','tests/test_qwen_pipeline.py','tests/test_source_evidence.py','Dockerfile')]
    sources += [ROOT/p for p in ('scripts/probe_upgrade_readiness.py',
                                'scripts/benchmark_decode_lookup.py','tests/test_upgrade_exploration.py',
                                'scripts/check_model_profiles.py','tests/test_model_profiles.py',
                                'scripts/fetch_upgrade_pilot.py','scripts/run_upgrade_model_pilot.py',
                                'scripts/replay_upgrade_pilots.py','scripts/evaluate_critic_pilot.py',
                                'scripts/build_upgrade_pilot_review.py','scripts/audit_upgrade_rasters.py','tests/test_upgrade_pilots.py',
                                'scripts/evaluate_best_of_k_pilot.py','tests/test_owned_type_calibration.py',
                                'scripts/evaluate_training_free_upgrades.py','scripts/build_expanded_layout_library.py',
                                'scripts/evaluate_tiled_selector.py','scripts/evaluate_structural_finalists.py',
                                'scripts/report_training_free_upgrades.py','scripts/audit_training_free_decisions.py',
                                'scripts/publish_selector_scale_review.py','scripts/benchmark_assisted_decode.py',
                                'tests/test_training_free_upgrades.py','tests/test_tiled_vision.py',
                                'scripts/explore_surgical_upgrades.py','scripts/report_surgical_upgrades.py',
                                'scripts/audit_target_glyph_ownership.py','tests/test_surgical_experiments.py')]
    sources += [ROOT/p for p in ('scripts/prepare_training_run.py','scripts/evaluate_trained_layouts.py','scripts/expand_training_candidates.py','tests/test_training_recipe.py')]
    sources += [ROOT/p for p in ('scripts/evaluate_scale_followup.py','scripts/report_scale_followup.py','tests/test_scale_followup.py')]
    sources += [ROOT/p for p in ('scripts/evaluate_industry_followup.py','scripts/report_industry_followup.py','tests/test_industry_followup.py')]
    sources += sorted((ROOT/'training').glob('*.py'))+[ROOT/'training/requirements.txt',ROOT/'training/colab.ipynb']
    sources += sorted((ROOT/'experiments').rglob('*.py'))
    sources += [ROOT/'experiments/layout_upgrade/requirements.txt']
    sources += sorted((ROOT/'tests/fixtures/user_level3_sources').glob('u*.json'))
    for p in sources:
        target=packet/'code'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('UPGRADE_EXPLORATION_AND_EVALUATION.md','GEMINI_FOLLOWUP_DECISIONS.md','NESTED_REFLOW_AUDIT.md','PROMPT_CAPACITY_AND_MODEL_DECISIONS.md','RUNTIME_AND_EXPERIMENTS.md','SKYSCRAPER_STRESS_RESULTS.md','COMPETITOR_REPOSITORY_AUDIT.md','CHAMELEON_ALGORITHM_UPGRADES.md','DENSE_REFLOW_AUDIT.md','PREFERENCE_CAMPAIGN_PLAN.md','ENGINE_UPGRADE_CHECKLIST.md','ENGINE_UPGRADE_RESULTS.md','REGRESSION_REVIEW_AND_NEXT_STEPS.md','PEER_ROW_ALIGNMENT.md','UNIVERSAL_REFLOW_RESEARCH.md','DAY4_5_AUDIT.md'):
        shutil.copy2(doc_path(name),packet/'docs'/name)
    shutil.copy2(ROOT/'config/regression_review.json',packet/'config/regression_review.json')
    shutil.copy2(ROOT/'config/fresh_design_review.json',packet/'config/fresh_design_review.json')
    shutil.copy2(ROOT/'config/upgrade_experiments.json',packet/'config/upgrade_experiments.json')
    shutil.copy2(ROOT/'config/layout_model_profiles.json',packet/'config/layout_model_profiles.json')
    shutil.copy2(ROOT/'config/upgrade_pilot.json',packet/'config/upgrade_pilot.json')
    shutil.copy2(ROOT/'config/training_candidates.json',packet/'config/training_candidates.json')
    shutil.copy2(ROOT/'config/algorithm_followup.json',packet/'config/algorithm_followup.json')
    shutil.copy2(ROOT/'docs/ALGORITHM_FOLLOWUP.md',packet/'docs/ALGORITHM_FOLLOWUP.md')
    shutil.copy2(ROOT/'config/industry_followup.json',packet/'config/industry_followup.json')
    shutil.copy2(ROOT/'docs/INDUSTRY_ALGORITHM_DECISIONS.md',packet/'docs/INDUSTRY_ALGORITHM_DECISIONS.md')
    shutil.copy2(ROOT/'docs/archive/UPGRADE_PILOT_RESULTS.md',packet/'docs/UPGRADE_PILOT_RESULTS.md')
    shutil.copy2(ROOT/'docs/archive/TRAINING_FREE_SCALE_RESULTS.md',packet/'docs/TRAINING_FREE_SCALE_RESULTS.md')
    shutil.copy2(ROOT/'docs/SURGICAL_ALGORITHM_DECISIONS.md',packet/'docs/SURGICAL_ALGORITHM_DECISIONS.md')
    for name in ('README.md','IMPLEMENTATION.md','EVALUATION.md','TRAINING_RUNBOOK.md','REVIEW_ARCHIVE.md','CRELLO_DATA_AND_TRAINING.md','MODEL_TRAINING_CANDIDATES.md'):
        shutil.copy2(doc_path(name),packet/'docs'/name)
    shutil.copytree(ROOT/'docs/archive',packet/'docs/archive',dirs_exist_ok=True)
    shutil.copy2(ROOT/'README.md',packet/'CURRENT_PROJECT_README.md')
    for src,name in [(viewer/'summary.json','unified_regression_summary.json'),(viewer/'run_manifest.json','unified_regression_manifest.json')]:
        shutil.copy2(src,packet/'evidence'/name)
    trace=ROOT/'data/qwen_lora/shared_column_audit'
    for p in trace.glob('*.json'):shutil.copy2(p,packet/'evidence'/('copy_axis_'+p.name))
    for p in trace.glob('*_comparison.jpg'):shutil.copy2(p,packet/'evidence'/('copy_axis_'+p.name))
    upgrade=ROOT/'data/qwen_lora/engine_upgrade'
    for name in ('dense_ab_summary.json','regression_pixels.json','peer_gaps.json','render_benchmark.json','human_labels_received.json','preferences_v2_summary.json','human_labels_v2_received.json','human_labels_v3_received.json','preference_corpus_readiness.json','preference_pilot.json','preferred_constraint_conflicts.json','preference_campaign_integrity.json','preference_complete_training.json','label_export_delta.json'):
        if (upgrade/name).is_file():shutil.copy2(upgrade/name,packet/'evidence'/('engine_upgrade_'+name))
    campaign=packet/'review/preferences_campaign'
    if (campaign/'summary.json').is_file():shutil.copy2(campaign/'summary.json',packet/'evidence/preference_campaign_summary.json')
    if (upgrade/'asset_lookup_production_benchmark/report.json').is_file():shutil.copy2(upgrade/'asset_lookup_production_benchmark/report.json',packet/'evidence/asset_lookup_benchmark.json')
    if (upgrade/'profile/profile.json').is_file():shutil.copy2(upgrade/'profile/profile.json',packet/'evidence/engine_upgrade_profile.json')
    for arm in ('0.6b','1.7b'):
        raw=ROOT/'data/qwen_lora/dense_ab_fresh_v1'/f'{arm}.manifest.json'
        if raw.is_file():shutil.copy2(raw,packet/'evidence'/f'engine_upgrade_raw_{arm}_manifest.json')
    for source,name in [(upgrade/'attachment_growth/report.json','attachment_growth_candidates.json'),(upgrade/'attachment_engine_replay/summary.json','attachment_growth_engine.json'),(upgrade/'attachment_engine_replay/contact.jpg','attachment_growth_comparison.jpg'),(upgrade/'attachment_regression_contact.jpg','attachment_regression_comparison.jpg')]:
        if source.is_file():shutil.copy2(source,packet/'evidence'/name)
    for name in ('dense_reflow_summary.json','dense_reflow_attachments.jpg','dense_reflow_reviewed.jpg'):
        if flow and (upgrade/name).is_file():shutil.copy2(upgrade/name,packet/'evidence'/name)
    # Metadata/reports only: normalized training library and weights remain local.
    study=ROOT/'data/chameleon_upgrade_audit'
    for path in ('summary.json','branches/branch_inventory.json','branches/isolated_gen_probes.json',
                 'library/summary.json','retrieval_benchmark.json','structural_ranker/report.json',
                 'viewer_before/report.json','viewer_merged/report.json','viewer_http/report.json','viewer_http_current/report.json','competitors/summary.json',
                 'native_fresh_comparison.json','runtime_diagnostics.json','regression_pixels.json'):
        src=study/path
        if src.is_file():shutil.copy2(src,packet/'evidence'/('chameleon_'+path.replace('/','_')))
    for name in ('readiness.json','prompt_lookup.json','verification.json','license_review.json',
                 'swappability.json','model_profiles.json'):
        src=ROOT/'data/upgrade_exploration'/name
        if src.is_file():shutil.copy2(src,packet/'evidence'/('upgrade_exploration_'+name))
    (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
        '\n## Upgrade research (no engine promotion)\n\n'
        'Read docs/archive/UPGRADE_EXPLORATION_AND_EVALUATION.md and config/upgrade_experiments.json. '
        'Every new Gemini/Google AI proposal has a disposition and evaluation gate, including newer small Qwen3.5, Qwen3-VL and LFM3 models. '
        'The initial research inspected metadata only; the authorized follow-up downloaded four pinned checkpoints and executed opt-in pilots. '
        'A short same-adapter prompt-lookup probe was slower locally and had a dense-case Metal recovery warning; '
        'it is not full-layout or reference-hardware evidence. The production model is unchanged. The later owned-label engine repair and its fresh regression are documented below. '
        'The follow-up audit prioritizes paired model packages: both local base/adapter pairs pass the existing compatibility checks, '
        'while the crossed 0.6B-adapter/1.7B-base pair is rejected. The new profile manifest/checker is audit tooling, not a hot-swap runtime implementation. '
        'Liquid models move to conditional license/runtime research; unmeasured causal explanations for Metal and decoding speed remain hypotheses. '
        'The exact approved-model list/size limit remains pending in the supplied brief. No new training or adapter publication is claimed.\n')
    pilots=packet/'review/upgrade_pilots'
    if (pilots/'report.json').is_file():
        shutil.copy2(pilots/'report.json',packet/'evidence/upgrade_exploration_pilots.json')
        shutil.copy2(pilots/'contact.jpg',packet/'evidence/upgrade_exploration_pilots.jpg')
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
            '\n## Executed upgrade pilots\n\nRead docs/archive/UPGRADE_PILOT_RESULTS.md and '
            'evidence/upgrade_exploration_pilots.json. Original/Resized images are in '
            'review/upgrade_pilots/index.html. New bases are untuned; validity is separate '
            'from visual quality. CP-SAT and visual critics have measured negative/mixed '
            'results; none has been silently promoted. These original pilot renders '
            'retain their earlier recorded engine snapshot; the current 59-case viewer and repair follow-up are separate.\n')
        for name in ('downloads.json','matrix_status.json','native_matrix_status.json','retry_status.json'):
            src=ROOT/'data/upgrade_exploration'/name
            if src.is_file():shutil.copy2(src,packet/'evidence'/('upgrade_exploration_'+name))
    (packet/'GEMINI_REVIEW_PROMPT.md').write_text((packet/'GEMINI_REVIEW_PROMPT.md').read_text()+
        '\nAlso review docs/archive/UPGRADE_EXPLORATION_AND_EVALUATION.md and config/upgrade_experiments.json. '
        'Distinguish researched hypotheses and future experiments from shipped engine changes. '
        'Check model total versus active counts, adapter/processor compatibility, source licensing, '
        'the adverse prompt-lookup prefix probe, and the separate fresh-model versus frozen-engine evaluation gates.\n')
    note=ROOT/'docs/archive/CHAMELEON_ALGORITHM_UPGRADES.md'
    if note.exists():
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+'\n## Latest algorithm and latency study\n\nRead docs/archive/CHAMELEON_ALGORITHM_UPGRADES.md, docs/archive/COMPETITOR_REPOSITORY_AUDIT.md and evidence/chameleon_summary.json. Retrieval/dual-granularity remain experimental; the calibrated ranker is not deployed. Native Metal selects FP16 and in-memory adapter merge automatically; ARM64 Linux alone does not expose Metal. New task generation and repeat caching are measured separately.\n')
    for src,name in [(ROOT/'data/user_design_stress/source_integrity.json','skyscraper_source_integrity.json'),
                     (ROOT/'data/user_design_stress/run_v2/summary.json','skyscraper_fresh_summary.json')]:
        if src.is_file():shutil.copy2(src,packet/'evidence'/name)
    (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+'\n## Supplied and skyscraper stress expansion\n\nRead docs/archive/SKYSCRAPER_STRESS_RESULTS.md. Four supplied designs use authorized local illustrative artwork/fonts. The original duplicate-ID input remains a rejection test. Thirteen additional conversions keep Qwen enabled; nine target 160×600. The initial prompt-refusal run is historical; current capacity/model results are documented separately. Residual output defects are listed in the current unified summary. A font-program measurement crash is fixed using renderer metrics on the exceptional path. Full original/source/output geometry remains available in audit files.\n')
    run_signature=sha256(json.dumps(manifest['fingerprint'],sort_keys=True).encode()).hexdigest()[:16]
    reuse=ROOT/'data/qwen_lora/regression_suite'/run_signature/'reused_engine_results.json'
    if reuse.is_file():shutil.copy2(reuse,packet/'evidence/unified_regression_reused_engine_results.json')
    capacity=ROOT/'data/prompt_capacity_ab'
    for path in ('capacity_probe.json','grouping_audit.json','prior46_render_check/report.json','selector/comparison.json','selector/manifest.json',
                 '0.6b/run_manifest.json','1.7b/run_manifest.json'):
        src=capacity/path
        if src.is_file():shutil.copy2(src,packet/'evidence'/('capacity_'+path.replace('/','_')))
    (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+'\n## Prompt capacity and model comparison\n\nRead docs/archive/PROMPT_CAPACITY_AND_MODEL_DECISIONS.md and evidence/capacity_selector_comparison.json. Dense prompts keep the leaf schema and pair-tuple relation vocabulary; a 900-token relation budget and dynamic output reservation replace the old 4096-token veto. The historical model arms used matching adapters and the same selector recorded in their manifests. They are not newly replayed after the nested-reflow changes; stale alternatives are hidden in the current viewer. Original incomplete inventories remain incomplete for raw-validity reporting. A zero-denominator recovery guard retains normal defect gates. The structural ranker and macro-only model contract remain experimental; no public adapter upload or new training occurred.\n')
    nested=ROOT/'data/nested_reflow_audit/final/summary.json'
    if nested.is_file():
        evidence=json.loads(nested.read_text())
        if evidence.get('run','').endswith(run_signature):
            shutil.copy2(nested,packet/'evidence/nested_reflow_summary.json')
            (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+'\n## Nested reflow release audit\n\nRead docs/archive/NESTED_REFLOW_AUDIT.md and evidence/nested_reflow_summary.json for source ownership, exact baseline/render identity checks and remaining limitations.\n')
    followup=ROOT/'data/gemini_followup_audit/final/summary.json'
    if followup.is_file() and json.loads(followup.read_text()).get('run','').endswith(run_signature):
        shutil.copy2(followup,packet/'evidence/followup_reflow_summary.json')
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+'\n## Gemini follow-up verification\n\nRead docs/archive/GEMINI_FOLLOWUP_DECISIONS.md and evidence/followup_reflow_summary.json for decisions on all 25 proposals, exact baseline comparisons and remaining defects. This is saved-proposal engine verification, not fresh inference or a model upgrade.\n')
    repair_audit=ROOT/'data/upgrade_exploration/joint_type_regression_audit/summary.json'
    if repair_audit.is_file() and json.loads(repair_audit.read_text()).get('run','').endswith(run_signature):
        shutil.copy2(repair_audit,packet/'evidence/upgrade_exploration_owned_type_regression.json')
        for filename in ('before.json','after.json','comparison.jpg'):
            path=ROOT/'data/upgrade_exploration/joint_type_repair_audit'/filename
            if path.exists():shutil.copy2(path,packet/'evidence'/('upgrade_exploration_owned_type_'+filename))
        selection=ROOT/'data/upgrade_exploration/best_of_k_v1/selection'
        for path in selection.glob('*.json'):
            shutil.copy2(path,packet/'evidence'/('upgrade_exploration_best_k_'+path.name))
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
            '\n## Adopted owned-label repair and sampling follow-up\n\n'
            'Read docs/archive/UPGRADE_PILOT_RESULTS.md. Rotated owned controls now fit glyphs in their local panel capacity, preserving source line count, tracking and existing readability floors. '
            'The known intermediate overflow is fixed; that candidate still has contacts and does not win. All 59 final JSONs and JPEG bytes are unchanged, with one previously flagged short conference task remaining. '
            'Four genuine resizing tasks with greedy plus two sampled grammar decodes yield the same final layouts. All eight extra realized proposals fail the existing physical gate, so no clean merged pool was available to compare. '
            'Multi-sampling stays experimental. Current two-panel follow-up: review/upgrade_repair_followup/index.html. No new training or public upload.\n')
    scale=ROOT/'data/upgrade_exploration/training_free_scale_v1'
    if (scale/'report.json').is_file():
        for filename in ('report.json','manifest.json','algorithms_manifest.json','decision_audit.json','coverage.json',
                         'family_bootstrap.json','visual_decisions.json','integrated_tiling_parity.json'):
            src=scale/filename
            if src.is_file():shutil.copy2(src,packet/'evidence'/('upgrade_exploration_training_free_'+filename))
        library_audit=json.loads((scale/'expanded_library.audit.json').read_text())
        library_audit.pop('local_provenance',None)
        (packet/'evidence/upgrade_exploration_training_free_library.json').write_text(json.dumps(library_audit,indent=2)+'\n')
        for folder in ('critic_retrospective_all','assisted_decode_v1'):
            for src in (ROOT/'data/upgrade_exploration'/folder).glob('*.json'):
                shutil.copy2(src,packet/'evidence'/('upgrade_exploration_'+folder+'_'+src.name))
        released=scale/'release_regression/summary.json'
        if released.is_file():shutil.copy2(released,packet/'evidence/upgrade_exploration_training_free_release.json')
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
            '\n## Expanded no-training comparisons\n\nRead docs/archive/TRAINING_FREE_SCALE_RESULTS.md. '
            'Six algorithm arms were evaluated on 134 distinct resize tasks, covering all 89 held-out reference pairs (83 distinct requests / 21 families), '
            'and fixed critics were scored retrospectively against all 178 existing human preferences. '
            'The existing structural head was evaluated without fitting on clean algorithm finalists. Several low-cost layouts visibly fracture source lockups; those are not promoted. '
            'Optional MAGICANVAS_SIGLIP_TILED=1 is integrated but OFF by default: 89-pair MAE improved about 2.3%, '
            'but improvements concentrate in a few families, confidence intervals cross zero and replay cost increased. '
            'Native speculative decoding was slower on three cases and matched exact target tokens on two; not promoted. '
            'All changed layouts received direct raster inspection and native contrast/visibility auditing; zero flags is not aesthetic proof. '
            'Original/Resized comparison pages: review/training_free_scale/index.html and review/training_free_selectors/index.html. '
            'No new training, labels or public upload. The 309 validation cases and quantized backends are not claimed evaluated.\n')
    surgical=ROOT/'data/upgrade_exploration/surgical_followup_v1'
    if (surgical/'diagnostic_summary.json').is_file():
        for filename in ('manifest.json','diagnostic_summary.json','topology.json','gated_critic.json',
                         'target_audit.json','target_owned_masks.json','reference_comparison.json','font_native.json','visual_review.json'):
            path=surgical/filename
            if path.is_file():shutil.copy2(path,packet/'evidence'/('upgrade_exploration_surgical_'+filename))
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
            '\n## Six surgical algorithm follow-ups\n\nRead docs/SURGICAL_ALGORITHM_DECISIONS.md for every suggestion, counterexample and revisit criterion. '
            '134 frozen tasks, 178 recorded preferences, all 1,231 distinct train/validation target geometries and 32 native target samples were analyzed. '
            'Source-width slack adds no edit; raw-model font slack changes ten proposals, four geometrically unsafe. '
            'A headline scale band creates nine newly flagged layouts. The suggested topology/occupancy gate and selective tiled critic do not earn promotion. '
            'The held-out negative-bound coordinate is floating-point residue; the real 320x50 conference failure remains flagged. '
            'An offline isolated/leave-one-out glyph auditor and line-count sidecars were added; training files and production behavior were NOT changed. '
            'Research Original/Resized proposals: review/surgical_followup/index.html. Full target sidecars, datasets and weights remain local.\n')
    training=ROOT/'data/training_next/audit_summary.json'
    if training.is_file():
        shutil.copy2(training,packet/'evidence/upgrade_exploration_training_preparation.json')
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+'\n## Current training preparation and submission docs\n\nRead docs/TRAINING_RUNBOOK.md and CURRENT_PROJECT_README.md. The portable recipe prepares six pinned providers and trains three distinct first-round packages: fresh 0.6B data control, Qwen3.5-0.8B and source-image-aware Qwen3-VL-2B, retains prompt v2 and compares fresh inference on 89+59 tasks before human review. Actual supervised targets are natively screened, with owned-mask confirmation for invisible-text exclusions. No new adapter was trained or promoted here. Full training/validation messages, line sidecars, fonts and weights remain local and are excluded from this packet. Public adapter upload and final offline Docker execution remain pending. Historical notes have moved under docs/archive; use docs/README.md for current instructions.\n')
    for name in ('candidate_preflight.json','new_model_smoke.json','frontier_screen.json','new_model_layout_summary.json'):
        src=ROOT/'data/training_next'/name
        if src.is_file():shutil.copy2(src,packet/'evidence'/('upgrade_exploration_training_'+name))
    followup=ROOT/'data/upgrade_exploration/scale_followup_20261029'
    for name in ('report.json','followup_manifest.json','native_summary.json','visual_decisions.json','uncertainty.json'):
        src=followup/name
        if src.is_file():shutil.copy2(src,packet/'evidence'/('upgrade_exploration_scale_followup_'+name))
    if (followup/'report.json').is_file():
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
            '\n## Current training-free follow-up\n\nRead docs/ALGORITHM_FOLLOWUP.md and evidence/upgrade_exploration_scale_followup_report.json. '
            'The 134-task paired replay tests shared-scale intact-container growth, a family-quota render funnel and exact glyph order statistics. '
            'All 89 reference pairs are covered, with saved Qwen predictions held fixed. Results and rejected proposals are visible at review/scale_followup/index.html. '
            'No production activation is implied by improved utilization or zero flags; detailed decisions retain failures and limitations.\n')
    industry=ROOT/'data/upgrade_exploration/industry_followup_20261029'
    for name in ('report.json','experiment_manifest.json','visual_decisions.json','competitor_recheck.json'):
        src=industry/name
        if src.is_file():shutil.copy2(src,packet/'evidence'/('upgrade_exploration_industry_'+name))
    if (industry/'report.json').is_file():
        (packet/'LATEST_UPDATE.md').write_text((packet/'LATEST_UPDATE.md').read_text()+
            '\n## Additional industry-algorithm study\n\nRead docs/INDUSTRY_ALGORITHM_DECISIONS.md. '
            'All 134 frozen requests were screened for measured paragraph balancing and intact-group neighborhood search. '
            'Nine paragraph outputs and one group-search output passed research gates; all ten were inspected and natively audited. '
            'All 89 reference outputs remained unchanged. Both arms are explored and rejected for production; current engine defaults and training are unchanged. '
            'Research comparisons: review/industry_followup/index.html. The preceding growth, quota and statistic arms are also explicitly rejected for production. '
            'The document distinguishes public Adobe/Canva behavior from unverified private implementation claims.\n')
    # The private adapter context contains weights and is deliberately never copied.
    overview=packet/'review/index.html'
    overview.write_text('<!doctype html><meta charset="utf-8"><title>MagiCanvas review</title><h1>Current engine review</h1><p><a href="regression_suite/index.html">Unified Original/Resized viewer</a></p><p><a href="attachment_growth/index.html">Source-attachment outliers (snapshot in audit)</a></p><p><a href="reviewed_outliers/index.html">Human-rejected tasks (snapshot in audit)</a></p><p><a href="dense_ab/index.html">Historical 20-case model A/B</a></p><p><a href="preferences_campaign/review.html">Large actual-engine preference campaign</a></p><p>Read LATEST_UPDATE.md and ENGINE_UPGRADE_RESULTS.md. No model-off arm, public upload or automatically generated approvals.</p>')
    page=viewer/'index.html'
    nav='<p id="upgrade-links"><a href="../training_candidates/index.html">New-model feasibility / training queue</a> · <a href="../surgical_followup/index.html">Surgical follow-ups</a> · <a href="../training_free_scale/index.html">134-task algorithm comparison</a> · <a href="../training_free_selectors/index.html">Selector comparisons</a> · <a href="../upgrade_repair_followup/index.html">Repair/sampling snapshot</a> · <a href="../upgrade_pilots/index.html">Earlier model pilots (recorded snapshot)</a> · <a href="../attachment_growth/index.html">Attachment outliers (audit snapshot)</a> · <a href="../reviewed_outliers/index.html">Human-rejected tasks (audit snapshot)</a> · <a href="../dense_ab/index.html">Historical model A/B</a> · <a href="../preferences_campaign/review.html">Human preference campaign</a></p>'
    body=page.read_text()
    if (followup/'report.json').is_file():nav=nav.replace('id="upgrade-links">','id="upgrade-links"><a href="../scale_followup/index.html">134-task growth/render follow-up</a> · ')
    if (industry/'report.json').is_file():nav=nav.replace('id="upgrade-links">','id="upgrade-links"><a href="../industry_followup/index.html">Industry-algorithm decisions</a> · ')
    if 'id="upgrade-links"' not in body:page.write_text(body.replace('</header>',nav+'</header>'))
    else:
        import re
        page.write_text(re.sub(r'<p id="upgrade-links">.*?</p>',nav,body,flags=re.S))
    if test_log:shutil.copy2(test_log,packet/'evidence/unified_regression_tests.log')
    small=ROOT/'data/gemini_review_current_compact';stage=small.with_name(small.name+'.next')
    if stage.exists():shutil.rmtree(stage)
    for name in ('images','cases','docs','evidence','code/engine','code/scripts','code/tests','config'):(stage/name).mkdir(parents=True,exist_ok=True)
    for name in ('LATEST_UPDATE.md','GEMINI_REVIEW_PROMPT.md','CURRENT_PROJECT_README.md'):shutil.copy2(packet/name,stage/name)
    for p in sources:
        dest=stage/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(ROOT/'config/regression_review.json',stage/'config/regression_review.json')
    shutil.copy2(ROOT/'config/fresh_design_review.json',stage/'config/fresh_design_review.json')
    shutil.copy2(ROOT/'config/upgrade_experiments.json',stage/'config/upgrade_experiments.json')
    shutil.copy2(ROOT/'config/layout_model_profiles.json',stage/'config/layout_model_profiles.json')
    shutil.copy2(ROOT/'config/upgrade_pilot.json',stage/'config/upgrade_pilot.json')
    shutil.copy2(ROOT/'config/training_candidates.json',stage/'config/training_candidates.json')
    shutil.copy2(ROOT/'config/algorithm_followup.json',stage/'config/algorithm_followup.json')
    shutil.copy2(ROOT/'docs/ALGORITHM_FOLLOWUP.md',stage/'docs/ALGORITHM_FOLLOWUP.md')
    shutil.copy2(ROOT/'config/industry_followup.json',stage/'config/industry_followup.json')
    shutil.copy2(ROOT/'docs/INDUSTRY_ALGORITHM_DECISIONS.md',stage/'docs/INDUSTRY_ALGORITHM_DECISIONS.md')
    shutil.copy2(ROOT/'docs/archive/UPGRADE_PILOT_RESULTS.md',stage/'docs/UPGRADE_PILOT_RESULTS.md')
    shutil.copy2(ROOT/'docs/archive/TRAINING_FREE_SCALE_RESULTS.md',stage/'docs/TRAINING_FREE_SCALE_RESULTS.md')
    shutil.copy2(ROOT/'docs/SURGICAL_ALGORITHM_DECISIONS.md',stage/'docs/SURGICAL_ALGORITHM_DECISIONS.md')
    shutil.copytree(packet/'docs/archive',stage/'docs/archive',dirs_exist_ok=True)
    for name in ('README.md','IMPLEMENTATION.md','EVALUATION.md','TRAINING_RUNBOOK.md','REVIEW_ARCHIVE.md','CRELLO_DATA_AND_TRAINING.md','MODEL_TRAINING_CANDIDATES.md'):
        shutil.copy2(packet/'docs'/name,stage/'docs'/name)
    for p in (packet/'docs').glob('*.md'):
        if p.name in ('UPGRADE_EXPLORATION_AND_EVALUATION.md','GEMINI_FOLLOWUP_DECISIONS.md','NESTED_REFLOW_AUDIT.md','PROMPT_CAPACITY_AND_MODEL_DECISIONS.md','RUNTIME_AND_EXPERIMENTS.md','SKYSCRAPER_STRESS_RESULTS.md','COMPETITOR_REPOSITORY_AUDIT.md','CHAMELEON_ALGORITHM_UPGRADES.md','DENSE_REFLOW_AUDIT.md','PREFERENCE_CAMPAIGN_PLAN.md','ENGINE_UPGRADE_CHECKLIST.md','ENGINE_UPGRADE_RESULTS.md','REGRESSION_REVIEW_AND_NEXT_STEPS.md','PEER_ROW_ALIGNMENT.md','DAY4_5_AUDIT.md'):shutil.copy2(p,stage/'docs'/p.name)
    for p in (packet/'evidence').glob('*'):
        if not flow and p.name.startswith('dense_reflow_'):continue
        if p.name.startswith(('upgrade_exploration_','chameleon_','unified_regression_','copy_axis_','engine_upgrade_','attachment_','dense_reflow_','preference_campaign_','asset_lookup_','skyscraper_','capacity_','nested_reflow_','followup_reflow_')) and p.is_file():shutil.copy2(p,stage/'evidence'/p.name)
    additional=packet/'review/reviewed_outliers'
    if additional.is_dir():
        for p in additional.glob('*_audit.json'):shutil.copy2(p,stage/'cases'/('additional_'+p.name))
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
    if (pilots/'report.json').is_file():
        pilot_report=json.loads((pilots/'report.json').read_text());seen_pilot_images={}
        for model in pilot_report['models']:
            arm=model['arm'];folder=pilots/arm
            if not (folder/'manifest.json').is_file():continue
            for i,row in enumerate(json.loads((folder/'manifest.json').read_text())):
                key=f'pilot_{arm}_{i:02d}'
                shutil.copy2(folder/row['audit'],stage/'cases'/f'{key}_audit.json')
                # Identical selected renders across arms share one paired sheet
                # in the compact upload. Every arm retains its full raw audit.
                digest=sha256((folder/row['source']).read_bytes()+(folder/row['image']).read_bytes()).hexdigest()
                if digest in seen_pilot_images:continue
                seen_pilot_images[digest]=key
                sheet=Image.new('RGB',(1200,700),'#182532');draw=ImageDraw.Draw(sheet)
                for col,(label,field) in enumerate((('Original','source'),('Resized · experimental','image'))):
                    with Image.open(folder/row[field]) as im:
                        im=im.convert('RGB');im.thumbnail((580,650));sheet.paste(im,(col*600+(600-im.width)//2,38+(650-im.height)//2))
                    draw.text((col*600+10,12),label,fill='white')
                image=f'images/{key}.jpg';sheet.save(stage/image,quality=82,optimize=True)
                cards.append(f'<article><h2>Opt-in pilot · {html.escape(arm)} · {html.escape(row["case"])}</h2><img loading="lazy" src="{image}" alt="Original and resized experimental output"><p>Not promoted. Identical images in other arms are deduplicated; all raw/intermediate per-arm audits remain in cases/.</p><a href="cases/{key}_audit.json">Pilot audit JSON</a></article>')
    scale_images=0
    for folder_name in ('training_free_scale','training_free_selectors','surgical_followup','training_candidates','scale_followup','industry_followup'):
        folder=packet/'review'/folder_name;manifest_path=folder/'manifest.json'
        if not manifest_path.is_file():continue
        for i,row in enumerate(json.loads(manifest_path.read_text())):
            key=f'{folder_name}_{i:02d}';source=row.get('source') or row['paths']['source'];resized=row.get('image') or row['paths']['after']
            audit=json.loads((folder/row['audit']).read_text())
            # Per-case source/raw/output geometry is useful; full training
            # prompt/target-message records are not an upload prerequisite.
            if isinstance(audit.get('case'),dict):audit['case'].pop('record',None)
            (stage/'cases'/f'{key}_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            sheet=Image.new('RGB',(1200,700),'#182532');draw=ImageDraw.Draw(sheet)
            for col,(label,image_name) in enumerate((('Original',source),('Resized · experimental',resized))):
                with Image.open(folder/image_name) as im:
                    im=im.convert('RGB');im.thumbnail((580,650));sheet.paste(im,(col*600+(600-im.width)//2,38+(650-im.height)//2))
                draw.text((col*600+10,12),label,fill='white')
            image=f'images/{key}.jpg';sheet.save(stage/image,quality=82,optimize=True);scale_images+=1
            status=html.escape(row.get('status','Not the production default.'))
            decision_doc='INDUSTRY_ALGORITHM_DECISIONS.md' if folder_name=='industry_followup' else 'ALGORITHM_FOLLOWUP.md' if folder_name=='scale_followup' else 'TRAINING_FREE_SCALE_RESULTS.md'
            cards.append(f'<article><h2>{html.escape(folder_name)} · {html.escape(str(row.get("case") or row.get("pair_id")))}</h2><img loading="lazy" src="{image}" alt="Original and experimental resized"><p>{status} See docs/{decision_doc} and visual decisions.</p><a href="cases/{key}_audit.json">Audit JSON</a></article>')
    alternatives=viewer/'model_17';extra_images=0
    comparison=ROOT/'data/prompt_capacity_ab/selector/comparison.json'
    if alternatives.is_dir() and comparison.is_file():
        different={r['pair_id'] for r in json.loads(comparison.read_text())['comparisons'] if not r['same_output']}
        for i,row in enumerate(json.loads((alternatives/'manifest.json').read_text())):
            key=f'model17_{i:02d}'
            shutil.copy2(alternatives/row['audit'],stage/'cases'/f'{key}_audit.json')
            shutil.copy2(alternatives/row['json'],stage/'cases'/f'{key}_output.json')
            if row['pair_id'] not in different:continue
            sheet=Image.new('RGB',(1200,700),'#182532');draw=ImageDraw.Draw(sheet)
            for col,(label,field) in enumerate((('Original','source'),('Resized · 1.7B','image'))):
                with Image.open(alternatives/row[field]) as im:
                    im=im.convert('RGB');im.thumbnail((580,650));sheet.paste(im,(col*600+(600-im.width)//2,38+(650-im.height)//2))
                draw.text((col*600+10,12),label,fill='white')
            image=f'images/{key}.jpg';sheet.save(stage/image,quality=82,optimize=True);extra_images+=1
            cards.append(f'<article><h2>1.7B alternative · {html.escape(row["case"])}</h2><img loading="lazy" src="{image}" alt="Original and resized with 1.7B"><p>Same current selector · flags {html.escape(str(row["defect_flags"]))}. Ten other final outputs are identical to the default arm.</p><a href="cases/{key}_audit.json">Audit JSON</a></article>')
    (stage/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current Gemini review</title><style>body{background:#101820;color:white;font:16px system-ui;margin:24px}img{width:100%;max-width:1200px}article{margin-bottom:40px}</style><h1>All current original/resized comparisons</h1><p>Read LATEST_UPDATE.md. Qwen enabled; saved model predictions replayed through the current engine. This compact packet excludes runtime weights and full assets. <a href="evidence/attachment_growth_comparison.jpg">Historical attachment outliers: before/after audit</a>. Read docs/archive/CHAMELEON_ALGORITHM_UPGRADES.md for current experiments; older dense outlier evidence retains its own snapshot.</p>'+''.join(cards))
    (stage/'README_FOR_GEMINI.md').write_text('Read LATEST_UPDATE.md, then GEMINI_REVIEW_PROMPT.md. Open index.html for all comparisons. Current engine code and per-case raw/source/output geometry are included. Historical archives, model weights, full assets and fonts are excluded. This packet is not runnable standalone.\n')
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
    print(json.dumps(dict(cases=len(rows),images=len(rows)+extra_images+scale_images,compact_zip=str(dest),zip_bytes=dest.stat().st_size,changed=len(changed),winner_flips=len(flips),new_flag_cases=len(added),flagged=len(flagged))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,default=PACK);parser.add_argument('--test-log',type=Path)
    args=parser.parse_args();package(args.packet,args.test_log)
