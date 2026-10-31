#!/usr/bin/env python3
"""Refresh current evidence in the existing compact Gemini packet."""
from pathlib import Path
import hashlib,html,json,shutil,sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PACK=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25'
RUN=ROOT/'data/qwen_lora/inventory_fixes_verified'

def jpeg(source,dest):
    with Image.open(source) as im:im.convert('RGB').save(dest,quality=92,optimize=True,progressive=True)

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,default=RUN);args=p.parse_args()
    run=args.run.resolve()
    from renderer.converter.convert import render_design_to_png as official_render
    from scripts.dataset_renderer import render_design_to_png as dataset_render
    reports={k:json.loads((run/k/'report.json').read_text()) for k in ('official','crello','crello_1p7b','heldout')}
    assert [len(reports[k]['results']) for k in ('official','crello','crello_1p7b','heldout')]==[18,12,12,89]
    rows={k:{r['pair_id']:r for r in d['results']} for k,d in reports.items()}
    gallery=PACK/'review/main_engine_qwen_outputs';manifest=[r for r in json.loads((gallery/'manifest.json').read_text()) if r.get('group') not in ('synthetic_reverse','dataset_diversity','novel_validation','novel_designs')];cards=[]
    for item in manifest:
        arm='official' if item['image'].startswith('official_') else 'crello_1p7b' if '_1p7b.' in item['image'] else 'crello'
        row=rows[arm][item['pair_id']];dest=gallery/(Path(item['image']).stem+'.jpg');temp=dest.with_suffix('.render.png')
        if arm=='official':official_render(row['output_design'],ROOT/'brand_kit/assets',temp,max_size=1200)
        else:dataset_render(row['output_design'],ROOT/'data'/('crello_'+row['source_split']),temp,ROOT/'data/crello_fonts.pickle',max_size=1200,adjust_tight_text_widths=False)
        jpeg(temp,dest);temp.unlink();(gallery/item['json']).write_text(json.dumps(row['output_design'],indent=2)+'\n')
        item.update(image=dest.name,winner=row['winner'],hard_checks_passed=row['hard_checks_passed'],
                    flagged=row['flagged'],defect_flags=row['defect_flags'],source_authored_issues=row.get('source_authored_issues',[]))
        title=html.escape(item.get('case',item['pair_id']));label=html.escape(arm)
        cards.append(f'<article><h3>{title} · {label}</h3><img loading="lazy" src="{dest.name}"><p>{html.escape(row["winner"])}</p><p>New flags: {html.escape(str(row["defect_flags"]))} · Authored issues: {len(row.get('source_authored_issues',[]))} · <a href="{item["json"]}">Selected JSON</a></p></article>')
    reverse=run/'reverse_level3'
    if (reverse/'manifest.json').exists():
        portable=PACK/'review/synthetic_level3_reverse';portable.mkdir(exist_ok=True)
        for path in reverse.iterdir():
            if path.suffix in ('.jpg','.html') or (path.suffix=='.json' and path.name!='report.json'):
                shutil.copy2(path,portable/path.name)
        for i,item in enumerate(json.loads((reverse/'manifest.json').read_text()),1):
            if (run/'novel_designs/manifest.json').exists():continue
            image=f'synthetic_reverse_{i:02d}.jpg';output=f'synthetic_reverse_{i:02d}.json'
            shutil.copy2(reverse/item['output'],gallery/image);shutil.copy2(reverse/item['json'],gallery/output)
            entry=dict(item,image=image,json=output,case=item['pair_id'],group='synthetic_reverse',flagged=bool(item['defect_flags']))
            manifest.append(entry)
            cards.append(f'<article><h3>Synthetic Level 3 · {html.escape(item["pair_id"])}</h3><img loading="lazy" src="{image}"><p>{html.escape(item["winner"])}</p><p>Flags: {html.escape(str(item["defect_flags"]))} · <a href="{output}">Selected JSON</a> · <a href="../synthetic_level3_reverse/index.html">Authored source / output</a></p></article>')
    from scripts.build_diverse_dataset_viewer import viewer as dataset_viewer
    newcards=[]
    sections=[('novel_validation/viewer','novel_validation','Validation / frozen 0.6B replay'),
              ('diverse_heldout','dataset_diversity','Additional held-out / saved 0.6B')]
    for relative,group,label in sections:
        origin=run/relative
        if not (origin/'manifest.json').exists():continue
        portable=PACK/'review'/group;portable.mkdir(exist_ok=True)
        items=json.loads((origin/'manifest.json').read_text());lite=[]
        for item in items:
            for key in ('contact','source_json','raw_json','output_json'):shutil.copy2(origin/item[key],portable/item[key])
            entry={k:v for k,v in item.items() if k not in ('source','raw','output','reference')};lite.append(entry)
            image=f'../{group}/{item["contact"]}';output=f'../{group}/{item["output_json"]}'
            manifest.append(dict(item,image=image,json=output,case=item['title'],group=group,flagged=bool(item['defect_flags'])))
            newcards.append(f'<article><h3>{html.escape(item["title"])} · {label}</h3><a href="../{group}/index.html"><img loading="lazy" src="{image}"></a><p>{html.escape(item["winner"])} · Qwen-family winner: {item["qwen_family_win"]}</p><p>New flags: {html.escape(str(item["defect_flags"]))} · <a href="{output}">Selected JSON</a> · <a href="../{group}/index.html">Source / human preview / raw proposal / engine</a></p></article>')
        (portable/'manifest.json').write_text(json.dumps(lite,indent=2)+'\n')
        (portable/'index.html').write_text(dataset_viewer(lite,True))
        shutil.copy2(origin/'selection.json',portable/'selection.json')
    novel=run/'novel_designs'
    if (novel/'manifest.json').exists():
        portable=PACK/'review/novel_designs';portable.mkdir(exist_ok=True)
        novelitems=json.loads((novel/'manifest.json').read_text());novelcards=[]
        for path in novel.iterdir():
            if path.name in ('index.html','manifest.json','run_manifest.json') or (path.suffix=='.jpg' and path.name[:2].isdigit()) or (path.suffix=='.json' and path.name[:2].isdigit()):
                shutil.copy2(path,portable/path.name)
        report=json.loads((novel/'report.json').read_text())
        summary=dict(summary=report['summary'],results=[{k:r.get(k) for k in ('pair_id','winner','qwen_family_win','raw_json_complete','raw_proposal_valid','defect_flags','raw_defects','geometry_mae','raw_geometry_mae')}|dict(qwen_off_winner=r.get('qwen_off',{}).get('winner'),qwen_off_flags=r.get('qwen_off',{}).get('defect_flags')) for r in report['results']],manual_review=report.get('manual_review',[]))
        (PACK/'evidence/novel_designs_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        for item in novelitems:
            image=f'../novel_designs/{item["contact"]}';output=f'../novel_designs/{item["output_json"]}'
            manifest.append(dict(item,image=image,json=output,case=item['title'],group='novel_designs',flagged=bool(item['defect_flags'])))
            novelcards.append(f'<article><h3>{html.escape(item["title"])} · {html.escape(item["pair_id"].split("__to__")[1])}</h3><a href="../novel_designs/index.html"><img loading="lazy" src="{image}"></a><p>Qwen ON: {html.escape(item["winner"])} · learned-family winner: {item["qwen_family_win"]}</p><p>ON flags: {html.escape(str(item["defect_flags"]))} · OFF: {html.escape(item["qwen_off_winner"])}</p><p><a href="../novel_designs/index.html">Source / raw / model ON / model OFF</a> · <a href="{output}">Selected JSON</a></p></article>')
        newcards=novelcards+newcards
    cards=newcards+cards
    (gallery/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current engine outputs</title><style>body{font:15px system-ui;background:#101820;color:#eee;margin:24px}a{color:#9ed4ff}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:#182532;padding:16px;border-radius:12px}img{width:100%;height:420px;object-fit:contain}p{overflow-wrap:anywhere}@media(max-width:800px){main{grid-template-columns:1fr}}</style><h1>Engine review · new authored designs and model ON/OFF first</h1><p>Eight new-composition tests reuse the exact earlier local model predictions and compare current Qwen ON/OFF engine outputs. The twelve validation designs and twenty-four additional held-out sources replay identical recorded predictions through the current engine. Each dataset card compares source, human preview, raw proposal and final output. Source-authored identity issues are disclosed separately; passing checks is not a guarantee of aesthetic quality. JSON remains beside every image.</p><p><a href="../index.html">Review overview</a> · <a href="../../docs/archive/MACRO_STRUCTURE_REVIEW.md">Current verification and limitations</a></p><main>'+''.join(cards)+'</main>')
    # Portable paired worst-case gallery, including external dataset previews.
    original=run/'heldout/viewer';target=PACK/'review/inventory_worst10';target.mkdir(parents=True,exist_ok=True)
    entries=json.loads((original/'manifest.json').read_text())
    for i,entry in enumerate(entries,1):
        for key in ('source','reference','candidate','baseline'):
            ref=entry.get(key)
            if not ref:continue
            src=(original/ref).resolve();dest=target/f'{i:02d}_{key}.jpg'
            jpeg(src,dest);entry[key]=dest.name
    payload=json.dumps(entries,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>Current worst 10</title><style>body{font:15px system-ui;background:#101820;color:#eee;margin:20px}a{color:#9ed4ff}header{position:sticky;top:0;background:#101820;padding:12px}button{padding:10px;margin-right:10px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}img{width:100%;height:420px;object-fit:contain;background:#182532}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><header><a href="../index.html">Overview</a> · <button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="count"></span></header><h2 id="title"></h2><div class="grid"><figure><figcaption>Source</figcaption><img id="source"></figure><figure id="ref"><figcaption>Human dataset preview</figcaption><img id="reference"></figure><figure><figcaption>Prior engine</figcaption><img id="baseline"></figure><figure><figcaption>Current engine</figcaption><img id="candidate"></figure></div><pre id="metrics"></pre><script>const rows=__ROWS__;let i=0;function show(){const r=rows[i];document.getElementById('count').textContent=`${i+1}/${rows.length}`;document.getElementById('title').textContent=r.pair_id;for(const k of ['source','reference','baseline','candidate'])document.getElementById(k).src=r[k]||'';document.getElementById('ref').hidden=!r.reference;document.getElementById('metrics').textContent=JSON.stringify(r.candidate_metrics,null,2)}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}show()</script>'''.replace('__ROWS__',payload)
    (target/'review.html').write_text(page);(target/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    evidence=PACK/'evidence/inventory_repairs';evidence.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run/'summary.json',evidence/'summary.json')
    compact_keys=('pair_id','source_split','target_canvas','winner','flagged','defect_flags','glyph_overflow',
                  'source_contrast_regressions','control_collisions','source_authored_issues','visual_contracts','text_hero_occlusion','geometry_mae','raw_geometry_mae',
                  'macro_reflow','comparison','layout_quality','before_defects','after_defects','contrast_dispatch_reexecuted','output_design')
    for name,d in reports.items():
        (evidence/(name+'.jsonl')).write_text(''.join(json.dumps({k:r[k] for k in compact_keys if k in r},ensure_ascii=False)+'\n' for r in d['results']))
        (evidence/(name+'_run_manifest.json')).write_text(json.dumps(d['manifest'],indent=2)+'\n')
    for name in ('novel_validation','reverse_level3'):
        report=json.loads((run/name/'report.json').read_text())
        (evidence/(name+'.jsonl')).write_text(''.join(json.dumps({k:r[k] for k in compact_keys if k in r},ensure_ascii=False)+'\n' for r in report['results']))
        if report.get('manifest'):
            (evidence/(name+'_run_manifest.json')).write_text(json.dumps(report['manifest'],indent=2)+'\n')
    shutil.copy2(evidence/'reverse_level3.jsonl',PACK/'evidence/synthetic_level3_reverse.jsonl')
    for path in ['engine/geometry_policy.py','engine/macro_layout.py','engine/source_structure.py','engine/clean_zones.py','engine/structural_reflow.py','engine/layout_contracts.py','engine/resize.py','engine/qwen_refine.py','engine/qwen_seed.py','engine/qwen_prompt.py',
                 'scripts/package_inventory_review.py','scripts/replay_structure_audit.py','scripts/build_structure_comparisons.py','scripts/evaluate_qwen_adapter.py','scripts/layout_defects.py','scripts/audit_inventory_fixes.py','scripts/finalize_visual_contract_audit.py',
                 'scripts/build_novel_level3_designs.py','scripts/evaluate_novel_designs.py','scripts/replay_validation_sample.py','scripts/build_diverse_dataset_viewer.py','scripts/evaluate_novel_validation.py','scripts/replay_qwen_engine.py','scripts/replay_visual_contract_delta.py','scripts/build_level3_reverse_fixtures.py','scripts/evaluate_level3_reverse.py','scripts/package_visual_contract_comparisons.py',
                 'tests/test_geometry_policy.py','tests/test_macro_layout.py','tests/test_novel_designs.py','tests/test_source_attachments.py','tests/test_qwen_pipeline.py','tests/test_source_evidence.py','tests/test_diverse_selection.py','tests/test_replay_pairs.py','tests/test_visual_contracts.py','tests/test_inventory_repairs.py','tests/test_stress_metrics.py','tests/test_layout_mechanics.py','pytest.ini',
                 'docs/archive/RIGID_ART_REVIEW.md','docs/archive/MACRO_STRUCTURE_REVIEW.md','docs/archive/VISUAL_CONTRACT_REPAIRS.md','docs/IMPLEMENTATION.md','docs/archive/INVENTORY_REPAIR_AUDIT.md','README.md']:
        dest=PACK/'code'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
    for fixture in (ROOT/'tests/fixtures/level3_reverse').glob('*.json'):
        dest=PACK/'code/tests/fixtures/level3_reverse'/fixture.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fixture,dest)
    for fixture in (ROOT/'tests/fixtures/novel_level3_sources').glob('*.json'):
        dest=PACK/'code/tests/fixtures/novel_level3_sources'/fixture.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fixture,dest)
    (PACK/'docs').mkdir(exist_ok=True);shutil.copy2(ROOT/'docs/archive/INVENTORY_REPAIR_AUDIT.md',PACK/'docs/archive/INVENTORY_REPAIR_AUDIT.md')
    shutil.copy2(ROOT/'docs/archive/VISUAL_CONTRACT_REPAIRS.md',PACK/'docs/archive/VISUAL_CONTRACT_REPAIRS.md')
    shutil.copy2(ROOT/'docs/archive/MACRO_STRUCTURE_REVIEW.md',PACK/'docs/archive/MACRO_STRUCTURE_REVIEW.md')
    shutil.copy2(ROOT/'docs/archive/RIGID_ART_REVIEW.md',PACK/'docs/archive/RIGID_ART_REVIEW.md')
    (PACK/'README_FOR_GEMINI.md').write_text("""# Compact current-engine review

Read `docs/archive/RIGID_ART_REVIEW.md`, then open `review/index.html`. `docs/archive/MACRO_STRUCTURE_REVIEW.md` records the earlier checkpoint.

- `review/novel_designs/index.html`: four genuinely new authored compositions, eight resize tasks. Exact earlier local 0.6B coordinates; current source / raw proposal / model ON / model OFF. Neutral IDs, frozen sources, all failures retained. These are new-composition stress tests, not unseen-asset or certified unseen-family claims.
- `evidence/novel_designs_summary.json`: completion, raw/selected/OFF defect counts, winner provenance and manual notes.
- `review/novel_validation/index.html`: twelve additional validation designs, exact earlier predictions replayed through current code.
- `review/dataset_diversity/index.html`: twenty-four held-out sources from thirteen families, selected without outcome scores.
- `review/main_engine_qwen_outputs/index.html`: all current images, new designs first. Every selected JSON is linked; learned-family and fallback winners are distinguished.
- `review/structure_changes/index.html`: every changed output versus the immediately previous frozen engine, including roundoff-only differences. `changes.json` retains exact canonical hashes and matching review-raster pixel hashes.
- `review/inventory_worst10/review.html`: worst ten held-out comparison cases.
- `review/visual_contract_fixes/index.html`: historical reported output / current output.
- `review/synthetic_level3_reverse/index.html`: legacy same-elements reverse fixtures, retained only for regression.
- `evidence/inventory_repairs/`: all 89 held-out, 18 official, paired 12-case adapters, twelve validation and eight legacy reverse outputs. Consult each report for current flags; authored identity issues remain disclosed. Replay is not new model inference.
- `code/`: current engine, tests, reproduction scripts and authored source fixtures. No weights or full font/dataset bundles.

The earlier prompt experiment remains historical; its raw prediction cache is retained. The active prompt and training adapter were not changed. No training was performed. Old documents are retained as labeled checkpoint history. Passing automatic checks does not imply aesthetic perfection; compare actual rendered sources and outputs.
""")
    (PACK/'GEMINI_REVIEW_PROMPT.md').write_text("""Read docs/archive/RIGID_ART_REVIEW.md first; the macro review is historical. Inspect structure_changes and admission diagnostics; separate substantive layout changes from floating-point normalization differences. Then compare the raw model panels against the current deterministic repairs. Inspect the eight genuinely new composition tests before the previously reviewed examples. Compare source, raw Qwen, selected Qwen-ON and Qwen-OFF. Do not credit deterministic fallback as raw model quality. Check edge anchors, parent containment, group scale, relative copy axes, rotated labels and negative space; all outputs, including failures, are retained. There is no human resized target for authored fixtures, so target MAE is null. Existing Crello training-split photographs are reused; do not claim unseen-asset generalization. The separate 151-case audit reuses identical model coordinates through the frozen current engine. Recommend general source-evidenced corrections only, with cross-family regression checks. Zero flags is not aesthetic perfection. Do not propose template-ID-specific rules.
""")
    (PACK/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current Gemini review</title><style>body{font:19px system-ui;background:#101820;color:#eee;max-width:900px;margin:60px auto}a{color:#9ed4ff}li{margin:22px}</style><h1>Current engine review</h1><ul><li><a href="structure_changes/index.html">Latest structural changes · every before / after comparison and unchanged hashes</a></li><li><a href="novel_designs/index.html">NEW: four different designs × two resizes · source / raw / Qwen ON / Qwen OFF</a></li><li><a href="novel_validation/index.html">12 additional validation-family designs · frozen prediction replay</a></li><li><a href="dataset_diversity/index.html">24 held-out sources · source / preview / proposal / output</a></li><li><a href="visual_contract_fixes/index.html">Previously reported defects: before / after</a></li><li><a href="main_engine_qwen_outputs/index.html">All current engine outputs</a></li><li><a href="inventory_worst10/review.html">89-case audit: worst 10 comparisons</a></li><li><a href="../docs/archive/MACRO_STRUCTURE_REVIEW.md">Current approach, verification and limitations</a></li><li><a href="synthetic_level3_reverse/index.html">Regression archive: older same-elements reverse fixtures</a></li></ul>')
    origin=run/'all_changes';portable=PACK/'review/structure_changes'
    if origin.exists():
        portable.mkdir(exist_ok=True)
        for path in origin.iterdir():
            if path.is_file():shutil.copy2(path,portable/path.name)
        changes=json.loads((origin/'changes.json').read_text())
        proof=PACK/'review/official_fixes';proof.mkdir(exist_ok=True)
        official_changes=[c for c in changes['changed'] if c['cohort']=='official' and not c['render_pixels_identical']]
        panels=[]
        for c in official_changes:
            for key in ('image','json'):shutil.copy2(origin/c[key],proof/c[key])
            panels.append(f'<article><h2>{html.escape(c["pair_id"])}</h2><img src="{c["image"]}"><p>Source · previous engine · current engine</p><a href="{c["json"]}">Current JSON</a></article>')
        (proof/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Official resize fixes</title><style>body{font:17px system-ui;background:#101820;color:#eee;margin:24px}a{color:#9ed4ff}article{background:#182532;padding:20px;margin:20px 0}img{width:100%;max-width:1500px}h2{overflow-wrap:anywhere}</style><h1>Official resizes · visual changes</h1><p>Exact earlier model coordinates; current repairs. All official outputs with changed review pixels are included. <a href="../main_engine_qwen_outputs/index.html?group=official">All 18 official outputs</a></p>'+''.join(panels))
    if (run/'art_policy_audit.json').exists():
        shutil.copy2(run/'art_policy_audit.json',PACK/'evidence/art_policy_audit.json')
    # Keep the portable packet small without resizing its comparison images.
    # Full-quality renders remain in the local run directory.
    from io import BytesIO
    for name in ('dataset_diversity','novel_validation','novel_designs','structure_changes'):
        pattern='*comparison.jpg' if name=='structure_changes' else '*contact.jpg'
        for image in (PACK/'review'/name).glob(pattern):
            with Image.open(image) as src:
                data=BytesIO();src.convert('RGB').save(data,format='JPEG',quality=82,optimize=True,progressive=True)
            if len(data.getvalue())<image.stat().st_size:image.write_bytes(data.getvalue())
    manifest=[]
    for p in sorted(PACK.rglob('*')):
        if p.is_file() and p.name not in ('MANIFEST.sha256','.DS_Store'):
            manifest.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(PACK).as_posix()}')
    (PACK/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    (PACK/'evidence/current_checkpoint.json').write_text(json.dumps(dict(checkpoint=run.name,run=str(run),raw_coordinates_reused=True),indent=2)+'\n')
    from scripts.refresh_review_viewer import refresh
    refresh(PACK)
    print('Refreshed',PACK,'files',len(manifest))

if __name__=='__main__':main()
