#!/usr/bin/env python3
"""Attach a completed fresh adapter comparison to the Gemini review packet."""
from pathlib import Path
from hashlib import sha256
import json,shutil,html,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def completed_run(pointer=None):
    pointer=pointer or ROOT/'data/training_next/latest_completed_evaluation.json'
    if not pointer.is_file():return None
    record=json.loads(pointer.read_text());run=Path(record['run'])
    for filename,key in [('manifest.json','manifest_sha256'),('paired_analysis.json','analysis_sha256')]:
        if sha256((run/filename).read_bytes()).hexdigest()!=record[key]:
            raise ValueError(f'Completed adapter evidence changed: {filename}')
    manifest=json.loads((run/'manifest.json').read_text());analysis=json.loads((run/'paired_analysis.json').read_text())
    for arm,digest in analysis['arms_sha256'].items():
        if sha256((run/(arm+'.jsonl')).read_bytes()).hexdigest()!=digest:
            raise ValueError('Adapter output records changed after final reporting')
    current={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'engine').glob('*.py'))}
    if manifest['engine']!=current:raise ValueError('Adapter comparison uses a different engine snapshot')
    if analysis['all']['tasks']!=148 or analysis['reference']['tasks']!=89 or analysis['review']['tasks']!=59:
        raise ValueError('Adapter comparison is incomplete')
    return run,analysis


def add_evidence(packet):
    for filename in ('recovered_prepared_corpus_coverage.json','recovered_partial_omissions.json'):
        path=ROOT/'data/training_next'/filename
        if path.is_file():shutil.copy2(path,packet/'evidence'/('recovered_adapter_'+filename))
    correction=ROOT/'data/training_next/realization_fault_regression_comparison.json'
    if correction.is_file():
        comparison=json.loads(correction.read_text())
        image_matches=sum(r['output_identical'] for r in comparison.get('image_comparison',[]))
        shutil.copy2(correction,packet/'evidence/recovered_adapter_realization_fault_regression.json')
        with (packet/'LATEST_UPDATE.md').open('a') as f:
            f.write(f"\n## Candidate realization fault isolation\n\nThe required-adapter path now distinguishes a load/generation failure from an exception after a proposal decoded. Required loading/generation still fails closed; realization failures preserve raw geometry and diagnostics and retain the normal fallback gates. Explicit old/new snapshot comparison: {comparison['cases']} tasks, {comparison['geometry_changes']} changed final geometries; {image_matches} exact resized JPEG matches. The known conference defect remains flagged. The earlier denominator-guard experiment was rejected because it worsened that output. Read docs/RECOVERED_ADAPTER_EVALUATION.md and evidence/recovered_adapter_realization_fault_regression.json.\n")
    result=completed_run()
    if not result:
        partial=ROOT/'data/training_next/recovered_partial_report.json'
        if partial.is_file():
            shutil.copy2(partial,packet/'evidence/recovered_adapter_interrupted_partial.json')
            with (packet/'LATEST_UPDATE.md').open('a') as f:
                f.write('\n## Recovered adapter: comparison still incomplete\n\nRead docs/RECOVERED_ADAPTER_EVALUATION.md and evidence/recovered_adapter_interrupted_partial.json. These are 33 matching tasks from the interrupted first snapshot, not the full 89+59 study or a promotion decision. The raw-font improvement and ID/box regressions are reported together. The engine realization-fault correction has its own refreshed 59-case verification; saved predictions will be re-realized before final comparison. Production remains on the earlier adapter.\n')
        return False
    run,analysis=result
    shutil.copy2(ROOT/'docs/RECOVERED_ADAPTER_EVALUATION.md',packet/'docs/RECOVERED_ADAPTER_EVALUATION.md')
    for filename in ('manifest.json','report.json','paired_analysis.json','parallel_arm_provenance.json','visual_review.json','native_raster_audit.json','stage_defect_audit.json','completed_review_subset.json','timing_limitations.json'):
        path=run/filename
        if path.is_file():shutil.copy2(path,packet/'evidence'/('recovered_adapter_'+filename))
    ref=analysis['reference'];review=analysis['review']
    note=(f"\n## Completed fresh recovered-adapter A/B\n\n"
          f"Read docs/RECOVERED_ADAPTER_EVALUATION.md and evidence/recovered_adapter_paired_analysis.json. "
          f"The comparison collects fresh predictions for all 59 review tasks plus 89 consumed reference pairs, with Qwen enabled, the same 0.6B base and unchanged selector. Resumed predictions are re-realized under the recorded engine; previous final outputs are not reused. "
          f"Reference geometry MAE: {ref['geometry_mae']['before']:.6f} → {ref['geometry_mae']['after']:.6f}; "
          f"font MAE: {ref['font_mae']['before']:.6f} → {ref['font_mae']['after']:.6f}. "
          f"Review flagged tasks: {review['before_flagged']} → {review['after_flagged']}. "
          "Do not infer gains from validation loss alone. Raw common-valid denominators, dense completion, new defects and family uncertainty are reported separately. "
          "The original Colab training manifest is missing; recovery does not invent its provenance. No weight files are included. "
          "Full local viewer: review/trained_adapters/index.html; model selection changes only Resized. Independent arm processes overlapped, so timing is not an isolated speed benchmark.\n")
    with (packet/'LATEST_UPDATE.md').open('a') as f:f.write(note)
    with (packet/'GEMINI_REVIEW_PROMPT.md').open('a') as f:
        f.write('\nAudit the fresh recovered-adapter comparison separately from frozen engine replays. Review visual regressions even without flags; do not recommend promotion solely on average MAE or training loss.\n')
    return True


def compact_cards(packet,stage):
    """All raw audits; review images plus ten largest reference changes."""
    result=completed_run()
    folder=packet/'review/trained_adapters'
    if not result or not (folder/'manifest.json').is_file():return []
    run,analysis=result;rows=json.loads((folder/'manifest.json').read_text())
    if len(rows)!=148:raise ValueError('Fresh adapter viewer is incomplete')
    differences={r['pair_id']:r for r in analysis['changes']}
    refs=[r for r in rows if r['pair_id'].startswith('reference:')]
    def priority(row):
        r=differences.get(row['pair_id'],{})
        return (bool(r.get('new_flags')),abs((r.get('after_geometry') or 0)-(r.get('before_geometry') or 0)))
    selected={r['pair_id'] for r in rows if r['pair_id'].startswith('review:')}
    selected.update(r['pair_id'] for r in sorted(refs,key=priority,reverse=True)[:10])
    from PIL import Image
    cards=[];image_cache={}
    for row in rows:
        i=row['index'];audit=f'cases/adapter_{i:03d}_audit.json'
        shutil.copy2(folder/row['audit'],stage/audit)
        if row['pair_id'] not in selected:continue
        paths={}
        for label,file in [('Original',row['source']),*row['outputs'].items()]:
            digest=sha256((folder/file).read_bytes()).hexdigest()
            if digest in image_cache:
                paths[label]=image_cache[digest];continue
            dest=f'images/adapter_{i:03d}_{label}.jpg'
            with Image.open(folder/file) as im:
                im=im.convert('RGB');im.thumbnail((900,900));im.save(stage/dest,quality=80,optimize=True)
            paths[label]=dest;image_cache[digest]=dest
        candidate=next(k for k in row['outputs'] if k!='baseline')
        options=''.join(f'<option value="{paths[k]}">{html.escape(k)}</option>' for k in (candidate,'baseline'))
        cards.append(f'''<article><h2>Fresh adapter A/B · {html.escape(row['pair_id'])}</h2><label>Resized <select onchange="document.getElementById('adapter-{i}').src=this.value">{options}</select></label>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px"><section><h3>Original</h3><img src="{paths['Original']}" alt="Original"></section><section><h3>Resized</h3><img id="adapter-{i}" src="{paths[candidate]}" alt="Resized"></section></div>
<p>Fresh model comparison, unchanged selector; see the recovery decision before interpreting these as production outputs. <a href="{audit}">Both raw/final audits</a>.</p></article>''')
    return cards
