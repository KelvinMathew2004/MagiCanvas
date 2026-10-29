#!/usr/bin/env python3
"""Publish completed pilot arms beside the unchanged production viewer."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse,html,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def flaws(result, source=None):
    """Instance counts; flag names alone are not flaw counts."""
    contracts=result.get('visual_contracts',[])
    stretched=set(result.get('image_ratio_violations',[]))
    stretched.update(i for d in contracts if d.get('kind')=='art_aspect_distortion' for i in d.get('ids',[]))
    # Both existing checks can describe the same asset. Count it once and
    # exclude a source full-bleed background from foreground deformation.
    if source:
        w,h=source['canvas']['width'],source['canvas']['height']
        backgrounds={e['id'] for e in source['elements'] if e['type']!='text' and not e.get('group_id')
            and e['x']<=.02*w and e['y']<=.02*h and e['x']+e['width']>=.98*w and e['y']+e['height']>=.98*h
            and all(e.get('z_index',0)<=t.get('z_index',0) for t in source['elements'] if t['type']=='text')}
        stretched-=backgrounds
    return dict(stretch=len(stretched),
        alignment=sum(d.get('kind') in ('primary_axis','control_label_axis','subpanel_axis','peer_gap_asymmetry') for d in contracts),
        overlap=sum(len(result.get(k,[])) for k in ('unintended_text_overlaps','control_collisions','text_hero_occlusion')),
        out_of_bounds=sum(any(k in str(d).lower() for k in ('outside','bounds','off-canvas')) for d in result.get('structural_errors',[])),
        text_fit=len(result.get('glyph_overflow',{})))


def inventory(row):
    from experiments.layout_upgrade.contracts import decode
    _,report=decode(row.get('response',''),row['source'],row['target'])
    return report


def normalized_source_drift(row):
    """Detect copying source coordinates, without consulting a target label."""
    source=row['source'];sw,sh=source['canvas']['width'],source['canvas']['height']
    tw,th=row['target']['width'],row['target']['height']
    originals={e['id']:e for e in source['elements']};diffs=[]
    for e in row.get('qwen_model_elements',[]):
        if e['id'] not in originals:continue
        for key,sd,td in (('x',sw,tw),('y',sh,th),('width',sw,tw),('height',sh,th)):
            diffs.append(abs(e[key]/td-originals[e['id']][key]/sd))
    return sum(diffs)/len(diffs) if diffs else None


def build(output):
    from scripts.build_focus_reflow_viewer import build as render,STYLE
    output.mkdir(parents=True,exist_ok=True);summaries=[];cards=[];contact=[]
    configurations=[]
    for arm,folder in (('qwen06_text_free','qwen06_text_free_v2'),('qwen06_text_grammar','qwen06_text_grammar'),('qwen17_text_grammar','qwen17_text_grammar')):
        native=ROOT/'data/upgrade_exploration/native_pilot_v1'
        configurations.append((native,arm,arm) if (native/'replay'/arm/'summary.json').exists()
                              else (ROOT/'data/upgrade_exploration/model_pilot_v1',arm,folder))
    coderroot=ROOT/'data/upgrade_exploration/new_model_native_v3'
    if not (coderroot/'replay/coder15_text_grammar/summary.json').exists():coderroot=ROOT/'data/upgrade_exploration/new_model_smoke_v1'
    configurations.append((coderroot,'coder15_text_grammar','coder15_text_grammar'))
    configurations += [(ROOT/'data/upgrade_exploration/new_model_native_v3',arm,arm) for arm in ('qwen35_text_grammar','qwen_vl_image_grammar','qwen_vl_text_grammar')]
    for root,arm,folder in configurations:
        replay=root/'replay'/folder
        if not (replay/'summary.json').is_file():continue
        raw={r['pair_id']:r for r in map(json.loads,(root/(arm+'.jsonl')).read_text().splitlines())}
        results=[json.loads(p.read_text()) for p in sorted(replay.glob('[0-9]*.json'))]
        trace_dir=root/'replay'/(arm+'_traced');trace_totals={}
        if (trace_dir/'summary.json').exists():
            traced={r['pair_id']:r for r in (json.loads(p.read_text()) for p in sorted(trace_dir.glob('[0-9]*.json')))}
            for result in results:
                trace=traced[result['pair_id']]
                if trace.get('output_design')!=result.get('output_design'):
                    raise ValueError('Repair tracing changed the selected output; investigate before publishing')
                result['repair_stages']=trace.get('repair_stages',{})
                for name,stage in result['repair_stages'].items():
                    total=trace_totals.setdefault(name,dict(cases=0,flaw_instances=Counter(),contrast_evaluated=False))
                    total['cases']+=1;total['flaw_instances'].update(flaws(stage['defects'],result['source_design']))
        manifest=json.loads((root/(arm+'.manifest.json')).read_text());records=output/'records'/arm;records.mkdir(parents=True,exist_ok=True)
        raster_path=replay/'native_rasters.json'
        rasters=json.loads(raster_path.read_text()) if raster_path.is_file() else {}
        raster_by={r['pair_id']:r for r in rasters.get('records',[])}
        for i,result in enumerate(results):
            if not result.get('output_design'):continue
            prediction=raw[result['pair_id']]
            audit=dict(name=result['pair_id'],source=result['source_design'],design=result['output_design'],
                target=result['target_canvas'],source_split=result['source_split'],frozen_model_prediction=True,
                diagnostics=dict(winner=result['winner'],model=manifest,raw=prediction,native_raster=raster_by.get(result['pair_id']),
                                 raw_schema=inventory(prediction),selection=result),
                defects=dict(defect_flags=result.get('defect_flags',[])))
            (records/f'{i:02d}.json').write_text(json.dumps(audit,indent=2)+'\n')
        label=arm+(' · matching trained adapter' if manifest['tuned'] else ' · UNTUNED base')
        rendered=[]
        if any(r.get('output_design') for r in results):
            rendered=render(records,output/arm,label)
            page=output/arm/'index.html'
            body=page.read_text().replace('Fresh Qwen-enabled engine run.','Fresh experimental generation; unchanged production repair and selection.')
            body=body.replace('</header>','<p>Experimental arm, not promoted. Source image conditioning: '+str(manifest['image_conditioned'])+'. Raw validity and proposals are in audit JSON. One case is an identity control; its final engine output does not use the proposal.</p></header>')
            page.write_text(body)
        before=Counter();after=Counter()
        for r in results:
            before.update(flaws(r.get('raw_defects',{}),r.get('source_design')));after.update(flaws(r,r.get('source_design')))
        contrast_counts={stage:dict(low_contrast=sum(len(r.get(stage,{}).get('low_contrast',[])) for r in raster_by.values()),
                 source_floor_regressions=sum(len(r.get(stage,{}).get('source_floor_regressions',[])) for r in raster_by.values()),
                 invisible_or_unpainted=sum(len(r.get(stage,{}).get('invisible_or_unpainted',[])) for r in raster_by.values())) for stage in ('raw','selected')}
        summary=dict(arm=arm,cases=len(results),tuned=manifest['tuned'],inventory_exact=sum(inventory(r)['inventory_exact'] for r in raw.values()),
            usable_geometry_complete=sum(inventory(r)['exact_ids'] for r in raw.values()),json_complete=sum(inventory(r)['json_complete'] for r in raw.values()),
            raw_flaws=dict(before),selected_flaws=dict(after),selected_flagged=sum(bool(r.get('defect_flags')) for r in results),
            selected_flags=dict(Counter(f for r in results for f in r.get('defect_flags',[]))),
            learned_winners=sum(bool(r.get('qwen_family_win')) for r in results),
            generation_seconds=[r.get('generation_seconds') for r in raw.values()],attention=manifest.get('attention','eager'),
            normalized_source_coordinate_drift={r['pair_id']:normalized_source_drift(r) for r in raw.values()},
            raw_contrast_evaluated=bool(raster_by) and not any('error' in r for r in raster_by.values()),
            native_raster=rasters,contrast_counts=contrast_counts,scope='Small development pilot, not the 89-case benchmark',
            repair_stage_totals=trace_totals,repair_trace_selected_output_parity=bool(trace_totals),
            code_sha256=manifest['files_sha256'],reference_geometry_mae=[r.get('geometry_mae') for r in results],
            inference_errors=[r.get('error') for r in raw.values() if r.get('error')])
        summaries.append(summary)
        cards.append(f'<tr><td><a href="{arm}/index.html">{html.escape(label)}</a></td><td>{len(results)}</td><td>{summary["inventory_exact"]}</td><td>{summary["usable_geometry_complete"]}</td><td>{summary["learned_winners"]}</td><td>{summary["selected_flagged"]}</td></tr>')
        contact.extend((label,row,output/arm) for row in rendered)
    critics={p.stem:json.loads(p.read_text()) for p in (ROOT/'data/upgrade_exploration/critic_pilot_v1').glob('*.json')}
    solver_path=ROOT/'data/upgrade_exploration/solver_pilot_v4/replay/cpsat/summary.json'
    if not solver_path.exists():solver_path=ROOT/'data/upgrade_exploration/solver_pilot_v3/replay/cpsat/summary.json'
    solver=json.loads(solver_path.read_text()) if solver_path.is_file() else None
    # The shared tasks permit a direct image-conditioning ablation and a
    # matched free/grammar comparison without averaging different case mixes.
    comparisons=[]
    for left,right in (('qwen06_text_free','qwen06_text_grammar'),('qwen06_text_grammar','qwen17_text_grammar'),
                       ('qwen_vl_text_grammar','qwen_vl_image_grammar')):
        a=next((s for s in summaries if s['arm']==left),None);b=next((s for s in summaries if s['arm']==right),None)
        if a is None or b is None:continue
        records_a={r['case']:r for r in json.loads((output/left/'manifest.json').read_text())}
        records_b={r['case']:r for r in json.loads((output/right/'manifest.json').read_text())}
        for key in sorted(records_a.keys()&records_b.keys()):
            da=json.loads((output/left/records_a[key]['json']).read_text());db=json.loads((output/right/records_b[key]['json']).read_text())
            comparisons.append(dict(a=left,b=right,pair_id=key,output_geometry_identical=da==db,
                rendered_jpeg_identical=(output/left/records_a[key]['image']).read_bytes()==(output/right/records_b[key]['image']).read_bytes(),
                a_winner=records_a[key]['winner'],b_winner=records_b[key]['winner']))
    report=dict(models=summaries,critics=critics,solver=solver,production_changed=False,
        paired_comparisons=comparisons,
        limitations=['Native glyph contrast coverage and exclusions are explicit per arm.','Automated zero flags do not imply perfect visual design.',
                    'Fresh inference overlapped CPU replay; times are not an isolated throughput benchmark.',
                    'New base models have no layout adapter; compare feasibility, not trained model quality.',
                    'Solver timeout is bounded wall time; its candidate selection need not be bit deterministic.'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    critic_rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["correct"]}/{v["cases"]}</td><td>{v["cohorts"]["campaign"]["correct"]}/{v["cohorts"]["campaign"]["cases"]}</td></tr>' for k,v in critics.items())
    defect_rows=''.join('<tr><td>'+html.escape(s['arm'])+'</td>'+''.join(f'<td>{s[stage][key]}</td>' for stage in ('raw_flaws','selected_flaws') for key in ('stretch','alignment','overlap','out_of_bounds','text_fit'))+'</tr>' for s in summaries)
    text='<!doctype html><meta charset="utf-8"><title>Upgrade pilots</title><style>'+STYLE+'table{border-collapse:collapse}td,th{padding:10px;text-align:left;border:1px solid #334552}.scroll{overflow-x:auto}</style><header><a href="../main_engine_qwen_outputs/index.html">Production regression viewer</a><h1>Upgrade pilots · no production promotion</h1><p>Original / Resized only. Each arm uses the same recorded pre-follow-up repair and selector; see the per-arm hashes. These historical model pilots are not rerendered after the owned-label fix. <a href="../upgrade_repair_followup/index.html">Current repair/sampling follow-up</a>. New base models are untuned. Inventory completion and usable numeric geometry are separate. One task per arm is an identity control and bypasses generator selection.</p><div class="scroll"><table><tr><th>Arm</th><th>Cases</th><th>All IDs emitted</th><th>Usable geometry</th><th>Learned winners</th><th>Flagged final</th></tr>'+''.join(cards)+'</table></div><h2>Raw → selected flaw instances</h2><p>S = stretch; A = alignment; O = unintended overlap; B = bounds; T = glyph overflow. Compare only arms with the same cases. Overlap contact classes may count the same contact separately; these are diagnostics, not an additive quality score.</p><div class="scroll"><table><tr><th>Arm</th>'+''.join('<th>'+stage+' '+key+'</th>' for stage in ('Raw','Selected') for key in ('S','A','O','B','T'))+'</tr>'+defect_rows+'</table></div><h2>Critic validation</h2><table><tr><th>Critic</th><th>All votes</th><th>Real campaign</th></tr>'+critic_rows+'</table><p>No test labels or fitting. Tiling improves one actual campaign vote; eight campaign comparisons are insufficient for promotion.</p><h2>Bounded solver</h2><p>'+html.escape(str(solver))+'</p><p><a href="report.json">Full counts, native glyph contrast, limitations and provenance</a> · <a href="contact.jpg">Selected output contact sheet</a></p><p>No 89-case release claim. Raw proposals still contain stretching, collisions and alignment defects despite schema-valid JSON. Model timings were collected alongside CPU work.</p></header>'
    (output/'index.html').write_text(text)
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(1050,max(1,(len(contact)+2)//3)*240),'#101820');draw=ImageDraw.Draw(sheet)
    for i,(label,row,folder) in enumerate(contact):
        x=(i%3)*350;y=(i//3)*240
        with Image.open(folder/row['image']) as im:
            im.thumbnail((330,196));sheet.paste(im,(x+(350-im.width)//2,y+30))
        draw.text((x+10,y+5),label[:47],fill='white');draw.text((x+10,y+220),row['case'][:43],fill='white')
    sheet.save(output/'contact.jpg',quality=88)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/upgrade_pilots')
    args=p.parse_args();report=build(args.output);print(json.dumps(dict(models=len(report['models']),output=str(args.output/'index.html'))))
