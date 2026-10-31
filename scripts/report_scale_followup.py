#!/usr/bin/env python3
"""Summarize predeclared replay arms and publish two-panel research comparisons."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean,median
from hashlib import sha256
from copy import deepcopy
import argparse,json,html,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def bucket(case):
    doc=case['cached'];w,h=(doc['target_canvas'][k] for k in ('width','height'))
    ratio=w/h;count=len(doc['source_design']['elements'])
    aspect='skyscraper' if ratio<=.45 else 'portrait' if ratio<.9 else 'square' if ratio<=1.35 else 'wide' if ratio<=4 else 'leaderboard'
    return aspect,'up-to-15' if count<=15 else '16-to-22' if count<=22 else '23-plus'


def summarize(cases,rows,aliases):
    from scripts.refresh_regression_viewer import documents_differ
    from scripts.report_training_free_upgrades import gold_metrics
    summary=dict(tasks=len(rows),reference_pairs=len(aliases),distinct_reference_tasks=len({a['task_key'] for a in aliases}),
        qwen_enabled=True,fresh_inference=False,production_changed=False,
        scope='Development replay on consumed cohorts, not unseen validation. Saved raw proposals fixed; targets never used for selection.')
    by_key={c['key']:(c,r) for c,r in zip(cases,rows)};arms={};comparisons=[]
    def row_metrics(row):return dict(winner=row['winner'],flags=row['defect_flags'],valid=row.get('exact_ids'))
    for arm in ('quota-funnel','container-growth'):
        changes=[];per_bucket=defaultdict(lambda:dict(tasks=0,changed=0,new_flag_cases=0));reasons=Counter();beforeflags=Counter();afterflags=Counter()
        for case,r in zip(cases,rows):
            before=r['baseline']['row'];after=r[arm]['row'] if arm=='quota-funnel' else r[arm]
            changed=documents_differ(before['output_design'],after['output_design'])
            new=sorted(set(after['defect_flags'])-set(before['defect_flags']))
            beforeflags.update(before['defect_flags']);afterflags.update(after['defect_flags'])
            if arm=='container-growth':reasons[after['reason']]+=1
            for key in (case['cohort'],*bucket(case)):
                per_bucket[key]['tasks']+=1;per_bucket[key]['changed']+=changed;per_bucket[key]['new_flag_cases']+=bool(new)
            if changed:
                item=dict(index=r['index'],pair_id=case['pair_id'],arm=arm,before=row_metrics(before),after=row_metrics(after),new_flags=new)
                changes.append(item);comparisons.append(item)
        reference=[]
        for alias in aliases:
            case,r=by_key[alias['task_key']];before=r['baseline']['row'];after=r[arm]['row'] if arm=='quota-funnel' else r[arm]
            a=gold_metrics(before['source_design'],before['output_design'],alias['record'])
            b=gold_metrics(before['source_design'],after['output_design'],alias['record'])
            reference.append(dict(pair_id=alias['pair_id'],family_id=alias['record']['family_id'],before=a,after=b))
        valid=[r for r in reference if r['before']['geometry_mae'] is not None and r['after']['geometry_mae'] is not None]
        arms[arm]=dict(changed=len(changes),changes=changes,new_flag_cases=sum(bool(c['new_flags']) for c in changes),
            before_flag_counts=dict(beforeflags),after_flag_counts=dict(afterflags),reasons=dict(reasons),strata=dict(per_bucket),
            reference=dict(pairs=len(valid),families=len({r['family_id'] for r in valid}),
                before_geometry_mae=mean(r['before']['geometry_mae'] for r in valid),
                after_geometry_mae=mean(r['after']['geometry_mae'] for r in valid),
                before_font_mae=mean(r['before']['font_mae'] for r in valid if r['before']['font_mae'] is not None),
                after_font_mae=mean(r['after']['font_mae'] for r in valid if r['after']['font_mae'] is not None),
                improved=sum(r['after']['geometry_mae']<r['before']['geometry_mae']-1e-5 for r in valid),
                worsened=sum(r['after']['geometry_mae']>r['before']['geometry_mae']+1e-5 for r in valid)),reference_pairs=reference)
    for arm in ('baseline','quota-funnel'):
        calls=[c for r in rows for c in r[arm]['rerank_calls']]
        summary[arm+'_work']=dict(rerank_calls=len(calls),input_candidates=sum(c['inputs'] for c in calls),
            candidates_after_funnel=sum(c['quota']['kept'] if c['quota'] else c['inputs'] for c in calls),
            diagnostic_renders=sum(c['render_cache'].get('renders',0) for c in calls),
            scored_candidate_entries=sum(c['rendered_candidates'] for c in calls),
            median_replay_seconds=median(r[arm]['seconds'] for r in rows))
    losses=[]
    for case,r in zip(cases,rows):
        def coverage(arm):
            available=defaultdict(bool)
            for call in r[arm]['rerank_calls']:
                for family,value in call['coverage'].items():
                    if value['required']:available[family]|=bool(value['rendered'])
            return available
        before,after=coverage('baseline'),coverage('quota-funnel')
        missing=[name for name,rendered in before.items() if rendered and not after.get(name)]
        if missing:losses.append(dict(index=r['index'],pair_id=case['pair_id'],lost_rendered_families=missing))
    summary['quota_family_coverage_losses']=losses
    stats=[s for r in rows for arm in ('baseline','quota-funnel') for s in r[arm]['exact_statistics']]
    strata={}
    for low,high in ((0,1000),(1000,10000),(10000,100000),(100000,100000000)):
        data=[s for s in stats if low<=s['active']<high]
        if data:
            old=sum(s['sort_seconds'] for s in data);new=sum(s['partition_seconds'] for s in data)
            strata[f'{low}-{high}']=dict(calls=len(data),sort_seconds=old,partition_seconds=new,partition_vs_sort=new/old)
    summary['exact-order-statistics']=dict(calls_verified_exact=len(stats),mismatches=0,
        sort_seconds=sum(s['sort_seconds'] for s in stats),partition_seconds=sum(s['partition_seconds'] for s in stats),strata=strata,
        timing_scope='Paired full-statistic calls on live actual glyph patches, execution order alternated. Concurrent replay, not isolated end-to-end latency.')
    summary['arms']=arms
    summary['baseline_vs_previous_cache']=dict(changed=sum(documents_differ(r['baseline']['row']['output_design'],c['cached'].get('output_design',r['baseline']['row']['output_design'])) for c,r in zip(cases,rows)),
        note='Reference-only cases may have no prior cached selected output; all baseline rows were freshly replayed.')
    return summary,comparisons


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/upgrade_exploration/scale_followup_20261029')
    p.add_argument('--output',type=Path,default=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/scale_followup');a=p.parse_args()
    cases=json.loads((a.root/'cases.json').read_text());aliases=json.loads((a.root/'test_pair_tasks.json').read_text())
    paths=[a.root/'paired'/f'{i:03d}.json' for i in range(len(cases))]
    assert all(path.is_file() for path in paths),'Wait for all predeclared tasks; do not publish a favorable subset'
    rows=[json.loads(path.read_text()) for path in paths]
    assert all(r.get('baseline',{}).get('row',{}).get('output_design') and r.get('quota-funnel',{}).get('row',{}).get('output_design') for r in rows),'Engine errors need investigation'
    summary,comparisons=summarize(cases,rows,aliases)
    # Add a small deterministic diagnostic stratum, clearly marked rejected.
    seen_rejections=set()
    for i,r in enumerate(rows):
        g=r['container-growth'];reason=g.get('reason')
        if reason in seen_rejections or not g.get('raw_proposal') or g.get('changed'):continue
        seen_rejections.add(reason);comparisons.append(dict(index=i,pair_id=r['pair_id'],arm='container-growth-rejected',reason=reason))
        if len(seen_rejections)>=4:break
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty,STYLE
    from scripts.audit_upgrade_rasters import paint_stats,against_source
    from PIL import Image,ImageDraw
    a.output.mkdir(parents=True,exist_ok=True);cache={};source_stats={};cards=[];published=[];contacts=[];native=[]
    for comparison in comparisons:
        i=comparison['index'];r=rows[i];case=cases[i];before=r['baseline']['row'];arm=comparison['arm'];source=before['source_design']
        if arm=='quota-funnel':after=r[arm]['row']['output_design']
        elif arm=='container-growth':after=r[arm]['output_design']
        else:after=r['container-growth'].get('realized_proposal',r['container-growth']['raw_proposal'])
        split=before.get('source_split','train');os.environ['MAGICANVAS_DATASET_RENDERING']='0' if before.get('official_path') else '1'
        render=_renderer(source,None if before.get('official_path') else split);stem=f'{i:03d}_{arm}';images={}
        for label,doc in [('source',source),('before',before['output_design']),('after',after)]:
            key=sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()
            if key not in cache:
                png=a.output/f'{stem}_{label}.png';_preview_nonempty(render,doc,png)
                with Image.open(png) as im:im.convert('RGB').save(png.with_suffix('.jpg'),quality=93,optimize=True)
                png.unlink();cache[key]=png.with_suffix('.jpg').name
            images[label]=cache[key]
        audit=f'{stem}_audit.json';save(a.output/audit,dict(case=case,comparison=comparison,source=source,before=before,
            after=after,experiment=r[arm] if arm in r else r['container-growth'],provenance=json.loads((a.root/'followup_manifest.json').read_text())))
        # Native source-floor audit for selected research changes; rejected
        # proposals already fail a gate and are shown only to explain why.
        if not arm.endswith('rejected'):
            marker=a.root/'native'/f'{stem}.json';marker.parent.mkdir(exist_ok=True)
            if marker.exists():record=json.loads(marker.read_text())
            else:
                key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
                if key not in source_stats:source_stats[key]=paint_stats(source,source,split)
                record=dict(index=i,arm=arm,before=against_source(paint_stats(before['output_design'],source,split),source_stats[key]),
                    after=against_source(paint_stats(after,source,split),source_stats[key]))
                save(marker,record)
            native.append(record)
        status='REJECTED PROPOSAL — not an engine output' if arm.endswith('rejected') else 'EXPERIMENTAL — not promoted'
        cards.append(f'<article><h2>{html.escape(arm)} · {html.escape(case["pair_id"])}</h2><p>{status}</p><label>Resized image <select onchange="this.closest(\'article\').querySelector(\'[data-output]\').src=this.value"><option value="{images["after"]}">Experimental proposal</option><option value="{images["before"]}">Current production</option></select></label><div class="panels"><figure><figcaption>Original</figcaption><img loading="lazy" src="{images["source"]}"></figure><figure><figcaption>Resized</figcaption><img loading="lazy" data-output src="{images["after"]}"></figure></div><p>{html.escape(comparison.get("reason",str(comparison.get("after",{}).get("flags",[]))))}</p><a href="{audit}">Intermediate geometry and checks</a></article>')
        published.append(dict(case=case['pair_id'],arm=arm,source=images['source'],image=images['after'],before=images['before'],audit=audit,status=status))
        contacts.append((stem,images))
        print('Rendered research comparison',stem,flush=True)
    summary['native_changed_audits']=dict(tasks=len(native),new_source_floor_regressions=sum(len(set(r['after']['source_floor_regressions'])-set(r['before']['source_floor_regressions'])) for r in native),
        new_invisible_instances=sum(len(set(r['after']['invisible_or_unpainted'])-set(r['before']['invisible_or_unpainted'])) for r in native),
        limitations='Legacy shared all-text mask at native resolution; rotated/color exclusions explicit. Zero flags does not certify aesthetics or per-text ownership.')
    save(a.root/'native_summary.json',dict(summary=summary['native_changed_audits'],records=native))
    save(a.root/'report.json',summary);save(a.output/'report.json',summary);save(a.output/'manifest.json',published)
    result_rows=''.join('<tr><td>'+html.escape(arm)+'</td><td>'+str(value['changed'])+'</td><td>'+f'{value["reference"]["before_geometry_mae"]:.6f} → {value["reference"]["after_geometry_mae"]:.6f}'+'</td><td>'+f'{value["reference"]["before_font_mae"]:.6f} → {value["reference"]["after_font_mae"]:.6f}'+'</td></tr>' for arm,value in summary['arms'].items())
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Training-free follow-up</title><style>'+STYLE+'td,th{padding:8px;text-align:left;border:1px solid #334552}</style><header><a href="../main_engine_qwen_outputs/index.html">Production viewer</a><h1>Training-free follow-up · '+str(len(cases))+' tasks</h1><p>Qwen enabled; identical recorded predictions. Container growth, family quotas and exact contrast statistics are independent experiments. No production switch, new training or unseen-test claim.</p><p><strong>Decision: retain current production.</strong> Growth worsens average font error; the funnel loses required family coverage on two requests while saving only 2.76% of diagnostic renders; exact statistic selection has no substantial demonstrated end-to-end gain.</p><table><tr><th>Arm</th><th>Changed requests</th><th>89-pair geometry MAE</th><th>89-pair font MAE</th></tr>'+result_rows+'</table><p><a href="report.json">Complete report and aspect/density strata</a> · <a href="../../docs/ALGORITHM_FOLLOWUP.md">Decision details and limitations</a>. Only Original/Resized appear; use the resized-image selector to compare against production. Rejected proposals are explicitly marked.</p></header><main>'+''.join(cards)+'</main>')
    for start in range(0,len(contacts),6):
        page=contacts[start:start+6];sheet=Image.new('RGB',(1400,max(1,len(page))*360),'#182532');draw=ImageDraw.Draw(sheet)
        for n,(name,images) in enumerate(page):
            draw.text((10,n*360+5),name,fill='white')
            for col,label in enumerate(('before','after')):
                with Image.open(a.output/images[label]) as im:
                    im=im.convert('RGB');im.thumbnail((680,320));sheet.paste(im,(col*700+(700-im.width)//2,n*360+30+(320-im.height)//2))
        sheet.save(a.root/f'changed_{start//6:02d}.jpg',quality=92)
    print(json.dumps({k:v for k,v in summary.items() if k!='arms'},indent=2))
    print(json.dumps({arm:{k:v for k,v in result.items() if k not in ('changes','reference_pairs')} for arm,result in summary['arms'].items()},indent=2))


if __name__=='__main__':main()
