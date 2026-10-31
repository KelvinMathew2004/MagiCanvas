#!/usr/bin/env python3
"""Publish source/resized-only evidence for surgical follow-up experiments."""
from pathlib import Path
from copy import deepcopy
from statistics import mean
from hashlib import sha256
import json,html,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'data/upgrade_exploration/training_free_scale_v1'
OUT=ROOT/'data/upgrade_exploration/surgical_followup_v1'
VIEW=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/surgical_followup'


def main():
    from engine.resize import _text_info
    from engine.qwen_prompt import decode_v2_completion,record_prompt_version
    from scripts.report_training_free_upgrades import gold_metrics
    from scripts.build_focus_reflow_viewer import _renderer,_preview_nonempty,STYLE
    from PIL import Image,ImageDraw
    cases=json.loads((BASE/'cases.json').read_text());bykey={c['key']:i for i,c in enumerate(cases)}
    diagnostics={i:json.loads((OUT/'diagnostics'/f'{i:03d}.json').read_text()) for i in range(len(cases))}
    baselines={i:json.loads((BASE/'baseline'/f'{i:03d}.json').read_text())['baseline'] for i in range(len(cases))}
    topo=json.loads((OUT/'topology.json').read_text());survivors={}
    for r in topo['records']:
        if r['owned_trigger_pass']:survivors.setdefault(r['index'],[]).append(r)
    references=[];line_counts=dict(text=0,source_lines_equal_gold=0,selected_lines_equal_gold=0,source_lines_less_than_gold=0)
    for alias in json.loads((BASE/'test_pair_tasks.json').read_text()):
        i=bykey[alias['task_key']];b=baselines[i];s=b['source_design'];old=b['output_design'];d=diagnostics[i]
        hierarchy=d['hierarchy_band']['output_design'] or old
        slack=d['raw_font_slack']['output_design'] or old
        algorithms=json.loads((BASE/'algorithms'/f'{i:03d}.json').read_text())['arms']
        filtered=old
        options=survivors.get(i,[])
        if options:
            # Keep the earlier proxy ordering, not a target-MAE oracle.
            chosen=min(options,key=lambda r:algorithms[r['arm']]['quality']['cost']);filtered=algorithms[chosen['arm']]['output_design']
        references.append(dict(pair_id=alias['pair_id'],family_id=alias['record']['family_id'],before=gold_metrics(s,old,alias['record']),
            hierarchy=gold_metrics(s,hierarchy,alias['record']),raw_font_slack=gold_metrics(s,slack,alias['record']),topology_filtered=gold_metrics(s,filtered,alias['record'])))
        gold=json.loads(alias['record']['messages'][-1]['content'])
        if gold.get('elements') and 'box' in gold['elements'][0]:gold=decode_v2_completion(gold,s,alias['record']['target_canvas'],record_prompt_version(alias['record']))
        if not isinstance(gold,dict):gold={'elements':gold}
        gb={e['id']:e for e in gold['elements']};ob={e['id']:e for e in old['elements']}
        for e in s['elements']:
            if e['type']!='text' or e['id'] not in gb:continue
            reconstructed=deepcopy(e);reconstructed.update(gb[e['id']]);a,c,z=_text_info(e)[2],_text_info(ob[e['id']])[2],_text_info(reconstructed)[2]
            line_counts['text']+=1;line_counts['source_lines_equal_gold']+=a==z;line_counts['selected_lines_equal_gold']+=c==z;line_counts['source_lines_less_than_gold']+=a<z
    summary=dict(tasks=len(cases),reference_pairs=len(references),line_counts=line_counts,
        geometry={arm:mean(r[arm]['geometry_mae'] for r in references) for arm in ('before','hierarchy','raw_font_slack','topology_filtered')},
        font={arm:mean(r[arm]['font_mae'] for r in references if r[arm]['font_mae'] is not None) for arm in ('before','hierarchy','raw_font_slack','topology_filtered')},
        scope='Counterfactual raw proposed edits/filter only; no automatic selection or promotion, no target-guided choice. Hierarchy includes unsafe edits to expose tradeoffs.',reference_records=references)
    (OUT/'reference_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    VIEW.mkdir(exist_ok=True);items=[]
    for i,d in diagnostics.items():
        if d['hierarchy_band']['changed']:items.append((i,'hierarchy-band',d['hierarchy_band']['output_design'],d['hierarchy_band']['flags'],d['hierarchy_band']))
        if d['raw_font_slack']['changed']:items.append((i,'raw-font-slack',d['raw_font_slack']['output_design'],d['raw_font_slack']['flags'],d['raw_font_slack']))
    for i,rs in survivors.items():
        for r in rs:
            doc=json.loads((BASE/'algorithms'/f'{i:03d}.json').read_text())['arms'][r['arm']]['output_design'];items.append((i,'topology-survivor-'+r['arm'],doc,[],r))
    for r in json.loads((OUT/'micro_strip.json').read_text())['records']:items.append((r['index'],r['arm'],r['output_design'],r['flags'],r))
    cards=[];manifest=[];contact=[];cache={}
    for n,(i,arm,doc,flags,details) in enumerate(items):
        b=baselines[i];paths={};render=_renderer(b['source_design'],None if b.get('official_path') else b['source_split']);stem=f'{n:02d}_{i:03d}'
        for label,value in [('source',b['source_design']),('before',b['output_design']),('after',doc)]:
            key=sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            if key not in cache:
                png=VIEW/f'{stem}_{label}.png';_preview_nonempty(render,value,png);jpg=png.with_suffix('.jpg')
                with Image.open(png) as im:im.convert('RGB').save(jpg,quality=93,optimize=True)
                png.unlink();cache[key]=jpg.name
            paths[label]=cache[key]
        audit=dict(index=i,pair_id=cases[i]['pair_id'],source=b['source_design'],before=b['output_design'],proposal=doc,details=details,production_changed=False)
        (VIEW/f'{stem}_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        cards.append(f'<article><h2>{html.escape(arm)} · {html.escape(cases[i]["pair_id"])}</h2><p>Research only; NOT promoted. Flags {html.escape(str(flags))}. A topology pass does not prove a coherent design.</p><label>Resized image <select onchange="this.closest(\'article\').querySelector(\'[data-result]\').src=this.value"><option value="{paths["after"]}">Experimental proposal</option><option value="{paths["before"]}">Current production</option></select></label><div class="panels"><figure><figcaption>Original</figcaption><img loading="lazy" src="{paths["source"]}"></figure><figure><figcaption>Resized</figcaption><img data-result loading="lazy" src="{paths["after"]}"></figure></div><a href="{stem}_audit.json">Audit geometry</a></article>')
        manifest.append(dict(case=cases[i]['pair_id'],arm=arm,source=paths['source'],image=paths['after'],before=paths['before'],audit=f'{stem}_audit.json'));contact.append((arm,paths,flags))
    (VIEW/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Surgical algorithm research</title><style>'+STYLE+'</style><header><a href="../main_engine_qwen_outputs/index.html">Current production</a><h1>Six surgical follow-ups</h1><p>Frozen 134-task development outputs; no training, fresh inference or production switch. Counterfactual proposals expose failure modes, including unsafe ones. Source and resized only; select current production for comparison. The full evidence ledger is docs/SURGICAL_ALGORITHM_DECISIONS.md in the review packet.</p></header><main>'+''.join(cards)+'</main>')
    (VIEW/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for page,start in enumerate(range(0,len(contact),3)):
        batch=contact[start:start+3];sheet=Image.new('RGB',(1400,len(batch)*400),'#182532');draw=ImageDraw.Draw(sheet)
        for n,(arm,paths,flags) in enumerate(batch):
            draw.text((10,n*400+8),arm+' flags '+str(flags),fill='white')
            for col,label in enumerate(('before','after')):
                with Image.open(VIEW/paths[label]) as im:
                    im=im.convert('RGB');im.thumbnail((680,350));sheet.paste(im,(col*700+(700-im.width)//2,n*400+40+(350-im.height)//2))
        sheet.save(OUT/f'inspect_{page:02d}.jpg',quality=94)
    print(json.dumps({k:v for k,v in summary.items() if k!='reference_records'},indent=2));print(len(items),'cards rendered')


if __name__=='__main__':
    import os
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import torch;torch.set_num_threads(1)
    main()
