#!/usr/bin/env python3
"""Filter the completed adapter comparison without adding public raw panels."""
from pathlib import Path
from hashlib import sha256
import json,html,argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def cohort(key,row):
    if key.startswith('reference:'):return 'reference'
    for name in ('authored','fresh','supplied','portrait'):
        if key.startswith('review:'+name+':'):return name
    if row.get('official_path'):return 'official'
    return 'dataset'


def metric_text(before,after):
    if before.get('geometry_mae') is None or after.get('geometry_mae') is None:return ''
    def number(value):return 'unavailable' if value is None else f'{value:.5f}'
    return (f"Final reference box MAE: {number(before['geometry_mae'])} → {number(after['geometry_mae'])}; "
            f"font MAE: {number(before.get('font_mae'))} → {number(after.get('font_mae'))}.")


def main():
    from scripts.report_recovered_adapter import load
    from scripts.refresh_regression_viewer import PACK,documents_differ
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--viewer',type=Path,default=PACK/'review/trained_adapters');a=p.parse_args()
    analysis=json.loads((a.run/'paired_analysis.json').read_text())
    for arm,digest in analysis['arms_sha256'].items():
        if sha256((a.run/(arm+'.jsonl')).read_bytes()).hexdigest()!=digest:raise ValueError('Stale paired analysis')
    before=load(a.run/'baseline.jsonl');after=load(a.run/'audited-0.6b-recovered.jsonl')
    manifest=json.loads((a.viewer/'manifest.json').read_text())
    if len(before)!=148 or before.keys()!=after.keys() or {r['pair_id'] for r in manifest}!=set(before):
        raise ValueError('Expected a complete matching 148-task viewer')
    cards=[];groups=set();candidate='audited-0.6b-recovered'
    for r in manifest:
        key=r['pair_id'];b,n=before[key],after[key];i=r['index'];group=cohort(key,b);groups.add(group)
        changed=documents_differ(b['output_design'],n['output_design']);newflags=sorted(set(n['defect_flags'])-set(b['defect_flags']))
        paths=r['outputs'];size=b['target_canvas'];src=b['source_design']['canvas']
        raw=f"Exact raw IDs: {b['raw_proposal_valid']} → {n['raw_proposal_valid']}; learned winner: {b['qwen_family_win']} → {n['qwen_family_win']}"
        metrics=metric_text(b,n)
        data=dict(group=group,changed=str(changed).lower(),flagged=str(bool(b['defect_flags'] or n['defect_flags'])).lower(),
            newflags=str(bool(newflags)).lower(),dense=str(len(b['source_design']['elements'])>=18).lower())
        attrs=' '.join(f'data-{k}="{v}"' for k,v in data.items())
        cards.append(f'''<article {attrs}><h2>{html.escape(key)}</h2>
<p>{src['width']}×{src['height']} → {size['width']}×{size['height']} · {len(b['source_design']['elements'])} source elements · changed: {changed}</p>
<label>Resized model <select data-arm onchange="change(this)"><option value="{candidate}">Recovered adapter</option><option value="baseline">Current champion</option></select></label>
<div class="pair"><figure><figcaption>Original</figcaption><a href="{r['source']}"><img loading="lazy" src="{r['source']}"></a></figure>
<figure><figcaption>Resized</figcaption><a data-resized-link href="{paths[candidate]}"><img loading="lazy" data-resized data-baseline="{paths['baseline']}" data-candidate="{paths[candidate]}" src="{paths[candidate]}"></a></figure></div>
<details><summary>Audit details</summary><p>{raw}</p><p>{html.escape(metrics)}</p><p>Champion: {html.escape(b['winner'])}; recovered: {html.escape(n['winner'])}.</p>
<p>Flags: {html.escape(str(b['defect_flags']))} → {html.escape(str(n['defect_flags']))}; new types: {html.escape(str(newflags))}.</p>
<p><a href="{r['audit']}">Raw/intermediate/final geometry and exact metrics</a></p></details></article>''')
    options=''.join(f'<option value="{g}">{html.escape(g)}</option>' for g in sorted(groups))
    ref=analysis['reference'];review=analysis['review']
    summary=f"89 reference pairs: box MAE {ref['geometry_mae']['before']:.5f} → {ref['geometry_mae']['after']:.5f}; font MAE {ref['font_mae']['before']:.5f} → {ref['font_mae']['after']:.5f}. 59 review tasks: flagged {review['before_flagged']} → {review['after_flagged']}."
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Completed recovered-adapter comparison</title>
<style>body{background:#101820;color:#edf2f7;font:17px/1.5 system-ui;margin:0}header,main{max-width:1500px;margin:auto;padding:24px}header{border-bottom:1px solid #435160}a{color:#9ed4ff}article{padding:20px;border:1px solid #435160;border-radius:12px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}figure{margin:0}img{width:100%;height:620px;object-fit:contain;background:#0b1118}select{font-size:17px;margin:8px;padding:8px}p{overflow-wrap:anywhere}details{margin-top:15px}[hidden]{display:none}@media(max-width:750px){.pair{grid-template-columns:1fr}img{height:480px}}</style>
<header><h1>Completed adapter A/B — Qwen enabled</h1><p>Both matching 0.6B packages use the same prompt, selector and recorded engine snapshot. Saved predictions from interrupted fresh collection were re-realized, never copied as final outputs. These are consumed development/reference cohorts. Zero flags does not prove design quality. Production has not switched.</p>
<p>__SUMMARY__</p><p><a href="../main_engine_qwen_outputs/index.html">Production regression viewer</a> · <a href="paired_analysis.json">Detailed paired report</a></p>
<label>Resized model for all <select id="all-arm"><option value="audited-0.6b-recovered">Recovered adapter</option><option value="baseline">Current champion</option></select></label>
<label>Cohort <select id="group"><option value="">All</option>__OPTIONS__</select></label>
<label>Filter <select id="condition"><option value="">All</option><option value="changed">Changed layouts</option><option value="flagged">Either arm flagged</option><option value="newflags">New flags</option><option value="dense">18+ source elements</option></select></label><p id="count"></p></header><main>__CARDS__</main>
<script>function change(s){const c=s.closest('article'),im=c.querySelector('[data-resized]');im.src=s.value==='baseline'?im.dataset.baseline:im.dataset.candidate;c.querySelector('[data-resized-link]').href=im.src}document.getElementById('all-arm').onchange=e=>document.querySelectorAll('[data-arm]').forEach(s=>{s.value=e.target.value;change(s)});function filter(){let n=0;const g=document.getElementById('group').value,f=document.getElementById('condition').value;document.querySelectorAll('article[data-group]').forEach(c=>{c.hidden=(g&&c.dataset.group!==g)||(f&&c.dataset[f]!=='true');if(!c.hidden)n++});document.getElementById('count').textContent=n+' of 148 tasks'}document.getElementById('group').onchange=filter;document.getElementById('condition').onchange=filter;filter()</script>'''
    (a.viewer/'paired_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    (a.viewer/'index.html').write_text(page.replace('__SUMMARY__',html.escape(summary)).replace('__OPTIONS__',options).replace('__CARDS__',''.join(cards)))
    print(json.dumps(dict(viewer=str(a.viewer/'index.html'),tasks=148)))


if __name__=='__main__':main()
