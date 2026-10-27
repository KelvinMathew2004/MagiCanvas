#!/usr/bin/env python3
"""Refresh navigation and filters over existing verified renders; never run inference."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'data/gemini_review_qwen_projection_fixes_2026-10-25'
STYLE = '''body{font:16px/1.5 system-ui;background:#101820;color:#edf2f7;margin:0}a{color:#9ed4ff}header,section,main{max-width:1450px;margin:auto;padding:24px}header{border-bottom:1px solid #334552}h1{margin:8px 0;font-size:30px}h2{margin:0 0 10px}p{margin:10px 0}.muted{color:#b9c8d3}.nav{display:flex;gap:16px;flex-wrap:wrap}.notice{border-left:3px solid #e1b46f;padding:8px 15px;background:#182532}input,select{font:inherit;color:inherit;background:#182532;border:1px solid #526b7d;border-radius:7px;padding:9px}input{width:min(480px,85vw)}.tools{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article,.tile{background:#182532;border:1px solid #334552;border-radius:12px;padding:20px}article img{width:100%;height:430px;object-fit:contain;background:#101820}article h2{font-size:21px}article p{overflow-wrap:anywhere}.tag{font-size:13px;color:#c4d6e3}.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article[hidden]{display:none}@media(max-width:800px){.grid,.tiles{grid-template-columns:1fr}article img{height:auto;max-height:600px}}'''


def refresh(packet: Path = PACK) -> None:
    review=packet/'review'
    gallery=review/'main_engine_qwen_outputs'
    rows=json.loads((gallery/'manifest.json').read_text())
    checkpoint_file=packet/'evidence/current_checkpoint.json'
    checkpoint=json.loads(checkpoint_file.read_text())['checkpoint'] if checkpoint_file.exists() else 'macro_structure_final'
    audit=ROOT/'data/chameleon_port_audit_2026-10-26'
    has_audit=(audit/'index.html').is_file()
    if has_audit:
        dest=review/'chameleon_audit';dest.mkdir(exist_ok=True)
        for path in audit.iterdir():
            if path.is_file():shutil.copy2(path,dest/path.name)
        shutil.copy2(ROOT/'docs/CHAMELEON_PORT_REVIEW.md',packet/'docs/CHAMELEON_PORT_REVIEW.md')
        for name in ('audit_chameleon_import.py','refresh_review_viewer.py'):
            dest=packet/'code/scripts'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'scripts'/name,dest)
    changes=json.loads((review/'structure_changes/changes.json').read_text())
    links=[('main_engine_qwen_outputs/index.html',f'All {len(rows)} current output cards','Search designs and filter official, new authored and dataset cases.'),
           ('structure_changes/index.html',f'{changes["material_changes"]} substantive before/after changes','Every changed result is retained, with roundoff-only differences identified.'),
           ('novel_designs/index.html','8 new-design resize tasks','Four different compositions: source, raw model, Qwen ON and Qwen OFF.'),
           ('novel_validation/index.html','12 additional validation designs','Source, human reference, recorded model proposal and selected engine output.'),
           ('dataset_diversity/index.html','24 diverse held-out designs','Additional dataset sources selected across families.'),
           ('inventory_worst10/review.html','89-case audit: worst 10','Compare prior and current output with the human dataset preview.'),
           ('visual_contract_fixes/index.html','Previously reported defects','Historical reported renders beside their current versions.'),
           ('synthetic_level3_reverse/index.html','Legacy reverse-fixture archive','Earlier same-elements cases, retained for regression review.')]
    if has_audit:links.insert(1,('chameleon_audit/index.html','Chameleon: all 20 branches reviewed','Verified components and prioritized port plan; no Chameleon ports are active in these renders.'))
    if (review/'official_fixes/index.html').exists():
        links.insert(0,('official_fixes/index.html','Official resize fixes · before / after','Artwork proportions and text/image clearance, compared with the frozen prior engine.'))
    tiles=''.join(f'<a class="tile" href="{url}"><h2>{html.escape(title)}</h2><p class="muted">{html.escape(detail)}</p></a>' for url,title,detail in links)
    (review/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MagiCanvas · latest verified review</title><style>'+STYLE+'</style><header><p class="tag">Updated October 26, 2026</p><h1>Latest verified engine outputs</h1><p class="muted">Current render checkpoint: '+html.escape(checkpoint)+'. Exact earlier model predictions replayed through the verified engine.</p><p class="notice">The Chameleon review is an implementation plan. These images show the current MagiCanvas engine. Passing automated checks does not establish aesthetic perfection.</p><nav class="nav"><a href="../docs/RIGID_ART_REVIEW.md">Engine changes and limitations</a><a href="../docs/CHAMELEON_PORT_REVIEW.md">Chameleon porting analysis</a></nav></header><main class="tiles">'+tiles+'</main></html>')
    labels={'novel_designs':'New authored designs','novel_validation':'Validation designs','dataset_diversity':'Additional held-out','official':'Official levels 1–3','crello':'Crello / 0.6B','crello_1p7b':'Crello / 1.7B'}
    def group(row):
        if row.get('group') in labels:return row['group']
        if row['image'].startswith('official_'):return 'official'
        return 'crello_1p7b' if '_1p7b.' in row['image'] else 'crello'
    order={g:i for i,g in enumerate(labels)}
    cards=[]
    for row in sorted(rows,key=lambda row:order[group(row)]):
        g=group(row);title=row.get('case',row.get('title',row['pair_id']))
        detail={'novel_designs':'../novel_designs/index.html','novel_validation':'../novel_validation/index.html','dataset_diversity':'../dataset_diversity/index.html'}.get(g)
        search=html.escape(' '.join(str(row.get(k,'')) for k in ('case','title','pair_id','winner')).casefold(),quote=True)
        links=f'<a href="{html.escape(row["json"],quote=True)}">Selected JSON</a>'
        if detail:links+=f' · <a href="{detail}">Source and model comparison</a>'
        flags=row.get('defect_flags',[])
        cards.append(f'<article data-group="{g}" data-search="{search}" data-flagged="{str(bool(flags)).lower()}"><p class="tag">{labels[g]}</p><h2>{html.escape(title)}</h2><a href="{html.escape(row["image"],quote=True)}"><img loading="lazy" src="{html.escape(row["image"],quote=True)}" alt="{html.escape(title,quote=True)} selected engine output"></a><p>{html.escape(row["winner"])}</p><p class="muted">New flags: {html.escape(str(flags))}</p><p>{links}</p></article>')
    counts={g:sum(group(r)==g for r in rows) for g in labels}
    options=''.join(f'<option value="{g}">{label} ({counts[g]})</option>' for g,label in labels.items())
    script='''const cards=[...document.querySelectorAll('article[data-group]')];const q=document.getElementById('search'),f=document.getElementById('group'),flag=document.getElementById('flagged');function update(){let n=0;for(const c of cards){const show=(!f.value||c.dataset.group===f.value)&&c.dataset.search.includes(q.value.trim().toLowerCase())&&(!flag.checked||c.dataset.flagged==='true');c.hidden=!show;if(show)n++;}document.getElementById('count').textContent=`${n} of ${cards.length} designs`;document.getElementById('empty').hidden=n!==0;}const initial=new URLSearchParams(location.search);if([...f.options].some(o=>o.value===initial.get('group')))f.value=initial.get('group');q.value=initial.get('q')||'';flag.checked=initial.get('flagged')==='1';for(const el of [q,f,flag])el.addEventListener('input',update);update();'''
    (gallery/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MagiCanvas · current output gallery</title><style>'+STYLE+'</style><header><nav class="nav"><a href="../index.html">Review overview</a><a href="../structure_changes/index.html">Before / after</a><a href="../chameleon_audit/index.html">Chameleon audit</a></nav><h1>Current engine output gallery</h1><p class="muted">Verified '+html.escape(checkpoint)+' renders · recorded model predictions · new designs first.</p><p class="notice">Chameleon mechanisms have been audited; these renders use the existing MagiCanvas engine. Open a comparison to distinguish raw model results from deterministic repairs.</p><div class="tools"><label>Search <input id="search" type="search" placeholder="Design, ID or winner"></label><label>Cases <select id="group"><option value="">All groups</option>'+options+'</select></label><label><input id="flagged" type="checkbox" style="width:auto"> Flagged only</label><span id="count" aria-live="polite"></span></div></header><main><p id="empty" hidden>No matching designs.</p><div class="grid">'+''.join(cards)+'</div></main><script>'+script+'</script></html>')
    hashes=[]
    for path in sorted(packet.rglob('*')):
        if path.is_file() and path.name not in ('MANIFEST.sha256','.DS_Store'):
            hashes.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(packet).as_posix()}')
    (packet/'MANIFEST.sha256').write_text('\n'.join(hashes)+'\n')
    print(json.dumps(dict(cards=len(rows),groups=counts,chameleon_audit=has_audit)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,default=PACK)
    refresh(p.parse_args().packet)
