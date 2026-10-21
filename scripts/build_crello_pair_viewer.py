#!/usr/bin/env python3
"""Build a self-contained browser viewer for mined Crello aspect-ratio pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def _components(pair_rows, node_index, accepted_only):
    parent = list(range(len(node_index)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in pair_rows:
        if accepted_only and pair[2] == 2:  # rejected
            continue
        union(pair[0], pair[1])

    members = defaultdict(list)
    for idx in range(len(node_index)):
        root = find(idx)
        members[root].append(idx)
    edges_by_root = defaultdict(list)
    for i, pair in enumerate(pair_rows):
        if accepted_only and pair[2] == 2:
            continue
        edges_by_root[find(pair[0])].append(i)
    out = [{"nodes": nodes, "edges": edges_by_root[root]}
           for root, nodes in members.items()
           if len(nodes) > 1 and edges_by_root[root]]
    out.sort(key=lambda group: (-len(group["nodes"]), min(group["nodes"])))
    return out


def build(data_root: Path, pairs_path: Path, output: Path) -> None:
    source = json.loads(pairs_path.read_text(encoding="utf-8"))
    manifest_index = {}
    for split in ("train", "validation", "test"):
        manifest = data_root / f"crello_{split}" / "manifest.jsonl"
        if not manifest.is_file():
            continue
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "exported" and row.get("preview"):
                    manifest_index[(split, row["template_id"])] = row["preview"]

    raw_pairs = source["pairs"]
    key_to_node = {}
    nodes = []

    def node_for(row):
        key = (row["split"], row["id"])
        if key not in key_to_node:
            key_to_node[key] = len(nodes)
            preview_rel = manifest_index.get(key)
            image = f"crello_{row['split']}/{preview_rel}" if preview_rel else None
            preview_exists = bool(image and (data_root / image).is_file())
            nodes.append({
                "id": row["id"], "split": row["split"], "row": row["row_index"],
                "title": row.get("title", ""), "width": row["width"],
                "height": row["height"], "preview": image if preview_exists else None,
            })
        return key_to_node[key]

    class_code = {"exact": 0, "core": 1, "rejected": 2}
    pairs = []
    for item in raw_pairs:
        ai, bi = node_for(item["a"]), node_for(item["b"])
        pairs.append([
            ai, bi, class_code[item["classification"]],
            item.get("match_ratio", 0), item.get("matched_count", 0),
            max(item.get("element_count_a", 0), item.get("element_count_b", 0)),
            item.get("text_matches", 0), item.get("visual_matches", 0),
            bool(item.get("order_preserved")), bool(item.get("primary_visuals_matched")),
        ])

    payload = {
        "meta": {
            "dataset": source.get("dataset"), "revision": source.get("revision"),
            "rows_scanned": source.get("rows_scanned"),
            "candidate_pairs": len(pairs), "counts": source.get("counts", {}),
            "preview_missing": sum(node["preview"] is None for node in nodes),
        },
        "nodes": nodes,
        "pairs": pairs,
        "verified_groups": _components(pairs, key_to_node, accepted_only=True),
        "all_groups": _components(pairs, key_to_node, accepted_only=False),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Protect the embedded JSON from closing its script tag or breaking JS parsing.
    encoded = (encoded.replace("<", "\\u003c").replace(">", "\\u003e")
              .replace("&", "\\u0026").replace("\u2028", "\\u2028")
              .replace("\u2029", "\\u2029"))

    html = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crello aspect-ratio pair viewer</title>
<style>
:root{color-scheme:dark;--bg:#0c1018;--panel:#151b27;--line:#293345;--muted:#a4b0c2;--text:#edf2fa;--accent:#92c6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:#101722f2;border-bottom:1px solid var(--line);padding:14px 20px;backdrop-filter:blur(12px)}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.title{font-weight:750;font-size:18px;margin-right:auto}.muted{color:var(--muted)}
button,select,input{font:inherit;color:var(--text);background:#1b2432;border:1px solid #39465b;border-radius:8px;padding:8px 11px}
button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{opacity:.4;cursor:default}
.nav{display:flex;align-items:center;gap:8px}.summary{margin-top:8px;color:var(--muted);font-size:13px}
.filters{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:10px}.filters label{display:inline-flex;gap:6px;align-items:center;cursor:pointer;color:#d4deeb}.filters input{accent-color:#85bfff}
main{padding:18px 20px 38px}.section-title{display:flex;align-items:baseline;gap:10px;margin:4px 0 12px}.section-title h2{font-size:16px;margin:0}
.design-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr));gap:14px;align-items:start}
.design-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;min-width:0;transition:border-color .15s,box-shadow .15s}
.design-card.active{border-color:var(--accent);box-shadow:0 0 0 2px #92c6ff33}.preview{height:280px;padding:10px;background:#0a0e15;display:flex;align-items:center;justify-content:center;overflow:hidden}
.preview img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px;background:#fff}.missing{color:var(--muted);text-align:center;padding:18px}
.card-info{padding:12px}.card-title{font-weight:650;margin-bottom:7px;overflow-wrap:anywhere}.dims{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted);font-variant-numeric:tabular-nums}.pill{border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:12px}
.pairs{margin-top:24px}.pair-list{display:grid;gap:8px}.pair{display:grid;grid-template-columns:minmax(130px,1fr) auto minmax(130px,1fr) minmax(110px,auto);gap:10px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px 12px;cursor:pointer}
.pair:hover,.pair.active{border-color:var(--accent)}.pair-side{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pair-arrow{color:var(--muted)}.kind{font-size:12px;font-weight:700;border-radius:999px;padding:3px 8px;width:max-content}.exact{color:#8ce0b0;background:#163a2b}.core{color:#ffd178;background:#46351a}.rejected{color:#ff9f9f;background:#482323}.pair-meta{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.empty{padding:25px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:10px}.hint{font-size:12px;color:var(--muted);margin-top:14px}
@media(max-width:700px){header{padding:12px}main{padding:14px}.pair{grid-template-columns:1fr auto 1fr}.pair-meta{grid-column:1/-1;text-align:left}.preview{height:240px}}
</style>
</head>
<body>
<header>
 <div class="top">
  <div class="title">Crello aspect-ratio comparisons</div>
  <label class="muted">Group scope <select id="scope"><option value="verified">Exact + core pairs</option><option value="all">All candidates, including rejected</option></select></label>
  <div class="nav"><button id="prev" title="Previous group (←)">←</button><select id="groups" aria-label="Comparison group"></select><button id="next" title="Next group (→)">→</button></div>
 </div>
 <div class="summary" id="summary"></div>
 <div class="filters">
  <span class="muted">Show pair types:</span>
  <label><input type="checkbox" data-type="exact" checked> Exact</label>
  <label><input type="checkbox" data-type="core" checked> Core matched</label>
  <label><input type="checkbox" data-type="rejected"> Rejected candidates</label>
  <input id="search" type="search" placeholder="Filter groups by title or ID" aria-label="Filter comparison groups">
 </div>
</header>
<main>
 <section>
  <div class="section-title"><h2>All designs in this connected group</h2><span id="design-count" class="muted"></span></div>
  <div class="design-grid" id="designs"></div>
 </section>
 <section class="pairs">
  <div class="section-title"><h2>Pair comparisons in this group</h2><span id="pair-count" class="muted"></span></div>
  <div class="pair-list" id="pairs"></div>
  <div class="hint">Previews are the PNGs saved from the Crello dataset. Use the arrows or keyboard ← / → to move between groups. Click a pair row to highlight its two designs.</div>
 </section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('payload').textContent);
const labels=['Exact 1:1','Core matched','Rejected candidate'];
const cls=['exact','core','rejected'];
let scope='verified',groups=[],filtered=[],position=0,selectedPair=null;
const $=id=>document.getElementById(id);
const nodeLabel=n=>`${n.width} × ${n.height} · ${n.split} · #${n.row+1}`;
function activeTypes(){return new Set([...document.querySelectorAll('[data-type]:checked')].map(x=>x.dataset.type))}
function groupLabel(g,index){
 const ns=g.nodes.map(i=>D.nodes[i]); const title=ns.find(n=>n.title)?.title||ns[0].id;
 const counts=[0,0,0];g.edges.forEach(i=>counts[D.pairs[i][2]]++);
 return `${String(index+1).padStart(4,'0')} · ${title.slice(0,54)} · ${ns.length} designs · ${counts[0]+counts[1]+counts[2]} pairs`;
}
function refreshGroups(keep=true){
 const old=groups[filtered[position]??0];
 scope=$('scope').value;groups=scope==='verified'?D.verified_groups:D.all_groups;
 const q=$('search').value.trim().toLowerCase();
 filtered=groups.map((g,i)=>i).filter(i=>{
  if(!q)return true;const g=groups[i];return g.nodes.some(n=>{const x=D.nodes[n];return `${x.title} ${x.id}`.toLowerCase().includes(q)})
 });
 const sel=$('groups');sel.replaceChildren();
 filtered.forEach(i=>{const opt=document.createElement('option');opt.value=String(i);opt.textContent=groupLabel(groups[i],i);sel.append(opt)});
 let next=0;if(keep&&old){const oldSet=new Set(old.nodes);const found=filtered.findIndex(i=>groups[i].nodes.some(n=>oldSet.has(n)));if(found>=0)next=found}
 position=Math.min(next,Math.max(0,filtered.length-1));sel.selectedIndex=position;render();
}
function escapeText(s){return String(s??'')}
function render(){
 const g=groups[filtered[position]];const designRoot=$('designs'),pairRoot=$('pairs');designRoot.replaceChildren();pairRoot.replaceChildren();selectedPair=null;
 if(!g){$('summary').textContent='No groups match this filter.';$('design-count').textContent='';$('pair-count').textContent='';$('prev').disabled=$('next').disabled=true;return}
 $('prev').disabled=$('next').disabled=filtered.length<2;
 $('summary').textContent=`${groupLabel(g,filtered[position])} · group ${position+1} of ${filtered.length} · ${D.meta.rows_scanned.toLocaleString()} dataset rows scanned · ${D.meta.preview_missing} previews unavailable`;
 const types=activeTypes(),shown=g.edges.map(i=>[i,D.pairs[i]]).filter(([,p])=>types.has(cls[p[2]]));
 $('design-count').textContent=`${g.nodes.length} designs`;$('pair-count').textContent=`${shown.length.toLocaleString()} shown / ${g.edges.length.toLocaleString()} total`;
 for(const i of g.nodes){const n=D.nodes[i],card=document.createElement('article');card.className='design-card';card.dataset.node=String(i);
  const preview=document.createElement('div');preview.className='preview';
  if(n.preview){const img=document.createElement('img');img.loading='lazy';img.alt=`Dataset preview ${n.id}`;img.src=n.preview;img.onerror=()=>{preview.innerHTML='<div class="missing">Preview file unavailable<br>'+escapeText(n.id)+'</div>'};preview.append(img)}else preview.innerHTML='<div class="missing">No preview path recorded<br>'+escapeText(n.id)+'</div>';
  const info=document.createElement('div');info.className='card-info';const title=document.createElement('div');title.className='card-title';title.textContent=n.title||n.id;
  const dims=document.createElement('div');dims.className='dims';dims.innerHTML=`<span class="pill">${escapeText(nodeLabel(n))}</span><span class="pill">${escapeText(n.id)}</span>`;info.append(title,dims);card.append(preview,info);designRoot.append(card);
 }
 if(!shown.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No pair rows of the selected types in this group.';pairRoot.append(empty)}
 for(const [idx,p] of shown){const [a,b,type,ratio,matched,total,texts,visuals,order,primary]=p,na=D.nodes[a],nb=D.nodes[b];
  const row=document.createElement('div');row.className='pair';row.dataset.pair=String(idx);row.dataset.a=String(a);row.dataset.b=String(b);
  const left=document.createElement('div');left.className='pair-side';left.textContent=`${na.width}×${na.height} · ${na.id}`;left.title=na.title;
  const arrow=document.createElement('div');arrow.className='pair-arrow';arrow.textContent='→';
  const right=document.createElement('div');right.className='pair-side';right.textContent=`${nb.width}×${nb.height} · ${nb.id}`;right.title=nb.title;
  const meta=document.createElement('div');meta.className='pair-meta';meta.innerHTML=`<span class="kind ${cls[type]}">${labels[type]}</span><br>${matched}/${total} elements · ${Math.round(ratio*100)}%<br>${texts} text · ${visuals} visual`;
  row.append(left,arrow,right,meta);row.addEventListener('click',()=>highlight(row,a,b));pairRoot.append(row);
 }
}
function highlight(row,a,b){document.querySelectorAll('.active').forEach(e=>e.classList.remove('active'));row.classList.add('active');for(const i of [a,b])document.querySelector(`[data-node="${i}"]`)?.classList.add('active');document.querySelector(`[data-node="${a}"]`)?.scrollIntoView({behavior:'smooth',block:'nearest'})}
function move(delta){if(!filtered.length)return;position=(position+delta+filtered.length)%filtered.length;$('groups').selectedIndex=position;render();window.scrollTo({top:0,behavior:'smooth'})}
$('scope').addEventListener('change',()=>refreshGroups(false));$('groups').addEventListener('change',e=>{position=e.target.selectedIndex;render()});
$('prev').addEventListener('click',()=>move(-1));$('next').addEventListener('click',()=>move(1));
$('search').addEventListener('input',()=>refreshGroups(false));document.querySelectorAll('[data-type]').forEach(x=>x.addEventListener('change',render));
window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&e.target.tagName!=='INPUT'&&e.target.tagName!=='SELECT')move(-1);if(e.key==='ArrowRight'&&e.target.tagName!=='INPUT'&&e.target.tagName!=='SELECT')move(1)});
if(D.meta.preview_missing)console.warn(`${D.meta.preview_missing} pair designs have no local preview image`);
refreshGroups(false);
</script>
</body>
</html>'''.replace("__PAYLOAD__", encoded)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Built {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"Embedded {len(nodes):,} designs and {len(pairs):,} pair rows; "
          f"{len(payload['verified_groups']):,} exact/core groups, {len(payload['all_groups']):,} all-candidate groups; "
          f"{payload['meta']['preview_missing']} previews missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data" / "crello_verified_pairs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "crello_pair_viewer.html")
    args = parser.parse_args()
    build(args.data_root, args.pairs, args.output)


if __name__ == "__main__":
    main()
