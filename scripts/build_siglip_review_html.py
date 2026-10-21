#!/usr/bin/env python3
"""Build a local side-by-side review page from SigLIP candidate manifests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "data" / "siglip_negative_pairs")
    args = parser.parse_args()
    manifest = args.folder / "manifest.jsonl"
    entries = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if "engine_image" not in row:
            continue
        preview_path=row.get("preview_fidelity_evidence") or row.get("positive_image")
        positive_views=row.get("positive_images") or []
        if not preview_path and positive_views:
            preview_path=positive_views[0]
        target_local_path=row.get("target_local_render")
        if not target_local_path and positive_views:
            target_local_path=(positive_views[0] if row.get("preview_fidelity_evidence")
                               else positive_views[1] if len(positive_views)>1 else None)
        if not preview_path:
            continue
        entries.append({
            "pair_id": row["pair_id"],
            "positive": os.path.relpath(preview_path, args.folder),
            "target_local": (os.path.relpath(target_local_path,args.folder)
                             if target_local_path else None),
            "candidate": os.path.relpath(row["engine_image"], args.folder),
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "canvas": row["target_canvas"],
            "pixels": row["preview_pixel_size"],
            "status": row.get("status",row.get("preference","unreviewed")),
            "defects": row.get("layout_defects",[]),
            "engine_revision": row.get("engine_revision"),
        })
    payload = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crello resize candidate review</title>
<style>
:root{color-scheme:dark;background:#0d1118;color:#eef2f8;font:15px system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;padding:18px}
header{position:sticky;top:0;background:#0d1118ed;padding:10px 0 16px;z-index:2}.bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.bar h1{font-size:19px;margin:0 auto 0 0}button{font:inherit;color:inherit;background:#202a39;border:1px solid #526178;border-radius:8px;padding:8px 14px;cursor:pointer}button:disabled{opacity:.4}.meta{color:#b9c5d5}.pair{font:13px ui-monospace,monospace;color:#aab7ca;margin-top:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}.grid.three{grid-template-columns:1fr 1fr 1fr}.card{background:#171e29;border:1px solid #303a4a;border-radius:10px;padding:10px;min-width:0}.card h2{font-size:14px;margin:0 0 8px}.imgbox{display:grid;place-items:center;background:#080b10;min-height:240px;border-radius:6px;overflow:auto}.imgbox img{display:block;max-width:100%;max-height:calc(100vh - 180px);object-fit:contain}.hint{color:#9eabc0;font-size:13px;margin-top:12px}@media(max-width:750px){.grid,.grid.three{grid-template-columns:1fr}.imgbox img{max-height:70vh}}
</style>
<header><div class="bar"><h1>Crello resize candidates</h1><button id="prev">← Previous</button><button id="next">Next →</button><span class="meta" id="count"></span></div><div class="pair" id="meta"></div></header>
<main class="grid" id="grid"><section class="card"><h2>Original dataset preview</h2><div class="imgbox"><img id="positive"></div></section><section class="card" id="target-card" hidden><h2>Locally rendered target JSON</h2><div class="imgbox"><img id="target-local"></div></section><section class="card"><h2>Current engine resize candidate</h2><div class="imgbox"><img id="candidate"></div></section></main>
<div class="hint">Use ← / → keys to browse. Candidate and preview files have identical pixel dimensions. The first rerendered batch is marked with its engine revision. These candidates are unreviewed; the original preview is linked directly from the local dataset.</div>
<script>const rows=__DATA__;let i=0;const pos=document.querySelector('#positive'),cand=document.querySelector('#candidate'),local=document.querySelector('#target-local'),localCard=document.querySelector('#target-card'),grid=document.querySelector('#grid'),meta=document.querySelector('#meta'),count=document.querySelector('#count');function show(){const r=rows[i];pos.src=r.positive;cand.src=r.candidate;localCard.hidden=!r.target_local;grid.classList.toggle('three',!!r.target_local);if(r.target_local)local.src=r.target_local;meta.textContent=`${r.pair_id} · ${r.status}${r.engine_revision?' · '+r.engine_revision:''} · canvas ${r.canvas.width}×${r.canvas.height} · image ${r.pixels[0]}×${r.pixels[1]} · source ${r.source_id} → target ${r.target_id}${r.defects.length?' · defects: '+r.defects.join(', '):''}`;count.textContent=`${i+1} / ${rows.length}`;document.querySelector('#prev').disabled=i===0;document.querySelector('#next').disabled=i===rows.length-1}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});show();</script>
</html>'''.replace("__DATA__", payload)
    (args.folder / "review.html").write_text(page, encoding="utf-8")
    pair_file = ROOT / "data" / "crello_verified_pairs.json"
    total_eligible = None
    if pair_file.is_file():
        pair_doc = json.loads(pair_file.read_text())
        total_eligible = sum(
            p.get("classification") == "exact"
            and p.get("a", {}).get("split") == "train"
            and p.get("b", {}).get("split") == "train"
            and not p.get("cross_split")
            for p in pair_doc.get("pairs", [])
        )
    rerendered = sum(bool(entry.get("engine_revision")) for entry in entries)
    summary = {"status": "review_batch_updated" if rerendered else "partial",
               "reviewable_pairs": len(entries), "rerendered_candidates": rerendered,
               "eligible_exact_train_pairs": total_eligible,
               "folder": str(args.folder.resolve()),
               "review_page": str((args.folder / "review.html").resolve()),
               "note": "The candidates marked with an engine revision were regenerated using the current resize engine." if rerendered else "Review candidates are available; inspect before deciding whether to continue."}
    (args.folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
