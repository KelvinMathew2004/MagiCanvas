#!/usr/bin/env python3
"""Replay four labelled attachment outliers through the complete Qwen-on engine.

Frozen raw model responses, not fresh inference. Review data names never enter
production logic. Existing preference images/labels are preserved byte-for-byte.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    os.environ.update(MAGICANVAS_QWEN_ENABLED='1',MAGICANVAS_CPU_THREADS='1')
    import torch
    torch.set_num_threads(1)
    from scripts.replay_qwen_engine import replay
    from scripts.dataset_renderer import render_design_to_png
    from scripts.build_focus_reflow_viewer import STYLE
    import html
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/qwen_lora/engine_upgrade/attachment_engine_replay')
    args=parser.parse_args()
    folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    page=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/attachment_growth';page.mkdir(exist_ok=True)
    fp={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
    marker=folder/'run_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise ValueError('Engine changed: choose a fresh replay folder')
    marker.write_text(json.dumps(fp,indent=2)+'\n');results=[];cards=[]
    for edition in ('00','01'):
        for aspect in ('leaderboard','widescreen'):
            task=f'pref-new-00-{edition}-{aspect}';saved=json.loads((ROOT/f'data/gemini_review_qwen_projection_fixes_2026-10-25/review/preferences_campaign/audits/{task}.json').read_text());source=saved['source'];meta=saved['task'];pid=meta['source_id']+'__to__'+aspect
            record=dict(pair_id=pid,source_split='train',family_id=meta['family_id'],target_canvas=meta['target_canvas'])
            cached=dict(pair_id=pid,source_design=source,target_canvas=meta['target_canvas'],qwen_model_elements=saved['diagnostics']['qwen_diagnostics']['model_elements'])
            path=folder/(task+'.json')
            if path.exists():row=json.loads(path.read_text())
            else:
                row=replay([record],[cached],ROOT/'models/qwen3-layout-lora-v2',ROOT/'models/qwen3-0.6b',None,source_for=lambda *a:deepcopy(source),allow_incomplete=True)[0]
                path.write_text(json.dumps(row,indent=2)+'\n')
            if row.get('engine_error'):raise ValueError(row['engine_error'])
            for suffix,doc in [('source',source),('output',row['output_design'])]:render_design_to_png(doc,ROOT/'data/crello_train',page/(task+'_'+suffix+'.png'),ROOT/'data/crello_fonts.pickle',max_size=1200,adjust_tight_text_widths=False)
            (page/(task+'_output.json')).write_text(json.dumps(row['output_design'],indent=2)+'\n')
            summary=dict(task=task,before_winner=saved['diagnostics']['winner'],winner=row['winner'],flags=row['defect_flags'],same_document=row['output_design']==saved['winner'],raw_ids_complete=row['raw_proposal_valid'])
            results.append(summary);cards.append(f'<article><h2>{html.escape(task)}</h2><div class="panels"><figure><figcaption>Original</figcaption><img src="{task}_source.png" alt="Original"></figure><figure><figcaption>Resized</figcaption><img src="{task}_output.png" alt="Resized"></figure></div><details><summary>Audit</summary><p>{html.escape(json.dumps(summary))}</p><a href="{task}_output.json">Selected JSON</a></details></article>')
            print(json.dumps(summary),flush=True)
    (folder/'summary.json').write_text(json.dumps(dict(scope='Four reviewed development outliers, cached raw Qwen; measured container packing and contact growth. No training or independent quality benchmark.',results=results),indent=2)+'\n')
    (page/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Attachment growth verification</title><style>'+STYLE+'</style><header><a href="../regression_suite/index.html">Unified regression viewer</a><h1>Measured attachment reflow</h1><p>Current complete engine, Qwen enabled with exact saved proposals. Four reviewed development outliers; no new inference or generalization claim. Preference comparisons remain unchanged.</p></header><main>'+''.join(cards)+'</main>')


if __name__=='__main__':main()
