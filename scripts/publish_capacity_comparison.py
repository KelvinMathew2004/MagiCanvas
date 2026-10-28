"""Add verified model alternatives to the unified two-panel development viewer."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'data/prompt_capacity_ab/selector/1.7b_report.json'


def augment(folder,run):
    if not REPORT.exists():return
    report=json.loads(REPORT.read_text())
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')}
    recorded=report['selector_manifest']['engine']
    if any(recorded.get(k)!=v for k,v in hashes.items()):
        # Never relabel old alternatives as current or block a new production
        # replay on an unrelated experimental model comparison.
        page=folder/'index.html'
        page.write_text(page.read_text().replace('</header>','<p class="muted">The optional 1.7B comparison is hidden because its selector snapshot is older than this engine. All displayed outputs use the current 0.6B replay.</p></header>'))
        return
    from scripts.build_focus_reflow_viewer import build
    records=run/'model17_records';records.mkdir(exist_ok=True)
    for i,row in enumerate(report['results']):
        audit=dict(name=row['pair_id'],source=row['source_design'],design=row['output_design'],target=row['target_canvas'],
                   source_split=None if row['source_split']=='official' else row['source_split'],frozen_model_prediction=True,
                   diagnostics=dict(winner=row['winner'],pair_id=row['pair_id'],source_model='Qwen3-1.7B',
                                    source_adapter='qwen3-layout-lora-v2-expanded-1.7b',raw_model_elements=row['qwen_model_elements'],
                                    original_generation=row['original_generation'],qwen_status=row.get('qwen_status'),qwen_enabled=True),
                   defects=dict(defect_flags=row['defect_flags']))
        (records/f'{i:02d}.json').write_text(json.dumps(audit,indent=2)+'\n')
    output=folder/'model_17';rows=build(records,output,'1.7B stress alternatives · Original and Resized')
    mapping={}
    for r,raw in zip(rows,report['results']):
        r['pair_id']=raw['pair_id']
        mapping[raw['pair_id']]=dict(image='model_17/'+r['image'],json='model_17/'+r['json'],audit='model_17/'+r['audit'],
            selection='Qwen enabled · 1.7B · selected '+r['winner']+' · automated flags '+str(r['defect_flags']),status=raw.get('qwen_status',''),render_notes='Render notes: '+str(r['render_warnings']))
    (output/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    data=json.dumps(mapping).replace('<','\\u003c')
    control='<p><label>Stress comparison model <select id="model-variant"><option value="06">0.6B (production default)</option><option value="17">1.7B alternative</option></select></label></p><p class="muted">This switch applies to the 13 supplied/skyscraper stress cases. The other 46 keep their recorded 0.6B predictions. Both arms use matching adapters and the same current selector. No live generation occurs here.</p>'
    script='''<script>(function(){const models=MODEL_DATA,select=document.getElementById('model-variant'),defaults=new Map();
    for(const card of document.querySelectorAll('article[data-pair-id]')){
      const id=card.dataset.pairId;if(!models[id])continue;
      const img=card.querySelector('[data-panel="Resized"] img');
      defaults.set(id,{image:img.getAttribute('src'),json:card.querySelector('[data-output]').getAttribute('href'),audit:card.querySelector('[data-audit]').getAttribute('href'),selection:card.querySelector('[data-selection]').textContent,status:card.querySelector('[data-qwen-status]').textContent,render_notes:card.querySelector('[data-render-notes]').textContent});
    }
    select.onchange=function(){for(const card of document.querySelectorAll('article[data-pair-id]')){
      const id=card.dataset.pairId;if(!defaults.has(id))continue;const v=select.value==='17'?models[id]:defaults.get(id);
      const img=card.querySelector('[data-panel="Resized"] img');img.src=v.image;img.parentElement.href=v.image;
      card.querySelector('[data-output]').href=v.json;card.querySelector('[data-audit]').href=v.audit;card.querySelector('[data-selection]').textContent=v.selection;card.querySelector('[data-qwen-status]').textContent=v.status;card.querySelector('[data-render-notes]').textContent=v.render_notes;
    }};
    })();</script>'''.replace('MODEL_DATA',data)
    page=folder/'index.html';s=page.read_text().replace('</header>',control+'</header>')
    page.write_text(s.replace('</html>',script+'</html>'))
