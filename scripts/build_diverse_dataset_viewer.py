#!/usr/bin/env python3
"""Pick diverse held-out designs without consulting outcome scores, then render.

The viewer shows only original and resized designs. Human references and decoded
proposals remain separate diagnostic artifacts. Portable contact sheets use the
same two panels. No new model inference is implied.
"""
from pathlib import Path
from collections import Counter
from hashlib import sha256
from copy import deepcopy
import argparse,json,html,sys,shutil
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def bucket(canvas):
    ratio=canvas['width']/canvas['height']
    return 'wide banner' if ratio>=2.4 else 'tall' if ratio<=.55 else 'portrait' if ratio<.85 else 'square' if ratio<=1.2 else 'landscape'


def proposal_design(source,canvas,proposal):
    """Attach raw coordinate predictions to source paint data without repairing them."""
    originals={e['id']:e for e in source['elements']}
    elements=[]
    for prediction in proposal:
        element=deepcopy(originals[prediction['id']])
        element.update(deepcopy(prediction))
        original=originals[prediction['id']]
        if element.get('type')=='text' and original.get('font_size',0)>0:
            element['letter_spacing']=float(original.get('letter_spacing',0))*element['font_size']/original['font_size']
        elements.append(element)
    result=deepcopy(source)
    result['canvas'].update(canvas)
    result['elements']=elements
    return result


def choose(rows,existing,count):
    sources={r['pair_id'].split('__to__')[0] for r in existing}
    reviewed_families={r.get('family_id') for r in rows if r['pair_id'].split('__to__')[0] in sources and r.get('family_id')}
    available=[r for r in rows if r['pair_id'].split('__to__')[0] not in sources and r.get('family_id') not in reviewed_families and not r.get('engine_error')]
    families=Counter();targets=Counter();source_formats=Counter();complexities=Counter();picked=[];used=set()
    def features(r):
        es=r['source_design']['elements'];n=sum(e['type']=='text' for e in es)
        return (r.get('family_id') or r['pair_id'].split('__to__')[0],bucket(r['target_canvas']),bucket(r['source_design']['canvas']),'dense' if n>=10 else 'medium' if n>=5 else 'sparse')
    while available and len(picked)<count:
        eligible=[r for r in available if r['pair_id'].split('__to__')[0] not in used and families[features(r)[0]]<2]
        if not eligible:break
        def score(r):
            f,t,s,c=features(r)
            stable=int(sha256(r['pair_id'].encode()).hexdigest()[:8],16)/2**32
            return 100*(families[f]==0)+15*(targets[t]==0)+15*(source_formats[s]==0)+10*(complexities[c]==0)-20*families[f]-3*targets[t]+stable
        row=max(eligible,key=score);picked.append(row);used.add(row['pair_id'].split('__to__')[0]);available.remove(row)
        f,t,s,c=features(row);families[f]+=1;targets[t]+=1;source_formats[s]+=1;complexities[c]+=1
    return picked,dict(families=len(families),source_formats=dict(source_formats),target_formats=dict(targets),complexities=dict(complexities),score_independent=True,distinct_sources=True,reviewed_families_excluded=True,max_per_family=2)


def viewer(rows,portable=False):
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    title='Additional validation designs' if any(r.get('inference_mode') in ('fresh','validation_replay') for r in rows) else 'Diverse held-out designs'
    return '''<!doctype html><meta charset="utf-8"><title>__TITLE__</title><style>body{font:16px system-ui;background:#101820;color:#eee;margin:20px}a{color:#9ed4ff}header{position:sticky;top:0;background:#101820;padding:12px;z-index:1}button,select{padding:10px;margin:5px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}figure{margin:0;background:#182532;padding:12px}img{width:100%;height:450px;object-fit:contain}.contact img{height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.bad{color:#ffab9f}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><header><a href="../main_engine_qwen_outputs/index.html">Main engine gallery</a> · <button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="count"></span><select id="select" onchange="i=Number(this.value);show()"></select></header><h1>__TITLE__</h1><p>Selected by family, complexity and aspect-ratio coverage before consulting outcome scores. Existing reviewed families are excluded. __INFERENCE_NOTE__ The human preview is for comparison, never an engine input.</p><h2 id="title"></h2><div id="grid"></div><p id="links"></p><pre id="metrics"></pre><script>const rows=__ROWS__,portable=__PORTABLE__;let i=0;document.getElementById('select').replaceChildren(...rows.map((r,j)=>{const o=document.createElement('option');o.value=j;o.textContent=`${j+1}. ${r.title}`;return o}));function show(){if(!rows.length)return;const r=rows[i];document.getElementById('count').textContent=`${i+1}/${rows.length}`;document.getElementById('select').value=i;document.getElementById('title').textContent=r.title;const cards=portable?[['Original · resized',r.contact]]:[['Original',r.source],['Resized',r.output]];document.getElementById('grid').className=portable?'contact':'grid';document.getElementById('grid').innerHTML=cards.filter(x=>x[1]).map(x=>`<figure><figcaption>${x[0]}</figcaption><a href="${x[1]}"><img src="${x[1]}"></a></figure>`).join('');document.getElementById('links').innerHTML=`<a href="${r.source_json}">Source JSON</a> · <a href="${r.raw_json}">Decoded proposal JSON</a> · <a href="${r.output_json}">Selected JSON</a>`;document.getElementById('metrics').className=r.defect_flags.length?'bad':'';document.getElementById('metrics').textContent=JSON.stringify({pair_id:r.pair_id,family:r.family_id,winner:r.winner,qwen_family_win:r.qwen_family_win,defect_flags:r.defect_flags,source_authored_issues:r.source_authored_issues,manual_review_notes:r.manual_review_notes,geometry_mae:r.geometry_mae,raw_geometry_mae:r.raw_geometry_mae,raw_json_complete:r.raw_json_complete,raw_proposal_valid:r.raw_proposal_valid,qwen_status:r.qwen_status},null,2)}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1)});show()</script>'''.replace('__TITLE__',title).replace('__ROWS__',payload).replace('__PORTABLE__',str(portable).lower()).replace('__INFERENCE_NOTE__', 'Fresh local 0.6B inference on the fixed validation sample.' if any(r.get('inference_mode')=='fresh' for r in rows) else 'Validation sample: exact coordinates from the earlier fresh local inference, replayed through the current engine.' if any(r.get('inference_mode')=='validation_replay' for r in rows) else 'Saved 0.6B proposals replayed through the current engine; these are not newly sampled predictions.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--records',type=Path,default=ROOT/'data/qwen_lora_expanded_v2/test.jsonl');p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=24);p.add_argument('--selection-plan',type=Path);p.add_argument('--portable',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from scripts.dataset_renderer import render_design_to_png
    existing=json.loads((ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25/review/main_engine_qwen_outputs/manifest.json').read_text())
    data=json.loads(args.report.read_text())
    if args.selection_plan:
        plan=json.loads(args.selection_plan.read_text());by={r['pair_id']:r for r in data['results']}
        picked=[by[p] for p in plan['pair_ids']];coverage=plan['coverage']
    else:picked,coverage=choose(data['results'],existing,args.count)
    entries=[];metadata={}
    records={r['pair_id']:r for r in map(json.loads,args.records.read_text().splitlines())}
    for split in ('train','validation','test'):
        folder=ROOT/'data'/('crello_'+split);manifest=folder/'manifest.jsonl'
        metadata[split]={r['template_id']:r for r in map(json.loads,manifest.read_text().splitlines())} if manifest.exists() else {}
    for i,row in enumerate(picked,1):
        source=row['source_design'];assets=ROOT/'data'/('crello_'+row['source_split']);target_id=row['pair_id'].split('__to__')[1];target_split=records[row['pair_id']].get('target_split',row['source_split']);preview=metadata[target_split].get(target_id,{}).get('preview')
        entry={k:row.get(k) for k in ['pair_id','family_id','winner','qwen_family_win','defect_flags','source_authored_issues','geometry_mae','raw_geometry_mae','raw_json_complete','raw_proposal_valid','qwen_status']}
        entry['inference_mode']='fresh' if data.get('fresh_model_inference') else 'validation_replay' if data.get('prediction_origin') else 'saved'
        title=next((e['content'].replace('\n',' ')[:95] for e in sorted(source['elements'],key=lambda e:-e.get('font_size',0)) if e['type']=='text'),'Untitled design');entry['title']=title
        raw=proposal_design(source,row['target_canvas'],row.get('qwen_model_elements') or [])
        docs={'source':source,'raw':raw,'output':row['output_design']}
        panels=[]
        for kind,doc in docs.items():
            png=args.output/f'{i:02d}_{kind}.png';jpg=png.with_suffix('.jpg')
            if doc['elements']:render_design_to_png(doc,assets,png,ROOT/'data/crello_fonts.pickle',max_size=1100,adjust_tight_text_widths=False)
            else:
                missing=Image.new('RGB',(500,500),'white');ImageDraw.Draw(missing).text((30,230),'No complete decoded proposal',fill='black');missing.save(png)
            with Image.open(png) as im:im.convert('RGB').save(jpg,quality=89,optimize=True)
            png.unlink();entry[kind]=jpg.name;name=f'{i:02d}_{kind}.json';(args.output/name).write_text(json.dumps(doc,indent=2)+'\n');entry[kind+'_json']=name
        reference=ROOT/'data'/('crello_'+target_split)/preview if preview else None
        if reference and reference.exists():
            dest=args.output/f'{i:02d}_reference.jpg'
            with Image.open(reference) as im:im=im.convert('RGB');im.thumbnail((1100,1100));im.save(dest,quality=89,optimize=True)
            entry['reference']=dest.name
        else:entry['reference']=None
        sheet=Image.new('RGB',(1200,470),'#182532');draw=ImageDraw.Draw(sheet)
        for j,(label,key) in enumerate([('Original','source'),('Resized','output')]):
            draw.text(((j%2)*600+12,(j//2)*470+8),label,fill='white')
            if entry.get(key):
                with Image.open(args.output/entry[key]) as im:
                    tile=ImageOps.contain(im.convert('RGB'),(580,430));sheet.paste(tile,((j%2)*600+(600-tile.width)//2,(j//2)*470+32+(430-tile.height)//2))
        contact=args.output/f'{i:02d}_contact.jpg';sheet.save(contact,quality=86,optimize=True);entry['contact']=contact.name;entries.append(entry)
    (args.output/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n');(args.output/'selection.json').write_text(json.dumps(coverage,indent=2)+'\n');(args.output/'index.html').write_text(viewer(entries))
    if args.portable:
        args.portable.mkdir(parents=True,exist_ok=True)
        for entry in entries:
            for key in ['contact','source_json','raw_json','output_json']:shutil.copy2(args.output/entry[key],args.portable/entry[key])
        (args.portable/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n');(args.portable/'selection.json').write_text(json.dumps(coverage,indent=2)+'\n');(args.portable/'index.html').write_text(viewer(entries,True))
    print(json.dumps(dict(cases=len(entries),coverage=coverage,flagged=sum(bool(e['defect_flags']) for e in entries)),indent=2))
if __name__=='__main__':main()
