#!/usr/bin/env python3
"""Fresh Qwen-enabled supplied-design and portrait-to-skyscraper evaluations."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/user_design_stress/run_v2'


def sources_and_records():
    sources={};records=[]
    for path in sorted((ROOT/'tests/fixtures/user_level3_sources').glob('u*.json')):
        f=json.loads(path.read_text());key='supplied:'+path.stem;sources[key]=f['source']
        for target in f['targets']:
            records.append(dict(pair_id=key+'__to__'+target['name'],family_id=key,source_split='official',target_canvas={k:target[k] for k in ('width','height')}))
    for key,path in [('portrait:n04',ROOT/'tests/fixtures/novel_level3_sources/n04.json')]:
        sources[key]=json.loads(path.read_text())['source'];records.append(dict(pair_id=key+'__to__skyscraper',family_id=key,source_split='train',target_canvas=dict(width=160,height=600)))
    # Feature-selected before generation: dense portrait sources from validation,
    # distinct families; neither candidate output nor target quality is consulted.
    eligible=[];base=ROOT/'data/crello_validation'
    for line in (base/'manifest.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('status')!='exported' or not 18<=row.get('element_count',0)<=40:continue
        canvas=row['canvas']
        if not .45<=canvas['width']/canvas['height']<=.85:continue
        source=json.loads((base/row['design']).read_text())
        if sum(e['type']=='text' for e in source['elements'])<8:continue
        eligible.append((row,source))
    families=set()
    for row,source in sorted(eligible,key=lambda item:(-item[0]['element_count'],item[0]['template_id'])):
        family=row.get('cluster_index',row['template_id'])
        if family in families:continue
        key='portrait:'+row['template_id'];families.add(family);sources[key]=source
        records.append(dict(pair_id=key+'__to__skyscraper',family_id=key,source_split='validation',target_canvas=dict(width=160,height=600)))
        if len(families)==4:break
    if len(families)!=4:raise ValueError('Need four eligible portrait families')
    return sources,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('0.6b','1.7b'),default='0.6b')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();output=args.output.resolve()
    model,adapter={'0.6b':('qwen3-0.6b','qwen3-layout-lora-v2'),
                   '1.7b':('qwen3-1.7b','qwen3-layout-lora-v2-expanded-1.7b')}[args.arm]
    os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    from scripts.evaluate_qwen_adapter import run_engine_mode
    sources,records=sources_and_records();output.mkdir(parents=True,exist_ok=True)
    # Official-path flag selects the same brand-font renderer as the live viewer.
    for r in records:
        if r['source_split']=='official':
            path=output/(r['family_id'].replace(':','_')+'.json');path.write_text(json.dumps(dict(source=sources[r['family_id']])));r['official_path']=str(path)
    fp=dict(engine={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')},sources={k:sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in sources.items()},records=records,qwen_enabled=True,fresh_inference=True,
            arm=args.arm,adapter_sha256=sha256((ROOT/'models'/adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),model_config_sha256=sha256((ROOT/'models'/model/'config.json').read_bytes()).hexdigest())
    marker=output/'run_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=fp:raise SystemExit('Snapshot changed; use a new run folder')
    marker.write_text(json.dumps(fp,indent=2)+'\n')
    rows=run_engine_mode(records,lambda split,key:deepcopy(sources[key]),ROOT/'models'/adapter,ROOT/'models'/model,output/'inference.jsonl')
    for r in rows:
        for k in ('geometry_mae','font_mae','raw_geometry_mae','raw_relative_font_mae'):r[k]=None
    (output/'report.json').write_text(json.dumps(dict(records=records,results=rows,manifest=fp,fresh_model_inference=True,human_target_available=False),indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),errors=[dict(case=r['pair_id'],error=r['engine_error']) for r in rows if r.get('engine_error')],flagged=sum(bool(r.get('defect_flags')) for r in rows))))


if __name__=='__main__':main()
