#!/usr/bin/env python3
"""No-training library expansion, preserving connected-family exclusions."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from scripts.build_layout_library import allowed_training,identities,bucket
    from engine.macro_layout import composites
    from engine.layout_descriptors import describe
    from scripts.evaluate_qwen_adapter import read_sources
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/upgrade_exploration/training_free_scale_v1/expanded_library.json');p.add_argument('--limit',type=int,default=600);a=p.parse_args()
    corpus=ROOT/'data/qwen_lora_expanded_v2';rows=[];inputs={}
    for split in ('train','validation','test'):
        f=corpus/f'{split}.jsonl';inputs[split]=sha256(f.read_bytes()).hexdigest()
        rows.extend(dict(json.loads(l),model_split=split) for l in f.read_text().splitlines() if l.strip())
    pref=ROOT/'data/siglip_reviewed_corpus/manifest.jsonl';prefs=list(map(json.loads,pref.read_text().splitlines()))
    allowed=allowed_training(rows,prefs);get=read_sources();seen=set();entries={};rejected=Counter()
    for row,family in allowed:
        for key,split in zip(identities(row),('source_split','target_split')):
            if key in seen:continue
            seen.add(key)
            try:doc=get(row[split],key)
            except (KeyError,OSError):rejected['unavailable']+=1;continue
            sw,sh=(doc['canvas'][k] for k in ('width','height'))
            for tight in (None,True,False):
                desc=describe(doc,composites(doc,tight=tight)['blocks'])
                if not 2<=len(desc)<=20:rejected['partition_size']+=1;continue
                zones=[]
                for d in desc:
                    l,t,r,b=d['box'];l=max(0.,l);t=max(0.,t);r=min(sw,r);b=min(sh,b)
                    if min(r-l,b-t)<=0 or (r-l)*(b-t)<.001*sw*sh:break
                    zones.append(dict(role=d['role'],kind=d['kind'],box=[round(v,6) for v in (l/sw,t/sh,(r-l)/sw,(b-t)/sh)]))
                if len(zones)!=len(desc):rejected['zone_geometry']+=1;continue
                def overlap(a,b):
                    x,y,w,h=a['box'];u,v,s,t=b['box'];return max(0.,min(x+w,u+s)-max(x,u))*max(0.,min(y+h,v+t)-max(y,v))>.01*min(w*h,s*t)
                if any(overlap(a,b) for i,a in enumerate(zones) for b in zones[:i]):rejected['exclusive_macro_overlap']+=1;continue
                zones.sort(key=lambda z:(z['box'][1],z['box'][0],z['role']))
                entry=dict(aspect=round(sw/sh,6),zones=zones,roles=dict(Counter(v['role'] for v in zones)))
                digest=sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest();entry['key']=digest
                entries.setdefault(digest,(entry,family,key,tight))
    pools={i:[] for i in range(6)}
    for entry,family,key,tight in sorted(entries.values(),key=lambda v:v[0]['key']):pools[bucket(entry['aspect'])].append((entry,family,key,tight))
    selected=[];provenance={};counts=Counter()
    while len(selected)<a.limit and any(pools.values()):
        for i in pools:
            while pools[i]:
                entry,family,key,tight=pools[i].pop(0)
                if counts[(family,i)]>=2:continue
                selected.append(entry);counts[(family,i)]+=1;provenance[entry['key']]=dict(family=family,template_id=key,tight=tight);break
            if len(selected)>=a.limit:break
    payload=dict(version=1,scope='Training-only geometry, default/tight/loose partitions. Evaluation and preference connected families excluded. No raster-clean certification.',layouts=selected)
    a.output.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    report=dict(inputs=inputs,training_pairs=len(allowed),families=len({v['family'] for v in provenance.values()}),unique_source_designs=len(seen),eligible_geometries=len(entries),exported=len(selected),
        buckets=dict(Counter(bucket(e['aspect']) for e in selected)),rejected=dict(rejected),library_sha256=sha256(a.output.read_bytes()).hexdigest(),local_provenance=provenance)
    a.output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='local_provenance'},indent=2))


if __name__=='__main__':main()
