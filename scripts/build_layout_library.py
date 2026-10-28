#!/usr/bin/env python3
"""Export geometry-only human macro zones from model-training families.

Validation/test connected families and human preference families are excluded.
This is a geometry-clean library, not a claim of raster-perfect human examples.
"""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def identities(row):
    pair=row.get('pair_id','').split('__to__')
    return (row.get('source_template_id',row.get('source_id',pair[0])),
            row.get('target_template_id',row.get('target_id',pair[-1])))


def allowed_training(rows,preference_rows=()):
    """Union all ancestry before excluding: indirect train->heldout bridges count."""
    parent={}
    def find(i):
        parent.setdefault(i,i)
        if parent[i]!=i:parent[i]=find(parent[i])
        return parent[i]
    def join(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    for row in list(rows)+list(preference_rows):
        a,b=identities(row);join('id:'+a,'id:'+b)
        if row.get('family_id'):join('id:'+a,'family:'+row['family_id'])
    excluded={find('id:'+identities(r)[0]) for r in rows if r.get('model_split')!='train'}
    # Keep all preference families out, including training votes. Retrieval
    # must not see their human target geometry when assessing the ranker.
    excluded|={find('id:'+identities(r)[0]) for r in preference_rows}
    return [(r,find('id:'+identities(r)[0])) for r in rows if r.get('model_split')=='train' and find('id:'+identities(r)[0]) not in excluded]


def zones(design):
    from engine.macro_layout import composites
    from engine.layout_descriptors import describe
    sw,sh=(design['canvas'][k] for k in ('width','height'))
    spec=composites(design);desc=describe(design,spec['blocks'])
    if not 2<=len(desc)<=20:return None
    result=[]
    for d in desc:
        l,t,r,b=d['box'];l=max(0.,l);t=max(0.,t);r=min(sw,r);b=min(sh,b)
        if min(r-l,b-t)<=0 or (r-l)*(b-t)<.001*sw*sh:return None
        result.append(dict(role=d['role'],kind=d['kind'],box=[round(v,6) for v in (l/sw,t/sh,(r-l)/sw,(b-t)/sh)]))
    # The packer treats zones as exclusive allocations. Reject macro overlap;
    # source-supported overlapping children already belong to one zone.
    for i,a in enumerate(result):
        x,y,w,h=a['box']
        for b in result[i+1:]:
            q,z,u,v=b['box'];overlap=max(0.,min(x+w,q+u)-max(x,q))*max(0.,min(y+h,z+v)-max(y,z))
            if overlap>.01*min(w*h,u*v):return None
    return sorted(result,key=lambda z:(z['box'][1],z['box'][0],z['role']))


def bucket(ar):
    ratios=[9/16,4/5,1,4,728/90,160/600]
    return min(range(6),key=lambda i:abs(math.log(ar/ratios[i])))


def main(args):
    rows=[];files=[]
    for split in ('train','validation','test'):
        p=args.corpus/f'{split}.jsonl';files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha256(p.read_bytes()).hexdigest()))
        rows += [dict(json.loads(l),model_split=split) for l in p.read_text().splitlines() if l.strip()]
    pref=ROOT/'data/siglip_reviewed_corpus/manifest.jsonl'
    preference_rows=[json.loads(l) for l in pref.read_text().splitlines() if l.strip()] if pref.exists() else []
    allowed=allowed_training(rows,preference_rows)
    from scripts.evaluate_qwen_adapter import read_sources
    get=read_sources();seen=set();entries={};rejected=Counter();provenance={}
    for row,family in allowed:
        for key,split_key in zip(identities(row),('source_split','target_split')):
            if key in seen:continue
            seen.add(key)
            try:design=get(row[split_key],key);z=zones(design)
            except (KeyError,ValueError,OSError):rejected['unavailable']+=1;continue
            if not z:rejected['geometry_or_partition']+=1;continue
            aspect=design['canvas']['width']/design['canvas']['height']
            entry=dict(aspect=round(aspect,6),zones=z,roles=dict(Counter(v['role'] for v in z)))
            digest=sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest();entry['key']=digest
            if digest not in entries:entries[digest]=(entry,family,key)
    # Round-robin aspect buckets, then at most two designs per family/aspect bucket; stable geometry
    # hashes make generation reproducible without shipping dataset IDs.
    pools={i:[] for i in range(6)}
    for entry,family,key in sorted(entries.values(),key=lambda v:v[0]['key']):pools[bucket(entry['aspect'])].append((entry,family,key))
    selected=[];families=set();family_buckets=Counter()
    while len(selected)<args.limit and any(pools.values()):
        for i in pools:
            while pools[i]:
                entry,family,key=pools[i].pop(0)
                if family_buckets[(family,i)]>=2:continue
                selected.append(entry);families.add(family);family_buckets[(family,i)]+=1;provenance[entry['key']]=dict(template_id=key,family=family);break
            if len(selected)>=args.limit:break
    payload=dict(version=1,scope='Geometry-only human zones from training-connected families; no evaluation/preference families; geometry checked, not raster-certified.',layouts=selected)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    args.audit.mkdir(parents=True,exist_ok=True)
    summary=dict(inputs=files,training_pairs=len(allowed),unique_human_designs_checked=len(seen),eligible_geometries=len(entries),exported=len(selected),
        families=len(families),rejected=dict(rejected),aspect_buckets=dict(Counter(str(bucket(e['aspect'])) for e in selected)),bytes=args.output.stat().st_size,
        library_sha256=sha256(args.output.read_bytes()).hexdigest(),evaluation_scope='Frozen 38 are development examples; library excludes all Qwen held-out/preference families, not a claim of unseen evaluation.')
    (args.audit/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(args.audit/'local_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,default=ROOT/'data/qwen_lora_expanded_v2');p.add_argument('--output',type=Path,default=ROOT/'engine/data/layout_library.json');p.add_argument('--audit',type=Path,default=ROOT/'data/chameleon_upgrade_audit/library');p.add_argument('--limit',type=int,default=300);main(p.parse_args())
