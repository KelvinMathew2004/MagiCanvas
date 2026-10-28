#!/usr/bin/env python3
"""Human-bound ten-feature Bradley-Terry pilot with sealed test-family gate.

Reconstruct exact reviewed candidates and require pixel equality before using
geometry. Fixed L2=.1, no validation hyperparameter sweep. Never activates a
production head. Test features/labels are consumed only if validation passes.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from collections import Counter
from tempfile import TemporaryDirectory
import argparse,json,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def reconstruct(row):
    if row.get('audit'):
        audit=json.loads(Path(row['audit']).read_text())
        if row['pair_id'].startswith('engine-v2-'):
            n=int(row['pair_id'].split('-')[-1]);return audit['source'],audit['winner'],audit['alternatives'][n]['design'],'engine'
        n=int(row['pair_id'].split('-')[-1]);p=audit['pairs'][n];return audit['source'],p['a'],p['b'],'campaign'
    # Original controlled shrink cohort: recreate the published preparation
    # recipe exactly. This cohort is reported separately from engine choices.
    index={r['template_id']:r for r in map(json.loads,(ROOT/'data/crello_train/manifest.jsonl').read_text().splitlines()) if r.get('status')=='exported'}
    target=json.loads((ROOT/'data/crello_train'/index[row['target_id']]['design']).read_text());bad=deepcopy(target)
    w,h=(target['canvas'][k] for k in ('width','height'));n=int(Path(row['positive_images'][0]).stem.split('_')[0]);s=.54 if n%2==0 else .72
    for e in bad['elements']:
        if e['type']!='text' and e['width']*e['height']>=.80*w*h:continue
        e.update(x=w/2+(e['x']-w/2)*s,y=h/2+(e['y']-h/2)*s,width=e['width']*s,height=e['height']*s)
        if e['type']=='text':e['font_size']*=s;e['letter_spacing']=e.get('letter_spacing',0)*s
    return target,target,bad,'controlled-shrink'


def render(doc,path,max_size):
    fonts={e.get('font_family','') for e in doc['elements'] if e['type']=='text'}
    if fonts and not fonts<={'Solstice Sans','Solstice Serif',''}:
        from scripts.dataset_renderer import render_design_to_png
        return render_design_to_png(doc,ROOT/'data/crello_train',path,ROOT/'data/crello_fonts.pickle',max_size=max_size,adjust_tight_text_widths=False)
    from renderer.converter.convert import render_design_to_png
    return render_design_to_png(doc,ROOT/'brand_kit/assets',path,fonts_dir=ROOT/'brand_kit/fonts',max_size=max_size)


def contrast(doc):
    from PIL import Image,ImageChops
    import numpy as np
    from engine.resize import _rgba
    from engine.raster_metrics import glyph_statistics
    # Native-resolution glyph paint against the same document without text.
    with TemporaryDirectory() as folder:
        a,b=Path(folder)/'full.png',Path(folder)/'under.png';size=max(doc['canvas']['width'],doc['canvas']['height']);render(doc,a,size)
        under=deepcopy(doc);under['elements']=[e for e in doc['elements'] if e['type']!='text'];render(under,b,size)
        with Image.open(a) as im,Image.open(b) as bg:
            full=im.convert('RGB');back=bg.convert('RGB');mask=np.asarray(ImageChops.difference(full,back).convert('L'));pixels=np.asarray(back)
        result={}
        for e in doc['elements']:
            if e['type']!='text':continue
            x,y,r,b=round(e['x']),round(e['y']),round(e['x']+e['width']),round(e['y']+e['height']);x=max(0,x);y=max(0,y);r=min(pixels.shape[1],r);b=min(pixels.shape[0],b)
            if r<=x or b<=y:result[e['id']]=1.;continue
            stat=glyph_statistics(pixels[y:b,x:r],mask[y:b,x:r],_rgba(e.get('color','#000000')),e.get('opacity',1))
            result[e['id']]=stat['lower_quintile'] if stat else 1.
        return result


def main(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    from scripts.train_siglip_ranker import reviewed_pairs,family_splits,image_embedding
    from engine.structural_ranker import FEATURES,features,fit,score
    from engine.vision import _POSITIVE,_NEGATIVE,_letterbox_square
    rows=list(map(json.loads,args.manifest.read_text().splitlines()));items=reviewed_pairs(rows,args.approvals);splits=family_splits(items,with_test=True)
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    modeldir=ROOT/'models/siglip';model=AutoModel.from_pretrained(str(modeldir),local_files_only=True).eval();processor=AutoProcessor.from_pretrained(str(modeldir),local_files_only=True)
    digest=sha256()
    for path in sorted(list(modeldir.glob('*.safetensors'))+list(modeldir.glob('*.json'))):
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    modelkey=digest.hexdigest();cache=ROOT/'data/siglip_feature_cache'/modelkey;cache.mkdir(exist_ok=True)
    with torch.inference_mode():
        ti=processor(text=list(_POSITIVE)+list(_NEGATIVE),padding='max_length',return_tensors='pt');tf=model.get_text_features(**ti)
        if not torch.is_tensor(tf):tf=tf.pooler_output
        tf=torch.nn.functional.normalize(tf.float(),dim=-1)
    def zero(path):
        key=sha256(Path(path).read_bytes()).hexdigest();p=cache/(key+'.pt')
        if p.exists():v=torch.load(p,weights_only=True,map_location='cpu')
        else:
            with Image.open(path) as im:inp=processor(images=_letterbox_square(im),return_tensors='pt')
            with torch.inference_mode():v=image_embedding(model,inp,torch)
            torch.save(v,p)
        logits=torch.nn.functional.normalize(v.float().reshape(-1),dim=-1)@tf.T*model.logit_scale.exp()
        return float((logits[:len(_POSITIVE)].mean()-logits[len(_POSITIVE):].mean()).item())
    skipped=[];source_stats={};prepared={};records=[]
    def prepare(split):
        pairs=[]
        selected=[i for i in items if splits[i[0]['pair_id']]==split]
        for num,(row,pos,neg) in enumerate(selected,1):
            try:
                source,a,b,cohort=reconstruct(row);vectors=[]
                key=sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
                if key not in source_stats:source_stats[key]=contrast(source)
                for doc,path in zip((a,b),(row['positive_images'][0],row['engine_image'])):
                    imagehash=sha256(Path(path).read_bytes()).hexdigest();doc_hash=sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest();ck=key+doc_hash+imagehash
                    if ck in prepared:vectors.append(prepared[ck]);continue
                    with TemporaryDirectory() as folder:
                        rendered=Path(folder)/'parity.png'
                        with Image.open(path) as voted:
                            render(doc,rendered,max(voted.size))
                            with Image.open(rendered) as now:
                                if now.size!=voted.size or not np.array_equal(np.asarray(now.convert('RGB')),np.asarray(voted.convert('RGB'))):raise ValueError('Reconstructed geometry does not reproduce voted pixels')
                    stats=contrast(doc);base=source_stats[key]
                    margin=min((v-min(3.,base.get(i,v)*.9) for i,v in stats.items()),default=0.)/3
                    vector=features(source,doc,margin,zero(path));vectors.append(vector);prepared[ck]=vector
                if str(Path(pos).resolve())!=str(Path(row['positive_images'][0]).resolve()):vectors.reverse()
                pairs.append(vectors);records.append(dict(pair_id=row['pair_id'],split=split,cohort=cohort,features=vectors,review_hash=row['review_hash']))
            except (ValueError,KeyError,OSError) as exc:skipped.append(dict(pair_id=row['pair_id'],split=split,reason=str(exc)))
            print(split,num,len(selected),'eligible',len(pairs),'skipped',len(skipped),flush=True)
        return pairs
    train=prepare('train');val=prepare('validation')
    if len(train)<50 or len(val)<10:raise RuntimeError('Insufficient verified geometry pairs; no fit or test access')
    artifact=fit(train,l2=.1)
    def evaluate(pairs):
        diffs=[score(p[0],artifact)-score(p[1],artifact) for p in pairs];baseline=[p[0][-1]-p[1][-1] for p in pairs]
        return dict(pairs=len(pairs),correct=sum(d>0 for d in diffs),accuracy=mean_positive(diffs),zero_shot_correct=sum(d>0 for d in baseline),zero_shot_accuracy=mean_positive(baseline))
    report=dict(features=list(FEATURES),split_counts=dict(Counter(splits.values())),verified_train=len(train),train=evaluate(train),validation=evaluate(val),skipped=skipped,test_scored=False,production_changed=False)
    gate=report['validation']['accuracy']>.60 and report['validation']['accuracy']>report['validation']['zero_shot_accuracy']
    if gate:
        test=prepare('test');report['test_scored']=True;report['test']=evaluate(test)
        gate=len(test)>=30 and report['test']['accuracy']>.60 and report['test']['accuracy']>report['test']['zero_shot_accuracy']
    else:report['test_not_scored_reason']='Fixed validation gate failed; test outcomes/features remain sealed.'
    report['human_gate_passed']=gate;report['deployment_gate_passed']=False;report['deployment_note']='Requires a separate safe-candidate engine ablation. No automatic production promotion.'
    report['cohorts']={c:{s:evaluate([r['features'] for r in records if r['cohort']==c and r['split']==s]) for s in ('train','validation','test') if any(r['cohort']==c and r['split']==s for r in records)} for c in sorted({r['cohort'] for r in records})}
    report['provenance']=dict(manifest_sha256=sha256(args.manifest.read_bytes()).hexdigest(),approvals_sha256=sha256(args.approvals.read_bytes()).hexdigest(),backbone_sha256=modelkey,trainer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),engine_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engine').glob('*.py')})
    (args.output/'candidate.json').write_text(json.dumps(dict(artifact,provenance=report['provenance'],human_gate_passed=gate,deployment_gate_passed=False),indent=2)+'\n')
    (args.output/'features.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def mean_positive(values):return sum(v>0 for v in values)/max(1,len(values))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'data/siglip_reviewed_corpus/manifest.jsonl');p.add_argument('--approvals',type=Path,default=ROOT/'data/siglip_reviewed_corpus/approvals.jsonl');p.add_argument('--output',type=Path,default=ROOT/'data/chameleon_upgrade_audit/structural_ranker');main(p.parse_args())
