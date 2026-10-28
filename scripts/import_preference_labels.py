#!/usr/bin/env python3
"""Validate human exports and prepare a local, family-separated ranking corpus.

No approvals are generated and no model is trained. Repeated checks, ties and
unsafe preferred layouts are reported separately rather than relabelled as good.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_siglip_ranker import reviewed_pairs,readiness,inconsistent_repeat_ids
PACK=ROOT/'data/gemini_review_qwen_projection_fixes_2026-10-25'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def validate(rows,choices):
    by={r['pair_id']:r for r in rows};latest={c['pair_id']:c for c in choices}
    errors=[]
    for pid,c in latest.items():
        r=by.get(pid)
        if not r:errors.append(f'{pid}: unknown pair');continue
        images=[Path(r['positive_images'][0]),Path(r['engine_image'])]
        if not all(p.is_file() for p in images):errors.append(f'{pid}: missing image');continue
        digest=hashlib.sha256(b''.join(p.read_bytes() for p in images)).hexdigest()
        if digest!=r.get('review_hash') or digest!=c.get('review_hash'):errors.append(f'{pid}: stale image hash')
        if c.get('decision') not in {'accept','prefer_candidate','skip'}:errors.append(f'{pid}: invalid choice')
        if 'display_swapped' in r and c.get('display_choice') in ('a','b'):
            expected='accept' if (c['display_choice']=='a')!=r['display_swapped'] else 'prefer_candidate'
            if c['decision']!=expected:errors.append(f'{pid}: side mapping disagrees')
    if errors:raise ValueError('\n'.join(errors))
    return latest


def main(args):
    campaign=PACK/'review/preferences_campaign'
    rows=read(campaign/'manifest.jsonl');choices=read(args.labels)
    latest=validate(rows,choices)
    # Incremental exports cannot erase earlier decisions. A new explicit skip
    # still revokes that pair's prior preference through latest-record semantics.
    existing=read(campaign/'approvals.jsonl')
    merged={c['pair_id']:c for c in existing+choices}
    latest=validate(rows,list(merged.values()))
    archive=campaign/'human_exports';archive.mkdir(exist_ok=True)
    digest=hashlib.sha256(args.labels.read_bytes()).hexdigest()
    (archive/(digest+'.jsonl')).write_bytes(args.labels.read_bytes())
    (campaign/'approvals.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in merged.values()))
    repeats=[];reasons=Counter();by={r['pair_id']:r for r in rows}
    ambiguous=inconsistent_repeat_ids(rows,latest)
    eligible=reviewed_pairs(rows,campaign/'approvals.jsonl');accepted={r['pair_id'] for r,_,_ in eligible}
    for pid,c in latest.items():
        r=by[pid]
        if r.get('reliability_repeat_of'):
            original=latest.get(r['reliability_repeat_of'])
            repeats.append(dict(pair_id=pid,original=r['reliability_repeat_of'],consistent=original is not None and (c['decision'],c.get('reject_both',False))==(original['decision'],original.get('reject_both',False))))
            reasons['consistency_repeat']+=1
        elif pid in ambiguous:reasons['inconsistent_original']+=1
        elif c['decision']=='skip':reasons['neither_good' if c.get('reject_both') else 'acceptable_tie']+=1
        elif pid not in accepted:reasons['unsafe_preferred_layout']+=1
    report=dict(received=len(choices),export_sha256=digest,unique_reviewed=len(latest),hash_validated=len(latest),decisions=dict(Counter(c['decision'] for c in latest.values())),excluded=dict(reasons),consistency=repeats,
        eligible=readiness(eligible),trained=False,production_promoted=False)
    neither=[by[pid] for pid,c in latest.items() if c.get('reject_both') and not by[pid].get('reliability_repeat_of')]
    report['neither_by_category']=dict(Counter(r['category'] for r in neither));report['neither_by_format']=dict(Counter(r['format'] for r in neither))
    evidence=ROOT/'data/qwen_lora/engine_upgrade';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'human_labels_v3_received.json').write_text(json.dumps(report,indent=2)+'\n')
    # Hash-bound labels only; preserve all original manifests/status and ancestry.
    all_rows=[];all_choices=[]
    for folder,label in [(ROOT/'data/siglip_human_review','approvals_occupancy.jsonl'),(ROOT/'data/siglip_engine_review_v2','approvals.jsonl'),(campaign,'approvals.jsonl')]:
        batch=read(folder/'manifest.jsonl');decisions=read(folder/label);validate(batch,decisions)
        all_rows.extend(batch);all_choices.extend(decisions)
    assert len({r['pair_id'] for r in all_rows})==len(all_rows),'Pair ID collision across campaigns'
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    for name,values in [('manifest.jsonl',all_rows),('approvals.jsonl',all_choices)]:
        (output/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
    corpus=reviewed_pairs(all_rows,output/'approvals.jsonl');summary=readiness(corpus)
    summary.update(human_exports=[str(args.labels)],trained=False,production_promoted=False,scope='Hash-verified eligible preferences from three human-reviewed campaigns. Ties, repeat checks and hard-ineligible positives excluded; fixed family folds. No evaluation fixture used as a training positive.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (evidence/'preference_corpus_readiness.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(batch=report,corpus=summary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'data/siglip_reviewed_corpus')
    main(p.parse_args())
