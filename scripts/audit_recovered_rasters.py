#!/usr/bin/env python3
"""Independent native glyph-contrast audit of changed adapter review outputs.

This is diagnostic only. It neither edits the engine nor changes selection.
The shared text-difference mask can be ambiguous for overlapping glyphs; rotated
text and unsupported colors are explicitly unmeasured, not called passing.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from scripts.report_recovered_adapter import load
    from scripts.refresh_regression_viewer import documents_differ
    from scripts.audit_upgrade_rasters import paint_stats,against_source
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    b=load(a.run/'baseline.jsonl',True);n=load(a.run/'audited-0.6b-recovered.jsonl',True)
    folder=a.run/'native_rasters';folder.mkdir(exist_ok=True)
    fingerprint=sha256((a.run/'manifest.json').read_bytes()+Path(__file__).read_bytes()+
        (ROOT/'scripts/audit_upgrade_rasters.py').read_bytes()).hexdigest()
    records=[];cache={}
    keys=[k for k in b if k in n and k.startswith('review:') and b[k].get('output_design') and n[k].get('output_design')
        and (documents_differ(b[k]['output_design'],n[k]['output_design']) or b[k]['defect_flags']!=n[k]['defect_flags'])]
    for k in keys:
        digest=sha256(json.dumps(dict(fingerprint=fingerprint,before=b[k]['output_design'],after=n[k]['output_design'],source=b[k]['source_design']),sort_keys=True).encode()).hexdigest()
        path=folder/(digest+'.json')
        if path.is_file():records.append(json.loads(path.read_text()));continue
        source=b[k]['source_design'];split=b[k].get('source_split','train')
        os.environ['MAGICANVAS_DATASET_RENDERING']='0' if b[k].get('official_path') else '1'
        source_key=sha256(json.dumps(dict(source=source,split=split),sort_keys=True).encode()).hexdigest()
        try:
            if source_key not in cache:cache[source_key]=paint_stats(source,source,split)
            record=dict(pair_id=k,before=against_source(paint_stats(b[k]['output_design'],source,split),cache[source_key]),
                after=against_source(paint_stats(n[k]['output_design'],source,split),cache[source_key]))
        except Exception as e:
            record=dict(pair_id=k,error=f'{type(e).__name__}: {e}')
        if not record.get('error'):path.write_text(json.dumps(record,indent=2)+'\n')
        records.append(record);print(len(records),len(keys),k,'error:',record.get('error'),flush=True)
    report=dict(native_resolution=True,partial_matching_tasks=len(set(b)&set(n)),changed_review_tasks=len(keys),
        fingerprint=fingerprint,records=records,production_changed=False,
        limitations='Shared text-difference mask is not individual glyph ownership. Rotated text/unknown colors are unmeasured. This diagnostic does not override exact engine gates.')
    (a.run/'native_raster_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(changed_review_tasks=len(keys),errors=sum('error' in r for r in records))))


if __name__=='__main__':main()
