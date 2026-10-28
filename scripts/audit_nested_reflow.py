#!/usr/bin/env python3
"""Record exact frozen-cohort deltas and source grouping evidence; no inference."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def audit(run, baseline, images, baseline_images, output):
    from engine.macro_layout import composites
    from engine.nested_structure import regional_surfaces,repeated_text_rows,unpainted_ids
    from scripts.refresh_regression_viewer import fingerprint
    manifest=json.loads((run/'run_manifest.json').read_text())
    current=fingerprint(ROOT/'config/regression_review.json')
    if any(manifest['fingerprint'].get(k)!=v for k,v in current.items()):
        raise ValueError('Engine/input snapshot changed; replay before auditing.')
    cases=[];groups=[]
    for f in sorted(run.glob('[0-9][0-9].json')):
        row=json.loads(f.read_text());old=json.loads((baseline/f.name).read_text())['after'];new=row['after']
        assert row['before']['output_design']==old['output_design']
        assert new['source_design']==old['source_design'] and new['target_canvas']==old['target_canvas']
        source_ids=[e['id'] for e in new['source_design']['elements']]
        output_ids=[e['id'] for e in new['output_design']['elements']]
        assert len(source_ids)==len(set(source_ids))
        assert Counter(source_ids)==Counter(output_ids),(row['case']['pair_id'],'element inventory changed')
        for e in new['output_design']['elements']:
            original=next(e0 for e0 in new['source_design']['elements'] if e0['id']==e['id'])
            # Styles, content and user metadata are not hidden repair knobs.
            for k in ('type','content','asset_id','font_family','color','fill_color','opacity','custom_metadata'):
                assert e.get(k)==original.get(k),(row['case']['pair_id'],e['id'],k)
        pixel_path=images/f'{f.stem}_output.jpg';prior=baseline_images/pixel_path.name
        same_pixels=(pixel_path.is_file() and prior.is_file() and sha256(pixel_path.read_bytes()).digest()==sha256(prior.read_bytes()).digest())
        old_elements={e['id']:e for e in old['output_design']['elements']}
        delta=max((abs(float(e.get(k,0))-float(old_elements[e['id']].get(k,0))) for e in new['output_design']['elements'] for k in ('x','y','width','height','font_size')),default=0.)
        cases.append(dict(max_geometry_delta=delta,same_geometry=delta<1e-6,index=int(f.stem),pair_id=row['case']['pair_id'],cohort=row['case']['cohort'],
            same_output=new['output_design']==old['output_design'],same_render_bytes=same_pixels,
            old_winner=old['winner'],winner=new['winner'],old_flags=old['defect_flags'],flags=new['defect_flags'],
            added_flags=sorted(set(new['defect_flags'])-set(old['defect_flags']))))
        if row['case']['cohort']=='supplied' and row['case']['pair_id'].endswith('__to__requested'):
            source=new['source_design'];spec=composites(source)
            groups.append(dict(pair_id=row['case']['pair_id'],elements=len(source['elements']),
                blocks=[[e['id'] for e in b] for b in spec['blocks']],sizes=[len(b) for b in spec['blocks']],
                links=dict(Counter(k for a,b,k in spec['links'])),regional_scopes=regional_surfaces(source),
                repeated_rows=repeated_text_rows(source['elements'],source['canvas']),source_unpainted=sorted(unpainted_ids(source))))
    assert len(cases)==len(manifest['cohort']['cases'])
    summary=dict(run=str(run.relative_to(ROOT)),baseline=str(baseline.relative_to(ROOT)),qwen_enabled=True,
        fresh_inference=False,cases=len(cases),changed=sum(not c['same_geometry'] for c in cases),exact_json_changed=sum(not c['same_output'] for c in cases),
        before_flagged=sum(bool(c['old_flags']) for c in cases),flagged=sum(bool(c['flags']) for c in cases),
        new_flag_cases=sum(bool(c['added_flags']) for c in cases),
        core_unchanged=sum(c['same_geometry'] for c in cases[:38]),
        official_and_core_dataset_unchanged=all(c['same_geometry'] and c['same_render_bytes'] for c in cases[:38] if c['cohort']!='alternate'),
        results=cases,source_grouping=groups)
    output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('results','source_grouping')},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--baseline-images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.run.resolve(),a.baseline.resolve(),a.images.resolve(),a.baseline_images.resolve(),a.output.resolve())
