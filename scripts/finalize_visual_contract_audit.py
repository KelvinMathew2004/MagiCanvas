#!/usr/bin/env python3
"""Combine independently replayed held-out shards and record cohort coverage."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    if (run/'heldout/report.json').exists():
        reports=[json.loads((run/'heldout/report.json').read_text())]
    else:
        reports=[json.loads((run/f'heldout_{s}/report.json').read_text()) for s in 'abcd']
    rows=[r for d in reports for r in d['results']]
    assert len(rows)==89 and len({r['pair_id'] for r in rows})==89
    hashes=reports[0]['manifest']['code_sha256'];assert all(d['manifest']['code_sha256']==hashes for d in reports)
    from scripts.evaluate_qwen_adapter import summarize_engine
    from scripts.layout_defects import build_ab_viewer
    base=json.loads((ROOT/'data/qwen_lora/inventory_fixes_verified/heldout/report.json').read_text())['results']
    records=[json.loads(s) for s in (ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines() if s.strip()]
    target=run/'heldout';target.mkdir(exist_ok=True)
    manifest=dict(reports[0]['manifest'],cases=89,pair_ids=[r['pair_id'] for r in rows],shards=len(reports))
    combined=dict(manifest=manifest,summary=summarize_engine(rows),results=rows)
    combined['viewer']=build_ab_viewer(rows,base,records,target/'viewer',10,candidate_label='Current visual contracts',comparison_label='Previous inventory engine')
    (target/'report.json').write_text(json.dumps(combined,indent=2)+'\n')
    from collections import Counter
    summary={}
    for cohort in ['heldout','official','crello','crello_1p7b']:
        d=combined if cohort=='heldout' else json.loads((run/cohort/'report.json').read_text());rs=d['results']
        assert d['manifest']['code_sha256']==hashes
        flags=Counter(f for r in rs for f in r.get('defect_flags',[]))
        summary[cohort]=dict(cases=len(rs),engine_errors=sum(bool(r.get('engine_error')) for r in rs),flagged=sum(bool(r.get('flagged')) for r in rs),flags=dict(flags),summary=d['summary'])
    (run/'summary.json').write_text(json.dumps(dict(cohorts=summary,code_sha256=hashes,raw_coordinates_reused=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
