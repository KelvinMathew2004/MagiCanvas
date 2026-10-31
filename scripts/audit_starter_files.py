#!/usr/bin/env python3
"""Compare every organizer-provided file with the exact starter-import commit."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INTENTIONAL = {
    'engine/resize.py': 'Team layout implementation behind the supplied resize() interface.',
    'README.md': 'Team approach, evaluator commands and operation limits.',
    'Dockerfile': 'Build-time offline model/dependency packaging and required-adapter verification.',
    'requirements.txt': 'Additional engine/model dependencies; expressly permitted by the brief.',
    '.gitignore': 'Keep generated outputs, private models/data and historical research out of the submission.',
}


def audit(baseline='a682b34'):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)
    commit = git('rev-parse', baseline + '^{commit}').decode().strip()
    paths = git('ls-tree', '-rz', '--name-only', commit).decode().split('\0')
    rows = []
    for name in filter(None, paths):
        original = git('show', commit + ':' + name)
        path = ROOT / name
        current = path.read_bytes() if path.is_file() else None
        status = 'unchanged' if current == original else 'modified' if current is not None else 'missing'
        rows.append(dict(path=name, status=status, original_sha256=sha256(original).hexdigest(),
                         current_sha256=sha256(current).hexdigest() if current is not None else None,
                         expected_change=name in INTENTIONAL,
                         reason=INTENTIONAL.get(name, 'Organizer-provided file; preserve byte for byte.')))
    flagged = [r['path'] for r in rows if r['status'] == 'missing' or
               r['status'] != 'unchanged' and not r['expected_change']]
    return dict(baseline_commit=commit, supplied_files=len(rows),
                unchanged=sum(r['status'] == 'unchanged' for r in rows),
                modified=[r['path'] for r in rows if r['status'] == 'modified'],
                flagged_unexpected_changes=flagged, files=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='a682b34')
    parser.add_argument('--output', type=Path, default=ROOT/'data/submission_audit/starter_files.json')
    args = parser.parse_args()
    result = audit(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))
    raise SystemExit(bool(result['flagged_unexpected_changes']))
