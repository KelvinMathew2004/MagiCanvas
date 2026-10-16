"""Inspect one official task's deterministic layout decisions (development only)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resize import _adapt_with_diagnostics
from renderer.converter.convert import render_design_to_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path, help='Official source + target_canvas task JSON')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'inspection')
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    output, report = _adapt_with_diagnostics(task['source'], task['target_canvas'])
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.task.stem
    (args.output / f'{stem}.json').write_text(json.dumps(output, indent=2))
    (args.output / f'{stem}-diagnostics.json').write_text(json.dumps(report, indent=2))
    render_design_to_png(output, ROOT / 'brand_kit' / 'assets', args.output / f'{stem}.png')
    print(json.dumps({key: report[key] for key in ('aspect', 'generated', 'feasible', 'starts', 'winner', 'hard_constraints_passed', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
