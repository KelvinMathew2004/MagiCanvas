# Working on MagiCanvas

Preserve unrelated local work. Use generic source- or candidate-evidenced rules;
never match template IDs, content strings, or named ornaments in engine logic.

For major engine/selector/renderer changes:

1. Keep Qwen enabled. A deterministic candidate winning is reported honestly;
   it is not a Qwen-off experiment.
2. Run appropriate tests and refresh the fixed regression suite with
   `.venv/bin/python scripts/refresh_regression_viewer.py --workers 2`.
   This reuses exact decoded predictions and is an engine regression check,
   not fresh inference, a training run, or a generation-latency benchmark.
3. Inspect changed winners and new flags. Report remaining defects; do not
   equate zero automated flags with a perfect design.
4. Refresh the Gemini audit packet with
   `.venv/bin/python scripts/package_regression_review.py` after the suite.
   Keep raw/intermediate geometry in audit JSON; public viewers show only
   Original and Resized. The canonical URL is
   `http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html`.
5. Use `scripts/refresh_regression_viewer.py --check` to detect stale code/input
   snapshots. Never relabel old renders as fresh after a code change. Restart
   the interactive preview server after engine edits.

The fixed 38-case development cohort is in `config/regression_review.json`:
18 official conversions, 8 independent alternate-design conversions, and 12
complex dataset cases. It complements the wider 89-case benchmark and is not
an unseen test set. Required datasets, cached predictions, fonts and weights
remain local and are not distributed in the Gemini review ZIP.
