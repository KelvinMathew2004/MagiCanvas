# Archived review artifacts

On October 25, 2026, 2,380 obsolete review and copied-code snapshot files were
archived outside the working repository before removal. The archive was reopened
and its complete file-name/size inventory matched against the originals.

- Archive: `/Users/kelvinmathew/Documents/MagiCanvas-archives/2026-10-25-pre-stress-reviews.tar.gz`
- Inventory and SHA-256: `/Users/kelvinmathew/Documents/MagiCanvas-archives/2026-10-25-pre-stress-reviews.manifest.json`
- Original content: 637,787,864 bytes; archive: 606,929,389 bytes.

This includes the superseded curated engine review runs, copied Gemini review
code, the restored v2 review, and the old expanded-data eight-case viewer. Some
historical audit documents reference those paths; their artifacts are recoverable
from the archive under their original repository-relative names.

The active training/evaluation JSONL, adapter weights, current full batch baseline,
V65 review, and deletion-audit replay evidence remain in the workspace. Runtime
modules and regression tests were not removed as part of this artifact cleanup.

Current verification and stress results are described in `DAY4_5_AUDIT.md`.

The two incomplete stress attempts (`day4_5_stress_v1` and `v2`) were also
archived with per-file SHA-256 verification at
`/Users/kelvinmathew/Documents/MagiCanvas-archives/2026-10-25-partial-stress-attempts.tar.gz`.
The completed v3 run is retained as evidence for the fixes in v4.

## October 29, 2026 cleanup

Removed 21 superseded copied Gemini packet folders/ZIPs (3,021 files,
90,143,933 bytes) only after independently verifying every archived file's
SHA-256 against the original. Recovery archive:
`/Users/kelvinmathew/Documents/MagiCanvas-archives/2026-10-29-stale-review-packets.tar.gz`.
The adjacent `.tar.manifest.json` records original paths, sizes and hashes.
The current full packet, compact packet/ZIP, all model weights, train/validation/test
corpora, human labels and regression prediction dependencies were retained.
A cleanup summary is local at `data/training_next/cleanup_summary.json`.

Historical Markdown now lives under `docs/archive/`, including the previous
README and viewer instructions. Current entry points are listed in
[README.md](README.md). These records remain for reproducibility and future
research; they are not current operating instructions. Competitor research
is ignored under the archive path as well.

Sixteen byte-identical `scripts/* 2.py` copies were also removed after archive
SHA-256 verification. Their canonical `.py` counterparts remain. Recovery:
`/Users/kelvinmathew/Documents/MagiCanvas-archives/2026-10-29-exact-script-duplicates.tar.gz`,
with its adjacent manifest. Non-identical copies were preserved.

## October 30, 2026 submission cleanup

All 43 previously tracked files under `docs/archive/` were removed from Git's
index and the entire folder was added to `.gitignore` and `.dockerignore`.
Every file remains on disk. The index removals require the next commit to
affect a fresh clone; earlier Git history is unchanged. Current documentation
is indexed in docs/README.md. Old model and experiment numbers are historical
records, not performance claims for the submitted adapter.

## VLM ablation closed — October 30, 2026

VLM-specific evaluation reports, viewer/packet builders, admission tracing and
partial-backing experiment code/tests are retained under the ignored
`docs/archive/vlm_research_20261030/` tree. The experiment's runtime hook was
removed. Existing local model weights, evaluation JSONL and rendered evidence
remain available; no weights were deleted. Generic training/inference provider
support is retained for reproducibility and model swapping, without production
VLM integration. Qwen3-0.6B v2 remains the production default. No further VLM
training, routing or repair work is planned for submission.
