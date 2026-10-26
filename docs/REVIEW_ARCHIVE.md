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
