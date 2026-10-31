# Organizer starter-file audit

Compared October 30, 2026 with the second commit,
`a682b343fd14640042794cd36a49111621baba3f` (“Adding starter repo content”).
The comparison reads every original Git blob and checks the working file bytes,
including binary assets/fonts; it does not rely on filenames or a diff summary.

**34 original files: 29 unchanged, five intentional modifications, no unexpected
remaining changes.**

| Modified original | Reason |
|---|---|
| `engine/resize.py` | Team layout engine, importing modular engine helpers |
| `README.md` | Short approach, evaluator commands and operation limits |
| `Dockerfile` | Install dependencies/download local models at build time; require the submitted adapter |
| `requirements.txt` | Engine/model dependencies |
| `.gitignore` | Exclude private/generated data, weights and local historical research |

The runner (`engine/main.py`), all renderer files, schema, three official sample
JSONs, rules, FAQ and supplied brand kit match the starter **byte for byte**.
New engine modules, tests, scripts and the preview are team additions, not
organizer files overwritten in place.

## Renderer helper found and relocated

Before this audit, `renderer/converter/assets.py` had 83 added lines for visible
artwork geometry; the original rendering functions were unchanged. Those added
helpers now live in `engine/asset_geometry.py`. Their function bodies were
retained, engine/test/tool imports were updated, and the organizer's asset loader
was restored exactly. This avoids a dependency on a modified official renderer.
The fixed suite is replayed to check this relocation separately from any layout
improvement; no model inference or new training is implied.

Verification: **487 tests and 34 subtests passed**. All 59 Qwen-enabled saved-
prediction cases retain their winners and meaningful geometry; all 118 Original/
Resized JPEGs are byte-identical. One JSON has only two Y-coordinate rounding
differences of about `4.55e-13` pixels. No new flags were added. The existing
flagged dense conference micro-banner remains a known defect, not an approved
or perfect output. Local evidence: `data/submission_audit/relocation_comparison.json`.

## Rules interpretation and release status

The challenge brief (§7.1) identifies `engine/resize.py` as the implementation
point; §7.2 and the Docker instructions explicitly permit build-time dependencies
and model weights. Modular engine helpers retain the same public `resize()`
interface. This is a code-integrity audit, not organizer approval of an approach.
The supplied schema, renderer and evaluation fixtures are preserved. The checked
PDF is “Final Chameleon Challenge.pdf”, SHA-256
`475b255088c663527c65cc66cc919a7fadcd8444659454ef08b1c657f9836887`;
§§5.1, 7.1 and 7.2 supply the deliverable and execution requirements.

The final command remains Linux/amd64 Docker with input/output mounts and
`--network none`. Public adapter publication and a successful final offline
container run remain pending. Native Mac Metal behavior must not be presented
as Docker CPU performance.

## Repeat the check

```bash
.venv/bin/python scripts/audit_starter_files.py
```

Detailed original/current SHA-256 hashes are saved locally in
`data/submission_audit/starter_files.json`. Any missing original file or unexpected
change makes this command fail.

Historical reports/notebooks in `docs/archive/` are retained on disk but ignored
and removed from Git's index; `data/`, models and competitor research remain
ignored too. These index removals take effect in the next commit; previous Git
history is not rewritten. `.gitignore` does not sanitize a manually created ZIP.
