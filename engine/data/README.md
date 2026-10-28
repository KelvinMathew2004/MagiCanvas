# Local retrieval library

`layout_library.json` is optional, generated training-derived macro geometry.
It stays local with the underlying Crello research data, rather than being
redistributed in Git or the Gemini packet. No runtime network calls are made.
The engine remains usable without it; that candidate family is unavailable.

With the locally converted dataset and family-isolated model corpus available:

```sh
.venv/bin/python scripts/build_layout_library.py
```

The current export has 134 geometry-clean layouts from 58 training families,
with no more than two per family/aspect bucket. It excludes connected Qwen
validation/test and human preference families. Sparse extreme-format coverage
is recorded honestly in `data/chameleon_upgrade_audit/library/summary.json`.
No source text, asset IDs, template IDs, font files or pixels enter the runtime
library. A SHA256 geometry key identifies each allocation, not a dataset ID.

Source: [CyberAgent Crello](https://huggingface.co/datasets/cyberagent/crello).
See [dataset notes](../../docs/CRELLO_DATA_AND_TRAINING.md) for the pinned local
export, attribution and distribution limits. Geometry checks do not certify
raster readability; every realized candidate still passes the engine's gates.
