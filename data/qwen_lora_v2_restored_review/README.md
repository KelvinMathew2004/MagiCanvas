# Restored v2 adapter review

The local default is `models/qwen3-layout-lora-v2` when `MAGICANVAS_QWEN_ADAPTER_DIR` is unset. It loaded successfully with prompt version 2.

`review.html` shows the same eight held-out pairs as the saved v11 review, rerun with the v2 adapter. `comparison_to_v11.json` compares model and engine candidates, selected winner, hard checks, and target geometry. The saved v11 viewer used the unadjusted starter renderer; this rerun used the current dataset-only text-width adjustment for display, so rendered pixels should not be compared as an exact image match.
