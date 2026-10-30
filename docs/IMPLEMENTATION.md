# Current architecture

The public entry point is `engine.resize.resize(source, target_canvas)`. The unchanged runner reads `input/*.json` and renders returned JSON to `output/` using the supplied converter and brand assets/fonts.

1. **Structure:** `source_structure.py`, `nested_structure.py` and `macro_layout.py` infer ownership, attachments, repeated siblings and copy stacks. Strong source lockups move together; ambient translucent paint does not indiscriminately merge a whole poster.
2. **Generate:** `qwen_seed.py` runs the local base plus matching LoRA with prompt v2. Qwen containers, bounded raw proposals and measured deterministic reflows join one candidate pool. Qwen stays enabled even when a deterministic candidate wins.
3. **Repair:** measured container flow, proportional artwork scaling, backing padding, shared axes and peer-gap equalization repair source-evidenced relationships. Existing intentional overlap may survive. Panels/background surfaces can resize; artwork does not freely stretch.
4. **Verify/select:** `resize.py`, `layout_contracts.py` and raster metrics compare actual rendered glyphs, clipping, contrast, contacts, IDs and bounds. SigLIP2 only breaks bounded ties. Returned geometry is the geometry verified; no unverified final mutation follows selection.

These are safeguards, not a proof that every complex input has a feasible, attractive layout. Tiny formats, ownership inference, background subjects and imperfect model proposals remain limitations. The known 320×50 conference case is kept flagged. No design ID, content string or named ornament selects engine behavior.

## Swappable model boundary

`qwen_seed.py` loads a model package through `MAGICANVAS_QWEN_DIR` and `MAGICANVAS_QWEN_ADAPTER_DIR`. A package must match architecture, hidden width, tokenizer and prompt marker. Existing 0.6B and 1.7B packages share the output contract, not adapter tensors. Use `scripts/check_model_profiles.py` for local compatibility checks. New model families require their own validated package; they are not arbitrary drop-ins.

The default remains the trained 0.6B package. Larger models, tiled vision, retrieval and learned preference ranking were studied separately and were not automatically promoted. Negative experiments and exact scopes are indexed in [README.md](README.md).

## Runtime

Native macOS selects MPS FP16 and merges LoRA in memory for inference. CUDA selects BF16 when supported, FP16 otherwise; CPU uses FP32. No hardware change rewrites saved adapter weights. Linux Docker cannot use the host Mac's Metal GPU. The interactive preview labels repeated-request cache hits; those timings are not fresh inference benchmarks.

## Training boundary

[TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) runs only the experimental VL-2B package from immutable, family-isolated, uniformly sampled v2 supervision plus source images. The completed 0.6B retraining was not promoted, and 0.8B is deferred. Line counts remain audit metadata. New adapters must pass raw and end-to-end comparisons before human review; training never replaces production automatically.
