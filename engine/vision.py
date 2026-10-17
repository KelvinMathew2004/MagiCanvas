"""Optional SigLIP perceptual re-ranker for already-valid candidate renders.

The engine generates several geometry-valid candidate layouts, renders the top
few to PNG, and picks a winner. This module adds a *learned* image/text judge
(``google/siglip-base-patch16-224``) to that final tie-break, as a closer proxy
for the AI evaluator that scores the rendered PNG than the deterministic
contrast/busyness heuristics alone.

Scope and guarantees:
- It ONLY reorders candidates that have already passed every hard constraint
  (bounds, non-overlap, element preservation). It never edits geometry, text,
  colors, or assets, and cannot add or remove elements. The scored v0.9
  contract is untouched.
- Every entry point degrades to ``None`` if torch/transformers or the baked
  weights are missing, so the deterministic heuristics remain the sole judge
  and the engine still runs under ``--network none`` with no model present.
- Weights are baked into the image at Docker build time (see
  ``scripts/fetch_siglip.py`` and the Dockerfile) and loaded with
  ``local_files_only=True``, so runtime needs no network access.

Inference runs on CPU in eval/inference mode, so repeated calls on the same
machine are deterministic (matching the engine's determinism contract).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


# Where the baked weights live. Overridable for local experiments; defaults to
# ``<repo>/models/siglip`` both locally and inside the container (WORKDIR /app).
MODEL_DIR = Path(os.environ.get(
    "MAGICANVAS_SIGLIP_DIR",
    Path(__file__).resolve().parents[1] / "models" / "siglip",
))

# Prompts framing "a well-adapted resized layout". SigLIP has a sigmoid head, so
# each prompt is scored independently; the quality signal is
# (mean positive logit - mean negative logit) per image.
_POSITIVE = (
    "a clean, professional graphic design layout with clear visual hierarchy",
    "a well-balanced marketing poster with readable text and tidy alignment",
    "a polished advertisement with elements spaced evenly and nothing cut off",
)
_NEGATIVE = (
    "a cluttered, messy design with overlapping elements and crowded text",
    "an unbalanced amateur layout with text running off the edge of the canvas",
    "a poorly aligned design with awkward empty gaps and misplaced elements",
)


@lru_cache(maxsize=1)
def _load():
    """Load SigLIP once. Returns ``(model, processor, torch)`` or ``None``.

    Any failure (missing deps, missing weights, incompatible files) returns
    ``None`` so callers transparently fall back to the heuristic ranking.
    """
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except Exception:
        return None
    if not MODEL_DIR.exists():
        return None
    try:
        model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
        processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model.eval()
        return model, processor, torch
    except Exception:
        return None


def available() -> bool:
    """True if the model and its weights can be loaded offline."""
    return _load() is not None


def _letterbox_square(image):
    """Preserve layout proportions before the processor's square resize.

    Use a sampled border median so the padding follows the render background.
    This preserves geometry, but does not establish aesthetic score accuracy.
    """
    from PIL import Image
    image=image.convert('RGB')
    width,height=image.size
    if width==height:return image.copy()
    border=[]
    for x in range(0,width,max(1,width//64)):
        border.extend((image.getpixel((x,0)),image.getpixel((x,height-1))))
    for y in range(0,height,max(1,height//64)):
        border.extend((image.getpixel((0,y)),image.getpixel((width-1,y))))
    color=tuple(sorted(p[k] for p in border)[len(border)//2] for k in range(3))
    side=max(width,height)
    canvas=Image.new('RGB',(side,side),color)
    canvas.paste(image,((side-width)//2,(side-height)//2))
    return canvas


def score_layouts(images):
    """Rank candidate renders by learned design quality.

    Args:
        images: list of PIL.Image candidate renders, in the caller's order.

    Returns:
        ``(scores, note)`` where ``scores`` is a ``list[float]`` the same length
        as ``images`` (higher = better looking), or ``None`` if the model is
        unavailable. ``note`` is a short status string for diagnostics.
    """
    if not images:
        return None, "No candidate renders to score"
    loaded = _load()
    if loaded is None:
        return None, (
            f"SigLIP unavailable (weights at {MODEL_DIR} missing, or "
            "torch/transformers not installed)"
        )
    model, processor, torch = loaded
    try:
        prompts = list(_POSITIVE) + list(_NEGATIVE)
        # One forward pass shares the identical prompt encoding across images.
        with torch.inference_mode():
            inputs = processor(
                text=prompts,
                images=[_letterbox_square(image) for image in images],
                padding="max_length",
                return_tensors="pt",
            )
            logits = model(**inputs).logits_per_image
            positive = logits[:, :len(_POSITIVE)].mean(dim=1)
            negative = logits[:, len(_POSITIVE):].mean(dim=1)
            scores = (positive-negative).tolist()
        return scores, f"SigLIP perceptual re-rank over {len(images)} candidate(s)"
    except Exception as exc:
        return None, f"SigLIP scoring failed: {exc}"
