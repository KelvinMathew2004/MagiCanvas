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


def _relationship_prompts(context):
    """Turn engine-inferred relationships into visual judging cues."""
    if not context:
        return (), ()
    roles=set(context.get('roles',()))
    positive=[]
    negative=[]
    format_prompts={
        'banner':('a compact horizontal banner with a readable inline sequence and balanced breathing room',
                  'a tall stacked poster compressed into a short narrow banner'),
        'wide':('a wide landscape advertisement with copy and artwork in balanced columns and consistent outer padding',
                'a wide landscape advertisement with awkwardly scattered sections and uneven gaps'),
        'square':('a balanced square social graphic with a compact clear hierarchy and even margins',
                  'a square graphic with content pushed to an edge and large accidental gaps'),
        'portrait':('a portrait advertisement with a coherent vertical reading flow and comfortable spacing',
                    'a portrait advertisement with its call to action stranded at the bottom far from related copy'),
        'story':('a tall story layout with an intentional vertical sequence and balanced negative space',
                 'a tall story layout with content awkwardly spread across excessive empty space'),
    }
    if context.get('bucket') in format_prompts:
        good,bad=format_prompts[context['bucket']]
        positive.append(good);negative.append(bad)
    if context.get('density')=='dense':
        positive.append('a content-rich design that preserves legibility by grouping related details and protecting the main headline')
        negative.append('a content-rich design with tiny crowded text and unrelated elements competing for attention')
    if 'logo' in roles and roles & {'headline','subheading','body'}:
        positive.append('the brand logo remains visible and distinct while the headline and supporting copy keep a clear reading hierarchy')
        negative.append('the brand logo overwhelms the headline or is lost among the supporting copy')
    if roles & {'headline','subheading','body'} and 'cta' in roles:
        positive.append('the headline, supporting text, and call to action read in a clear order with comfortable spacing')
        negative.append('the call to action crowds or visually merges with the main text')
    if roles & {'hero','art'} and 'cta' in roles:
        positive.append('the product artwork and call to action are both prominent with clear separation')
        negative.append('the product artwork collides with or crowds the call to action')
    if context.get('grouped_units'):
        positive.append('related grouped illustration pieces stay together and read as one coherent visual')
        negative.append('pieces of one grouped illustration are scattered apart')
    quadrants=context.get('callout_quadrants',())
    for quadrant in quadrants:
        positive.append(f'the promotional callout remains anchored in the {quadrant} corner, outside the main reading flow')
        negative.append(f'the promotional callout is displaced from the {quadrant} corner into the main content')
    return tuple(positive),tuple(negative)


def score_layouts(images, reference_image=None, layout_context=None):
    """Rank candidate renders by learned design quality.

    Args:
        images: list of PIL.Image candidate renders, in the caller's order.
        reference_image: optional render of the original design. Its visual
            embedding is used as a restrained identity/palette continuity cue.
        layout_context: optional compact role and relationship summary inferred
            by the deterministic engine. It conditions the image/text prompts.

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
        contextual_positive,contextual_negative=_relationship_prompts(layout_context)
        positive_prompts=list(_POSITIVE)+list(contextual_positive)
        negative_prompts=list(_NEGATIVE)+list(contextual_negative)
        prompts=positive_prompts+negative_prompts
        scored_images=list(images)+([reference_image] if reference_image is not None else [])
        # One forward pass shares the identical prompt encoding across images.
        with torch.inference_mode():
            inputs = processor(
                text=prompts,
                images=[_letterbox_square(image) for image in scored_images],
                padding="max_length",
                return_tensors="pt",
            )
            output=model(**inputs)
            logits=output.logits_per_image
            if reference_image is not None:
                logits=logits[:len(images)]
            positive=logits[:,:len(positive_prompts)].mean(dim=1)
            negative=logits[:,len(positive_prompts):].mean(dim=1)
            quality=(positive-negative)
            reference_similarity=None
            image_features=getattr(output,'image_embeds',None)
            if reference_image is not None and image_features is not None:
                normalized=torch.nn.functional.normalize(image_features.float(),dim=-1)
                reference_similarity=(normalized[:len(images)] @ normalized[-1]).clamp(-1,1)
                # Similarity is deliberately a modest tie-breaker: layouts
                # should retain brand/art identity without copying the source
                # composition into a very different aspect ratio.
                quality=quality+.35*reference_similarity
            scores=quality.tolist()
        note=f"SigLIP judged {len(images)} candidate(s)"
        if contextual_positive or contextual_negative:
            note+=f" with {len(contextual_positive)} inferred relationship cue(s)"
        if reference_similarity is not None:
            note+=" against the original design"
        return scores,note
    except Exception as exc:
        return None, f"SigLIP scoring failed: {exc}"
