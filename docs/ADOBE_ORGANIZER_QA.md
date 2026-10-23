# Adobe organizer Q&A notes

**Date recorded:** October 22, 2026  
**Source:** Team notes from a Q&A session with Adobe organizers. These are
reported verbal clarifications, not a replacement for the challenge brief,
official rules, or later written updates. If wording conflicts, ask Adobe which
instruction governs before submission.

## Model and evaluation guidance

- The organizers did not state a numeric model-size ceiling in this Q&A. They
  said a model of any size may be considered if its weights are available or
  downloadable from Hugging Face. The model still needs to operate smoothly on
  the stated M5 Mac evaluation hardware; exact RAM and runtime limits were not
  provided.
- A large model or extensive fine-tuning is not expected given the project
  timeline. A model-swappable design is a bonus, so keep model loading behind a
  small interface and make the no-model path usable.
- For each test aspect ratio, the judge LLM has a reference design and compares
  the engine output against it. Human evaluation of visual appeal follows.
  Flexibility, including the ability to swap models, and the size limit are also
  part of the evaluation discussion. No numerical weights or score formula
  were provided.
- The team has observed that Qwen-enabled layouts improve on the Qwen-off
  engine in many reviewed cases, but still fall short of the original/reference
  designs. Treat that as a current internal observation, not an Adobe score.

## Visibility, crop, and bleed guidance

- Preserve every design element and keep its meaningful visible content
  visible. Intentional bleed beyond the canvas is acceptable when the design
  calls for it. Transparent padding around a PNG should not be mistaken for
  visible artwork when deciding whether content is clipped.
- Keep the distinction between an image element's rectangular frame and its
  alpha-visible pixels. Do not crop meaningful opaque pixels just to make the
  frame fit. Conversely, a frame extending beyond the canvas because its PNG
  has transparent margins does not by itself mean that visible content is lost.
- The supplied stage-one materials also describe an on-canvas structural
  check. Until Adobe confirms how intentional bleed is handled by that check,
  preserve visible pixels and inspect both the JSON bounds and alpha-visible
  bounds when reviewing any intentional bleed case.

## Judge-facing README

Keep the root README short and procedural: prerequisites, one local run path,
one Docker build/run path, which model weights are fetched at build time, how to
enable/disable optional models, and where outputs appear. Put architecture,
training, and dataset details in linked docs. The container should need no
network access at runtime; model weights and dependencies must be present at
build time.

## Dates and team follow-ups from the Q&A notes

- Winners are expected to be notified by November 23, 2026.
- Add Ajith as a repository collaborator.
- Ask Treverus for general feedback by October 25 or 26, 2026; the feedback is
  not expected to be design-specific.

These are team action items, not judge instructions.
