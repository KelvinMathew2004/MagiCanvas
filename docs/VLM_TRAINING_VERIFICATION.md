# VL training verification — October 29, 2026

The production champion is identified exactly. The VL package is verified as
an image-conditioned training experiment; improved layout quality is not yet
proven. Production remains on the existing adapter.

## Historical identity

The weight file added in Git commit `ae0fe91` matches the installed champion
byte for byte: SHA-256
`6ebbbe8d20e88504a1ecbb9fb6d5c9466df1c111bde84f099a6f788089d3c63b`.
The config and prompt-version marker also match. Both historical notebooks and
exporters are preserved in [the recovered history](archive/champion_training_history/README.md).
The exact executed notebook, uploaded dataset and original package versions
were not recorded. Finding the correct weights does not resolve those missing
run details or establish the cause of the rejected retraining's regression.

## Frozen VL bundle checks

Run `.venv/bin/python scripts/verify_vlm_training_bundle.py` locally. Results
are in `data/training_next/vlm_full_confirmation.json`; this checks the actual
ZIP and every training/validation example using the locally installed pinned
model's processor, not a mock tokenizer.

- Archive SHA-256 remains
  `fbe34f10f418cf3d738af84927beca348eeb08c1ac15d44aa363e887298b513c`.
  Archive contents equal the hashed extracted bundle and current trainer/provider.
- All **1,043 training and 309 validation rows** have unit loss weights and one
  sampler visit. Training has 1,043 distinct target IDs. Validation retains its
  original 309 directions across 109 targets; it does not supply gradients.
- All **971 source PNG mappings** are checked against source JSON and the
  preparation audit fingerprint. No target image is supplied to the model.
- Actual visual patch tensors, image-grid metadata and masked image tokens
  survive collation. Images contribute 69–256 visual tokens per example.
- Full answers plus EOS are supervised; system/user/image tokens are masked.
  Exact element aliases are retained. No sequence is truncated: maximum full
  lengths are 5,003 training / 3,751 validation, below the 6,144-token limit.
- Training, validation and the 89-reference test split have disjoint template
  endpoints; train/validation family isolation is also checked.
- The actual 2B architecture has all seven language projection types covered
  in each of its 28 layers: 196 LoRA target modules. Vision modules are excluded.
- The 26 focused training/continuation tests pass, including an image-conditioned
  tiny-model backward pass, agreement with native response loss, nonzero LoRA
  gradients and frozen vision weights. Changing pixels while keeping text fixed
  changes answer logits in that executable test. This is not a full 2B CUDA run.

The image/grid inputs follow the [Transformers Qwen3-VL interface](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).
Qwen's [fine-tuning reference](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
supports selecting language versus vision training; it also stresses that image
resolution affects performance. These references support the architecture
choices, not a claim that our hyperparameters are optimal.

## What remains unproven

This is a T4-budget QLoRA pilot: frozen pretrained vision, language LoRA, a
262,144-pixel image-area cap, and source renders capped at a 768-pixel long edge.
It preserves pretrained visual input and DeepStack processing, but fine print
can lose detail. It is not maximum-resolution or full-parameter fine-tuning.
The Crello-only training set has limited dense examples and no native shape
elements. Uniform sampling removes a confound; it does not prove that sampling
caused the old retraining's failure. Retaining screened exclusions is also a
data-policy choice, not a demonstrated optimum.

Actual full-model CUDA forward/backward/optimizer feasibility is checked in
Colab before training. Train from the fresh pinned VL base, save checkpoints,
then compare against the frozen champion on raw IDs, geometry/font errors and
end-to-end renders. Follow [the post-training gates](POST_TRAINING_EVALUATION_PLAN.md).
Do not promote on lower validation loss alone. No new adapter is guaranteed
regression-free, and the existing production adapter will not be replaced
unless those gates and visual review pass.

The runbook, notebook and private training ZIP are unchanged by this verification.
