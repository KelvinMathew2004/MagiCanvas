# Dataset and training provenance

Source: CyberAgent's [Crello dataset](https://huggingface.co/datasets/cyberagent/crello), version 5.1.0. Local conversion scripts preserve structured text, asset references, geometry and the supplied preview separately. Credit the dataset authors. Images, extracted assets, fonts and full converted data remain local, ignored by Git and excluded from the Gemini review ZIP. The dataset card's terms and third-party asset rights still apply; a review packet is not permission to redistribute a dataset.

## Frozen pair export

`data/qwen_lora_expanded_v2` contains 1,122 training, 309 validation and 89 reference-test rows. Pair mining matches element inventories/content; connected design families define split isolation. Training is capped at one pair per target template ID. These are inferred correspondences, not independently human-verified resized labels. The 89 reference rows and 59-case viewer have been used during development and are not unseen benchmarks.

The current production adapter remains `models/qwen3-layout-lora-v2`. A matching 1.7B adapter exists locally, but its measured final-layout trade-offs did not justify making it the default. Public adapter upload is pending.

## Next controlled training run

Use [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md), not the archived Colab recipe.

`scripts/prepare_training_run.py` leaves the frozen source export unchanged and writes a separate bundle. It reconstructs the actual supervised target with source styles/assets, checks every train/validation row natively, verifies suspect invisible text with isolated/leave-one-out renders, and records line counts as metadata. Confirmed newly invisible training labels are excluded; severe new overflow is downweighted. Validation is preserved; test data stays outside the training ZIP. Dense rows receive extra sampler visits, not new unique target IDs.

Only 35 of the original training rows have ≥18 elements; repeat sampling cannot create missing layout diversity. A future expansion should add independently clean dense families while preserving family isolation. Do not use official, alternate-review or test layouts as training positives merely because they look correct.

## Human preferences

178 eligible comparisons were reviewed across the local preference campaigns. The image-embedding ranking head did not pass its promotion gate. A ten-feature structural ranker improved aggregate pair accuracy but did not improve the actual-engine validation subset enough to deploy. These preferences and consumed test results remain evidence, not fresh independent approval of a new selector.

Full historical counts, filters and experiments are retained in [the data history](archive/CRELLO_DATA_HISTORY_2026-10-28.md). Current raw/intermediate evaluation geometry stays in local audit JSON; public comparison pages show Original/Resized only.
