# Container flow: research, failure analysis and implementation decisions

October 27, 2026. This note records a targeted investigation, not a claim that every design is solved. Acceptance results below refer to the fingerprinted October 27 run. The findings support a bounded engine change, not universal visual perfection.

Latest follow-up: [peer-row shared-axis alignment](PEER_ROW_ALIGNMENT.md) records the later 277-test run and refreshed official outputs. The container-reflow benchmark counts below describe the preceding checkpoint.

## What the reference systems actually establish

CSS Flexbox supports directional flow, wrapping and flexible sizes, but its default non-scrollable items have content-based minimum sizes. Flex shrink is not permission to reduce every item without regard to its contents. The useful transfer is to measure content before allocating space, then choose a feasible arrangement. A fixed raster canvas has no guarantee that every required item fits at a desired font floor. [W3C Flexbox, sections 4.5, 6 and 9](https://www.w3.org/TR/css-flexbox-1/).

CSS Grid derives intrinsic track sizes from the participating items and incorporates gutters in sizing. A fixed number of equal cells chosen only from the canvas aspect ratio is not equivalent to that algorithm; words, media aspect and padding still constrain cell sizes. [W3C Grid, sections 12.3–12.5](https://www.w3.org/TR/css-grid-2/).

Figma distinguishes fixed size, content hugging, filling available space, and minimum/maximum dimensions. It also explicitly allows objects to ignore auto layout so overlays can remain positioned relative to a parent. Nested frames may use different flows. This supports retaining an atomic badge/photo/card interior while reflowing its siblings; it does not support forcing every graphic layer into normal flow. [Figma auto-layout guide](https://help.figma.com/hc/en-us/articles/360040451373-Guide-to-auto-layout-in-Figma).

O'Donovan, Agarwala and Hertzmann's work combines alignment detection, perceived importance and hierarchical segmentation with an optimized layout energy. It supports joint structural analysis and scored alternatives, not a claim that a short universal formula eliminates all graphic-design defects. [Project and paper](https://www.dgp.toronto.edu/~donovan/layout/), [paper PDF](https://www.dgp.toronto.edu/~donovan/layout/designLayout.pdf).

These references inform engineering choices. MagiCanvas does not implement the complete CSS or Figma algorithms, and the paper is not evidence that our thresholds are optimal.

## Findings against the actual code

| Proposed change or claim | Verified current behavior and decision |
| --- | --- |
| Infer a macro box from Qwen leaves | `project_container_tree` already estimates a source-relative transform from substantial members using weighted medians, excluding tiny rules from the fit. Keep this robust estimate; a raw union box is sensitive to one noisy leaf. |
| Recover 1–3 missing IDs | `generate_seed` already allows up to `max(2, int(0.20*N))` omissions, provided some valid proposals exist. `_missing_proposal` chooses a matched explicit group, containing parent or nearby anchor, otherwise the supplied fallback. Recovery is useful but not risk-free. |
| Mark a recovered raw proposal valid | Reject this accounting change. The generation diagnostics preserve the originally missing IDs. Evaluation requires complete raw IDs when reporting `raw_proposal_valid`; repaired completion must not be credited to the model. |
| Replace every deterministic seed with one realizer | Reject a wholesale replacement. Add independently ranked flow candidates and preserve existing source-backed interiors and established working candidates. |
| Give every shape that overlaps text ownership | Reject. Incidental translucent paint can cross several otherwise unrelated containers. Explicit grouping and strict containment provide stronger evidence than overlap alone. |
| Fix tiny results solely with more occupancy weight | Existing costs already include utilization, font scale and attachment terms. The investigated failures require correcting source decomposition and candidate availability before increasing a global weight. Bounding boxes alone can be inflated by detached decorations. |
| Switch to 1.7B because one font metric improved | Do not promote automatically. The recorded 89-case A/B in `docs/DAY4_5_AUDIT.md` had geometry MAE 0.06347 versus 0.06155, font MAE 0.20039 versus 0.17535, and Qwen-family wins 65.2% versus 67.4% for 1.7B versus 0.6B. Those historical results did not pass the promotion gate; they are not a fresh evaluation of this revision. |

No retraining, output-contract migration or adapter promotion is necessary to test the diagnosed engine changes. A new public model configuration still requires its own runtime and held-out promotion check.

## Four frozen baseline investigations

The baseline records are under `data/qwen_lora/universal_reflow_research/before/`. `code.json` records engine hashes. The two poster cases used actual local Qwen inference; After Dark and Winter reuse exact recorded Qwen coordinates. The replay's one-token stub is not the original inference prompt length or a runtime measurement of model generation.

| Case | Baseline result | What the record establishes |
| --- | --- | --- |
| Stacked poster, 1600×400 | `readable-text-floor-editorial-copy-first-0.3`; median font 9.48px; no automated flags | Qwen had a 3,360-token prompt, completed JSON and did not hit its token limit. Two omitted labels were recovered from their panel anchors. Qwen was not absent because of those two omissions. The macro pass was considered but recorded no rendered macro alternatives. |
| Stacked poster, 728×90 | `deterministic-aspect-reflow-editorial-copy-first-0.45`; median font 2.28px; no automated flags | Qwen returned complete IDs. Main content occupied too little useful area, although separated corner material made its width/height envelope appear large. Zero flags did not mean usable typography. |
| After Dark, Story | `source-attached-proportional`; width use 0.910, height use 0.517; no automated flags | One label was recovered from its source badge. The preserved square composition used only about half the tall target's height. The earlier macro opportunity gate did not trigger a render sweep. |
| Winter, Square | `source-attached-proportional`; width use 0.680, height use 0.940; no automated flags | Three IDs were recovered. The aspect change is moderate and the source-derived quality cost is already low. This is a preservation/control case, not evidence that all unused space should be filled. |

The poster's two tall rectangular rays have layer opacity 1.0 but fill `rgba(..., 0.2)`. The earlier overlap/grouping check considered layer opacity and only exempted certain ornament types. It therefore treated the rectangular paint as substantial source overlap and fused the headline, subhead and three cards into a hero unit. A fixed topology then shrank the stack to fit the target short side. This is a source-analysis bug, not a failure that more missing-ID synthesis would resolve.

## Bounded implementation

1. Compute effective paint alpha from layer opacity and RGBA fill alpha. Prevent ungrouped ambient paint from becoming an inferred atomic bridge. Explicit groups and enclosing copy panels remain stronger evidence; an actual translucent multi-text card must remain a container.
2. Keep ambient paint outside normal macro flow and exclude it from content-utilization measurements. Carry it with uniform scaling and canvas-relative placement; do not replace asset pixels or delete IDs.
3. Add intrinsic horizontal and vertical macro-flow candidates alongside existing band and media candidates. Each source-backed card, rotated label or media attachment remains atomic. Sibling movement changes topology; internal leaf offsets and aspect stay source-relative.
4. Extend the macro opportunity gate only for sufficiently changed aspect ratios with both low axis use and reduced relative type scale. Existing clean layouts still require a materially better, clean alternative before replacement.
5. Continue to render and measure the final realized geometry. A higher utilization score cannot excuse new clipping, text/control collisions or lost attachments. No selected geometry should mutate after verification.

This is not yet a general grid-divisor optimizer, nor a maximal-free-rectangle orphan solver. Those are possible later additive candidates, not missing-ID fixes claimed complete in this revision. A universal minimum pixel size can be physically impossible in a dense leaderboard; infeasibility must be reported honestly rather than hidden by arbitrary shrinkage.

## Acceptance and presentation

Final tests: **271 passed, 34 subtests passed**. The October 27 run is `data/qwen_lora/universal_reflow_research/accepted/`; its combined summary is `data/qwen_lora/universal_reflow_research/summary.json`.

- 89 held-out outputs: 89 passed the automated checks, zero engine errors. Of these, 88 reuse exact saved raw coordinates. One historical record had no raw prediction; a fresh 0.6B run completed JSON, omitted two IDs recovered by existing relation anchoring, and selected a deterministic output. That row is excluded from paired-model claims.
- On the 88 comparable rows, mean geometry MAE changed from **0.0449766 to 0.0449117** and relative font MAE from **0.1676861 to 0.1673475**. Both arms had zero automated flags. Five final geometries changed; each was rendered beside the historical output for inspection. These tiny aggregate differences do not establish a general improvement in model prediction.
- All 18 official combinations passed. Five geometries changed, chiefly peer placement; all changed outputs were visually compared with the previous checkpoint. Identity-sized tasks can bypass generation even though Qwen is enabled.
- All eight authored conversions passed. Seven are geometry-identical to the previous authored checkpoint; After Dark → Story selects the new intrinsic flow. The previously working After Dark leaderboard remains unchanged. All eight selected deterministic candidates despite Qwen being enabled; omitted-ID recovery does not imply the learned proposal won.
- Four focused checks passed. Stacked-poster median font increased from **9.48 to 14.95px** at 1600×400, and **2.28 to 5.46px** at 728×90. Occupied union area increased from 8.5% to 25.2%, and 7.7% to 29.9%, respectively. The leaderboard still has small text. After Dark retains nearly the same headline and photo scale while using more height, but its editorial balance remains imperfect. Winter Square is unchanged.

The comparator for the 89/18 cohort is the historical `art_proportions_verified` checkpoint, not an isolated immediate-parent revision. These repeatedly inspected cases are a development benchmark, not a blind generalization estimate. No Qwen-off arm was run and no adapter was retrained or promoted.

One defensive engine delta followed the batch freeze: ambient layers that are already backgrounds or edge strips now have only one emission path. `ambient_path_impact.json` scans source and ink-prepared forms for all 115 batch inputs and finds no affected layers. The run records both macro-module hashes and the exact batch module snapshot; the current full test suite includes the duplicate-path regression. The fresh missing-record run uses current code.

Visual review also rejected earlier prototypes. Bounding-box fill can reward overlapping or detached layers, so occupancy now uses rectangle union and excludes ambient paint. An independent median-font guard incorrectly rejected a much larger hero because annotation fonts were smaller. Admission instead permits hero/headline trade-offs, rejects joint primary-content shrinkage and substantial union-area loss for clean incumbents, and retains the existing defect and quality-margin checks. These are conservative heuristics, not proven optimal thresholds.

The user-facing review should show only original and selected resized output with Qwen enabled. Raw proposals, missing-ID recovery, per-candidate rejection diagnostics and previous outputs remain audit artifacts. A deterministic winner must be labelled as such; engine improvement is not evidence of better raw model predictions.
