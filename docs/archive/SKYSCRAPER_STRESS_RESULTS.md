> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

Historical initial stress run. For the newer capacity change and model comparison, see [capacity/model decisions](PROMPT_CAPACITY_AND_MODEL_DECISIONS.md).

# Supplied design and skyscraper stress evaluation — October 28

The unified viewer now contains 59 development conversions: the fixed 38,
eight earlier fresh independent conversions, and 13 additional tasks. Nine
of the new tasks target 160×600: four supplied designs, one existing portrait
alternate and four dense portrait validation designs. The supplied designs
also retain their four requested target sizes.

## Fixture provenance

The four supplied JSONs reference unavailable artwork and invented fonts.
With user authorization, derived test copies use illustrative local Crello
photographs, fixture-owned artwork and Solstice fonts. Text, geometry, stacking
and styling are retained. The festival original has duplicate IDs; its derived
copy renames only the duplicate occurrence. Tests still require the original
to fail validation. Original files and substitution hashes are retained locally
in `data/user_design_stress/`. Asset binaries are excluded from Git and the
review ZIP. They are not official samples or confirmed hidden-test designs.

## Fresh run and limitations

`data/user_design_stress/run_v2/report.json` records all 13 Qwen-enabled tasks.
There are no engine exceptions after the font-metric fix. Seven outputs have
automated flags: control readability, text/button clearance, collisions,
containment, glyph overflow, image overlap and axis alignment. All selected
winners are deterministic candidates.

Twelve tasks exceed the existing 4,096-token prompt safety limit (4,175–5,549
tokens). Qwen is enabled but explicitly declines those proposals. The remaining
21-element alternate generates a proposal with two recovered omissions, but
its deterministic candidate wins. These results do not demonstrate successful
learned reflow on dense supplied designs. The viewer exposes the actual status
under Audit details. No adapter training or model promotion occurred.

The supplied sources also contain intentional stressors: off-canvas or zero-size
elements, rotations, dense copy, long unbreakable strings, unsupported justify
alignment, mixed scripts and malformed style runs. Asset/font substitution
changes their visual appearance. The audit distinguishes authored source
problems from output flags rather than silently rewriting content to pass.

## Generic crash fix

One validation source triggered Pillow/FreeType's `execution context too long`
while measuring a glyph at the auxiliary 1000px reference size. That exceptional
path now uses the same Skia glyph advances as the renderer. Successful font
measurements retain their original behavior. A regression test exercises the
failure with styled text. Frozen replay also preserves a recorded no-proposal
outcome without manufacturing model coordinates or running inference.

## Next useful work

Measure a larger prompt/output budget against the base model's real context
capacity, or compact the prompt while retaining the adapter's trained contract.
Evaluate runtime and JSON completion before changing defaults. Then address
recurring narrow-format containment and readable-size failures across this
cohort and the wider benchmark. Zero flags on the other six tasks still does
not imply visual quality. Do not tune rules to these design names or IDs.

The verification suite passes 344 tests and 34 subtests. Frozen regression
comparison results are recorded in the unified viewer summary after replay.

## Visual review of all 13 additions

The supplied outdoor layout remains an undersized island; the conference
fallback separates its speaker cards from the headline and becomes illegible
at 320×50. The menu skyscraper detaches prices from their rows and moves the
heading away from the list even though automated flags are empty. The festival
skyscraper leaves excessive vertical gaps. The existing portrait alternate is
more coherent but still small. The dense resume and bar menu remain narrow
islands, the invoice fragments, and the planner separates some day labels from
their columns. These are unresolved layout defects, including false negatives
in the current metrics. Zero automated flags on six additions is not success.

All previous 46 selected JSON documents, winners and rendered JPEG bytes remain
identical, with no added automated flag types. The initial 59-case snapshot is
`d3a107ed30a3f68b`. Its completed engine results were reused for preview-only
zero-area handling after verifying identical engine/model/input fingerprints;
no new inference was substituted during replay.
