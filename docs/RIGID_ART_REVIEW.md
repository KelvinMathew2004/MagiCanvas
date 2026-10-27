# Rigid artwork and glyph-to-art clearance

October 26, 2026. Historical checkpoint: `data/qwen_lora/art_proportions_verified`; subsequent code changes require fresh verification.

## Reported failures and causes

The official Level 2 Story candidate bypassed the older circle-only repair. Its two 1:1 source ellipses became 540×576 and 324×384. A shared pre-render policy now applies to raw Qwen, container-projected, deterministic and macro candidates. Assembly also enforces it, so a background-classified decorative shape cannot bypass the rule.

The Level 3 Square headline intersected the bottle by 614 actual Skia glyph pixels. This covered only 1.09% of the entire multiline headline, below the old 2% paragraph threshold. The peak line coverage was 1.51%. The detector now retains per-line coverage, with minimum painted-pixel support and comparison against source-supported intersections. The stricter line threshold only applies where the source had effectively no intersection; an existing frame/text intersection is evaluated against the aggregate source floor after wrapping. Repair uses the artwork's visible bounds, avoiding excessive movement caused by transparent image padding. The final new-contact repair runs after font calibration and before rendering/selection, without independently moving leaves of source-attached or macro-owned containers. No template ID or text content triggers either repair.

## Aspect policy

- Decorative shapes, ellipses, icons and foreground artwork preserve their source aspect ratio. A proposed box is a limit: the artwork fits inside it uniformly, retaining its center where legal.
- Generated resize candidates do not use raster `fill`. Foreground fill requests become `contain`; backgrounds use uniform `cover` cropping. An exact-size identity operation preserves an already-authored design. An absent background asset retains a non-stretching authored contain/cover mode. Graphic backgrounds use uniform cover cropping without creating or replacing image assets.
- Native rectangular text panels/buttons may change aspect to fit their copy. Native thin dividers and rectangular edge strips may extend along their layout axis. Rectangular canvas fills may cover the canvas. These are layout surfaces, not deformable artwork.
- Container projection uses the same source-evidenced elasticity policy. Merely having a shape in a group no longer makes that group elastic.
- Explicit source-supported overlaps remain permitted. This does not claim that every authored design is aesthetically ideal, or that background cropping understands subjects.

## Verification

Five targeted regressions cover artwork versus flexible panels, foreground image fit modes, assembly bypass protection, graphic-background proportions, and a first-line collision hidden by paragraph averaging. The full pytest suite passed 175 tests and 34 subtests after the source-overlap guard and background-fit regression checks. All 18 official cases were rerendered from frozen predictions and pass current hard/defect checks; both reported outputs were visually inspected. The broader frozen-prediction replay covered 159 outputs, including all 89 held-out cases and eight new-composition cases; no rigid-aspect violations or newly flagged defects remained. This is a mechanical check, not a guarantee of aesthetic quality. Before-and-after comparisons are recorded beside this checkpoint.

Replay keeps the adapter, prompt and decoded model coordinates fixed. Changed selection or geometry is engine behavior, not evidence of improved model inference. The intentionally missing Level 3 holiday asset remains missing and disclosed; it was not replaced. Historical reports remain available under their original checkpoint names.
