# Review of Gemini's engine feedback

Reviewed against the actual `engine/resize.py`, `engine/main.py`,
`renderer/converter/convert.py` and `requirements.txt` on October 16, 2026.
The feedback was supplied by the user. Its production-grade assessment is not
established by this review; hidden-case quality remains unproven.

## 1. Rotated versus raw bounds — agree, fixed

The runner checks unrotated coordinates; the engine previously constrained only
rotated visual bounds. At some angles one envelope can fit while the other
extends outside. `_bounds()` now uses their union throughout packing, fitting,
projection, repair and final feasibility checks. This is conservative but keeps
explicit groups together and avoids a final per-element clamp that would break
relationships. The runner remains unchanged.

## 2. Kiwi import and rejected topology — partly agree, hardened

Pillow and `kiwisolver==1.5.1` were already explicit requirements. Build-time
installation is the normal supported deployment. A missing Kiwi import now
returns an unsuccessful projection, allowing feasible raw seeds and bounded
repair/fallback instead of crashing immediately.

Greedy separating-axis choices can reject a topology. This is not evidence that
all sample topologies were failing: the previous diagnostics recorded feasible
candidates for every supplied task. The engine now tries nearest separation,
center-directed axes, and alternate axes for overlapping pairs. It also retains
an already feasible raw seed when projection fails and attempts bounded repair
on up to three rejected seeds.

Separation stays required. Softening it without a hard final gate could select
an overlapping result. Bounded search is not a proof that every packable layout
will be found.

## 3. Implicit grouping — agree, narrowed

Infer a backplate/text relationship only when the rectangle contains exactly
that one text element and no other source element. Multi-text cards are not
implicitly grouped. Elastic label treatment is also limited to exactly one
shape and one text; explicit complex groups keep their source arrangement under
a common transform. An explicit text/icon group being rigid is intentional,
not itself a bug. More flexible card semantics would need explicit rules.

## 4. Renderer argument mismatch — disagree with diagnosis; hardening added

The actual renderer signature accepts `Union[str, Path]` for assets/output and
an optional `max_size` keyword. There was no argument-order or signature mismatch.
`Path` conversion is already performed by the renderer.

The optional diagnostic-render boundary now catches ordinary `Exception`,
records the error, and falls back to geometry ranking if all diagnostic renders
fail. Interrupts/shutdown (`BaseException`) still propagate. This isolates
third-party diagnostic failures; it does not suppress errors in core layout
logic or the runner's required final render.

## 5. Repairing infeasible starts — agree with the goal, change the mechanism

Previously `_polish()` only refined feasible starts. It now has a bounded repair
phase for rejected seeds. Repair prioritizes an explicit constraint-violation
measure including safety margins and required separation gaps. IoU/layout energy
alone is insufficient: two boxes can have zero overlap and still miss the gap.
Once a state becomes feasible, infeasible moves are rejected as before. Only
feasible repairs become ranked candidates; unrepairable seeds are discarded.
No guarantee is made that 60 iterations repairs every arrangement.

## Observed execution

- All three official tasks rendered successfully with no runner self-check
  warnings after these changes.
- A 90-degree tall artwork example fitted a 160 × 320 target with both bounding
  envelopes inside, and produced no runner warning.
- A near-touching pair's normalized constraint violation decreased from 0.005
  to zero during bounded repair.

These are concrete reproductions, not comprehensive hidden-case validation.
