# MAGICANVAS

This is the scaffolding for your engine. It handles input loading, output
writing, rendering, and Docker packaging — so the only file you need to
edit is **`engine/resize.py`**.

```
input:  a task JSON (source design + target canvas)
              |
              v
        engine/resize.py   <-- YOU IMPLEMENT THIS
              |
              v
output: <task>_output.json  (schema-valid resized design)
        <task>_output.png   (rendered preview)
```

Both the JSON and the PNG are what get evaluated — see `RULES.md`.

## Team handoff

This README combines the starter instructions with our implementation and
development tools. The [original starter README](docs/STARTER_README.md) is also
preserved verbatim for reference. Official rules remain in [RULES.md](RULES.md),
[FAQ.md](FAQ.md), and the design schema; team implementation notes do not replace
those requirements.

See [REQUIREMENTS_AUDIT.md](REQUIREMENTS_AUDIT.md) for implemented requirements,
remaining quality gaps, viewer/finale work, and organizer confirmations.

Start with [IMPLEMENTATION.md](IMPLEMENTATION.md) for the engine approach,
prototype-to-Python mapping, preserved starter comments, validation results,
and limitations. Read [viewer/README.md](viewer/README.md) for both viewer flows.

The current engine is deterministic and offline. It uses no AI model, provider,
API key, or Project Chameleon runtime. The earlier prototype remains in its
separate repository; this repository runs independently.

## Repo layout

```
engine/
  resize.py         <- scored engine: analysis, reflow, text fitting, ranking
  main.py           <- orchestrator (input -> resize() -> output). Don't need to edit this.
renderer/           <- the real renderer (transformer-convertor + cr-renderer), already vendored
brand_kit/
  assets/           <- image files referenced by asset_id in the sample designs
  fonts/            <- font files matching the schema's font_family values
input/              <- drop task JSON files here for local testing (3 samples included)
output/             <- generated JSON/PNG results (Git-ignored)
tests/
  test_resize.py    <- contract and relationship regression checks
  test_theme.mjs    <- frontend palette, contrast, and color sampling checks
  test_viewer.py    <- native PNG export resolution regression
viewer/             <- separate local testing UI; not part of engine execution
  server.py         <- loopback preview server, calls the Python engine/renderer
  index.html        <- side-by-side original/adapted preview
  app.js
  theme.mjs         <- content-derived light/dark theme (frontend only)
  style.css
  README.md         <- viewer startup and Express workflow
  express/          <- separate minimal Adobe Express viewer add-on
    src/            <- manifest v2, panel HTML, SDK integration
    package.json
    package-lock.json
    tsconfig.json
docs/
  STARTER_README.md  <- unchanged original starter instructions
IMPLEMENTATION.md   <- engine approach, provenance, rules interpretation, limits
REQUIREMENTS_AUDIT.md <- supplied Adobe requirements and remaining work
chameleon_design_schema_v0.9.json   <- full field-by-field spec for the design format
Dockerfile
requirements.txt
RULES.md
FAQ.md
```

## Quick start (no Docker needed for iterating on `resize()`)

From the repository root, use Python **3.10 or newer**. Set up an isolated
environment once, then activate it in each new terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main
```

On Windows, activate with `.venv\Scripts\activate` instead. The original
`pip install -r requirements.txt` / `python -m engine.main` workflow still applies
after activation. Installing dependencies requires internet; engine execution
does not.

This processes every `*.json` file in `input/` (the 3 sample designs are
already there — one per difficulty level) and writes an `_output.json` and
`_output.png` for each into `output/`.

The original uniform-stretch baseline has been replaced in
`engine/resize.py` with an offline, deterministic candidate layout engine:
group-aware reflow, local font measurement, proportional artwork, flexible
text/backplate pairs, and geometric ranking. It retains the v0.9 input/output
contract and requires no model or API key. See [IMPLEMENTATION.md](IMPLEMENTATION.md)
for the approach, validation commands, and known limits.

For interactive testing, run `.venv/bin/python viewer/server.py` and open
http://127.0.0.1:5050. The separate [viewer](viewer/README.md) also includes a
minimal Adobe Express add-on for inspecting generated results. These optional
tools are outside the headless container path; the starter runner, renderer,
schema, input fixtures, requirements, and Dockerfile remain unchanged.

## Development and validation

Run from the repository root with the environment active:

```bash
python -m unittest discover -s tests -v
python -m engine.main
python viewer/server.py
```

The final command keeps the preview server running at http://127.0.0.1:5050.
It accepts challenge tasks `{source, target_canvas}` and standalone official
designs `{canvas, elements}`. Brand assets and fonts come from `brand_kit/`.
Changes to dimensions apply on Enter or field blur. Downloadable comparison
bundles include both rendered previews and the adapted JSON.

The tests use the existing engine dependencies. For additional schema validation
in development, optionally run `pip install jsonschema`; the tests detect it
automatically. This is not a new runtime requirement. Tests cover the three
samples across six target formats, source immutability, groups, text fitting,
image proportions, rotations, unusual content, and determinism.

### Inspect the layout engine

The engine now combines feature-based semantic roles, aspect-ratio layout
families, Cassowary bounds/separation constraints, deterministic multi-start
energy refinement, balanced typography, repeated icon/caption affinities,
background-busyness-guided candidates, and rendered contrast/busyness checks. It uses no AI model.
Artwork groups remain atomic; simple text/backplate pairs can reflow together.
The scorer is heuristic, so visual quality still needs review on varied designs.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for algorithm details and limits.

```bash
python scripts/inspect_layout.py input/chameleon_sample_level3_advanced.json
```

The output includes the design, PNG and a diagnostic report with roles, candidate
energies, the selected layout and actual timings. There is no fixed 80 ms claim.

For the Express viewer, use a separate terminal with Node.js/npm installed:

```bash
cd viewer/express
npm ci
npm run build
npm run start
```

Adobe's local server uses HTTPS and may require certificate setup. Follow
[viewer/README.md](viewer/README.md) for sideloading. `npm run package` produces
`viewer/express/dist.zip` when run from that directory. This viewer loads engine
results; it does not adapt live Express documents or import editable elements.
Its insertion action adds one flattened preview image.

## Working boundaries for teammates

- **Engine work:** keep the public `resize(source, target_canvas)` API and
  official v0.9 JSON contract. Helpers can live in `engine/resize.py`. Preserve
  every element ID, content, fonts, asset references, and group relationships.
- **Viewer work:** use `viewer/` for local preview and `viewer/express/` for the
  add-on. Both are outside the existing Dockerfile's COPY paths and entry point.
- **Starter plumbing:** the runner, renderer, schema, fixtures, requirements, and
  Dockerfile have not been rewritten. Coordinate changes to these files because
  they affect the evaluation contract. New engine dependencies, if needed later,
  must install during the Docker build and work offline at runtime.
- **Fixtures/assets:** keep the supplied samples intact for regression checks.
  The missing Level 3 holiday badge is intentional; do not invent a replacement.
  Do not tailor engine logic to known element IDs, sample text, or sample sizes.
- **Sharing:** share source files and the Express lockfile through Git. Virtual
  environments, `node_modules`, generated previews, Express build files, and the
  packaged ZIP are ignored and must be recreated on each teammate's machine.

The engine is an initial deterministic port, not a trained model. Very dense
content in a 728 × 90 banner still produces tiny text. The local viewer reports
small/unfitted text and missing assets. Automated checks and ranking are local
diagnostics, not the official evaluator. Local tests, real renders, an offline
Docker run, and Express packaging have passed; live Express sideloading and
insertion remain unverified.

The documents differ on viewer timing: the main brief places the full Express
add-on in the finale, while `FAQ.md` calls for an unscored stage-one viewer.
We included a separate minimal viewer to cover the stricter reading; confirm
the final submission checklist with organizers. Approved-model details and
hardware limits are still marked unfinished in the supplied documents.

## What the self-check does (and doesn't do)

`main.py` runs a quick structural check after every task — same element
`id`s as the source, required fields present, nothing off-canvas. **This
is not the official scorer.** It exists to catch obvious mistakes before
you waste a submission on them. See `RULES.md` for what's actually
checked and graded.

A self-check warning does not fail the run — your JSON and PNG are still
written either way. Read the warning; decide whether it's a real problem
or, like the Level 2 sample's intentionally corner-bleeding decorative
circle, an expected one.

## The renderer

`main.py` calls `renderer/converter/convert.py`'s `render_design_to_png()`
(a vendored copy of `transformer-convertor`, wrapping `cr-renderer`) to
produce each task's PNG — already wired in, no setup needed. It reads
fonts/assets from `brand_kit/` and writes a real rendered preview, not a
placeholder. If you ever see plain labeled boxes instead of real text and
images in `output/*.png`, that means the import in `main.py` failed (e.g.
`cr-renderer` didn't install — check `requirements.txt` installed cleanly)
and it silently fell back to the box-and-label placeholder; don't judge
your layout quality by that fallback.

## Docker (how you'll actually be run)

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  my-chameleon-engine
```

`--network none` matches how stage-one scoring runs: everything your
engine needs must be installed at **build** time (add it to
`requirements.txt`), not fetched at run time. If you're using a locally
hosted model, bake its weights into the image during the build step.

## See also

- `chameleon_design_schema_v0.9.json` — full input/output field spec
- `RULES.md` — what's checked, what's graded, and what disqualifies a submission
- FAQ — for anything not covered above
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — current engine, port provenance, and validation
- [viewer/README.md](viewer/README.md) — local and Express viewers
- [docs/STARTER_README.md](docs/STARTER_README.md) — original starter README, verbatim
