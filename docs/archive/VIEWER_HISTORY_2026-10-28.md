# Layout lab and Express viewer

These viewers are separate from the headless engine. Neither is required to run
`python -m engine.main` or the existing Docker container. No editor or model code
from Project Chameleon is used here.

## Local layout lab

From the MagiCanvas root, with the engine requirements installed:

```sh
.venv/bin/python viewer/server.py
```

Open http://127.0.0.1:5050. Use `--port 5051` if 5050 is already occupied.

- Use **Demo** for the supplied challenge fixtures and brand-kit assets. Use
  **Test** to browse every compatible design JSON under `data/`; dataset asset
  folders are resolved alongside those designs. The selector lists each JSON
  by filename. You can still import a standalone v0.9 task
  `{source, target_canvas}` or design `{canvas, elements}`.
- Choose a format or enter width/height. Changes apply on Enter or field blur.
- Click either preview for a lightbox with fit-to-window and actual-pixel views.
  The viewer prepares a separate detail render on demand, aiming for a 1080px
  short side when the format allows it. It caps detail renders at 40 million
  pixels and 24,000px per side; very extreme aspect ratios may stay below the
  short-side target. Close with Escape, the close button, or the backdrop.
- Compare original and adapted previews from the **same real renderer** used by
  `engine.main`. Failures are shown explicitly; there is no box fallback here.
- Download the exact output JSON, native-resolution PNG, or a comparison bundle
  for Express. Local exports bypass the renderer's default 2048px preview cap;
  the submission runner and converter remain unchanged. A 728 x 90 target still
  exports at 728 x 90, not an artificial upscale.
- The lightbox's detail render is temporary and separate from exported files.
- Demo assets resolve from `brand_kit/assets`. Test assets resolve from the
  matching dataset folder under `data/` (for example,
  `data/crello_sample/assets`). Asset paths in downloaded JSON remain relative
  and portable. Test mode uses `data/crello_fonts.pickle` when present; this is
  the pinned Crello v5.1 font resource and is ignored by Git with the rest of
  `data/`. Otherwise the viewer falls back to the brand-kit fonts. The viewer
  does not fetch fonts from the network.

The local server listens only on loopback, writes previews in temporary folders,
and doesn't change the source, sample inputs, brand kit, or output fixtures. It
uses standard-library HTTP serving, not a new backend framework. Browser preview
limits (12,000px per side / 40 million pixels / 500 elements / 4 MB task JSON) are
resource guards for development only, not engine constraints.

## Repeated-request latency and timings

The preview keeps up to eight exact results / 32 MiB in memory. A repeat uses
that exact prior Qwen-enabled result and is explicitly labelled **cached previous
result**; it is not described as fresh inference. Keys bind source, target,
assets, fonts and model configuration. Changes invalidate the result; failed
requests are never cached. Concurrent preview computations are serialized so
one local model/GPU is not oversubscribed. This does not parallelize tokens or
make a new target size instantaneous.

Fresh status shows model time separately from layout/verification. Timings are
also in response JSON (`timings`, `inference`, `cache_hit`, `computed_seconds`).
For generation benchmarks set `MAGICANVAS_PREVIEW_CACHE=0`. Weights/fonts/code
remain loaded in process: restart the server after replacing them.

Native Metal automatically uses an in-memory LoRA merge. Eight fresh independent
conversions retained identical final outputs while taking about 8.6% less time
in one M1 Pro comparison. Five raw completions changed, so this is not a bit-exact
model equivalence promise. `MAGICANVAS_QWEN_MERGE_ADAPTER=0` disables merging for
comparison; `=1` explicitly enables it elsewhere. Downloaded weights stay unchanged.
See [full measurements](../docs/CHAMELEON_ALGORITHM_UPGRADES.md).

## Content-derived appearance

The viewer starts with neutral light/dark surfaces and follows the operating
system's appearance. A background rectangle's opaque `fill_color` supplies the
color seed when available. For image backgrounds or other compositions, a small
in-browser sample of the real source preview finds the dominant color, weighting
edges more strongly so headlines and central artwork have less influence.

OKLCH tones derived from that seed supply accents and a restrained surface tint.
There are no sample names, asset filenames, or predefined design palettes in the
selection logic. Foreground/action contrast is checked; grayscale sources remain
neutral. Colors transition over 650ms when designs change. Reduced-motion users
receive immediate updates. Failed sampling keeps the previous or neutral theme.
This changes only viewer styling, never the design JSON or rendered assets.

Theme logic is in `theme.mjs`. Run its focused tests with:

```sh
node tests/test_theme.mjs
```

Run that command from the repository root. It checks light/dark contrast,
background selection, image precedence, and edge-weighted pixel sampling.

References:

- https://linear.app/now/how-we-redesigned-the-linear-ui
- https://m3.material.io/styles/color/the-color-system

## Adobe Express viewer add-on

`express/` uses Adobe's official JavaScript add-on tooling and manifest v2.
It loads locally generated results; it never calls an external AI provider or
requires a local HTTP engine connection from the Adobe iframe.

```sh
cd viewer/express
npm ci
npm run build
npm run start
```

Adobe's local add-on server uses HTTPS. Follow the CLI's certificate setup when
running it for the first time; certificate trust changes need your participation.
Enable Adobe Express Add-on Development, then use **Test your local add-on** to
connect to the URL printed by the CLI (normally https://localhost:5241).

In the panel:

1. Load `magicanvas-viewer-bundle.json`, exported by the local layout lab, to see
   both previews, dimensions, notes, and the official output design JSON.
2. Alternatively, select a matching `_output.json` and `_output.png` together.
3. Optionally insert the adapted preview into Express. This inserts one flattened
   PNG, not editable scene elements. This is a viewer, not the full finale editor.

Package for sideloading/submission with:

```sh
npm run package
```

The Express host/SDK requires Adobe and a browser session. The **engine** remains
fully offline; the add-on is not a way to bypass the offline engine requirement.
A successful build validates the package, not an in-Express acceptance test. The
panel still needs a live Adobe Express sideload and insertion check by the team.

Official references:

- https://developer.adobe.com/express/add-ons/docs/guides/getting-started/hello-world
- https://developer.adobe.com/express/add-ons/docs/references/manifest/
- https://developer.adobe.com/express/add-ons/docs/guides/learn/how-to/use-images
