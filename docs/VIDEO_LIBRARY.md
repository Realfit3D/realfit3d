# Precomputed Video Library

[繁體中文](VIDEO_LIBRARY.zh-TW.md)

## Flow

A job file selects garments, body measurements, motion, and opacity. The CLI
expands those choices into videos, then uses the same packaged ContourCraft
worker as on-demand try-on: fit SMPL-X, prepare and untangle garments, rebind
them to the target body, optionally repair initialization, simulate, and render
with the pinned GG fork. It skips the preview turntable and publishes an MP4
plus a JSON record only after checking the rendered frame count.

`/library/` reads `manifest.json` and `catalog.json` to offer only available
combinations. Selecting opacity switches between pre-rendered V3 videos; it
does not execute a model in the browser. The original on-demand page remains
at `/`. Both pages are served by the same Flask process.

## Inputs and Configuration

Use the existing `configs/paths.local.json` runtime configuration and add:

```json
{
  "package_root": "/data/tryon-package",
  "library_root": "/data/realfit-videos"
}
```

These fields supplement, not replace, the GG/CC Python, checkpoint, body-model,
data-root and output-root settings in `configs/paths.example.json`.
Paths are relative to the configuration file unless absolute. Use dedicated
writable `gg_output_root`, `output_root`, and `library_root` directories outside
the input package. No installer copies code over GG or CC checkouts.

The input package has this layout:

```text
catalog.json                       # garments and motions: objects keyed by ID
motions/<motion-file>.npz
thumbnails/<garment-id>.png
garments/<garment-id>/
  stage1/template_uv.obj
  stage1/cameras.json
  stage3/ckpt/net.pt
  stage3/ckpt/info.txt
  cache/garment_dict.pkl
  cache/source_body_fit.pkl
  opacity_controller_v3/controller.pt  # required for selected V3 targets
```

Each garment catalog entry supplies `slot` (`lower`/`upper`), `source_gender`,
and `label`. Each motion supplies `file` relative to `motions/` and `label`.
Optional `heights`/`weights` lists contain `{id, label, value}`; job measurements
must match their representative values. Height is in **cm** for batch jobs,
converted to meters for the existing worker. This catalog is separate from
the list-based on-demand `configs/catalog.json`.

Raw delivery ZIPs are archival inputs, not this ready-to-run package: they do
not necessarily contain body-fit caches or motion clips. Preparing those inputs
is required before a video batch can run. Only use trusted pickle/checkpoint files.

## Commands

Edit the garment/motion IDs in `configs/library-jobs.example.json` to match the
package. Start with a short supported motion and a single body/outfit.

```bash
realfit3d --config configs/paths.local.json library render \
  --jobs configs/library-jobs.example.json

realfit3d --config configs/paths.local.json library render \
  --jobs configs/library-jobs.example.json --execute

realfit3d --config configs/paths.local.json library index --execute

REALFIT3D_CONFIG=configs/paths.local.json python -m apps.api.server
```

Without `--execute`, the CLI validates inputs, reads motion length using the CC
Python, hashes dependencies, and prints outputs; it does not fit, simulate,
render, or write assets. Open `http://127.0.0.1:5000/library/` after indexing.
Existing compatible video libraries can be indexed without rerendering.

The batch copies model inputs into `gg_output_root`, refusing conflicting
existing models. Mutable garment caches are **copies**, not symlinks into the
input package. Runs live in `output_root/precompute/<fingerprint>/`; trajectories,
repair reports, and rendered frames remain there for inspection. Body fits and
untangle results are reused within the same input/code fingerprint. This does
not interpolate between body heights or eliminate simulation of a new motion.

Re-running a batch skips completed videos only when parameters and dependency
fingerprints match. Failed attempts retain diagnostics and restart in a new run
directory. A library lock prevents concurrent batch publishers. There is no
cross-process GPU scheduler: do not run a batch and on-demand GPU job together.

## Optional Initialization Repair

Add `garment_repairs` to the runtime configuration:

```json
{
  "garment_repairs": {
    "Actor08_shirt": {
      "reference_dict": "/data/references/Actor08_shirt.pkl",
      "profile": "repair_profiles/upper.example.json",
      "genders": ["male", "female"]
    }
  }
}
```

The repair receives the rebound garment dictionary and a same-topology,
same-source-body reference. SMPL-X skinning weights identify anatomical
regions. Reference-guided restoration and a topology-based harmonic transition
adjust damaged initialization, then transfer body binding weights. Displacement,
edge strain, orientation, and configured clearance gates decide whether to
publish the candidate. Failure stops that video before simulation.

Repair changes only the run-local initialization/binding dictionary. It preserves
rest geometry, face indices, UVs, Stage 3 appearance weights, and source caches.
The previous dictionary and a JSON report are saved next to the repaired result.
Repair is disabled unless the garment and target gender are explicitly listed.
Included profiles are starting configurations, not validated defaults for every
garment. The same settings also apply to on-demand Web/CLI try-on.

## Opacity

`opacity_garments` selects which garment(s) receive the requested opacity. Omit
it to control the entire outfit. V3 requires each selected garment's own controller,
including for endpoint requests. The unselected garment stays unchanged.
The player labels target garments when multiple opacity variants exist.

GG captures reference Gaussian positions after loading Stage 3, before applying
any trajectory. V3 reads these fixed positions and the checkpoint normalization;
the rendered Gaussian positions still follow the simulated mesh.

Integration is covered by CPU command, file-isolation, API, and browser tests.
The combined repaired-cloth/V3 GPU pipeline still requires asset-level acceptance.
