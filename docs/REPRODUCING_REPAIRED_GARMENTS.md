# Repaired Garments: Training and Inference

[繁體中文](REPRODUCING_REPAIRED_GARMENTS.zh-TW.md)

Stage 1.1 reads a reconstructed mesh, a capture-specific JSON recipe, and optional
source-body geometry. It rebuilds or smooths selected openings, repairs local
surfaces, checks the candidate, and generates UV coordinates. Stage 2 consumes the
reviewed mesh and UVs; Stage 3 learns appearance on that registered asset; Stage 4
fine-tunes one dynamics predictor. During try-on, optional contact correction acts
on the simulated positions before Stage 3 appearance rendering.

## Environments and assets

Use a private checkout and separate Python environments for reconstruction,
Gaussian Garments, and ContourCraft. Configure paths using
[`configs/paths.example.json`](../configs/paths.example.json). Licensed SMPL-X
models, raw RGB/masks/calibration, motion files, and model weights are separate
inputs; they are not included in Git. An MP4 alone cannot reconstruct training
inputs. Pickle/checkpoint files must come from a trusted source.

For the CPU repair environment (Python 3.10 on Linux):

```bash
python3.10 -m venv /your/work/repair-env
/your/work/repair-env/bin/python -m pip install -r requirements/repair.txt
/your/work/repair-env/bin/python -m pip install -e .
```

Open3D requires the platform EGL/OpenGL runtime libraries even for these CPU
operations. The tested dependency versions are in `requirements/repair.txt`.
Set `reconstruction_python` to this environment for repair. Reconstruction itself
also requires the dependencies of the pinned ActorsHQ checkout. The optional
calibrated Stage 1 projection patch is documented in [STAGE1_CAMERA.md](STAGE1_CAMERA.md).

Contact rollout also needs NumPy, SciPy and Open3D in the private ContourCraft
environment. MP4 assembly uses MoviePy 2's top-level `ImageSequenceClip`; install
`requirements/render-video.txt` in the private GG rendering environment and test
short-clip encoding before a long render. Geometry packages do not replace CUDA,
PyTorch, rasterizer, or collision-extension dependencies.

## Build and inspect a repair candidate

Recipes are versioned in [`configs/repair/`](../configs/repair). They encode the
inspected capture's Y-up coordinate frame in metres. Crop heights are settings,
not automatically estimated garment semantics. Preserve the source frame/body
alignment when using a preset.

| Preset | Input and operations |
| --- | --- |
| Actor05 upper | Original four-opening shirt; ellipse/ARAP opening fit, hem-band rebuild, local remeshing |
| Actor05 lower | Fused pants; waist/cuff cuts, three-opening preservation, bounded local fold repair |
| Actor08 upper/lower | Fused mesh and source body; band reconstruction, local remeshing and body clearance; upper preserves the raised collar/bent-arm geometry |
| Actor06 upper | Mesh plus matching female SMPL-X body and NPZ weights; body-supported sleeveless topology prior, then opening repair; occluded regions are inferred surfaces |
| Actor06 lower / Actor07 upper/lower | Existing template and source body; small-hole repair, bounded opening smoothing and body clearance |

These are parameterized versions of the experimental operations, not a promise
of byte-identical replay across geometry-library versions. They do not mirror
bent sleeves in world coordinates. Different captures may need different cuts,
opening selections, and displacement limits.

```bash
python -m realfit3d.quality.repair_template \
  --source /your/input/lower_largest.ply \
  --recipe configs/repair/actor05_lower.json --inspect

python -m realfit3d.quality.repair_template \
  --source /your/input/lower_largest.ply \
  --recipe configs/repair/actor05_lower.json \
  --output /your/work/actor05_lower_candidate
```

Add `--body /your/input/source_body.ply` for recipes with body clearance. Actor06
upper additionally requires `--body-model /your/models/smplx/SMPLX_FEMALE.npz`.
`--reference` optionally supplies the original shirt for hem dimensions.
`--inspect` is read-only. Candidate generation refuses an existing output folder.

Outputs are `template.obj`, `template_uv.obj`, `uv_diagnostics.npz`, and
`quality.json`. The report records source/configuration/code hashes, package
versions, opening counts, components, non-manifold structures, degenerate faces,
self-intersections, triangle quality, bounds, dimensional change, and UV face
correspondence. Failed candidates retain a diagnostic report and are not
installable. Nearest-vertex distances are labeled separately from displacement
because remeshing changes vertex identity. Cropping can shorten trousers; check
`dimension_change_m` rather than assuming original dimensions are preserved.

For CLI scheduling, configure a `template_repairs` entry:

```json
{
  "template_repairs": {
    "Actor05_lower_v2": {
      "source": "/your/input/lower_largest.ply",
      "recipe": "repair/actor05_lower.json",
      "output": "/your/work/actor05_lower_candidate"
    }
  }
}
```

Paths are relative to the deployment JSON. `offline repair --subject
Actor05_lower_v2` prints the plan; add `--execute` to build the candidate.

## Install the reviewed version and train

Inspect openings and dimensions, then install into a **new** GG output subject.
The installer checks output hashes and copies calibration/initialization colors
alongside the new mesh. It never overwrites an existing asset.

```bash
python -m realfit3d.quality.install_repaired_template \
  --candidate /your/work/actor05_lower_candidate \
  --destination /your/gg_output/Actor05_lower_v2 \
  --cameras /your/input/cameras.json \
  --point-cloud /your/input/point_cloud.ply --accept-visual-review
```

Prepare the matching RGB, garment masks, `cameras.json`, and SMPL-X sequence at
`<gg_data_root>/Actor05_lower_v2/Sequence1/`. Images, masks, body parameters and
registration outputs must use the same contiguous frame indices. Preserve an
index mapping when extracting a clip. The point cloud supplies initialization
colors; use the matching source capture. See the upstream
[GG data format](https://github.com/eth-ait/Gaussian-Garments/blob/ceb66fc36598da05b3b3391f4562cdc54d2a2ed2/DataPreparation.md).

Run these stages in order, inspecting existing outputs before execution:

```bash
python -m realfit3d.cli --config configs/paths.local.json offline register-template --subject Actor05_lower_v2 --sequence Sequence1 --frame 0 --garment-type lower
python -m realfit3d.cli --config configs/paths.local.json offline register-sequence --subject Actor05_lower_v2 --sequence Sequence1 --frame 0 --garment-type lower
python -m realfit3d.cli --config configs/paths.local.json offline appearance --subject Actor05_lower_v2 --sequence Sequence1 --garment-type lower
python -m realfit3d.cli --config configs/paths.local.json offline convert --subject Actor05_lower_v2 --sequence Sequence1 --frame 0 --garment-type lower
python -m realfit3d.cli --config configs/paths.local.json offline dynamics --subject Actor05_lower_v2 --garment-type lower
```

Commands above are plans; add `--execute` to the stage being run. Existing Stage 2/3
scripts retain upstream resume/overwrite semantics; the CLI does not prove that
an existing stage is complete. Review its checkpoint and log before rerunning.
Stage 1.1 does not add a new Stage 2 boundary loss. Inspect registered boundaries
again, since registration may introduce folds or self-intersections.

Set `training_frame_rate` to the capture rate (25 for these extracted Actor clips),
`validation_frames` to 10, and optionally `training_updates` (1000),
`training_rollout_steps` (8), `training_split_repeat` (100), and
`training_checkpoint_every` (100). Conversion uses the selected checkpoint for
initial relaxation and training restart, installs lower-waist handles before
relaxation, and validates the support report against the exact UV mesh hash.
It alternates registration updates with physics rollouts on the training split.
The process-local `cmu_npz_smplx_preserve` loader preserves captured body/hand
poses and rejects resampling or body randomization. Run `dynamics` through this
CLI so that loader is registered. See [training conversion](TRAINING_CONVERSION.md).

The optional CPU graph backend for Pascal hardware is a patch against
ContourCraft `d606432c2f7e96b05664e6a4ffb9d1cb98705baf`:

```bash
git -C upstream/ContourCraft apply --check ../../patches/contourcraft-cpu/cpu-graph.patch
git -C upstream/ContourCraft apply ../../patches/contourcraft-cpu/cpu-graph.patch
export REALFIT_CC_GRAPH_BACKEND=cpu
```

Apply it only in a private checkout. It replaces connected-component labeling;
CUDA collision extensions are still required. `check_checkout.py` reports this
intentionally patched submodule as dirty.

## Render with contact correction

Keep the existing `tryon` / `library render` flow. Set `checkpoint` to the selected
pretrained or Stage 4 checkpoint, and add this deployment setting:

```json
{
  "contact_correction": {
    "schema_version": 1,
    "relaxation_steps": 30,
    "lower_impulse_passes": 2,
    "max_waist_growth_ratio": 1.06,
    "projector": {
      "clearance": 0.003,
      "minimum_clearance": 0.002,
      "max_displacement": 0.035
    }
  }
}
```

Omit `contact_correction` for the previous rollout. The contact path requires
exactly two garments in lower→upper order, lower-waist HANDLE vertices, an SMPL-X
motion, and the configured target-gender body model. It fits waist targets,
separates the initial waist layers, performs static relaxation, then projects
predictions against local body-surface constraints with lower-only impulse
passes. Corrected positions and velocities feed the next step. Solver settings,
waist measurements, and per-step correction metrics are saved beside the
trajectory in `*.contact/`.

The complete outfit uses one dynamics checkpoint. Each garment retains its own
Stage 3 appearance model. For a Stage 4-only comparison, keep garment assets,
target betas, motion, contact settings, camera, opacity, and appearance weights
identical. Changing template repair and contact together compares the combined
pipeline. Contact constraints are local surface tests, not continuous collision
detection or a guarantee of no inter-garment/self intersections.

## Validation scope

CPU contract, contact-solver and synthetic geometry tests accompany the code.
Run `PYTHONPATH=src python -m unittest discover -s tests -v`; geometry tests require
the repair environment. Run `node --test tests/library.test.cjs` for playback.
The integrated GPU training/rollout path still requires a smoke run in a configured
private runtime before launching a new full training job. Record geometry,
collision, and rendered appearance results separately; a completed job is not a
visual-quality pass.

Implementation provenance and experiment-to-module mapping are in the
[maintainer record](maintainer/REPAIR_INTEGRATION_20261113.md).
