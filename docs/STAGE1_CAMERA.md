# Calibrated Stage 1 Cameras

Stage 1 reads calibrated multi-view images, trains a static Gaussian scene,
renders stereo pairs, estimates depth with DLNR, and fuses the masked depth
maps into a TSDF surface. The surface is remeshed into the garment template
used by registration.

The calibrated projection option carries each camera's `cx` and `cy` from
COLMAP into Gaussian training and stereo rendering. The rendered pixel
coordinates then agree with the calibrated intrinsics already used by TSDF.
The patch accounts for the rasterizer's half-pixel mapping and scales principal
points with resized images. Camera poses and focal lengths are unchanged.

## Install

Use a dedicated checkout for a new reconstruction run. The installer checks
both pinned repositories before applying changes and supports repeated runs.
Conflicting local changes or a different base commit stop installation.
Applied patches remain visible as local changes in Git; the source checkout
checker reports them as dirty. The installer dry run and recorded source hashes
verify this patch set separately from the unmodified dependency check.

```bash
python scripts/apply_stage1_camera_patch.py \
  --actorshq-root upstream/ActorsHQ-for-Gaussian-Garments
python scripts/apply_stage1_camera_patch.py \
  --actorshq-root upstream/ActorsHQ-for-Gaussian-Garments --apply
```

Add this setting to the existing runtime JSON configuration:

```json
{"stage1_projection": "calibrated"}
```

Then use the existing `offline initialize` and `offline reconstruct` commands.
The reconstruction plan adds `--GS_calibrated_principal_point` and checks the
installed source hashes before execution. Gaussian and stereo output names
include `_calibrated`. Set a separate `gg_output_root` when comparing runs:
the final `stage1/template.obj` path is still selected by subject and output root.

## Checkpoint Compatibility

The default remains `legacy` for existing runs. A calibrated reconstruction
requires a newly trained calibrated Gaussian checkpoint, not an old checkpoint
rendered with a different camera. The renderer checks the mode recorded in
`cfg_args` and rejects mismatches. This option affects the static Stage 1
Gaussian scene, not the Stage 3 appearance checkpoint or opacity controller.

This patch does not change segmentation, mask erosion, TSDF voxel size, or
remeshing. A sharper render does not by itself guarantee a valid mesh boundary.

## Sources

- [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments),
  revision `46651f55030407ea25463ed070eed9f7e837a9af`, `mesh_reconstruction.py`:
  forwards the calibrated mode to Gaussian training.
- The nested GS2Mesh checkout at `b08a68f2b896301472d25fe3f5287b5b1bf1b737`:
  `third_party/gaussian-splatting/{arguments,scene,utils}` carries principal points;
  `gs2mesh_utils/{argument_utils,renderer_utils,eval_utils}.py` handles the option,
  checkpoint-mode validation, and distinct output names.

Exact changes are in [the patch bundle](../patches/stage1-camera/).
See [the validation record](maintainer/STAGE1_CAMERA_20261110.md) for measured
image improvements and remaining mesh defects.
