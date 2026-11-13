# Stage 1 Camera Validation: Actor05

## Scope

We corrected the principal-point mapping in Stage 1 Gaussian training and
stereo rendering, then reconstructed Actor05's upper garment again. This change
belongs to the ActorsHQ/GS2Mesh dependency, not the Gaussian Garments Stage 3
appearance model. Installation and source revisions are in
[the camera guide](../STAGE1_CAMERA.md).

The comparison uses frame 156, 60 calibrated views at 746 x 1020 pixels, and an
RTX 4090. The corrected run uses the same input images and initial point cloud.
Through 30,000 iterations, the training change is the camera projection. The
60,000-iteration extension retains the original learning-rate schedule and
densification cutoff at 15,000 iterations.

## Image Results

All measurements below are averages over the 60 training views, not held-out
views. Person-mask scores cover the person, not only the garment. PSNR measures
agreement with source photographs; it does not measure mesh accuracy.

| Run | Full-image PSNR (dB) | Person-mask PSNR (dB) | Gaussians |
| --- | ---: | ---: | ---: |
| Original projection, 30k | 27.99 | 24.32 | 134,213 |
| Calibrated projection, 7k | 34.81 | 30.40 | 170,888 |
| Calibrated projection, 30k | 40.28 | 35.83 | 290,641 |
| Calibrated projection, 60k | 41.83 | 37.36 | 290,641 |

Cam007's selected collar crop improves from 20.73 dB to 38.33 dB at the same
30k iteration count. The visible double edges and blurred face are substantially
reduced. The crop also includes adjacent skin and is not a garment-only metric.

Corrected training reaches the 30k snapshot in approximately 4m13s and completes
60k in 8m15s. These times exclude depth, segmentation, fusion, and remeshing.
Extending to 60k gives a smaller improvement than correcting the projection at
30k. We did not train an original-projection 60k control or measure held-out views.

## Mesh Results

For each corrected checkpoint, we regenerated stereo images, DLNR depth, and
DINO/SAM2 masks before TSDF fusion. We tested erosion kernel sizes 10 and 3 and
remeshing targets of 8k and 16k vertices. Other reconstruction settings were held
fixed. Nonmanifold counts here mean edges shared by more than two triangles;
they are not a complete manifoldness or self-intersection test.

| Projection / iterations | Erosion | Vertex target | Nonmanifold edges | Within collar region |
| --- | ---: | ---: | ---: | ---: |
| Original / 30k | 10 | 8k | 54 | 8 |
| Calibrated / 30k | 10 | 8k | 2 | 2 |
| Calibrated / 30k | 10 | 16k | 4 | 2 |
| Calibrated / 30k | 3 | 8k | 1 | 1 |
| Calibrated / 30k | 3 | 16k | 4 | 3 |
| Calibrated / 60k | 10 | 8k | 1 | 0 |
| Calibrated / 60k | 10 | 16k | 3 | 2 |
| Calibrated / 60k | 3 | 8k | 1 | 0 |
| Calibrated / 60k | 3 | 16k | 3 | 1 |

The collar region is defined in this dataset's coordinates by y > 1.48 m and
abs(x) < 0.18 m, with both edge endpoints inside. Corrected raw and cleaned TSDF
meshes have zero edges with incidence above two; the remaining such defects
appear during remeshing. Collar and hem irregularities remain visible even when
the collar's nonmanifold-edge count is zero. No candidate is accepted for Stage 2.

## Validation and Next Check

- Six GPU/numerical checks passed: native/half/quarter-resolution projection,
  unchanged legacy projection, invalid parameter rejection, and CUDA rasterizer
  centroid/backpropagation. Maximum projection error was 0.000031 pixels;
  rasterized centroid error was 0.00168 pixels.
- RealFit3D's 57 CPU tests passed, including five new installer/plan tests.
- The installer dry run passed against the pinned dependency checkouts. The GPU
  experiment invoked the modified reconstruction components directly; the full
  RealFit3D offline CLI reconstruction has not been rerun with this patch bundle.
- SHA-256 checks confirmed that the seven protected baseline files, including
  original checkpoints, camera files, and template, were unchanged.

Use corrected 30k training as the next comparison baseline. Retain 60k as a
quality candidate, not a universal default. Inspect the raw TSDF boundary and
the affected remeshing edges before registration; increasing vertex count did
not remove the defects in this experiment. Do not replace the accepted asset
catalog with these experimental templates.

## Artifacts

Local experiment directory:
`actor05_validation_108_20261110/camera_fix_20261110/` in the workspace alongside
RealFit3D. It contains scripts, logs, `projection_tests.json`,
`baseline_integrity.json`, and `evaluation/metrics.json`. Images are
`evaluation/comparison.png` and `evaluation/summary_mesh_comparison.png`.

Server 108 experiment directory:
`/home/vtongroup/lin/actor05_upper_full_20261110/camera_fix_20261110/`.
Training weights remain in `models/calibrated_60k/` there. Reconstruction outputs
are under `/mnt/nas/vtongroup/lin/actor05_upper_full_20261110/camera_fix_20261110/`,
in `iter_30000/` and `iter_60000/`. No large weights were copied to the local PC.
