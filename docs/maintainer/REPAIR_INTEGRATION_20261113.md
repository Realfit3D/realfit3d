# Repair and Contact Integration, 2026-11-13

Source snapshots are recorded in [repair_sources_20261113.json](repair_sources_20261113.json).
The integration began at main `fb289d1481a9274f32533a45b706a74d29b16a94` and GG
`cbfd9c33463b57af10d1192e5d009f733c3fbf8f`. Source attribution:

- ContourCraft, `utils/gaugar.py`, `datasets/sequence_loaders/cmu_npz_smplx.py`,
  `utils/validation.py`, and `utils/icontour.py`, revision
  `d606432c2f7e96b05664e6a4ffb9d1cb98705baf`: upstream conversion, pose-loader
  schema, rollout APIs and optional CPU-component patch locations.
- Gaussian Garments, reconstruction/appearance pipeline, upstream baseline
  `ceb66fc36598da05b3b3391f4562cdc54d2a2ed2`; maintained fork adds training I/O fixes.
- Project experiment snapshots in the JSON record: opening ellipse fit, ARAP,
  band reconstruction, local remeshing, body-supported prior, bounded repair,
  UV packing, waist fitting and contact projection.

## Module mapping

| Source behavior | Integrated entry |
| --- | --- |
| Actor05 opening/hem repair | `repair/geometry.py`, `repair/pipeline.py`, `configs/repair/actor05_*.json` |
| Actor08 bands and UV | Same repair pipeline with upper/lower recipes |
| Actor06/07 bounded repair | Same pipeline; Actor06 upper includes an explicit body-supported torso prior |
| Template installation | `quality/install_repaired_template.py`; matching colors/calibration and new subject required |
| Lower waist before relaxation | `repair/support.py`, `workers/contourcraft/convert_training_assets.py` |
| Registered-body preservation | Process-local `registered_sequence.py`, initialized by `runtime_entry.py` |
| Bounded training configuration | Conversion worker: per-sequence split, matching physics loader, update/rollout controls |
| Paired body contact | `contact.py`, `contact_projection.py`, `contact_rollout.py`, existing preview/library workers |
| Camera calibration patch | `adapters/stage1.py`, `patches/stage1-camera/` |
| Pascal graph fallback | `patches/contourcraft-cpu/cpu-graph.patch` |

Recipes preserve the investigated operations but are not byte-identical exports of
all historical candidates. Shared UV packing, strict OBJ position-index loading,
configuration-driven cuts, independent operation budgets and collision-preserving
clearance line search make the generalized path explicit. Actor06 upper uses a
body-supported topology prior; observed-surface offsets were not applied in its
selected historical v10 candidate because they introduced intersections.

OBJ import must preserve position indices across UV seams. Open3D's general OBJ
loader duplicated seam positions in the Actor05 source test, changing a connected
four-opening template into disconnected patches. The repair reader now parses
OBJ position indices directly, including negative indices, and rejects polygons.

## Validation and remaining acceptance

CPU tests cover dry-run behavior, output/hash protection, complete waist loops,
pose preservation, motion-rate rejection, contact feasibility/displacement limits,
UV correspondence, functional openings, and dimension-changing cuts. The geometry
suite uses synthetic three/four-opening surfaces. Actor05 source-frame 156 is also
used as a local numerical regression; results are recorded in the release log.
Actor08 fused sources and Actor06/07 complete raw inputs are not bundled in Git,
so those presets still require capture-specific regression runs.

No remote GPU training or video queue was launched for this integration. The
parameterized GPU conversion/rollout requires a configured-runtime smoke run;
historical experiment completion alone does not validate this refactoring.
Appearance quality, body contact, self-intersections and garment-to-garment
collisions remain separate acceptance categories.

## Recorded checks

[Validation results](repair_validation_20261113.json): all 84 primary test cases
passed across the Python 3.13 and Python 3.10 environments (geometry dependencies
were isolated in the latter). GG: 28 passed. Existing initialization-repair tests:
19 passed, 3 dependency skips. Browser-library JavaScript: passed. MoviePy 2.2.1
encoded and decoded a four-frame, 25 fps H.264 smoke clip.

Actor05 frame 156 repair produced one component, four simple openings, no detected
self-intersections/non-manifold/degenerate faces, and valid UV face correspondence.
Height extent decreased by 21.1 mm. Local body-normal measurements still counted
1693 vertices more than 2 mm inward; this is a geometry/UV regression pass, not a
body-contact or visual-quality acceptance. Input Stage 1/2 face correspondence
also passed on the recorded baseline pair.
