# Registered Sequence to Simulation Training

[Traditional Chinese](TRAINING_CONVERSION.zh-TW.md)

The `offline convert` command reads one registered garment sequence and its
SMPL-X pose files. Our worker extends ContourCraft's `utils/gaugar.py` converter:
it packages the meshes and body motion, transfers body skinning weights to the
garment, and uses the configured pretrained simulator to relax the unposed
garment. The resulting garment dictionary, motion arrays, registration targets,
and generated YAML are the inputs to `offline dynamics`.

The upstream implementation is
[Dolorousrtur/ContourCraft, utils/gaugar.py](https://github.com/Dolorousrtur/ContourCraft/blob/d606432c2f7e96b05664e6a4ffb9d1cb98705baf/utils/gaugar.py)
at revision `d606432c2f7e96b05664e6a4ffb9d1cb98705baf`.

The worker selects the requested sequence rather than treating Stage 2's
`Template/` directory as another motion. It preserves 45-dimensional hand
axis-angle poses; six-dimensional PCA poses are expanded with the selected body
model's hand components. Mesh and body frame IDs must match and be contiguous
from zero. Conversion refuses existing Stage 4 outputs or generated configuration.

Set `training_frame_rate` to the source capture rate and `validation_frames` to
the number of trailing frames reserved for validation in the deployment JSON.
The defaults are 30 fps and 10 validation frames. Both partitions need at least
three frames. The current ContourCraft dataset schema requires integer fps.
This temporal split checks rollout on a held-out part of the same motion; it does
not measure generalization to another actor or action.

Output locations:

- `<gg_output_root>/<subject>/stage4/`: full motion and mesh arrays, plus matching
  `_train` and `_valid` partitions.
- `<ccraft_data_root>/aux_data/garment_dicts/finetune/`: garment dictionary.
- `<ccraft_data_root>/aux_data/datasplits/finetuning/<subject>/`: split CSVs.
- `<contourcraft_root>/configs/finetune/<subject>.yaml`: training configuration.

The generated configuration is consumed by ContourCraft training. Motion sampling,
rollout length, iteration budget, and material settings determine the experiment;
conversion does not itself train the dynamics model.

For lower garments, use `--garment-type lower`. The worker reads waist indices
from `stage1/repair_quality.json`, validates the UV mesh hash and complete boundary
loop, and installs handles before relaxation. Generated training settings alternate
registration and physics updates on the training partition: defaults are 1000
updates, 8-step rollouts, checkpoints every 100 steps, and split repetition 100.
The process-local `cmu_npz_smplx_preserve` loader retains source poses and rejects
resampling/body randomization. Use `offline dynamics` so this loader is registered,
rather than calling upstream `train.py` directly. See the
[reproduction guide](REPRODUCING_REPAIRED_GARMENTS.md) for settings and repair inputs.
