"""Convert registered meshes and body poses into ContourCraft training assets."""

import argparse
import math
from pathlib import Path


def validate_sequence(
    data_root, output_root, subject, sequence, frame, validation_frames, frame_rate
):
    """Reject mismatched registration inputs before creating simulation assets."""
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("Frame rate must be finite and positive")
    if not float(frame_rate).is_integer():
        raise ValueError("ContourCraft requires an integer frame rate")
    bodies = sorted((data_root / subject / sequence / "smplx").glob("*.pkl"))
    meshes = sorted(
        (output_root / subject / "stage2" / sequence / "meshes").glob("frame_*.obj")
    )
    body_ids = [int(path.stem) for path in bodies]
    mesh_ids = [int(path.stem.removeprefix("frame_")) for path in meshes]
    if not body_ids or body_ids != mesh_ids:
        raise ValueError("Body and registration frame IDs must match")
    if body_ids != list(range(len(body_ids))):
        raise ValueError("Input frames must be contiguous and indexed from zero")
    if frame not in body_ids:
        raise ValueError("Template frame is outside the registered sequence")
    if validation_frames < 3 or len(body_ids) - validation_frames < 3:
        raise ValueError("Reserve at least 3 training and 3 validation frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "gg-data-root",
        "gg-output-root",
        "body-model-root",
        "checkpoint",
        "garment-dicts",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--support-report",
        type=Path,
        help="Stage 1.1 quality report containing matching waist indices",
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--gender", choices=["male", "female"], required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--frame-rate", type=float, default=30)
    parser.add_argument("--training-updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--split-repeat", type=int, default=100)
    parser.add_argument("--validation-frames", type=int, default=10)
    args = parser.parse_args()
    if (
        min(
            args.training_updates,
            args.rollout_steps,
            args.checkpoint_every,
            args.split_repeat,
        )
        < 1
    ):
        parser.error("Training counts must be positive")
    if args.frame_rate <= 0 or args.validation_frames < 0:
        parser.error("Frame rate must be positive and validation frames nonnegative")
    validate_sequence(
        args.gg_data_root,
        args.gg_output_root,
        args.subject,
        args.sequence,
        args.frame,
        args.validation_frames,
        args.frame_rate,
    )
    from realfit3d.repair.registration import validate_registered_topology

    validate_registered_topology(
        args.gg_output_root / args.subject / "stage1/template_uv.obj",
        sorted(
            (
                args.gg_output_root / args.subject / "stage2" / args.sequence / "meshes"
            ).glob("frame_*.obj")
        ),
    )
    import numpy as np
    import yaml
    from utils.defaults import DEFAULTS
    from utils.gaugar import GauGarConverter
    from utils.io import pickle_load, pickle_dump

    config = Path(DEFAULTS.project_dir) / "configs/finetune" / (args.subject + ".yaml")
    stage4 = args.gg_output_root / args.subject / "stage4"
    if config.exists() or stage4.exists():
        raise FileExistsError(
            "Refusing to overwrite existing Stage4/config; review and archive it first"
        )

    support = None
    if args.support_report:
        from realfit3d.repair.support import waist_indices

        support = waist_indices(
            args.support_report,
            args.gg_output_root / args.subject / "stage1/template_uv.obj",
        )
    garment_dict = args.garment_dicts / (args.subject + ".pkl")
    split_root = Path(DEFAULTS.aux_data) / "datasplits/finetuning" / args.subject
    if garment_dict.exists() or split_root.exists():
        raise FileExistsError(
            "Existing garment dictionary or training split; use a new subject"
        )

    class RegisteredSequenceConverter(GauGarConverter):
        def _import_garment(
            self,
            subject_out,
            template_sequence,
            template_frame,
            gender,
            pinned_indices=None,
        ):
            return super()._import_garment(
                subject_out,
                template_sequence,
                template_frame,
                gender,
                pinned_indices=support if support is not None else pinned_indices,
            )

        def _convert_smplx_dir(self, source, target, gender):
            frames = [pickle_load(path) for path in sorted(source.glob("*.pkl"))]
            if not frames:
                raise ValueError(f"No body pose files: {source}")
            poses = {
                key: np.stack([np.asarray(frame[key]).reshape(-1) for frame in frames])
                for key in frames[0]
            }
            hands = []
            for side in ("left", "right"):
                pose = poses[f"{side}_hand_pose"]
                if pose.shape[1] == 6:
                    components = getattr(
                        self.body_models_dict[gender], f"{side}_hand_components"
                    )
                    pose = pose @ components.cpu().numpy()
                elif pose.shape[1] != 45:
                    raise ValueError(
                        f"Unsupported {side} hand dimensions: {pose.shape[1]}"
                    )
                hands.append(pose)
            np.savez_compressed(
                target,
                betas=poses["betas"],
                expression=poses["expression"],
                trans=poses["transl"],
                root_orient=poses["global_orient"],
                pose_body=poses["body_pose"],
                pose_hand=np.concatenate(hands, axis=-1),
                pose_jaw=poses["jaw_pose"],
                pose_eye=np.concatenate(
                    [poses["leye_pose"], poses["reye_pose"]], axis=-1
                ),
                mocap_frame_rate=args.frame_rate,
            )

        def _convert_garment_subject(self, subject):
            source = (
                self.gaugar_output_root / subject / "stage2" / args.sequence / "meshes"
            )
            if not any(source.glob("*.obj")):
                raise ValueError(f"No registered meshes: {source}")
            target = (
                self.gaugar_output_root
                / subject
                / "stage4/registrations"
                / f"{args.sequence}.pkl"
            )
            self._convert_garment_dir(source, target)

        def _convert_smplx_subject(self, subject, subject_out, gender):
            source = self.gaugar_data_root / subject / args.sequence / "smplx"
            target = (
                self.gaugar_output_root
                / subject_out
                / "stage4/smplx"
                / f"{args.sequence}.npz"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            self._convert_smplx_dir(source, target, gender)

        def _create_datasplits(self, subject, gender, n_validation=1):
            directory = self.gaugar_output_root / subject / "stage4"
            source = directory / "smplx" / f"{args.sequence}.npz"
            poses = dict(np.load(source))
            count = len(poses["pose_body"])
            held_out = args.validation_frames
            if held_out < 3 or count - held_out < 3:
                raise ValueError(
                    "A single sequence requires --validation-frames with at least 3 training and 3 validation frames"
                )
            garment = pickle_load(directory / "registrations" / f"{args.sequence}.pkl")
            if len(garment["vertices"]) != count:
                raise ValueError("Registration and body frame counts differ")
            split_root = Path(DEFAULTS.aux_data) / "datasplits/finetuning" / subject
            split_root.mkdir(parents=True, exist_ok=True)
            for name, selection in [
                ("train", slice(0, count - held_out)),
                ("valid", slice(count - held_out, count)),
            ]:
                stem = f"{args.sequence}_{name}"
                target = directory / "smplx" / f"{stem}.npz"
                np.savez_compressed(
                    target,
                    **{
                        k: (v if k == "mocap_frame_rate" else v[selection])
                        for k, v in poses.items()
                    },
                )
                subset = {
                    k: (v[selection] if k in ("vertices", "pred", "uv_coords") else v)
                    for k, v in garment.items()
                }
                pickle_dump(subset, directory / "registrations" / f"{stem}.pkl")
                self._create_datasplit([target], subject, gender).to_csv(
                    split_root / f"{name}.csv", index=False
                )

        def _create_config(self, subject, multigarment=False):
            target = super()._create_config(subject, multigarment)
            config = yaml.safe_load(target.read_text())
            dataset = config["dataloaders"]["finetune"]["dataset"]["finetune"]
            dataset["garment_dicts_dir"] = str(self.garment_dicts_dir)
            dataset["fps"] = int(args.frame_rate)
            config["runner"]["finetune"]["regular_ts"] = 1 / args.frame_rate
            config["runner"]["finetune"]["initial_ts"] = 10 / args.frame_rate
            dataset.update(
                sequence_loader="cmu_npz_smplx_preserve",
                repeat_datasplit=args.split_repeat,
                body_model_root=str(args.body_model_root),
            )
            config["dataloaders"]["finetune"]["num_workers"] = 0
            import pandas as pd

            split_root = Path(DEFAULTS.aux_data) / "datasplits/finetuning" / subject
            physics_split = split_root / "physics_train.csv"
            pd.concat(
                [pd.read_csv(split_root / "train.csv")] * args.split_repeat,
                ignore_index=True,
            ).to_csv(physics_split, index=False)
            start = int(config["restart"]["step_start"])
            config["restart"]["checkpoint_path"] = str(args.checkpoint)
            config["experiment"].update(
                use_writer=False,
                n_epochs=math.ceil(args.training_updates / (2 * args.split_repeat)),
                max_iter=start + args.training_updates,
                save_checkpoint_every=args.checkpoint_every,
                save_checkpoint_every_wlong=args.checkpoint_every,
            )
            config["runner"]["finetune"]["long_rollout_steps"] = args.rollout_steps
            config["dataloaders"]["long"] = dict(
                num_workers=0,
                batch_size=1,
                pyg_data=True,
                dataset=dict(
                    ccraft=dict(
                        data_root=str(
                            self.gaugar_output_root / subject / "stage4/smplx"
                        ),
                        body_model_root=str(args.body_model_root),
                        model_type="smplx",
                        sequence_loader="cmu_npz_smplx_preserve",
                        garment_dicts_dir=str(self.garment_dicts_dir),
                        obstacle_dict_file=None,
                        split_path=str(physics_split),
                        pinned_verts=True,
                        n_coarse_levels=4,
                        separate_arms=False,
                        wholeseq=True,
                        nobody_freq=0.0,
                        fps=int(args.frame_rate),
                        swap_axes=False,
                    )
                ),
            )
            target.write_text(yaml.safe_dump(config, sort_keys=False))
            return target

    converter = RegisteredSequenceConverter(
        args.gg_data_root,
        args.gg_output_root,
        args.body_model_root,
        args.garment_dicts,
        args.checkpoint,
    )
    converter.convert_subject(args.subject, args.gender, args.sequence, args.frame)


if __name__ == "__main__":
    main()
