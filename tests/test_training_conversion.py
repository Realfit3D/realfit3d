import tempfile
import ast
import pickle
from pathlib import Path
from types import SimpleNamespace
import unittest

from realfit3d.workers.contourcraft import convert_training_assets
from realfit3d.workers.contourcraft.convert_training_assets import validate_sequence

try:
    import numpy as np
except ImportError:
    np = None


class TrainingConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = self.root / "input/Actor/Sequence1/smplx"
        self.mesh = self.root / "output/Actor/stage2/Sequence1/meshes"
        self.body.mkdir(parents=True)
        self.mesh.mkdir(parents=True)
        for index in range(6):
            (self.body / f"{index:05d}.pkl").touch()
            (self.mesh / f"frame_{index:05d}.obj").touch()

    def validate(self, frame=0, held_out=3, fps=25):
        validate_sequence(
            self.root / "input",
            self.root / "output",
            "Actor",
            "Sequence1",
            frame,
            held_out,
            fps,
        )

    def test_accepts_matching_reindexed_clip_without_writing(self):
        before = sorted(self.root.rglob("*"))
        self.validate()
        self.assertEqual(before, sorted(self.root.rglob("*")))

    def test_rejects_missing_mesh(self):
        (self.mesh / "frame_00005.obj").unlink()
        with self.assertRaisesRegex(ValueError, "IDs must match"):
            self.validate()

    def test_equal_counts_do_not_imply_matching_ids(self):
        (self.mesh / "frame_00005.obj").rename(self.mesh / "frame_00006.obj")
        with self.assertRaisesRegex(ValueError, "IDs must match"):
            self.validate()

    def test_rejects_gap_in_both_inputs(self):
        (self.mesh / "frame_00005.obj").rename(self.mesh / "frame_00006.obj")
        (self.body / "00005.pkl").rename(self.body / "00006.pkl")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.validate()

    def test_rejects_invalid_split_frame_and_rate(self):
        for kwargs in (
            {"frame": 6},
            {"held_out": 0},
            {"held_out": 4},
            {"fps": 0},
            {"fps": 29.97},
            {"fps": float("nan")},
            {"fps": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.validate(**kwargs)


@unittest.skipIf(np is None, "NumPy is required for pose-conversion tests")
class PoseConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tree = ast.parse(Path(convert_training_assets.__file__).read_text())
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        cls = next(node for node in main.body if isinstance(node, ast.ClassDef))
        module = ast.Module(body=[cls], type_ignores=[])
        self.support = [0, 1, 2]

        class Base:
            def _import_garment(base_self, *args, **kwargs):
                return kwargs.get("pinned_indices")

        namespace = dict(
            np=np,
            GauGarConverter=Base,
            support=self.support,
            args=SimpleNamespace(frame_rate=25),
            pickle_load=lambda path: pickle.loads(path.read_bytes()),
        )
        exec(compile(module, "<converter-test>", "exec"), namespace)
        self.namespace = namespace
        self.converter = namespace["RegisteredSequenceConverter"]()
        self.components = np.arange(270, dtype=float).reshape(6, 45) / 270
        components = SimpleNamespace(
            cpu=lambda: SimpleNamespace(numpy=lambda: self.components)
        )
        self.converter.body_models_dict = {
            "male": SimpleNamespace(
                left_hand_components=components, right_hand_components=components
            )
        }

    def convert(self, hand_dimensions):
        frame = {
            key: np.zeros((1, count))
            for key, count in dict(
                betas=10,
                expression=10,
                transl=3,
                global_orient=3,
                body_pose=63,
                jaw_pose=3,
                leye_pose=3,
                reye_pose=3,
            ).items()
        }
        hand = np.arange(hand_dimensions, dtype=float).reshape(1, -1) / 100
        frame.update(left_hand_pose=hand, right_hand_pose=hand + 0.01)
        (self.root / "00000.pkl").write_bytes(pickle.dumps(frame))
        target = self.root / "sequence.npz"
        self.converter._convert_smplx_dir(self.root, target, "male")
        with np.load(target) as archive:
            return hand, dict(archive)

    def test_physics_config_uses_training_split_and_requested_checkpoint(self):
        try:
            import yaml
            import pandas as pd
        except ImportError:
            self.skipTest("PyYAML/pandas required for generated training config test")
        import math
        from realfit3d.settings import ROOT

        config = yaml.safe_load(
            (ROOT / "upstream/ContourCraft/configs/finetune/base.yaml").read_text()
        )
        target = self.root / "Actor.yaml"
        target.write_text(yaml.safe_dump(config))
        base = type(self.converter).__bases__[0]
        base._create_config = lambda obj, subject, multigarment=False: target
        split = self.root / "aux/datasplits/finetuning/Actor"
        split.mkdir(parents=True)
        (split / "train.csv").write_text("id,gender\nSequence1_train,male\n")
        (split / "valid.csv").write_text("id,gender\nSequence1_valid,male\n")
        self.namespace.update(
            yaml=yaml,
            math=math,
            Path=Path,
            DEFAULTS=SimpleNamespace(aux_data=self.root / "aux"),
            args=SimpleNamespace(
                frame_rate=25,
                split_repeat=100,
                training_updates=1000,
                rollout_steps=8,
                checkpoint_every=100,
                body_model_root=self.root / "models",
                checkpoint=self.root / "selected.pth",
            ),
        )
        self.converter.garment_dicts_dir = self.root / "garments"
        self.converter.gaugar_output_root = self.root / "outputs"
        self.converter._create_config("Actor")
        generated = yaml.safe_load(target.read_text())
        self.assertEqual(
            generated["restart"]["checkpoint_path"], str(self.root / "selected.pth")
        )
        self.assertEqual(generated["experiment"]["max_iter"], 46000)
        self.assertEqual(generated["experiment"]["n_epochs"], 5)
        self.assertEqual(generated["runner"]["finetune"]["long_rollout_steps"], 8)
        self.assertEqual(generated["runner"]["finetune"]["regular_ts"], 0.04)
        physics = generated["dataloaders"]["long"]["dataset"]["ccraft"]
        self.assertEqual(physics["sequence_loader"], "cmu_npz_smplx_preserve")
        rows = pd.read_csv(physics["split_path"])
        self.assertEqual(len(rows), 100)
        self.assertEqual(set(rows["id"]), {"Sequence1_train"})

    def test_waist_handles_are_passed_before_upstream_relaxation(self):
        self.assertEqual(
            self.converter._import_garment("Actor", "Sequence1", 0, "male"),
            self.support,
        )

    def test_full_axis_angle_hands_are_preserved(self):
        hand, result = self.convert(45)
        np.testing.assert_array_equal(result["pose_hand"][:, :45], hand)
        np.testing.assert_array_equal(result["pose_hand"][:, 45:], hand + 0.01)
        self.assertEqual(result["pose_body"].shape, (1, 63))
        self.assertEqual(result["mocap_frame_rate"], 25)

    def test_pca_hands_are_expanded_with_model_components(self):
        hand, result = self.convert(6)
        np.testing.assert_allclose(result["pose_hand"][:, :45], hand @ self.components)
        np.testing.assert_allclose(
            result["pose_hand"][:, 45:], (hand + 0.01) @ self.components
        )

    def test_unsupported_hand_dimensions_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "hand dimensions"):
            self.convert(12)


if __name__ == "__main__":
    unittest.main()
