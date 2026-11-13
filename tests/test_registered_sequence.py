import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None
from realfit3d.settings import ROOT
from realfit3d.workers.contourcraft.registered_sequence import install


@unittest.skipIf(np is None, "NumPy is required for pose loader tests")
class RegisteredLoaderTests(unittest.TestCase):
    def setUp(self):
        source = (
            ROOT / "upstream/ContourCraft/datasets/sequence_loaders/cmu_npz_smplx.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        namespace = dict(np=np)
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        upstream = ModuleType("datasets.sequence_loaders.cmu_npz_smplx")
        upstream.SequenceLoader = namespace["SequenceLoader"]
        with patch.dict(
            sys.modules, {"datasets.sequence_loaders.cmu_npz_smplx": upstream}
        ):
            install()
            self.loader = sys.modules[
                "datasets.sequence_loaders.cmu_npz_smplx_preserve"
            ].SequenceLoader
        self.cfg = SimpleNamespace(model_type="smplx", fps=25, separate_arms=False)

    def raw(self):
        return dict(
            betas=np.ones(10),
            expression=np.zeros(10),
            root_orient=np.ones((4, 3)),
            pose_body=np.ones((4, 63)),
            pose_hand=np.ones((4, 90)),
            pose_jaw=np.ones((4, 3)),
            pose_eye=np.ones((4, 6)),
            trans=np.ones((4, 3)),
            mocap_frame_rate=np.array(25),
        )

    def test_source_pose_keeps_last_body_joints_and_hands(self):
        loader = self.loader(self.cfg, "unused")
        raw = self.raw()
        seq = loader.process_sequence(loader.convert_seq_to_hood_format(raw))
        np.testing.assert_array_equal(seq["body_pose"], raw["pose_body"])
        np.testing.assert_array_equal(seq["left_hand_pose"], raw["pose_hand"][:, :45])
        self.assertEqual(seq["betas"].shape, (4, 10))
        seq["body_pose"][0] = 0
        self.assertTrue((raw["pose_body"] == 1).all())

    def test_registration_never_silently_resamples_or_changes_body(self):
        raw = self.raw()
        raw["mocap_frame_rate"] = np.array(30)
        with self.assertRaisesRegex(ValueError, "same frame rate"):
            self.loader(self.cfg, "").convert_seq_to_hood_format(raw)
        self.cfg.separate_arms = True
        with self.assertRaisesRegex(ValueError, "separate_arms"):
            self.loader(self.cfg, "").process_sequence(
                self.loader(self.cfg, "").convert_seq_to_hood_format(self.raw())
            )


if __name__ == "__main__":
    unittest.main()
