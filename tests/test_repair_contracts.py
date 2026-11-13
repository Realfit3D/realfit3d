import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from realfit3d.cli import main
from realfit3d.contact import contact_config, arguments
from realfit3d.pipelines.offline import build_plan
from realfit3d.quality.install_repaired_template import install
from realfit3d.repair.pipeline import load_recipe
from realfit3d.repair.support import waist_indices
from realfit3d.settings import ROOT, load_settings


class RepairContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_presets_keep_capture_specific_openings(self):
        for path in (ROOT / "configs/repair").glob("*.json"):
            recipe = load_recipe(path)
            self.assertEqual(
                recipe["expected_openings"],
                3 if recipe["garment_type"] == "lower" else 4,
            )
            self.assertNotIn("/mnt/", path.read_text())
        upper = load_recipe(ROOT / "configs/repair/actor08_upper.json")
        bands = next(op for op in upper["operations"] if op["kind"] == "extend_bands")
        self.assertEqual(len(bands["bands"]), 1)
        self.assertLess(bands["bands"][0]["max_y_m"], 1)

    def test_repair_plan_does_not_run_or_write(self):
        cfg = dict(
            reconstruction_python="/missing/python",
            gg_data_root=str(self.root / "data"),
            gg_output_root=str(self.root / "output"),
            output_root=str(self.root / "jobs"),
            template_repairs=dict(
                Test=dict(source="source.obj", recipe="recipe.json", output="candidate")
            ),
        )
        path = self.root / "paths.json"
        path.write_text(json.dumps(cfg))
        settings = load_settings(path)
        self.assertEqual(
            settings["template_repairs"]["Test"]["source"],
            str(self.root / "source.obj"),
        )
        with (
            patch("subprocess.run") as child,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                main(["--config", str(path), "offline", "repair", "--subject", "Test"]),
                0,
            )
        child.assert_not_called()
        self.assertEqual([p.name for p in self.root.iterdir()], ["paths.json"])

    def candidate(self):
        src = self.root / "candidate"
        src.mkdir()
        # Two open triangular rims connected by six faces.
        obj = "v 0 1 0\nv 1 1 0\nv 0 1 1\nv 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 4 2\nf 2 4 5\nf 2 5 3\nf 3 5 6\nf 3 6 1\nf 1 6 4\n"
        for name in ("template.obj", "template_uv.obj"):
            (src / name).write_text(obj)
        report = dict(
            status="geometry_passed_pending_visual_review",
            source_sha256="source",
            recipe_sha256="recipe",
            waist_indices=[0, 1, 2],
            outputs_sha256={
                name: hashlib.sha256((src / name).read_bytes()).hexdigest()
                for name in ("template.obj", "template_uv.obj")
            },
        )
        (src / "quality.json").write_text(json.dumps(report))
        return src, report

    def test_installed_support_is_bound_to_exact_mesh(self):
        src, report = self.candidate()
        stage = install(src, self.root / "NewActor")
        self.assertEqual(
            waist_indices(stage / "repair_quality.json", stage / "template_uv.obj"),
            [0, 1, 2],
        )
        with self.assertRaises(FileExistsError):
            install(src, self.root / "NewActor")
        (stage / "template_uv.obj").write_text("different")
        with self.assertRaisesRegex(ValueError, "does not match"):
            waist_indices(stage / "repair_quality.json", stage / "template_uv.obj")

    def test_rejects_changed_candidate_before_install(self):
        src, _ = self.candidate()
        (src / "template.obj").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            install(src, self.root / "NewActor")
        self.assertFalse((self.root / "NewActor").exists())

    def test_rejects_nonboundary_pins(self):
        src, report = self.candidate()
        report["waist_indices"] = [0, 1, 4]
        (src / "quality.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "boundary loop"):
            waist_indices(src / "quality.json", src / "template_uv.obj")

    def test_registration_faces_must_keep_template_correspondence(self):
        from realfit3d.repair.registration import validate_registered_topology

        src, _ = self.candidate()
        template = src / "template.obj"
        registered = self.root / "frame_00000.obj"
        registered.write_text(template.read_text().replace("v 0 1 0", "v 0 1.1 0"))
        validate_registered_topology(template, [registered])
        registered.write_text(registered.read_text().replace("f 1 4 2", "f 1 2 4"))
        with self.assertRaisesRegex(ValueError, "correspondence"):
            validate_registered_topology(template, [registered])

    def test_contact_is_explicit_and_validated(self):
        self.assertEqual(arguments({}), [])
        config = contact_config({})
        self.assertEqual(config["relaxation_steps"], 30)
        self.assertAlmostEqual(config["projector"]["max_displacement"], 0.035)
        for value in (
            {"typo": 1},
            {"relaxation_steps": 0},
            {"projector": {"clearance": float("nan")}},
            {"projector": {"minimum_clearance": 0.02}},
            {"lower_impulse_passes": 1.5},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                contact_config(value)


if __name__ == "__main__":
    unittest.main()
