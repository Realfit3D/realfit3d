"""CPU geometry regressions; run in the optional repair environment."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_GEOMETRY = all(
    importlib.util.find_spec(k) for k in ("open3d", "trimesh", "pymeshlab", "xatlas")
)


@unittest.skipUnless(
    HAS_GEOMETRY, "Install requirements/repair.txt in a geometry environment"
)
class TemplateGeometryTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import trimesh

        self.np = np
        self.trimesh = trimesh
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, openings):
        np = self.np
        mesh = self.trimesh.creation.icosphere(subdivisions=3, radius=0.3)
        centers = mesh.triangles_center
        unit = centers / np.linalg.norm(centers, axis=1)[:, None]
        directions = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])[:openings]
        mesh.update_faces((unit @ directions.T < 0.91).all(1))
        mesh.remove_unreferenced_vertices()
        source = self.root / f"mesh{openings}.obj"
        mesh.export(source)
        return source

    def recipe(self, openings):
        path = self.root / "recipe.json"
        path.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    garment_type="upper" if openings == 4 else "lower",
                    expected_openings=openings,
                    operations=[
                        dict(
                            kind="smooth_openings",
                            iterations=2,
                            max_displacement_m=0.002,
                        )
                    ],
                )
            )
        )
        return path

    def test_three_and_four_opening_candidates_keep_uv_face_correspondence(self):
        from realfit3d.repair.pipeline import run, sha256

        for count in (3, 4):
            with self.subTest(openings=count):
                source = self.source(count)
                before = sha256(source)
                out = self.root / f"candidate{count}"
                report = run(source, self.recipe(count), out)
                self.assertEqual(report["geometry"]["openings"], count)
                self.assertEqual(report["geometry"]["components"], 1)
                self.assertEqual(report["geometry"]["self_intersections"], 0)
                self.assertGreater(report["uv_min_double_area"], 0)
                with self.np.load(out / "uv_diagnostics.npz") as uv:
                    self.np.testing.assert_array_equal(
                        uv["mapping"][uv["indices"]], uv["faces"]
                    )
                self.assertEqual(sha256(source), before)
                with self.assertRaises(FileExistsError):
                    run(source, self.recipe(count), out)

    def test_inspection_is_read_only_and_wrong_openings_fail_with_report(self):
        from realfit3d.repair.pipeline import run

        src = self.source(3)
        recipe = self.recipe(4)
        before = set(self.root.iterdir())
        report = run(src, recipe, inspect_only=True)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(report["baseline"]["openings"], 3)
        out = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "opening"):
            run(src, recipe, out)
        self.assertEqual(
            json.loads((out / "quality.json").read_text())["status"], "failed"
        )
        self.assertFalse((out / "template_uv.obj").exists())

    def test_obj_uv_seams_do_not_split_geometry(self):
        from realfit3d.repair.geometry import read_mesh

        path = self.root / "seams.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt .5 .5\n"
            "f 1/1 2/2 3/3\nf 1/5 3/3 4/4\n"
        )
        mesh = read_mesh(path)
        self.assertEqual(len(mesh.vertices), 4)
        self.np.testing.assert_array_equal(
            self.np.asarray(mesh.triangles), [[0, 1, 2], [0, 2, 3]]
        )

    def test_cut_reports_shortening_and_does_not_cap_functional_openings(self):
        from realfit3d.repair.pipeline import apply_operation
        from realfit3d.repair.geometry import boundaries

        np = self.np
        # Open tube with two rings; crop its lower part without making a cap.
        theta = np.arange(32) * 2 * np.pi / 32
        ring = np.c_[0.2 * np.cos(theta), np.zeros(32), 0.2 * np.sin(theta)]
        v = np.vstack([ring, ring + [0, 1, 0]])
        f = []
        for i in range(32):
            j = (i + 1) % 32
            f += [[i, j, i + 32], [j, j + 32, i + 32]]
        result, faces = apply_operation(
            v,
            np.array(f),
            dict(kind="crop", height_m=0.2, direction=1),
            dict(expected_openings=2),
        )
        self.assertAlmostEqual(result[:, 1].min(), 0.2)
        self.assertEqual(len(boundaries(faces)[1]), 2)


if __name__ == "__main__":
    unittest.main()
