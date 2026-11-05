"""Anatomical core tests: synthetic upper/lower garments, not motion validation."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/realfit3d/workers/contourcraft"))
import repair_garment_lbs as repair


def fixture():
    v = np.array([[.8, 0, 0], [.9, 0, 0], [.8, .1, 0],
                  [.1, -.8, 0], [.2, -.8, 0], [.1, -.9, 0],
                  [0, .1, 0], [.1, .1, 0], [0, .2, 0]], dtype=np.float32)
    n = len(v)
    w = np.zeros((n, 55), np.float32)
    w[:3, 18], w[3:6, 4], w[6:, 3] = 1, 1, 1
    g = dict(faces=np.arange(n).reshape(-1, 3), rest_pos=v.copy(), node_type=np.zeros((n, 1)),
             gender="male", fixed_size_metadata=dict(source_betas=np.zeros(10), target_betas=np.zeros(10)),
             lbs=dict(v=v.copy(), lbs_weights=w.copy(), shapedirs=np.zeros((n, 3, 10)),
                      posedirs=np.zeros((486, n * 3))))
    j = np.zeros((55, 3))
    for chain, axis, sign in [([16, 18, 20], 0, 1), ([17, 19, 21], 0, -1),
                              ([1, 4, 7], 1, -1), ([2, 5, 8], 1, -1),
                              ([0, 3, 6, 9, 12], 1, 1)]:
        for i, idx in enumerate(chain):
            j[idx, axis] = sign * i * .25
    body_pd = np.ones((486, n * 3), np.float32) * .123
    return g, copy.deepcopy(g), v.copy(), g["faces"], j, w, body_pd


def execute(data, config, apply=True):
    return repair.repair_with_body(*data, config, apply)


class AnatomicalRepair(unittest.TestCase):
    def test_healthy_region_is_unchanged(self):
        data = fixture()
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertTrue(report["gates_passed"])
        self.assertEqual(report["proposed_vertices"], 0)
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_restores_arm_and_preserves_other_regions_and_inputs(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += .15
        before = copy.deepcopy(data[0])
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])
        for key in ("rest_pos", "faces", "node_type"):
            np.testing.assert_array_equal(result[key], before[key])
        np.testing.assert_array_equal(data[0]["lbs"]["v"], before["lbs"]["v"])
        np.testing.assert_allclose(result["lbs"]["posedirs"].reshape(486, -1, 3)[:, :3], .123)
        np.testing.assert_array_equal(result["lbs"]["posedirs"].reshape(486, -1, 3)[:, 3:], 0)

    def test_same_algorithm_restores_trouser_leg(self):
        data = fixture()
        data[0]["lbs"]["v"][3:6, 2] += .15
        result, report, _ = execute(data, {"regions": ["left_leg"]})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])

    def test_same_algorithm_restores_torso(self):
        data = fixture()
        data[0]["lbs"]["v"][6:, 2] += .15
        result, report, _ = execute(data, {"regions": ["torso"]})
        self.assertTrue(report["applied"])

    def test_diagnose_does_not_change_dictionary(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += .15
        result, report, _ = execute(data, {"regions": ["left_arm"]}, False)
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_global_rotation_translation_do_not_select_wrong_vertices(self):
        data = list(fixture())
        data[0]["lbs"]["v"][:3, 2] += .15
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        shift = np.array([.15, -.2, .1])
        for d in data[:2]:
            d["lbs"]["v"] = d["lbs"]["v"] @ rot.T + shift
        data[2] = data[2] @ rot.T + shift
        data[4] = data[4] @ rot.T + shift
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])

    def test_overlarge_displacement_rejects_entire_candidate(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += 1
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertFalse(report["gates_passed"])
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_binding_only_keeps_geometry(self):
        data = fixture()
        data[0]["lbs"]["lbs_weights"][:3] = 0
        data[0]["lbs"]["lbs_weights"][:3, 3] = 1
        result, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"]})
        self.assertTrue(report["applied"])
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])
        np.testing.assert_array_equal(result["lbs"]["lbs_weights"][:3], data[5][:3])

    def test_apply_requires_reviewed_regions(self):
        with self.assertRaisesRegex(ValueError, "explicitly reviewed"):
            execute(fixture(), {})

    def test_axial_trigger_restores_length_collapse(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 0] -= .15
        result, report, _ = execute(data, {"regions": ["left_arm"], "restore_trigger": "axial"})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])

    def test_axial_trigger_does_not_treat_lateral_offset_as_length_collapse(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += .15
        result, report, _ = execute(data, {"regions": ["left_arm"], "restore_trigger": "axial"})
        self.assertEqual(report["proposed_vertices"], 0)
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_unknown_and_nonfinite_settings_are_rejected(self):
        for config in ({"regions": ["sleeve"]}, {"clearance_m": float("nan")},
                       {"max_edge_strain": -1}, {"magic": True}):
            with self.assertRaises(ValueError):
                repair.validate_profile(config)

    def test_wrong_reference_topology_is_rejected(self):
        data = fixture()
        data[1]["faces"][0] = [2, 1, 0]
        with self.assertRaisesRegex(ValueError, "topology"):
            execute(data, {"regions": ["left_arm"]})

    def test_surface_distance_gate_catches_wrong_target(self):
        data = list(fixture())
        data[2] += 10
        _, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"]})
        self.assertFalse(report["gates_passed"])

    def test_large_edge_distortion_rejected_even_in_rebind_mode(self):
        data = fixture()
        data[0]["lbs"]["v"][0, 0] += .15
        _, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"]})
        self.assertFalse(report["gates_passed"])
        self.assertGreater(report["max_edge_strain"], .35)

    def test_reference_triangle_reversal_is_rejected(self):
        data = fixture()
        data[0]["lbs"]["v"][[0, 1]] = data[0]["lbs"]["v"][[1, 0]]
        _, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"], "max_edge_strain": 1})
        self.assertFalse(report["gates_passed"])
        self.assertGreater(report["flipped_triangles"], 0)

    def test_harmonic_transition_preserves_outside_and_removes_seam(self):
        from reference_transition import harmonic_transition, quality
        x, y = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, .3, 6))
        ref = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        faces = []
        for row in range(5):
            for col in range(17):
                a = row * 18 + col
                faces.extend([[a, a + 1, a + 18], [a + 1, a + 19, a + 18]])
        faces = np.array(faces)
        old = ref.copy()
        old[:, 2] = .10
        old[ref[:, 0] > .6, 2] += .12
        region = ref[:, 0] > .4
        seed = ref[:, 0] > .8
        result, selected, info = harmonic_transition(old, ref, faces, seed, region)
        self.assertTrue(info["gates_converged"])
        np.testing.assert_allclose(result[seed], ref[seed], atol=1e-8)
        _, strain, flags = quality(result, ref, faces, selected)
        self.assertLessEqual(strain.max(), .35)
        self.assertFalse(flags.any())
        np.testing.assert_array_equal(result[ref[:, 0] < .2], old[ref[:, 0] < .2])

    def test_harmonic_budget_failure_is_explicit(self):
        from reference_transition import harmonic_transition
        data = fixture()
        v = data[2]
        region = np.ones(len(v), bool)
        seed = np.zeros(len(v), bool)
        seed[[0, 3, 6]] = True
        _, _, info = harmonic_transition(v + .1, v, data[3], seed, region, max_support_fraction=.1)
        self.assertFalse(info["gates_converged"])

    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh is not installed")
    def test_limb_centering_translates_rings_without_resizing(self):
        from reference_transition import centered_anchors
        angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        v = np.array([[x, -.06 + .05 * np.cos(a), .05 * np.sin(a)]
                      for x in (.32, .40, .50, .58) for a in angle])
        faces = []
        for ring in range(3):
            for i in range(16):
                a, b = ring * 16 + i, ring * 16 + (i + 1) % 16
                faces.extend([[a, b, a + 16], [b, b + 16, a + 16]])
        faces = np.array(faces)
        joints = np.array([[0., 0, 0], [.3, 0, 0], [.6, 0, 0]])
        centered = centered_anchors(v, faces, np.ones(len(v)), joints, .3)
        np.testing.assert_allclose(centered[:, 1], v[:, 1] + .06, atol=1e-6)
        np.testing.assert_allclose(centered[:, [0, 2]], v[:, [0, 2]], atol=1e-6)
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        moved = centered_anchors(v @ rot.T, faces, np.ones(len(v)), joints @ rot.T, .3)
        np.testing.assert_allclose(moved, centered @ rot.T, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh is not installed")
    def test_short_limb_uses_covered_segment_without_extending_hem(self):
        from reference_transition import centered_anchors
        angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        v = np.array([[x, -.03 + .01 * np.cos(a), .01 * np.sin(a)]
                      for x in (.08, .13, .19, .25) for a in angle])
        faces = []
        for ring in range(3):
            for i in range(16):
                a, b = ring * 16 + i, ring * 16 + (i + 1) % 16
                faces.extend([[a, b, a + 16], [b, b + 16, a + 16]])
        joints = np.array([[0., 0, 0], [.3, 0, 0], [.3, .3, 0]])
        centered = centered_anchors(v, np.array(faces), np.ones(len(v)), joints, .3)
        np.testing.assert_allclose(centered[:, 1], v[:, 1] + .03, atol=1e-6)
        np.testing.assert_allclose(centered[:, [0, 2]], v[:, [0, 2]], atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("open3d"), "open3d is not installed")
    def test_clearance_escapes_narrow_gap_without_moving_unselected_points(self):
        import open3d as o3d
        cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
        cf = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                       [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                       [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
        body = np.concatenate((cube * [.02, .1, .02] + [-.024, -.05, -.01],
                               cube * [.02, .1, .02] + [.004, -.05, -.01]))
        faces = np.concatenate((cf, cf + 8))
        v = np.array([[0., 0, 0], [0, 0, .1]], dtype=np.float32)
        result = repair.project_clearance(v, np.array([True, False]), body, faces, .006)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(body.astype('float32')), o3d.core.Tensor(faces.astype('uint32')))
        sd = scene.compute_signed_distance(o3d.core.Tensor(result)).numpy()
        self.assertGreaterEqual(sd[0], .006 - 1e-4)
        np.testing.assert_array_equal(result[1], v[1])

    def test_small_query_body_hole_is_closed_without_editing_source(self):
        from body_collider import prepare_body_collider
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                      [.45, .45, 1], [.55, .45, 1], [.55, .55, 1], [.45, .55, 1]], dtype=np.float32)
        f = [[0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
             [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
        for i in range(4):
            f.extend([[4 + i, 4 + (i + 1) % 4, 8 + (i + 1) % 4],
                      [4 + i, 8 + (i + 1) % 4, 8 + i]])
        f = np.array(f)
        old_v, old_f = v.copy(), f.copy()
        cv, cf, report = prepare_body_collider(v, f, np.ones(len(v), bool))
        self.assertTrue(report["query_watertight"])
        self.assertEqual(report["sealed_loops"], 1)
        self.assertEqual(len(cv), len(v) + 1)
        np.testing.assert_array_equal(v, old_v)
        np.testing.assert_array_equal(f, old_f)
        with self.assertRaisesRegex(ValueError, "non-facial"):
            prepare_body_collider(v, f, np.zeros(len(v), bool))


if __name__ == "__main__":
    unittest.main()
