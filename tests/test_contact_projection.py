import unittest

try:
    import numpy as np
    from realfit3d.workers.contourcraft.contact_projection import ContactProjector
except ImportError:
    np = None


class Plane:
    def query(self, points):
        normals = np.zeros_like(points)
        normals[:, 2] = 1
        return points[:, 2], np.abs(points[:, 2]), normals


@unittest.skipIf(np is None, "NumPy/SciPy are required for contact solver tests")
class ContactTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [x, y, -0.01 if x < 0.05 else 0.008]
                for y in np.linspace(-0.1, 0.1, 7)
                for x in np.linspace(-0.1, 0.1, 7)
            ]
        )
        faces = []
        for y in range(6):
            for x in range(6):
                i = y * 7 + x
                faces += [[i, i + 1, i + 7], [i + 1, i + 8, i + 7]]
        self.faces = np.array(faces)
        self.body = Plane()

    def test_contact_clearance_support_and_tangential_preservation(self):
        solver = ContactProjector(self.faces, len(self.vertices))
        result, metrics = solver.project(self.vertices, self.body)
        self.assertGreaterEqual(result[:, 2].min(), 0.003 - 4e-5)
        np.testing.assert_allclose(result[:, :2], self.vertices[:, :2], atol=1e-6)
        pins = np.arange(7)
        fixed, _ = solver.project(self.vertices, self.body, pins, result[pins])
        np.testing.assert_array_equal(fixed[pins], result[pins])
        self.assertEqual(metrics["inward_after"], 0)

    def test_displacement_gate_and_infeasible_support_fail(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            ContactProjector(
                self.faces, len(self.vertices), max_displacement=0.001
            ).project(self.vertices, self.body)
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            ContactProjector(self.faces, len(self.vertices)).project(
                self.vertices, self.body, np.arange(7), self.vertices[:7]
            )

    def test_invalid_input_and_deep_penetration_fail(self):
        for config in (
            {"clearance": -0.01},
            {"minimum_clearance": 0.1},
            {"smoothness": -1},
        ):
            with self.assertRaises(ValueError):
                ContactProjector(self.faces, len(self.vertices), **config)
        solver = ContactProjector(self.faces, len(self.vertices))
        v = self.vertices.copy()
        v[:, 2] = -0.1
        with self.assertRaisesRegex(RuntimeError, "Deep inward"):
            solver.project(v, self.body)
        v[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            solver.project(v, self.body)

    def test_support_only_contact_does_not_force_unselected_vertices(self):
        pins = np.arange(7)
        result, _ = ContactProjector(self.faces, len(self.vertices)).project(
            self.vertices, self.body, contact_indices=pins
        )
        self.assertGreaterEqual(result[pins, 2].min(), 0.003 - 4e-5)
        self.assertLess(result[-7:, 2].min(), 0.003)


if __name__ == "__main__":
    unittest.main()
