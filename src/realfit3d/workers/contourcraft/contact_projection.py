"""Bounded, spatially smooth contact correction for an oriented body surface.

Closest triangle points and normals define local half-space constraints. An
ADMM solve minimizes displacement and graph differences, then refreshes the
closest triangles. This is a local surface constraint, not a watertight-volume
test or a continuous collision detector. Connectivity and stored rest positions are preserved; posed edge lengths can change.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def edges_of(faces):
    return np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )


class BodySurface:
    def __init__(self, vertices, faces):
        import open3d as o3d

        self.o3d = o3d
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(vertices, np.float32)),
            o3d.core.Tensor(np.asarray(faces, np.uint32)),
        )
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(mesh)

    def query(self, points):
        o3d = self.o3d
        answer = self.scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=2
        )
        closest = answer["points"].numpy().astype(np.float64)
        normals = answer["primitive_normals"].numpy().astype(np.float64)
        delta = np.asarray(points) - closest
        return (delta * normals).sum(1), np.linalg.norm(delta, axis=1), normals


class ContactProjector:
    def __init__(
        self,
        faces,
        n_vertices,
        clearance=0.003,
        smoothness=12.0,
        max_displacement=0.035,
        near_radius=0.05,
        tolerance=2e-5,
        minimum_clearance=0.002,
    ):
        if (
            n_vertices < 3
            or not np.isfinite(
                [
                    clearance,
                    smoothness,
                    max_displacement,
                    near_radius,
                    tolerance,
                    minimum_clearance,
                ]
            ).all()
        ):
            raise ValueError("Invalid contact parameters")
        if (
            not 0 <= minimum_clearance <= clearance < near_radius
            or smoothness < 0
            or max_displacement <= 0
            or tolerance <= 0
        ):
            raise ValueError("Invalid contact parameter ranges")
        faces = np.asarray(faces)
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or faces.min() < 0
            or faces.max() >= n_vertices
        ):
            raise ValueError("Invalid triangle indices")
        self.n_vertices = n_vertices
        self.faces = np.asarray(faces)
        self.clearance, self.limit = clearance, max_displacement
        self.radius, self.tol = near_radius, tolerance
        self.minimum_clearance = minimum_clearance
        edges = edges_of(self.faces)
        a, b = edges.T
        adjacency = sparse.coo_matrix(
            (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])),
            shape=(n_vertices, n_vertices),
        ).tocsc()
        laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
        self.rho = 20.0
        self.system = splu(
            ((1 + self.rho) * sparse.eye(n_vertices) + smoothness * laplacian).tocsc()
        )

    def project(
        self, vertices, body, fixed=None, fixed_positions=None, contact_indices=None
    ):
        source = np.asarray(vertices, np.float64)
        if source.shape != (self.n_vertices, 3) or not np.isfinite(source).all():
            raise ValueError("Invalid contact vertices")
        result = source.copy()
        fixed = np.asarray([], dtype=int) if fixed is None else np.asarray(fixed)
        active = np.ones(len(source), dtype=bool)
        if contact_indices is not None:
            active[:] = False
            active[contact_indices] = True
        if len(fixed):
            fixed_positions = np.asarray(fixed_positions, dtype=float)
            if (
                fixed_positions.shape != (len(fixed), 3)
                or not np.isfinite(fixed_positions).all()
            ):
                raise ValueError("Invalid support positions")
            result[fixed] = fixed_positions
        initial_signed, initial_dist, _ = body.query(source)
        iterations = 0
        target_clearance = self.clearance
        for outer in range(16):
            # Closely opposing body surfaces can leave less than 6 mm of room.
            # Prefer 3 mm clearance, but retain a separate 2 mm hard minimum.
            if outer == 8:
                target_clearance = self.minimum_clearance
            signed, distance, normals = body.query(result)
            near = (distance < self.radius) & active
            if np.any(active & (distance >= self.radius) & (signed < -self.radius)):
                raise RuntimeError(
                    "Deep inward point exceeds local contact search radius"
                )
            deficit = np.where(near, target_clearance - signed, -1e3)
            if deficit.max() <= self.tol:
                break
            # Constraints are expressed relative to the original proposal so
            # successive closest-point updates do not accumulate shrinkage.
            bound = deficit + np.einsum("ij,ij->i", normals, result - source)
            z = result - source
            dual = np.zeros_like(z)
            for it in range(400):
                x = self.system.solve(self.rho * (z - dual))
                candidate = x + dual
                violation = np.maximum(
                    bound - np.einsum("ij,ij->i", normals, candidate), 0
                )
                next_z = candidate + normals * violation[:, None]
                if len(fixed):
                    next_z[fixed] = fixed_positions - source[fixed]
                residual = max(np.abs(next_z - x).max(), np.abs(next_z - z).max())
                dual += x - next_z
                z = next_z
                if residual < 1e-6:
                    break
            result = source + z
            iterations += it + 1
        signed, distance, _ = body.query(result)
        displacement = np.linalg.norm(result - source, axis=1)
        if not np.isfinite(result).all() or displacement.max() > self.limit:
            raise RuntimeError(
                f"Contact correction exceeds {self.limit} m: {displacement.max()}"
            )
        residual = np.maximum(
            target_clearance - signed[(distance < self.radius) & active], 0
        ).max(initial=0.0)
        if residual > self.tol * 2:
            raise RuntimeError(f"Unresolved contact clearance: {residual} m")
        return result.astype(np.float32), dict(
            max_displacement_m=float(displacement.max()),
            mean_displacement_m=float(displacement.mean()),
            inward_before=int(
                np.sum((initial_signed < -0.002) & (initial_dist < self.radius))
            ),
            inward_after=int(np.sum((signed < -0.002) & (distance < self.radius))),
            min_near_clearance_m=float(signed[distance < self.radius].min(initial=1.0)),
            target_clearance_m=target_clearance,
            solver_iterations=iterations,
        )


def self_test():
    # A planar patch includes penetrating supported vertices and an unaffected
    # far edge. Check contact, fixed support, and displacement-limit rejection.
    grid = np.array(
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
    body = BodySurface(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]]
    )
    projector = ContactProjector(np.asarray(faces), len(grid))
    fitted, metrics = projector.project(grid, body)
    assert fitted[:, 2].min() >= 0.003 - 4e-5
    assert np.max(np.abs(fitted[:, :2] - grid[:, :2])) < 1e-6
    pinned = np.arange(7)
    fitted2, _ = projector.project(grid, body, pinned, fitted[pinned])
    assert np.array_equal(fitted2[pinned], fitted[pinned])
    try:
        ContactProjector(np.asarray(faces), len(grid), max_displacement=0.001).project(
            grid, body
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Oversized correction was not rejected")
    return dict(
        status="passed",
        checks=[
            "planar_clearance",
            "no_tangential_drift",
            "corrected_fixed_support",
            "displacement_gate",
        ],
        metrics=metrics,
    )
