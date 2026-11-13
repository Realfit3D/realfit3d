"""Opening priors, band reconstruction and UV packing used by repair recipes."""

import numpy as np
import open3d as o3d
import trimesh
import xatlas
from scipy.optimize import least_squares, minimize_scalar


def boundaries(f):
    edges, counts = np.unique(
        np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
        axis=0,
        return_counts=True,
    )
    adj = {}
    for a, b in edges[counts == 1]:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(n) != 2 for n in adj.values()) or (counts > 2).any():
        raise ValueError("Non-manifold source")
    left, loops = set(adj), []
    while left:
        start = min(left)
        ids, last, now = [], None, start
        while True:
            ids.append(now)
            left.remove(now)
            nxt = next(k for k in adj[now] if k != last)
            if nxt == start:
                break
            last, now = now, nxt
        loops.append(np.array(ids))
    return edges, loops


def ellipse_target(points):
    center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    xy = (points - center) @ basis[:2].T
    radius = np.maximum(np.quantile(np.abs(xy), 0.8, axis=0), 0.01)

    def residual(p):
        return np.sqrt(np.sum(((xy - p[:2]) / np.exp(p[2:])) ** 2, axis=1)) - 1

    fitted = least_squares(
        residual, np.r_[np.zeros(2), np.log(radius)], loss="soft_l1", f_scale=0.1
    )
    c2, rad = fitted.x[:2], np.exp(fitted.x[2:])
    signed_area = np.sum(
        xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)
    )
    angles = (
        np.arange(len(points))
        * 2
        * np.pi
        / len(points)
        * (1 if signed_area > 0 else -1)
    )

    def sample(phase):
        return (
            center
            + (
                c2
                + np.column_stack([np.cos(angles + phase), np.sin(angles + phase)])
                * rad
            )
            @ basis[:2]
        )

    # A coarse phase search avoids picking a local minimum on the circle.
    phases = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    phase = min(phases, key=lambda t: np.sum((sample(t) - points) ** 2))
    result = minimize_scalar(
        lambda t: np.sum((sample(t) - points) ** 2),
        bounds=(phase - 0.1, phase + 0.1),
        method="bounded",
    )
    return sample(result.x), dict(
        center_3d=center.tolist(),
        plane_axes=basis[:2].tolist(),
        radii_m=rad.tolist(),
        phase=float(result.x),
    )


def as_o3d(v, f):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )


def clip(v, f, height, direction):
    v, f, _ = trimesh.intersections.slice_faces_plane(
        v,
        f,
        plane_normal=np.array([0.0, direction, 0.0]),
        plane_origin=np.array([0.0, height, 0.0]),
    )
    m = trimesh.Trimesh(v, f, process=False)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.faces)


def extend(v, f, rim, target_y):
    directed = set(
        map(tuple, np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]))
    )
    if (rim[0], rim[1]) not in directed:
        rim = rim[::-1]
    points = v[rim]
    target, fit = ellipse_target(points)
    target[:, 1] = target_y
    height = float(points[:, 1].mean())
    rows = max(3, int(np.ceil(abs(height - target_y) / 0.008)))
    vertices, faces, previous = [v], [f], rim
    for row in range(1, rows + 1):
        t = row / rows
        blend = t * t * (3 - 2 * t)
        current = points * (1 - blend) + target * blend
        current[:, 1] = height * (1 - t) + target_y * t
        ids = np.arange(len(rim)) + len(v) + (row - 1) * len(rim)
        vertices.append(current)
        b, d = np.roll(previous, -1), np.roll(ids, -1)
        faces.extend(
            [np.column_stack([previous, ids, b]), np.column_stack([b, ids, d])]
        )
        previous = ids
    return np.concatenate(vertices), np.concatenate(faces), fit


def parameterize(v, f):
    isolated = set()
    for attempt in range(4):
        atlas = xatlas.Atlas()
        primary = np.array([i for i in range(len(f)) if i not in isolated])
        # xatlas rejects very small metric-space triangles; use centimetres for UV solving only.
        atlas.add_mesh((v * 100).astype(np.float32), f[primary].astype(np.uint32))
        for index in sorted(isolated):
            atlas.add_mesh(
                (v[f[index]] * 100).astype(np.float32),
                np.array([[0, 1, 2]], dtype=np.uint32),
            )
        packing = xatlas.PackOptions()
        packing.resolution, packing.padding = 4096, 4
        atlas.generate(pack_options=packing)
        mapping, indices, uv = atlas[0]
        all_indices = np.empty_like(f)
        all_indices[primary] = indices
        mappings, coords = [mapping], [uv]
        offset = len(uv)
        for part, index in enumerate(sorted(isolated), start=1):
            local_mapping, local_indices, local_uv = atlas[part]
            all_indices[index] = local_indices[0] + offset
            mappings.append(f[index][local_mapping])
            coords.append(local_uv)
            offset += len(local_uv)
        mapping, uv = np.concatenate(mappings), np.concatenate(coords)
        areas = np.abs(
            np.cross(
                uv[all_indices[:, 1]] - uv[all_indices[:, 0]],
                uv[all_indices[:, 2]] - uv[all_indices[:, 0]],
            )
        )
        assert np.array_equal(mapping[all_indices], f)
        bad = set(np.flatnonzero(areas <= 0).tolist())
        if not bad:
            return mapping, all_indices, uv, areas, sorted(isolated)
        # Unmapped faces get separate packed charts, without changing geometry indices.
        if bad.issubset(isolated) or len(isolated | bad) > 64:
            raise ValueError("UV packing failed even with independent triangle charts")
        isolated.update(bad)
    raise ValueError("UV packing did not converge")


def quality(v, f):
    e = np.stack(
        [v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 1]], v[f[:, 0]] - v[f[:, 2]]],
        axis=1,
    )
    area = np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1) / 2
    return 4 * np.sqrt(3) * area / (e**2).sum(axis=(1, 2))


def rebuild(v, f, height, bottom_y, center, radii):
    cv, cf, _ = trimesh.intersections.slice_faces_plane(
        v,
        f,
        plane_normal=np.array([0.0, 1.0, 0.0]),
        plane_origin=np.array([0.0, height, 0.0]),
    )
    cut = trimesh.Trimesh(cv, cf, process=False)
    cut.merge_vertices()
    cut.remove_unreferenced_vertices()
    top, faces = np.asarray(cut.vertices), np.asarray(cut.faces)
    _, loops = boundaries(faces)
    rims = [ids for ids in loops if np.max(np.abs(top[ids, 1] - height)) < 1e-6]
    if len(rims) != 1 or len(loops) != 4:
        raise ValueError("Expected exactly one torso cut and three retained openings")
    rim = rims[0]
    directed = set(
        map(
            tuple,
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        )
    )
    if (rim[0], rim[1]) not in directed:
        rim = rim[::-1]
    p = top[rim]
    angle = np.arctan2(
        (p[:, 2] - center[1]) / radii[1], (p[:, 0] - center[0]) / radii[0]
    )
    delta = np.angle(np.exp(1j * (np.roll(angle, -1) - angle)))
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(
            "Torso cut backtracks around its center; choose a different cut"
        )
    lower = np.column_stack(
        [
            center[0] + radii[0] * np.cos(angle),
            np.full(len(rim), bottom_y),
            center[1] + radii[1] * np.sin(angle),
        ]
    )
    rows = max(3, int(np.ceil((height - bottom_y) / 0.008)))
    vertices = [top]
    new_faces = [faces]
    previous = rim
    for row in range(1, rows + 1):
        t = row / rows
        blend = t * t * (3 - 2 * t)
        current = p * (1 - blend) + lower * blend
        current[:, 1] = height * (1 - t) + bottom_y * t
        ids = np.arange(len(rim)) + len(top) + (row - 1) * len(rim)
        vertices.append(current)
        b = np.roll(previous, -1)
        d = np.roll(ids, -1)
        new_faces.extend(
            [np.column_stack([previous, ids, b]), np.column_stack([b, ids, d])]
        )
        previous = ids
    return np.concatenate(vertices), np.concatenate(new_faces)


def read_mesh(path):
    """Keep OBJ position indices independent from UV seams and normal indices."""
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() != ".obj":
        return o3d.io.read_triangle_mesh(str(path))
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        values = line.split("#", 1)[0].split()
        if not values:
            continue
        if values[0] == "v":
            vertices.append([float(x) for x in values[1:4]])
        elif values[0] == "f":
            if len(values) != 4:
                raise ValueError("Repair requires triangulated OBJ faces")
            ids = [int(x.split("/")[0]) for x in values[1:]]
            if any(i == 0 for i in ids):
                raise ValueError("OBJ indices cannot be zero")
            faces.append([i - 1 if i > 0 else len(vertices) + i for i in ids])
    v, f = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Empty or invalid OBJ geometry")
    if f.min() < 0 or f.max() >= len(v) or not np.isfinite(v).all():
        raise ValueError("Invalid OBJ positions or face indices")
    return as_o3d(v, f)
