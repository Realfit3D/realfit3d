"""Watertight query-only body copy; never alter the simulation SMPL-X mesh."""
import numpy as np


def prepare_body_collider(vertices, faces, allowed_boundary_vertices=None):
    v, f = np.asarray(vertices), np.asarray(faces)
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Body collider is non-manifold; inside/outside queries are unreliable")
    boundary = directed[counts[inverse] == 1]
    report = dict(original_watertight=not len(boundary), boundary_edges=int(len(boundary)),
                  sealed_loops=0, query_watertight=not len(boundary), query_only=True, ray_samples=5)
    if not len(boundary):
        return v.copy(), f.copy(), report
    outgoing = {}
    for a, b in boundary:
        if int(a) in outgoing:
            raise ValueError("Body boundary does not form simple consistently wound loops")
        outgoing[int(a)] = int(b)
    if len(set(outgoing.values())) != len(outgoing):
        raise ValueError("Body boundary winding is inconsistent")
    remaining = set(outgoing)
    vertices_out = list(v.copy())
    faces_out = list(f.copy())
    max_span = .15 * np.linalg.norm(v.max(0) - v.min(0))
    while remaining:
        start = min(remaining)
        loop, point = [], start
        while point in remaining:
            loop.append(point)
            remaining.remove(point)
            point = outgoing[point]
        if point != start or len(loop) < 3 or len(loop) > 128:
            raise ValueError("Unsupported body boundary; refusing an arbitrary hole fill")
        if allowed_boundary_vertices is not None and not np.asarray(allowed_boundary_vertices)[loop].all():
            raise ValueError("Body has a non-facial opening; refusing to change its collider")
        if np.linalg.norm(np.ptp(v[loop], axis=0)) > max_span:
            raise ValueError("Body opening is too large for a query-only facial cap")
        center = len(vertices_out)
        vertices_out.append(v[loop].mean(0))
        # Reverse the existing directed boundary so the cap has consistent winding.
        faces_out.extend([[outgoing[a], a, center] for a in loop])
        report["sealed_loops"] += 1
    vv = np.asarray(vertices_out, dtype=v.dtype)
    ff = np.asarray(faces_out, dtype=f.dtype)
    edges = np.sort(np.concatenate((ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]])), axis=1)
    if not np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2):
        raise ValueError("Query-only body collider is still not watertight")
    report["query_watertight"] = True
    return vv, ff, report
