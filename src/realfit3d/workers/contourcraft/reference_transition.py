"""Topology-aware interpolation of reference-relative displacements.

The Dirichlet solve preserves reference edge vectors up to a smooth residual,
instead of blending two potentially folded meshes using discontinuous weights.
No changes to rest positions, faces, or physical material parameters.
"""
import numpy as np


def edges_of(faces):
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]])), axis=1), axis=0)


def quality(vertices, reference, faces, selected):
    edges = edges_of(faces)
    active = selected[edges].any(1)
    lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    active &= lengths > 1e-7
    actual = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    strain = np.zeros(len(edges))
    strain[active] = np.abs(actual[active] / lengths[active] - 1)
    tr, tn = reference[faces], vertices[faces]
    nr = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0])
    nn = np.cross(tn[:, 1] - tn[:, 0], tn[:, 2] - tn[:, 0])
    reversed_normals = (selected[faces].any(1) & (np.sum(nr * nn, axis=1) <= 0)
                        & (np.linalg.norm(nr, axis=1) > 1e-10))
    return edges, strain, reversed_normals


def constrain_lengths(vertices, reference, edges, free, active, limit, iterations=1200, faces=None):
    """Project edge lengths into the admissible interval with fixed anchors.

    Jacobi updates distribute edge corrections among incident free vertices.
    This changes the candidate, never the acceptance threshold or reference.
    """
    v = vertices.copy()
    e = edges[active]
    rest = np.linalg.norm(reference[e[:, 1]] - reference[e[:, 0]], axis=1)
    useful = rest > 1e-7
    e, rest = e[useful], rest[useful]
    mobility = free[e].astype(float)
    total = mobility.sum(1)
    movable = total > 0
    e, rest, mobility, total = e[movable], rest[movable], mobility[movable], total[movable]
    if not len(e):
        return v
    # Small numerical margin avoids accepting float64 solutions that fail once
    # stored as float32. The actual gate remains the user's original limit.
    bound = max(limit - 5e-4, limit * .99)
    degree = np.bincount(e.ravel(), minlength=len(v)).clip(1)[:, None]
    scatter_ids = np.concatenate((e[:, 0], e[:, 1]))
    if faces is not None:
        tri = reference[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(normals, axis=1)
        face_active = free[faces].any(1) & (area > 1e-10)
        f = faces[face_active]
        area = area[face_active]
        normals = normals[face_active] / area[:, None]
    for step in range(iterations):
        delta = v[e[:, 1]] - v[e[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        target = np.clip(length, rest * (1 - bound), rest * (1 + bound))
        error = length - target
        edge_ok = np.max(np.abs(error) / rest) < 1e-6
        direction = delta / np.maximum(length[:, None], 1e-12)
        collapsed = length < 1e-12
        direction[collapsed] = (reference[e[collapsed, 1]] - reference[e[collapsed, 0]]) / rest[collapsed, None]
        correction = direction * (error / total)[:, None]
        contributions = np.concatenate((correction * mobility[:, 0, None],
                                        -correction * mobility[:, 1, None]))
        update = np.column_stack([np.bincount(scatter_ids, weights=contributions[:, k],
                                              minlength=len(v)) for k in range(3)])
        v += .95 * update / degree
        normal_ok = True
        if faces is not None and len(f):
            tri = v[f]
            ba, ca = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            signed = np.sum(np.cross(ba, ca) * normals, axis=1)
            deficit = np.maximum(.05 * area - signed, 0)
            normal_ok = not np.any(deficit > 1e-12)
            if not normal_ok:
                gb = np.cross(ca, normals)
                gc = np.cross(normals, ba)
                gradients = np.stack((-gb - gc, gb, gc), axis=1)
                fm = free[f].astype(float)
                denom = np.sum(np.sum(gradients ** 2, axis=2) * fm, axis=1)
                scale = deficit / np.maximum(denom, 1e-20)
                corrections = gradients * (scale[:, None] * fm)[:, :, None]
                contributions = corrections.reshape(-1, 3)
                update = np.column_stack([np.bincount(f.ravel(), weights=contributions[:, k],
                                                      minlength=len(v)) for k in range(3)])
                counts = np.bincount(f[deficit > 1e-12].ravel(), minlength=len(v)).clip(1)[:, None]
                v += .8 * update / counts
        if edge_ok and normal_ok:
            break
    return v


def centered_anchors(reference, faces, membership, chain_joints, fraction, align_end=False):
    """Center limb cross-sections on their bone axis without resizing rings.

    Works for either arm or leg, including garments ending before the joint.
    Bone-length-relative search/shift limits replace garment-specific crop boxes.
    """
    import trimesh
    from scipy.ndimage import gaussian_filter1d
    mask = membership > .25
    if not mask.any():
        return reference.copy()
    whole_axis = chain_joints[-1] - chain_joints[0]
    whole_axis /= np.linalg.norm(whole_axis)
    observed_end = np.quantile((reference[mask] - chain_joints[0]) @ whole_axis, .98)
    middle = np.dot(chain_joints[-2] - chain_joints[0], whole_axis)
    # A short sleeve/shorts tube belongs to shoulder-elbow/hip-knee, not the
    # extrapolated forearm/shin axis. The decision uses coverage, not garment ID.
    start, end = chain_joints[:2] if observed_end < middle else chain_joints[-2:]
    axis = end - start
    length = np.linalg.norm(axis)
    axis /= length
    s = (reference - start) @ axis
    lo, hi = np.quantile(s[mask], [.02, .98])
    if hi - lo < 1e-7:
        return reference.copy()
    epsilon = (hi - lo) * .001
    samples = np.linspace(lo + epsilon, hi - epsilon, 25)
    region_faces = faces[mask[faces].any(1)]
    mesh = trimesh.Trimesh(reference, region_faces, process=False)
    shifts = []
    for position in samples:
        origin = start + position * axis
        lines = trimesh.intersections.mesh_plane(mesh, plane_origin=origin, plane_normal=axis)
        if not len(lines):
            shifts.append(np.full(3, np.nan)); continue
        mids = lines.mean(1)
        lens = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
        nearby = (np.linalg.norm(mids - origin, axis=1) < .75 * length) & (lens > 1e-9)
        if not nearby.any():
            shifts.append(np.full(3, np.nan)); continue
        center = np.average(mids[nearby], weights=lens[nearby], axis=0)
        shift = origin - center
        shift -= np.dot(shift, axis) * axis
        shifts.append(shift)
    shifts = np.asarray(shifts)
    valid = np.isfinite(shifts).all(1)
    if not valid.any():
        return reference.copy()
    shifts = np.stack([np.interp(samples, samples[valid], shifts[valid, k]) for k in range(3)], axis=1)
    shifts = gaussian_filter1d(shifts, 1, axis=0)
    shift = np.stack([np.interp(s, samples, shifts[:, k]) for k in range(3)], axis=1)
    norm = np.linalg.norm(shift, axis=1)
    shift *= np.minimum(1, fraction * length / np.maximum(norm, 1e-8))[:, None]
    if align_end:
        # Explicit long-sleeve/long-trouser profile only. Translate the tube
        # end back to its wrist/ankle landmark; do not scale its radius/length.
        # Endpoint alignment is explicit: do not silently stop short at a
        # radial-centering limit. The outer whole-region displacement gate
        # rejects an excessive translation instead of leaving it half aligned.
        excess = max(float(hi - length), 0.)
        shift -= excess * axis
    return reference + shift


def harmonic_transition(original, reference, faces, seed, region, max_strain=.35,
                        max_support_fraction=.8, steps=32, anchor_positions=None):
    """Restore seed vertices exactly and solve a smooth residual in a growing halo.

    Outside the halo, preserve the original geometry exactly. Expansion stops at
    the configured vertex budget. Quality gates are never relaxed by this solve.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    n = len(original)
    edges = edges_of(faces)
    lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    # Clamp near-zero lengths to avoid degenerate edges dominating the solve.
    positive = lengths[lengths > 1e-7]
    floor = max(float(np.median(positive)) * .1, 1e-7) if len(positive) else 1e-7
    weights = 1 / np.maximum(lengths, floor)
    i = np.concatenate((edges[:, 0], edges[:, 1]))
    j = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix((np.concatenate((weights, weights)), (i, j)), shape=(n, n)).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    _, component = connected_components(adjacency, directed=False)
    touched = np.isin(component, np.unique(component[seed]))
    domain = (region | seed) & touched
    candidate = original.astype(np.float64).copy()
    residual = original.astype(np.float64) - reference
    history = []
    for attempt in range(steps):
        if domain.sum() > max_support_fraction * n:
            break
        fixed = ~domain | seed
        free = np.flatnonzero(domain & ~seed)
        d = residual.copy()
        d[seed] = 0 if anchor_positions is None else anchor_positions[seed] - reference[seed]
        if len(free):
            rhs = -laplacian[free][:, fixed] @ d[fixed]
            solved = spsolve(laplacian[free][:, free].tocsc(), rhs)
            if not np.isfinite(solved).all():
                raise ValueError("Singular topology transition; inspect disconnected reference components")
            d[free] = solved
        candidate = reference.astype(np.float64) + d
        candidate[~domain] = original[~domain]
        selected = domain
        e, strain, normal_flags = quality(candidate, reference, faces, selected)
        if np.any(strain > max_strain) or normal_flags.any():
            candidate = constrain_lengths(candidate, reference, e, domain & ~seed,
                                          domain[e].any(1), max_strain, faces=faces)
            e, strain, normal_flags = quality(candidate, reference, faces, selected)
        history.append(dict(attempt=attempt + 1, support_vertices=int(domain.sum()),
                            max_edge_strain=float(strain.max()) if len(strain) else 0.,
                            normal_reversal_proxy=int(normal_flags.sum())))
        if not np.any(strain > max_strain) and not normal_flags.any():
            break
        frontier = np.zeros(n, bool)
        frontier[e[strain > max_strain].ravel()] = True
        frontier[faces[normal_flags].ravel()] = True
        # Two topological rings per step, never across disconnected fragments.
        for _ in range(2):
            frontier[e[frontier[e].any(1)].ravel()] = True
        grow = domain | frontier
        grow &= touched
        if np.array_equal(grow, domain) or grow.sum() > max_support_fraction * n:
            break
        domain = grow
    selected = np.linalg.norm(candidate - original, axis=1) > 1e-8
    return candidate.astype(original.dtype), selected, dict(
        method="harmonic_reference_residual", seed_vertices=int(seed.sum()),
        support_vertices=history[-1]["support_vertices"] if history else 0,
        support_budget_fraction=max_support_fraction,
        gates_converged=bool(history and history[-1]["max_edge_strain"] <= max_strain
                             and history[-1]["normal_reversal_proxy"] == 0), history=history)
