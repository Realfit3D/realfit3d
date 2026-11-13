"""Create a separate, audited Stage 1.1 candidate from a versioned recipe."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema_version") != 1:
        raise ValueError("Repair recipe requires schema_version=1")
    if recipe.get("garment_type") not in ("upper", "lower", "dress"):
        raise ValueError("Invalid garment_type")
    if (
        not isinstance(recipe.get("expected_openings"), int)
        or recipe["expected_openings"] < 1
    ):
        raise ValueError("expected_openings must be positive")
    known = {
        "body_torso_prior",
        "crop",
        "clean",
        "fill_small_holes",
        "smooth_openings",
        "arap_openings",
        "extend_bands",
        "rebuild_hem",
        "remesh",
        "body_clearance",
        "local_intersections",
        "merge",
    }
    if not isinstance(recipe.get("operations"), list):
        raise ValueError("operations must be a list")
    for operation in recipe["operations"]:
        if operation.get("kind") not in known:
            raise ValueError(f"Unknown repair operation: {operation.get('kind')}")
        if "symmetry" in operation:
            raise ValueError("Pose-normalized symmetry is not implemented")
    return recipe


def inspect_mesh(v, f):
    import numpy as np
    from .geometry import as_o3d, quality
    from realfit3d.quality.repair_mesh_boundaries import (
        boundary_components,
        ordered_cycle,
    )

    m = as_o3d(v, f)
    groups = boundary_components(v, f)
    simple = True
    for _, group, adjacency in groups:
        try:
            ordered_cycle(group, adjacency)
        except RuntimeError:
            simple = False
    q = quality(v, f)
    areas = (
        np.linalg.norm(
            np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1
        )
        / 2
    )
    return dict(
        vertices=len(v),
        faces=len(f),
        openings=len(groups),
        simple_openings=simple,
        components=len(m.cluster_connected_triangles()[1]),
        nonmanifold_edges=len(m.get_non_manifold_edges()),
        nonmanifold_vertices=len(m.get_non_manifold_vertices()),
        self_intersections=len(m.get_self_intersecting_triangles()),
        degenerate_faces=int((areas < 1e-12).sum()),
        triangle_quality_p01=float(np.quantile(q, 0.01)),
        bounds_m=[v.min(0).tolist(), v.max(0).tolist()],
        dimensions_m=np.ptp(v, axis=0).tolist(),
        opening_perimeters_m=[p for p, _, _ in groups],
    )


def apply_operation(
    v, f, op, recipe, body_scene=None, reference=None, body_mesh=None, model_path=None
):
    import numpy as np
    import open3d as o3d
    import pymeshlab
    import trimesh
    from .geometry import as_o3d, boundaries, clip, ellipse_target, extend, rebuild
    from realfit3d.quality.repair_mesh_boundaries import (
        boundary_components,
        ordered_cycle,
    )

    kind = op["kind"]
    expected = recipe["expected_openings"]
    if kind == "body_torso_prior":
        if body_mesh is None or model_path is None:
            raise ValueError("body_torso_prior requires --body and --body-model")
        base_faces = np.asarray(body_mesh.triangles)
        with np.load(model_path, allow_pickle=False) as archive:
            weights = archive["weights"]
        if len(weights) != len(body_mesh.vertices):
            raise ValueError("Body skinning weights do not match body vertices")
        mesh = body_mesh.subdivide_midpoint(op.get("subdivisions", 2))
        mesh.compute_vertex_normals()
        points, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)
        hit = body_scene.compute_closest_points(
            o3d.core.Tensor(points.astype(np.float32))
        )
        tri = base_faces[hit["primitive_ids"].numpy()]
        uv = hit["primitive_uvs"].numpy()
        bary = np.c_[1 - uv.sum(1), uv]
        semantic = (weights[tri] * bary[:, :, None]).sum(1)
        arms = semantic[:, op.get("arm_joint_start", 16) :].sum(1)
        front = np.clip(
            (points[:, 2] - op["front_z_start_m"]) / op["front_z_width_m"], 0, 1
        )
        neck = (
            op["neck_y_m"]
            + op["front_neck_raise_m"] * front
            + op["neck_slope"] * np.abs(points[:, 0] - op["center_x_m"])
        )
        keep = (
            (points[:, 1] >= op["hem_y_m"])
            & (points[:, 1] <= neck)
            & (arms <= op["arm_weight_max"])
        )
        mesh = as_o3d(points + normals * op["clearance_m"], faces[keep[faces].all(1)])
        mesh.remove_unreferenced_vertices()
        labels, counts, _ = mesh.cluster_connected_triangles()
        if not len(counts):
            raise ValueError("Body prior selects no surface")
        mesh.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
        mesh = mesh.simplify_quadric_decimation(op.get("target_faces", 16000))
        mesh.orient_triangles()
        return np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    if kind == "crop":
        return clip(v, f, op["height_m"], op["direction"])
    if kind in ("clean", "merge"):
        mesh = as_o3d(v, f)
        if op.get("target_faces"):
            mesh = mesh.simplify_quadric_decimation(op["target_faces"])
        if kind == "merge":
            mesh.merge_close_vertices(op["distance_m"])
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        if op.get("remove_nonmanifold_edges", False):
            mesh.remove_non_manifold_edges()
        if op.get("largest_component", False):
            labels, counts, _ = mesh.cluster_connected_triangles()
            mesh.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
        if op.get("taubin_iterations", 0):
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=op["taubin_iterations"]
            )
        v, f = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
        if op.get("split_nonmanifold_vertices", False):
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(v, f))
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0.05)
            v, f = (
                ms.current_mesh().vertex_matrix().copy(),
                ms.current_mesh().face_matrix().copy(),
            )
        return v, f
    if kind == "fill_small_holes":
        groups = boundary_components(v, f)
        if len(groups) < expected:
            raise ValueError("Fewer openings than expected")
        holes = groups[expected:]
        if not holes:
            return v, f
        if max(p for p, _, _ in holes) > op["max_perimeter_m"]:
            raise ValueError(
                "Extra opening exceeds small-hole threshold; review required"
            )
        if op.get("remove_puncture_band_above_y") is not None:
            ids = [
                list(g)
                for _, g, _ in holes
                if v[list(g), 1].mean() > op["remove_puncture_band_above_y"]
            ]
            if ids:
                f = f[~np.isin(f, np.concatenate(ids)).any(axis=1)]
                mesh = trimesh.Trimesh(v, f, process=False)
                mesh.remove_unreferenced_vertices()
                v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                groups = boundary_components(v, f)
                holes = groups[expected:]
                if len(groups) < expected or any(
                    p > op["max_perimeter_m"] for p, _, _ in holes
                ):
                    raise ValueError("Puncture-band removal changes protected openings")
        if not holes:
            return v, f
        limit = max(len(g) for _, g, _ in holes)
        if min(len(g) for _, g, _ in groups[:expected]) <= limit:
            raise ValueError(
                "Hole edge counts overlap protected openings; review required"
            )
        if op.get("method", "meshlab") == "fan":
            directed = set(
                map(tuple, np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]))
            )
            added = []
            for _, group, adj in holes:
                rim = ordered_cycle(group, adj)
                center = len(v)
                v = np.vstack([v, v[rim].mean(0)])
                for a, b in zip(rim, rim[1:] + rim[:1]):
                    added.append(
                        [b, a, center] if (a, b) in directed else [a, b, center]
                    )
            f = np.vstack([f, added])
        else:
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(v, f))
            ms.meshing_close_holes(
                maxholesize=limit + 1, selfintersection=True, refinehole=False
            )
            v, f = (
                ms.current_mesh().vertex_matrix().copy(),
                ms.current_mesh().face_matrix().copy(),
            )
        if len(boundary_components(v, f)) != expected:
            raise ValueError("Hole fill did not preserve expected openings")
        return v, f
    if kind == "smooth_openings":
        _, loops = boundaries(f)
        if len(loops) != expected:
            raise ValueError("Unexpected opening count")
        top = max(loops, key=lambda ids: v[ids, 1].mean())
        for rim in loops:
            p = v[rim].copy()
            for _ in range(op.get("iterations", 6)):
                p = 0.5 * p + 0.25 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
            if "top_y_m" in op:
                p[:, 1] = (
                    op["top_y_m"] if np.array_equal(rim, top) else op["bottom_y_m"]
                )
            delta = p - v[rim]
            lengths = np.linalg.norm(delta, axis=1)
            delta *= np.minimum(
                1,
                op.get("max_displacement_m", float("inf")) / np.maximum(lengths, 1e-12),
            )[:, None]
            v[rim] += delta
        return v, f
    if kind == "arap_openings":
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        edges, loops = boundaries(f)
        if len(loops) != expected:
            raise ValueError("Unexpected opening count")
        target = v.copy()
        for ids in loops:
            target[ids], _ = ellipse_target(v[ids])
        boundary = np.concatenate(loops)
        lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        graph = coo_matrix(
            (
                np.r_[lengths, lengths],
                (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
            ),
            shape=(len(v), len(v)),
        ).tocsr()
        distances = dijkstra(graph, directed=False, indices=boundary, min_only=True)
        ids = np.r_[boundary, np.flatnonzero(distances > op.get("band_width_m", 0.1))]
        mesh = as_o3d(v, f)
        steps = op.get("steps", 5)
        for step in range(1, steps + 1):
            positions = (
                v[ids] + (target[ids] - v[ids]) * op.get("fraction", 1.0) * step / steps
            )
            mesh = mesh.deform_as_rigid_as_possible(
                o3d.utility.IntVector(ids.tolist()),
                o3d.utility.Vector3dVector(positions),
                max_iter=op.get("iterations", 50),
                energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
                smoothed_alpha=0.01,
            )
        return np.asarray(mesh.vertices).copy(), f
    if kind == "rebuild_hem":
        rv, rf = reference if reference is not None else (v, f)
        _, loops = boundaries(rf)
        rim = min(loops, key=lambda ids: rv[ids, 1].mean())
        lo, hi = np.quantile(rv[rim][:, [0, 2]], [0.02, 0.98], axis=0)
        return rebuild(
            v,
            f,
            op["cut_y_m"],
            float(np.median(rv[rim, 1])),
            (lo + hi) / 2,
            (hi - lo) / 2,
        )
    if kind == "extend_bands":
        _, loops = boundaries(f)
        if len(loops) != expected:
            raise ValueError("Unexpected opening count")
        for rim in loops:
            y = v[rim, 1].mean()
            matches = [
                rule
                for rule in op["bands"]
                if rule.get("min_y_m", -float("inf"))
                <= y
                <= rule.get("max_y_m", float("inf"))
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous band selector")
            if matches:
                v, f, _ = extend(v, f, rim, matches[0]["target_y_m"])
            else:
                p = v[rim].copy()
                for _ in range(op.get("other_opening_iterations", 8)):
                    p = 0.5 * p + 0.25 * (
                        np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
                    )
                v[rim] = p
        return v, f
    if kind == "remesh":
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(v, f))
        ms.compute_selection_by_condition_per_face(condselect=op["face_selection"])
        absolute = getattr(pymeshlab, "PureValue", None) or pymeshlab.AbsoluteValue
        ms.meshing_isotropic_explicit_remeshing(
            iterations=op.get("iterations", 10),
            selectedonly=True,
            targetlen=absolute(op.get("edge_m", 0.008)),
            maxsurfdist=absolute(op.get("surface_distance_m", 0.002)),
            checksurfdist=True,
            featuredeg=175,
            reprojectflag=True,
        )
        mesh = trimesh.Trimesh(
            ms.current_mesh().vertex_matrix(),
            ms.current_mesh().face_matrix(),
            process=False,
        )
        trimesh.repair.fix_normals(mesh)
        return np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy()
    if kind == "body_clearance":
        if body_scene is None:
            raise ValueError("body_clearance requires --body")
        original = v.copy()
        for _ in range(op.get("iterations", 9)):
            query = o3d.core.Tensor(v.astype(np.float32))
            hit = body_scene.compute_closest_points(query)
            points, normals = hit["points"].numpy(), hit["primitive_normals"].numpy()
            distance = ((v - points) * normals).sum(1)
            update = (
                normals
                * np.clip(
                    op.get("clearance_m", 0.002) - distance, 0, op.get("step_m", 0.004)
                )[:, None]
            )
            delta = v + update - original
            length = np.linalg.norm(delta, axis=1)
            proposed = (
                original
                + delta
                * np.minimum(
                    1, op.get("max_displacement_m", 0.025) / np.maximum(length, 1e-12)
                )[:, None]
            )
            update = proposed - v
            old = len(as_o3d(v, f).get_self_intersecting_triangles())
            scales = (
                (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
                if op.get("line_search", True)
                else (1.0,)
            )
            for scale in scales:
                trial = v + scale * update
                if (
                    not op.get("line_search", True)
                    or len(as_o3d(trial, f).get_self_intersecting_triangles()) <= old
                ):
                    v = trial
                    break
        return v, f
    if kind == "local_intersections":
        original = v.copy()
        _, loops = boundaries(f)
        locked = set(np.concatenate(loops)) if op.get("lock_openings", True) else set()
        adjacency = [set() for _ in v]
        for a, b, c in f:
            adjacency[a].update((b, c))
            adjacency[b].update((a, c))
            adjacency[c].update((a, b))
        for _ in range(op.get("rounds", 3)):
            pairs = np.asarray(as_o3d(v, f).get_self_intersecting_triangles())
            if not len(pairs):
                break
            region = set(f[pairs].ravel().tolist())
            for _ in range(op.get("neighbor_rings", 2)):
                region.update(n for i in list(region) for n in adjacency[i])
            ids = np.array(sorted(region - locked), dtype=int)
            if not len(ids):
                break
            for _ in range(op.get("iterations", 12)):
                average = np.array([v[list(adjacency[i])].mean(0) for i in ids])
                v[ids] = 0.7 * v[ids] + 0.3 * average
                delta = v[ids] - original[ids]
                length = np.linalg.norm(delta, axis=1)
                v[ids] = (
                    original[ids]
                    + delta
                    * np.minimum(
                        1,
                        op.get("max_displacement_m", 0.015) / np.maximum(length, 1e-12),
                    )[:, None]
                )
        return v, f
    raise ValueError(kind)


def run(
    source,
    recipe_path,
    output=None,
    body=None,
    reference=None,
    inspect_only=False,
    body_model=None,
):
    import numpy as np
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    from .geometry import parameterize, boundaries, read_mesh

    recipe = load_recipe(recipe_path)
    source = Path(source)
    mesh = read_mesh(source)
    v, f = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    if not len(v) or not len(f) or not np.isfinite(v).all():
        raise ValueError("Input must be a nonempty finite triangle mesh")
    report = dict(
        schema_version=1,
        recipe=recipe,
        source_sha256=sha256(source),
        recipe_sha256=sha256(recipe_path),
        baseline=inspect_mesh(v, f),
        operations=[],
    )
    if inspect_only:
        return report
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("numpy", "scipy", "open3d", "trimesh", "pymeshlab", "xatlas"):
        versions[name] = importlib.metadata.version(name)
    report["versions"] = versions
    report["implementation_sha256"] = {
        p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
    }
    scene = None
    b = None
    if body:
        b = read_mesh(body)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(b))
        report["body_sha256"] = sha256(body)
    if body_model:
        report["body_model_sha256"] = sha256(body_model)
    rv, rf = v.copy(), f.copy()
    if reference:
        r = read_mesh(reference)
        rv, rf = np.asarray(r.vertices), np.asarray(r.triangles)
        report["reference_sha256"] = sha256(reference)
    original = v.copy()
    try:
        for i, operation in enumerate(recipe["operations"]):
            before = v.copy()
            v, f = apply_operation(
                v.copy(), f.copy(), operation, recipe, scene, (rv, rf), b, body_model
            )
            if not len(v) or not len(f) or not np.isfinite(v).all():
                raise ValueError("Repair produced an invalid mesh")
            report["operations"].append(
                dict(
                    index=i,
                    settings=operation,
                    max_displacement_m=float(np.linalg.norm(v - before, axis=1).max())
                    if v.shape == before.shape
                    else None,
                    geometry=inspect_mesh(v, f),
                )
            )
        report["geometry"] = q = inspect_mesh(v, f)
        report["dimension_change_m"] = (
            np.ptp(v, axis=0) - np.ptp(original, axis=0)
        ).tolist()
        report["max_nearest_source_vertex_distance_m"] = float(
            cKDTree(original).query(v)[0].max()
        )
        report["max_nearest_result_vertex_distance_m"] = float(
            cKDTree(v).query(original)[0].max()
        )
        if scene is not None:
            hit = scene.compute_closest_points(o3d.core.Tensor(v.astype(np.float32)))
            signed = (
                (v - hit["points"].numpy()) * hit["primitive_normals"].numpy()
            ).sum(1)
            report["contact"] = dict(
                normal_inside_gt2mm=int((signed < -0.002).sum()),
                minimum_normal_distance_m=float(signed.min()),
            )
        trimesh.Trimesh(v, f, process=False).export(output / "template.obj")
        if (
            q["openings"] != recipe["expected_openings"]
            or q["components"] != 1
            or not q["simple_openings"]
        ):
            raise ValueError("Opening/component gate failed")
        if any(
            q[k]
            for k in (
                "nonmanifold_edges",
                "nonmanifold_vertices",
                "self_intersections",
                "degenerate_faces",
            )
        ):
            raise ValueError("Topology/intersection gate failed")
        max_change = recipe.get("gates", {}).get("max_dimension_change_m")
        if (
            max_change is not None
            and max(map(abs, report["dimension_change_m"])) > max_change
        ):
            raise ValueError("Dimension-change gate failed")
        mapping, indices, uv, areas, isolated = parameterize(v, f)
        if (
            not np.array_equal(mapping[indices], f)
            or not np.isfinite(uv).all()
            or not (areas > 0).all()
        ):
            raise ValueError("UV correspondence/area gate failed")
        with (output / "template_uv.obj").open("x") as stream:
            for p in v:
                stream.write("v " + " ".join(map(str, p)) + "\n")
            for p in uv:
                stream.write("vt " + " ".join(map(str, p)) + "\n")
            for face, tex in zip(f, indices):
                stream.write(
                    "f "
                    + " ".join(f"{a + 1}/{b + 1}" for a, b in zip(face, tex))
                    + "\n"
                )
        np.savez(
            output / "uv_diagnostics.npz",
            mapping=mapping,
            indices=indices,
            uv=uv,
            faces=f,
        )
        if recipe["garment_type"] == "lower":
            _, loops = boundaries(f)
            report["waist_indices"] = max(
                loops, key=lambda ids: v[ids, 1].mean()
            ).tolist()
        report.update(
            status="geometry_passed_pending_visual_review",
            uv_min_double_area=float(areas.min()),
            uv_independent_triangle_charts=isolated,
            outputs_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        )
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        (output / "quality.json").write_text(json.dumps(report, indent=2))
        raise
    (output / "quality.json").write_text(json.dumps(report, indent=2))
    return report
