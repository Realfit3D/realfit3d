#!/usr/bin/env python3
"""Opt-in SMPL-X sleeve initialization repair; no simulation or training here.

Input is a *target-bound* garment dictionary. restore-rebind additionally needs
the same-topology, pre-relax/pre-untangle reference dictionary. See SLEEVE_LBS.md.
"""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

SIDES = (("left", 1, 18, 20, [13, 16, 18, 20], [18, 20] + list(range(25, 40))),
         ("right", -1, 19, 21, [14, 17, 19, 21], [19, 21] + list(range(40, 55))))


def load_dict(path):
    # Garment dictionaries are pickle files: only load trusted project assets.
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_garment(g, reference=None):
    v = array(g["lbs"]["v"])
    faces = array(g["faces"])
    weights = array(g["lbs"]["lbs_weights"])
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("lbs.v must be a finite V x 3 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a nonempty F x 3 triangle array")
    if faces.min() < 0 or faces.max() >= len(v):
        raise ValueError("Face indices are outside lbs.v")
    if weights.shape != (len(v), 55) or not np.isfinite(weights).all():
        raise ValueError("This implementation requires SMPL-X V x 55 LBS weights")
    if np.any(weights < -1e-6) or not np.allclose(weights.sum(1), 1, atol=1e-4):
        raise ValueError("LBS weights must be nonnegative and normalized")
    if array(g["lbs"]["shapedirs"]).shape[:2] != v.shape:
        raise ValueError("shapedirs do not match lbs.v")
    if array(g["lbs"]["posedirs"]).shape != (486, len(v) * 3):
        raise ValueError("Expected SMPL-X posedirs with shape 486 x (V * 3)")
    if array(g["rest_pos"]).shape != v.shape:
        raise ValueError("rest_pos does not match lbs.v")
    meta = g.get("fixed_size_metadata", {})
    for name in ("source_betas", "target_betas"):
        if name not in meta or array(meta[name]).shape != (10,):
            raise ValueError(f"Input needs fixed_size_metadata.{name} (10 numeric betas)")
        if not np.isfinite(array(meta[name])).all():
            raise ValueError(f"Non-finite {name}")
    if not np.allclose(array(g["lbs"]["shapedirs"]), 0):
        raise ValueError("Input must already be fixed-size bound (zero garment shapedirs)")
    if reference is not None:
        if array(reference["lbs"]["v"]).shape != v.shape:
            raise ValueError("Reference has a different vertex count")
        if not np.array_equal(array(reference["faces"]), faces):
            raise ValueError("Reference topology/face ordering differs; do not reuse vertex indices")
        if array(reference["lbs"]["lbs_weights"]).shape != weights.shape:
            raise ValueError("Reference skinning layout differs")
        if array(reference["lbs"]["shapedirs"]).shape != (len(v), 3, 10):
            raise ValueError("Reference must contain V x 3 x 10 shapedirs")
        source_beta = reference.get("reference_metadata", {}).get("source_betas")
        if source_beta is not None and not np.allclose(array(source_beta), array(meta["source_betas"]), atol=1e-5):
            raise ValueError("Reference/source fit betas differ; use the matching source_body_fit.pkl")


def rebind_local(g, vertices, mask, body_vertices, joints, body_weights, body_posedirs):
    """Same nearest-body-vertex transfer as fixed-size prepare, restricted distally.

    Update both weights and posedirs together. Outside mask, preserve the input.
    """
    from scipy.spatial import cKDTree
    ids = cKDTree(body_vertices).query(vertices)[1]
    for _, sign, elbow, wrist, _, distal_joints in SIDES:
        axis = joints[wrist] - joints[elbow]
        length = np.linalg.norm(axis)
        if length < 1e-6:
            raise ValueError("Degenerate SMPL-X forearm")
        axis /= length
        distal = mask & (vertices[:, 0] * sign > 0) & (
            ((vertices - joints[elbow]) @ axis) > .25 * length)
        candidates = np.flatnonzero(body_weights[:, distal_joints].sum(1) > .5)
        if not len(candidates):
            raise ValueError("No matching forearm/hand body surface vertices")
        if distal.any():
            ids[distal] = candidates[cKDTree(body_vertices[candidates]).query(vertices[distal])[1]]
    pd = body_posedirs.reshape(body_posedirs.shape[0], -1, 3)
    gp = array(g["lbs"]["posedirs"]).reshape(pd.shape[0], -1, 3).copy()
    gp[:, mask] = pd[:, ids[mask]]
    weights = array(g["lbs"]["lbs_weights"]).copy()
    weights[mask] = body_weights[ids[mask]]
    g["lbs"]["posedirs"] = gp.reshape(pd.shape[0], -1).astype(np.float32)
    g["lbs"]["lbs_weights"] = weights.astype(np.float32)
    g["lbs"]["v"] = vertices.astype(np.float32)


def radial_clearance(g, body_vertices, body_faces, joints, clearance=.006):
    """Validated v1 radial recentering and body clearance; never fill mesh holes."""
    import trimesh
    import open3d as o3d
    from scipy.ndimage import gaussian_filter1d
    v = array(g["lbs"]["v"]).copy()
    original = v.copy()
    mesh = trimesh.Trimesh(v, array(g["faces"]), process=False)
    weights = array(g["lbs"]["lbs_weights"])
    selected = np.zeros(len(v), bool)
    for _, _, elbow, wrist, arm_joints, _ in SIDES:
        axis = joints[wrist] - joints[elbow]
        length = np.linalg.norm(axis)
        axis /= length
        s = (v - joints[elbow]) @ axis
        mask = (weights[:, arm_joints].sum(1) > .35) & (s > -.14) & (s < length + .10)
        selected |= mask
        samples = np.linspace(-.12, length + .04, 25)
        centers = []
        for position in samples:
            origin = joints[elbow] + position * axis
            lines = trimesh.intersections.mesh_plane(mesh, plane_origin=origin, plane_normal=axis)
            mids = lines.mean(1)
            lens = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
            nearby = np.linalg.norm(mids - origin, axis=1) < .18
            if nearby.any() and lens[nearby].sum() > 0:
                center = np.average(mids[nearby], axis=0, weights=lens[nearby])
                shift = origin - center
                shift -= np.dot(shift, axis) * axis
            else:
                shift = np.zeros(3)
            centers.append(shift)
        centers = gaussian_filter1d(np.asarray(centers), 1, axis=0)
        shift = np.stack([np.interp(s, samples, centers[:, k]) for k in range(3)], 1)
        blend = np.clip((s + .14) / .12, 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        norm = np.linalg.norm(shift, axis=1)
        shift *= np.minimum(1, .08 / np.maximum(norm, 1e-8))[:, None]
        v[mask] += shift[mask] * blend[mask, None]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(body_vertices.astype(np.float32)),
                        o3d.core.Tensor(body_faces.astype(np.uint32)))

    def distance(x):
        return scene.compute_signed_distance(o3d.core.Tensor(x.astype(np.float32))).numpy()

    before = distance(original)
    for _ in range(4):
        sd = distance(v)
        bad = selected & (sd < clearance)
        if not bad.any():
            break
        query = scene.compute_closest_points(o3d.core.Tensor(v[bad].astype(np.float32)))
        closest = query["points"].numpy()
        direction = v[bad] - closest
        direction[sd[bad] < 0] *= -1
        lengths = np.linalg.norm(direction, axis=1)
        valid = lengths > 1e-7
        direction[valid] /= lengths[valid, None]
        direction[~valid] = query["primitive_normals"].numpy()[~valid]
        v[bad] += direction * (clearance - sd[bad])[:, None]
    after = distance(v)
    return v, selected, {"sleeve_inside_over_2mm_before": int(np.sum(selected & (before < -.002))),
                         "sleeve_inside_over_2mm_after": int(np.sum(selected & (after < -.002)))}


def repair(input_dict, reference_dict, body_model_root, gender=None, mode="restore-rebind", clearance=.006):
    """Return a new garment dict, a JSON-safe report, and preview geometry."""
    if mode not in ("restore-rebind", "radial"):
        raise ValueError(f"Unknown repair mode: {mode}")
    if mode == "restore-rebind" and reference_dict is None:
        raise ValueError("restore-rebind requires the unrelaxed reference dictionary")
    validate_garment(input_dict, reference_dict if mode == "restore-rebind" else None)
    gender = gender or input_dict.get("gender")
    if gender not in ("male", "female") or input_dict.get("gender") != gender:
        raise ValueError("Target gender must match the already-bound input dictionary")
    meta = input_dict["fixed_size_metadata"]
    if meta.get("target_gender", gender) != gender:
        raise ValueError("fixed_size_metadata.target_gender does not match input gender")
    import torch
    import smplx
    g = copy.deepcopy(input_dict)
    original = array(g["lbs"]["v"]).copy()
    model = smplx.create(str(body_model_root), model_type="smplx", gender=gender, use_pca=False)
    with torch.no_grad():
        output = model(betas=torch.as_tensor(array(meta["target_betas"])[None], dtype=torch.float32))
    body = output.vertices[0].numpy()
    joints = output.joints[0].numpy()
    body_weights = model.lbs_weights.detach().numpy()
    body_posedirs = model.posedirs.detach().numpy()
    restore_mask = np.zeros(len(original), bool)
    reference_vertices = None
    if mode == "restore-rebind":
        source_lbs = reference_dict["lbs"]
        reference_vertices = array(source_lbs["v"]) + np.einsum(
            "vck,k->vc", array(source_lbs["shapedirs"]), array(meta["source_betas"]))
        source_weights = array(source_lbs["lbs_weights"])
        arm_weight = source_weights[:, [13, 16, 18, 20, 14, 17, 19, 21]].sum(1)
        t = np.clip((np.abs(reference_vertices[:, 0]) - .23) / .20, 0, 1)
        blend = t * t * (3 - 2 * t) * np.clip(arm_weight / .5, 0, 1)
        restore_mask = blend > 0
        if not restore_mask.any():
            raise ValueError("A08 profile found no sleeve vertices; verify canonical axes/units")
        vertices = original + (reference_vertices - original) * blend[:, None]
        rebind_local(g, vertices, restore_mask, body, joints, body_weights, body_posedirs)
    vertices, radial_mask, distances = radial_clearance(g, body, model.faces, joints, clearance)
    if mode == "restore-rebind":
        rebind_local(g, vertices, restore_mask, body, joints, body_weights, body_posedirs)
    else:
        g["lbs"]["v"] = vertices.astype(np.float32)
    # These invariants are deliberate: simulation material/rest mesh is unchanged.
    for field in ("rest_pos", "faces", "node_type"):
        if field in input_dict and not np.array_equal(array(g[field]), array(input_dict[field])):
            raise ValueError(f"Repair unexpectedly changed {field}")
    validate_garment(g)
    changed = np.linalg.norm(array(g["lbs"]["v"]) - original, axis=1)
    report = dict(mode=mode, target_gender=gender, profile="actor08_shirt_validated_v2",
                  clearance_m=clearance, changed_vertices=int(np.sum(changed > 1e-8)),
                  restored_vertices=int(restore_mask.sum()), radial_vertices=int(radial_mask.sum()),
                  max_displacement_cm=float(changed.max() * 100),
                  rest_pos_unchanged=True, topology_unchanged=True, node_type_unchanged=True,
                  lbs_weights_updated=mode == "restore-rebind", posedirs_updated=mode == "restore-rebind",
                  **distances)
    if reference_vertices is not None:
        report["cuff_center_to_canonical_wrist_cm"] = []
        for side, sign, _, wrist, _, _ in SIDES:
            x = reference_vertices[:, 0] * sign
            selected = x > np.quantile(x, .96)
            report["cuff_center_to_canonical_wrist_cm"].append(dict(
                side=side, before=float(np.linalg.norm(original[selected].mean(0) - joints[wrist]) * 100),
                after=float(np.linalg.norm(vertices[selected].mean(0) - joints[wrist]) * 100)))
    g["sleeve_repair"] = copy.deepcopy(report)
    return g, report, (body, model.faces, original, array(g["lbs"]["v"]), array(g["faces"]))


def preview(path, geometry, label="Sleeve repair", full_body=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    body, body_faces, before, after, faces = geometry
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, v, title in zip(axes, (before, after), ("Before", label)):
        polygons = np.concatenate((body[body_faces][:, :, [0, 1]], v[faces][:, :, [0, 1]]))
        depths = np.concatenate((body[body_faces, 2].mean(1), v[faces, 2].mean(1)))
        colors = np.concatenate((np.tile([.7, .75, .8, 1], (len(body_faces), 1)),
                                 np.tile([1, .45, .12, .8], (len(faces), 1))))
        order = np.argsort(depths)
        ax.add_collection(PolyCollection(polygons[order], facecolors=colors[order], edgecolors="none"))
        ax.set_xlim(.95, -.95)
        ax.set_ylim(-.6, .9)
        if full_body:
            all_vertices = np.concatenate((body, before, after))
            lo, hi = all_vertices.min(0), all_vertices.max(0)
            margin = max(float(np.max(hi - lo)) * .05, .02)
            ax.set_xlim(hi[0] + margin, lo[0] - margin)
            ax.set_ylim(lo[1] - margin, hi[1] + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-garment-dict", type=Path, required=True)
    ap.add_argument("--reference-garment-dict", type=Path)
    ap.add_argument("--output-garment-dict", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, required=True)
    ap.add_argument("--target-gender", choices=("male", "female"))
    ap.add_argument("--mode", choices=("restore-rebind", "radial"), default="restore-rebind")
    ap.add_argument("--clearance", type=float, default=.006, help="body clearance in meters")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--preview", type=Path)
    args = ap.parse_args(argv)
    if not 0 < args.clearance <= .03:
        ap.error("clearance must be in (0, 0.03] meters")
    inputs = [args.input_garment_dict, args.reference_garment_dict]
    outputs = [args.output_garment_dict, args.report, args.preview]
    destinations = [p.resolve() for p in outputs if p is not None]
    if len(destinations) != len(set(destinations)):
        ap.error("output, report and preview must use different paths")
    for path in outputs:
        if path is None:
            continue
        if any(src is not None and path.resolve() == src.resolve() for src in inputs):
            ap.error("Never overwrite an input/reference file; choose a separate output")
        if path.exists() or path.is_symlink():
            ap.error(f"Output already exists: {path}; choose a new output")
    result, report, geometry = repair(load_dict(args.input_garment_dict),
        load_dict(args.reference_garment_dict) if args.reference_garment_dict else None,
        args.body_model_root, args.target_gender, args.mode, args.clearance)
    for key, path in (("input_sha256", args.input_garment_dict),
                      ("reference_sha256", args.reference_garment_dict)):
        if path:
            report[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview(args.preview, geometry)
    args.output_garment_dict.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects shared source dictionaries and prior runs.
    with args.output_garment_dict.open("xb") as handle:
        pickle.dump(result, handle, protocol=4)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
