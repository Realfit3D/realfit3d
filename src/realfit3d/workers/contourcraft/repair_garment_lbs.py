#!/usr/bin/env python3
"""Reference-guided, anatomical-region garment initialization repair (SMPL-X).

Default CLI operation diagnoses only. An explicit region profile and --apply
are required to write a repaired dict. Does not change rest mesh or simulation.
"""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from repair_sleeve_lbs import array, load_dict, validate_garment, preview

# Anatomical membership, not world-coordinate crop boxes. SMPL-X 55-joint order.
REGIONS = {
    "left_arm": ([13, 16, 18, 20] + list(range(25, 40)), (16, 18, 20)),
    "right_arm": ([14, 17, 19, 21] + list(range(40, 55)), (17, 19, 21)),
    "left_leg": ([1, 4, 7, 10], (1, 4, 7)),
    "right_leg": ([2, 5, 8, 11], (2, 5, 8)),
    "torso": ([0, 3, 6, 9, 12, 15], (0, 3, 6, 9, 12)),
}
DEFAULTS = {
    "mode": "restore-rebind", "regions": [],
    "membership_threshold": .45, "body_membership_threshold": .5,
    "damage_fraction": .10, "max_displacement_fraction": .75,
    "max_binding_distance_fraction": .5, "max_edge_strain": .35,
    "max_support_fraction": .8,
    "centering_fraction": 0.0,
    "align_limb_ends": False,
    "restore_trigger": "spatial",
    "clearance_m": 0.0,
}


def validate_profile(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Profile must contain only: " + ", ".join(DEFAULTS))
    cfg = {**DEFAULTS, **config}
    if cfg["mode"] not in ("restore-rebind", "rebind"):
        raise ValueError("mode must be restore-rebind or rebind")
    if cfg["restore_trigger"] not in ("spatial", "axial"):
        raise ValueError("restore_trigger must be spatial or axial")
    if (not isinstance(cfg["regions"], list) or
            any(r not in REGIONS for r in cfg["regions"]) or
            len(set(cfg["regions"])) != len(cfg["regions"])):
        raise ValueError("regions must be unique anatomical region names: " + ", ".join(REGIONS))
    if not isinstance(cfg["align_limb_ends"], bool):
        raise ValueError("align_limb_ends must be a boolean")
    for key in set(DEFAULTS) - {"mode", "regions", "align_limb_ends", "restore_trigger"}:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if not 0 < value <= 1 and key not in ("clearance_m", "centering_fraction"):
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 <= cfg["clearance_m"] <= .03:
        raise ValueError("clearance_m must be in [0, .03]")
    if not 0 <= cfg["centering_fraction"] <= .5:
        raise ValueError("centering_fraction must be in [0, .5]")
    if cfg["membership_threshold"] >= 1:
        raise ValueError("membership_threshold must be less than 1")
    return cfg


def nearest(points, candidates):
    """Nearest vertex transfer with a bounded-memory NumPy fallback."""
    if not len(candidates):
        raise ValueError("No anatomically matching target body vertices")
    try:
        from scipy.spatial import cKDTree
        return cKDTree(candidates).query(points)
    except ImportError:
        distances, indices = [], []
        for start in range(0, len(points), 32):
            d2 = np.sum((points[start:start + 32, None] - candidates[None]) ** 2, axis=2)
            ids = d2.argmin(1)
            indices.extend(ids.tolist())
            distances.extend(np.sqrt(d2[np.arange(len(ids)), ids]).tolist())
        return np.asarray(distances), np.asarray(indices, dtype=int)


def edge_strain(vertices, reference, faces, selected):
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]])), axis=1), axis=0)
    edges = edges[selected[edges].any(1)]
    lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    valid = lengths > 1e-7
    actual = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(np.max(np.abs(actual[valid] / lengths[valid] - 1))) if valid.any() else 0.


def project_clearance(vertices, selected, body, faces, clearance):
    if not clearance or not selected.any():
        return vertices
    import open3d as o3d
    from body_collider import prepare_body_collider
    body, faces, _ = prepare_body_collider(body, faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(body.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    v = vertices.copy()
    def distance(points):
        return scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)), nsamples=5).numpy()
    active = np.flatnonzero(selected)
    for _ in range(20):
        sd = distance(v[active])
        bad = sd < clearance - 1e-5
        if not bad.any():
            break
        active = active[bad]
        sd = sd[bad]
        result = scene.compute_closest_points(o3d.core.Tensor(v[active].astype(np.float32)))
        direction = v[active] - result["points"].numpy()
        direction[sd < 0] *= -1
        length = np.linalg.norm(direction, axis=1)
        valid = length > 1e-7
        direction[valid] /= length[valid, None]
        direction[~valid] = result["primitive_normals"].numpy()[~valid]
        v[active] += direction * (clearance - sd)[:, None]
    remaining = distance(v[selected])
    # Near an armpit or a narrow gap, alternating closest-surface normals can
    # oscillate. Search bounded escape directions only for those stuck points.
    stuck = np.flatnonzero(selected)[remaining < clearance - 1e-4]
    if len(stuck):
        k = np.arange(64)
        z = 1 - 2 * (k + .5) / 64
        theta = k * np.pi * (3 - np.sqrt(5))
        directions = np.column_stack((np.sqrt(1 - z * z) * np.cos(theta),
                                      np.sqrt(1 - z * z) * np.sin(theta), z))
        for index in stuck:
            origin = v[index].copy()
            for radius in (.003, .006, .012, .024, .048, .096):
                points = origin + radius * directions
                values = distance(points)
                valid = np.flatnonzero(values >= clearance)
                if not len(valid):
                    continue
                direction = directions[valid[np.argmax(values[valid])]]
                lo, hi = 0., radius
                for _ in range(12):
                    mid = (lo + hi) / 2
                    value = distance((origin + mid * direction)[None])[0]
                    if value >= clearance:
                        hi = mid
                    else:
                        lo = mid
                v[index] = origin + hi * direction
                break
        remaining = distance(v[selected])
    if np.any(remaining < clearance - 1e-4):
        raise ValueError(f"Requested body clearance could not be reached: min {remaining.min():.6f} m")
    return v


def repair_with_body(input_dict, reference, body, body_faces, joints, body_weights, body_posedirs,
                     config, apply=False):
    """Pure repair core; caller supplies the personalized canonical body arrays.

    Every anatomical mask is derived from the trusted reference, never the
    potentially damaged input binding. A failed gate rejects the whole candidate.
    """
    cfg = validate_profile(config)
    validate_garment(input_dict, reference)
    if apply and not cfg["regions"]:
        raise ValueError("Apply requires explicitly reviewed regions")
    body, joints, body_weights, body_posedirs = map(array, (body, joints, body_weights, body_posedirs))
    if (body.ndim != 2 or body.shape[1] != 3 or body_weights.shape != (len(body), 55)
            or body_posedirs.shape != (486, len(body) * 3) or joints.shape[0] < 55):
        raise ValueError("Expected canonical SMPL-X body geometry and 55-joint skinning")
    if not all(np.isfinite(x).all() for x in (body, joints, body_weights, body_posedirs)):
        raise ValueError("Non-finite target body")
    ref_lbs = reference["lbs"]
    ref = array(ref_lbs["v"]) + np.einsum("vck,k->vc", array(ref_lbs["shapedirs"]),
                                         array(input_dict["fixed_size_metadata"]["source_betas"]))
    ref_weights = array(ref_lbs["lbs_weights"])
    if (not np.isfinite(ref).all() or not np.isfinite(ref_weights).all() or
            np.any(ref_weights < 0) or not np.allclose(ref_weights.sum(1), 1, atol=1e-4)):
        raise ValueError("Invalid canonical reference vertices/weights")
    original = array(input_dict["lbs"]["v"])
    vertices = original.copy()
    faces = array(input_dict["faces"])
    chosen = np.zeros(len(vertices), bool)
    restoration = np.zeros(len(vertices), bool)
    seed = np.zeros(len(vertices), bool)
    anchors = ref.copy()
    transition_report = None
    assigned = np.full(len(vertices), -1, dtype=int)
    scale_at = np.ones(len(vertices))
    rows, rejected = [], []
    # Winner-takes-membership prevents contradictory left/right restoration.
    membership = np.stack([ref_weights[:, ids].sum(1) for ids, _ in REGIONS.values()], axis=1)
    winner = membership.argmax(1)
    source_ref = ref.copy()
    axial_loss = {}
    spatial_deviation = {}
    for index, (name, (_, chain)) in enumerate(REGIONS.items()):
        axis = joints[chain[-1]] - joints[chain[0]]
        axis /= max(np.linalg.norm(axis), 1e-8)
        scale = sum(np.linalg.norm(joints[b] - joints[a]) for a, b in zip(chain, chain[1:]))
        mask = (winner == index) & (membership[:, index] > cfg["membership_threshold"])
        axial_loss[name] = float(np.quantile(((source_ref - original) @ axis)[mask] / max(scale, 1e-8), .9)) if mask.any() else 0.
        spatial_deviation[name] = float(np.quantile(np.linalg.norm(source_ref[mask] - original[mask], axis=1) / max(scale, 1e-8), .9)) if mask.any() else 0.
    fit_only = (cfg["mode"] == "restore-rebind" and cfg["restore_trigger"] == "axial"
                and bool(cfg["centering_fraction"] or cfg["clearance_m"])
                and any(spatial_deviation[name] > cfg["damage_fraction"] for name in cfg["regions"])
                and not any(axial_loss[name] > cfg["damage_fraction"] for name in cfg["regions"]))
    if fit_only:
        # Lateral displacement is not evidence of sleeve/leg length collapse.
        # Preserve the valid input's intrinsic geometry while fitting it to the
        # target body. Report the distinct strain-reference frame explicitly.
        ref = original.copy()
        anchors = ref.copy()
    for region_index, (name, (joint_ids, chain)) in enumerate(REGIONS.items()):
        scale = float(sum(np.linalg.norm(joints[b] - joints[a]) for a, b in zip(chain, chain[1:])))
        if scale < 1e-5:
            raise ValueError(f"Degenerate anatomical chain: {name}")
        mass = membership[:, region_index]
        mask = (winner == region_index) & (mass > cfg["membership_threshold"])
        row = dict(region=name, vertices=int(mask.sum()), chain_length_m=scale,
                   requested=name in cfg["regions"], proposed=False)
        if not mask.any():
            if row["requested"]:
                rejected.append(f"{name}: no matching reference vertices")
            rows.append(row)
            continue
        delta = np.linalg.norm(original[mask] - source_ref[mask], axis=1) / scale
        row["reference_deviation_p90_fraction"] = float(np.quantile(delta, .9))
        row["reference_deviation_max_fraction"] = float(delta.max())
        row["axial_loss_p90_fraction"] = axial_loss[name]
        candidates = np.flatnonzero(body_weights[:, joint_ids].sum(1) >= cfg["body_membership_threshold"])
        distance, local = nearest(original[mask], body[candidates])
        row["binding_distance_before_p95_fraction"] = float(np.quantile(distance / scale, .95))
        row["binding_region_agreement_before"] = float(np.mean(
            array(input_dict["lbs"]["lbs_weights"])[mask][:, joint_ids].sum(1)))
        if not row["requested"]:
            rows.append(row)
            continue
        if cfg["mode"] == "restore-rebind":
            trigger = axial_loss[name] if cfg["restore_trigger"] == "axial" else row["reference_deviation_p90_fraction"]
            if trigger <= cfg["damage_fraction"] and not fit_only:
                row["reason"] = "below restoration trigger; kept unchanged"
                rows.append(row)
                continue
            row["repair_action"] = "input-preserving body fit" if fit_only else "reference restoration"
            restoration |= mask
            # Distal anatomical vertices anchor restoration. The rest of the
            # transition is solved on mesh connectivity, not skinning-weight cuts.
            axis = joints[chain[-1]] - joints[chain[0]]
            length = np.linalg.norm(axis)
            if length < 1e-6:
                raise ValueError(f"Degenerate anatomical direction: {name}")
            progress = (ref - joints[chain[0]]) @ (axis / length)
            core = mask & (progress >= np.quantile(progress[mask], .65))
            if cfg["centering_fraction"] and name != "torso":
                from reference_transition import centered_anchors
                target = centered_anchors(ref, faces, mass, joints[list(chain)], cfg["centering_fraction"], cfg["align_limb_ends"])
                anchors[core] = target[core]
            seed |= core
        chosen |= mask
        scale_at[mask] = scale
        assigned[mask] = candidates[local]
        row["proposed"] = True
        rows.append(row)
    if restoration.any():
        from reference_transition import harmonic_transition
        vertices, changed, transition_report = harmonic_transition(
            original, ref, faces, seed, restoration, cfg["max_edge_strain"], cfg["max_support_fraction"],
            anchor_positions=anchors)
        chosen |= changed
        if not transition_report["history"]:
            rejected.append("Topology transition exceeded its support budget before solving")
    contact_history = []
    collider_report = None
    try:
        collision_body, collision_faces = body, body_faces
        if cfg["clearance_m"] and chosen.any():
            from body_collider import prepare_body_collider
            allowed = body_weights[:, [12, 15, 22, 23, 24]].sum(1) > .5
            collision_body, collision_faces, collider_report = prepare_body_collider(body, body_faces, allowed)
        vertices = project_clearance(vertices, chosen, collision_body, collision_faces, cfg["clearance_m"])
        if cfg["clearance_m"] and chosen.any():
            from reference_transition import constrain_lengths, quality
            for contact_step in range(64):
                e, _, _ = quality(vertices, ref, faces, chosen)
                vertices = constrain_lengths(vertices, ref, e, chosen, chosen[e].any(1), cfg["max_edge_strain"], iterations=1200, faces=faces)
                checked = project_clearance(vertices, chosen, collision_body, collision_faces, cfg["clearance_m"])
                remaining = float(np.linalg.norm(checked - vertices, axis=1).max())
                _, current_strain, normal_flags = quality(vertices, ref, faces, chosen)
                contact_history.append(dict(step=contact_step + 1, clearance_correction_m=remaining,
                                            max_edge_strain=float(current_strain.max()),
                                            normal_reversal_proxy=int(normal_flags.sum()),
                                            support_vertices=int(chosen.sum())))
                if remaining <= 1e-4 and current_strain.max() <= cfg["max_edge_strain"] and not normal_flags.any():
                    break
                # Free only locked neighbors implicated by a contact/edge
                # conflict. Otherwise a fixed penetrating boundary can make
                # the two constraints infeasible, regardless of iteration count.
                ce, cs, cf = quality(checked, ref, faces, chosen)
                grow = chosen.copy()
                grow[ce[cs > cfg["max_edge_strain"]].ravel()] = True
                grow[faces[cf].ravel()] = True
                if grow.sum() <= cfg["max_support_fraction"] * len(vertices):
                    chosen = grow
                if contact_step < 63:
                    vertices = checked
            if remaining > 1e-4:
                rejected.append("Body clearance and intrinsic edge constraints did not converge")
    except ValueError as exc:
        rejected.append(str(exc))
    for row, (name, (joint_ids, _)) in zip(rows, REGIONS.items()):
        mask = chosen & (winner == list(REGIONS).index(name))
        if not mask.any():
            continue
        chain = REGIONS[name][1]
        scale_at[mask] = sum(np.linalg.norm(joints[b] - joints[a]) for a, b in zip(chain, chain[1:]))
        row["rebound_support_vertices"] = int(mask.sum())
        candidates = np.flatnonzero(body_weights[:, joint_ids].sum(1) >= cfg["body_membership_threshold"])
        distance, local = nearest(vertices[mask], body[candidates])
        assigned[mask] = candidates[local]
        row["binding_distance_after_max_fraction"] = float((distance / scale_at[mask]).max())
        row["max_displacement_fraction"] = float((np.linalg.norm(vertices[mask] - original[mask], axis=1) / scale_at[mask]).max())
        if row["binding_distance_after_max_fraction"] > cfg["max_binding_distance_fraction"]:
            rejected.append(f"{name}: target body too far from candidate")
        if row["max_displacement_fraction"] > cfg["max_displacement_fraction"]:
            rejected.append(f"{name}: excessive initialization displacement")
    strain = edge_strain(vertices, ref, faces, chosen)
    if strain > cfg["max_edge_strain"]:
        rejected.append("Excessive edge distortion relative to source reference")
    # Conservative reference-normal proxy; not proof of a topological inversion.
    tri_ref, tri_new = ref[faces], vertices[faces]
    nr = np.cross(tri_ref[:, 1] - tri_ref[:, 0], tri_ref[:, 2] - tri_ref[:, 0])
    nn = np.cross(tri_new[:, 1] - tri_new[:, 0], tri_new[:, 2] - tri_new[:, 0])
    flipped = chosen[faces].any(1) & (np.sum(nr * nn, axis=1) <= 0) & (np.linalg.norm(nr, axis=1) > 1e-10)
    if flipped.any():
        rejected.append("Candidate triggered reference-normal reversal/collapse proxy")
    report = dict(profile="anatomical_reference_v2_harmonic_experimental", config=cfg, regions=rows,
                  gates_passed=not rejected, rejection_reasons=rejected,
                  proposed_vertices=int(chosen.sum()), applied=bool(apply and not rejected),
                  max_edge_strain=strain, flipped_triangles=int(flipped.sum()),
                  normal_reversal_proxy=int(flipped.sum()), transition=transition_report,
                  shape_reference="input_preserving_fit" if fit_only else "unrelaxed_source",
                  contact_refinement=contact_history if cfg["clearance_m"] else [],
                  body_collider=collider_report,
                  rest_pos_unchanged=True, topology_unchanged=True)
    result = copy.deepcopy(input_dict)
    if report["applied"]:
        ids = assigned[chosen]
        weights = array(result["lbs"]["lbs_weights"]).copy()
        pd = array(result["lbs"]["posedirs"]).reshape(486, -1, 3).copy()
        weights[chosen] = body_weights[ids]
        pd[:, chosen] = body_posedirs.reshape(486, -1, 3)[:, ids]
        result["lbs"]["v"] = vertices.astype(original.dtype)
        result["lbs"]["lbs_weights"] = weights
        result["lbs"]["posedirs"] = pd.reshape(486, -1)
        result["garment_repair"] = copy.deepcopy(report)
        validate_garment(result)
    geometry = (body, body_faces, original, vertices, faces)
    return result, report, geometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-garment-dict", type=Path, required=True)
    parser.add_argument("--reference-garment-dict", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--output-garment-dict", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.apply != bool(args.output_garment_dict):
        parser.error("Use --apply and --output-garment-dict together; omit both to diagnose")
    paths = [p for p in (args.report, args.preview, args.output_garment_dict) if p]
    inputs = {p.resolve() for p in (args.input_garment_dict, args.reference_garment_dict, args.profile)}
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Outputs must have distinct paths")
    if any(p.exists() or p.is_symlink() or p.resolve() in inputs for p in paths):
        parser.error("Never overwrite an input or an existing output; use a new run directory")
    config = validate_profile(json.loads(args.profile.read_text(encoding="utf-8")))
    g, reference = load_dict(args.input_garment_dict), load_dict(args.reference_garment_dict)
    validate_garment(g, reference)
    gender = g.get("gender")
    if gender not in ("male", "female") or g["fixed_size_metadata"].get("target_gender", gender) != gender:
        parser.error("Input gender/target gender mismatch")
    import torch
    import smplx
    model = smplx.create(str(args.body_model_root), model_type="smplx", gender=gender, use_pca=False)
    with torch.no_grad():
        output = model(betas=torch.as_tensor(array(g["fixed_size_metadata"]["target_betas"])[None], dtype=torch.float32))
    result, report, geometry = repair_with_body(g, reference, output.vertices[0].numpy(), model.faces,
        output.joints[0].numpy(), model.lbs_weights.detach().numpy(), model.posedirs.detach().numpy(), config, args.apply)
    report["target_gender"] = gender
    for key, path in (("input", args.input_garment_dict), ("reference", args.reference_garment_dict), ("profile", args.profile)):
        report[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if args.preview:
        preview(args.preview, geometry, label="Candidate: " + ("gates passed" if report["gates_passed"] else "REJECTED"),
                full_body=True)
    if args.apply:
        if not report["gates_passed"]:
            raise SystemExit("Candidate rejected; inspect report. No repaired dictionary written.")
        result["garment_repair"] = copy.deepcopy(report)
        with args.output_garment_dict.open("xb") as handle:
            pickle.dump(result, handle, protocol=4)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
