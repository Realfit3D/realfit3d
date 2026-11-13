"""Fit waist supports and feed bounded contact corrections back into rollout."""

from collections import defaultdict
import json
import numpy as np
import torch
from contact_projection import ContactProjector, BodySurface


def cpu(value):
    return value.detach().cpu().numpy()


def rollout(runner, sequence, n_steps, garments, out, config):
    from utils.io import pickle_dump

    out.mkdir(parents=True, exist_ok=False)
    c, b = sequence["cloth"], sequence["obstacle"]
    steps = n_steps
    if b.lookup.shape[1] < steps:
        raise ValueError("Motion is shorter than the requested rollout")
    faces, bf = cpu(c.faces_batch.T), cpu(b.faces_batch.T)
    ids = cpu(c.garment_id).ravel()
    pins = np.flatnonzero(cpu(c.vertex_type).ravel() == 3)
    if not len(pins) or not np.all(ids[pins] == 0):
        raise ValueError(
            "Contact rollout requires pinned lower garment first, then unpinned upper garment"
        )
    if set(ids.tolist()) != {0, 1}:
        raise ValueError("Contact rollout supports exactly two garments: lower, upper")
    raw = np.concatenate(
        [
            cpu(c.prev_pos)[None],
            cpu(c.pos)[None],
            cpu(c.lookup[:, :steps]).transpose(1, 0, 2),
        ]
    )
    bodies = np.concatenate(
        [
            cpu(b.prev_pos)[None],
            cpu(b.pos)[None],
            cpu(b.lookup[:, :steps]).transpose(1, 0, 2),
        ]
    )
    if len(raw) != steps + 2:
        raise ValueError("Unexpected trajectory length")
    projector = ContactProjector(faces, len(ids), **config["projector"])
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    waist = edges[(counts == 1) & np.isin(edges, pins).all(1)]
    if not len(waist):
        raise ValueError("Pinned boundary loop is missing")
    perimeter = lambda v: float(
        np.linalg.norm(v[waist[:, 0]] - v[waist[:, 1]], axis=1).sum()
    )
    target = raw.copy()
    fit_reports = []
    for i in range(len(raw)):
        fitted, report = projector.project(
            raw[i], BodySurface(bodies[i], bf), contact_indices=pins
        )
        ratio = perimeter(fitted) / perimeter(raw[i])
        if ratio > config["max_waist_growth_ratio"]:
            raise RuntimeError(f"Waist growth exceeds configured limit at {i}: {ratio}")
        target[i, pins] = fitted[pins]
        report.update(
            frame=i,
            waist_growth_ratio=ratio,
            waist_before_m=perimeter(raw[i]),
            waist_after_m=perimeter(fitted),
        )
        fit_reports.append(report)
        if i % 30 == 0:
            print("FIT", "rollout", i, report, flush=True)
    (out / "fit_metrics.json").write_text(json.dumps(fit_reports, indent=2))
    initial = target[0].copy()
    initial[pins] = target[0, pins]
    initial, _ = projector.project(
        initial, BodySurface(bodies[0], bf), fixed=pins, fixed_positions=target[0, pins]
    )
    lower = np.flatnonzero(ids == 0)
    upper = np.flatnonzero(ids == 1)
    if not np.array_equal(np.r_[lower, upper], np.arange(len(ids))):
        raise ValueError("Garment vertices must be concatenated in lower,upper order")
    lf = faces[np.all(ids[faces] == 0, axis=1)] - lower[0]
    uf = faces[np.all(ids[faces] == 1, axis=1)] - upper[0]
    upper_projector = ContactProjector(uf, len(upper), **config["projector"])
    lower_surface = BodySurface(initial[lower], lf)
    body_surface = BodySurface(bodies[0], bf)

    class WaistLayer:
        def query(self, points):
            import open3d as o3d

            answer = lower_surface.scene.compute_closest_points(
                o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=2
            )
            closest = answer["points"].numpy()
            _, _, normal = body_surface.query(closest)
            delta = np.asarray(points) - closest
            return (delta * normal).sum(1), np.linalg.norm(delta, axis=1), normal

    signed, distance, _ = WaistLayer().query(initial[upper])
    band = np.flatnonzero(
        (distance < config["waist_layer_radius_m"])
        & (initial[upper, 1] > initial[pins, 1].min() - config["waist_band_below_m"])
    )
    if len(band):
        initial[upper], _ = upper_projector.project(
            initial[upper], WaistLayer(), contact_indices=band
        )
    initial, _ = projector.project(
        initial, body_surface, fixed=pins, fixed_positions=target[0, pins]
    )
    tensor = lambda a: torch.as_tensor(a, dtype=c.pos.dtype, device=c.pos.device)
    c.lookup = tensor(target[2:].transpose(1, 0, 2))
    b.lookup = b.lookup[:, :steps]
    for i, key in enumerate(["prev_pos", "pos", "target_pos"]):
        c[key] = tensor(target[i])
    phase = ["relaxation"]
    reports = []
    lower_mask = torch.as_tensor(np.all(ids[faces] == 0, axis=1), device=c.pos.device)[
        None
    ]

    def hook(module, args, result):
        cloth, body = result["cloth"], result["obstacle"]
        prediction = cpu(cloth.pred_pos)
        surface = BodySurface(cpu(body.target_pos), bf)
        support = cpu(cloth.target_pos)[pins]
        value, record = projector.project(
            prediction, surface, fixed=pins, fixed_positions=support
        )
        cloth.pred_pos = tensor(value)
        cloth.pred_velocity = cloth.pred_pos - cloth.pos
        impulses = defaultdict(list)
        for _ in range(config["lower_impulse_passes"] if phase[0] == "motion" else 0):
            mask = cloth.faces_cutout_mask_batch
            cloth.faces_cutout_mask_batch = mask & lower_mask
            try:
                result = runner.safecheck_solver.safecheck_impulses(result, impulses)
            finally:
                cloth.faces_cutout_mask_batch = mask
            value, _ = projector.project(
                cpu(cloth.pred_pos), surface, fixed=pins, fixed_positions=support
            )
            cloth.pred_pos = tensor(value)
            cloth.pred_velocity = cloth.pred_pos - cloth.pos
        distance = float(np.linalg.norm(cpu(cloth.pred_pos) - prediction, axis=1).max())
        if distance > config["projector"]["max_displacement"]:
            raise RuntimeError(
                f"Combined contact displacement exceeds limit: {distance}"
            )
        record.update(
            phase=phase[0],
            step=sum(r["phase"] == phase[0] for r in reports),
            combined_correction_max_m=distance,
        )
        reports.append(record)
        if record["step"] % 20 == 0:
            (out / "contact_metrics.json").write_text(json.dumps(reports, indent=2))
            print("CONTACT", "rollout", record, flush=True)
        return result

    handle = runner.model.register_forward_hook(hook)
    try:
        static = sequence.clone()
        for obj, vertices in [("cloth", initial), ("obstacle", bodies[0])]:
            v = tensor(vertices)
            for key in ["prev_pos", "pos", "target_pos"]:
                static[obj][key] = v.clone()
            static[obj].lookup = v[:, None].repeat(1, config["relaxation_steps"], 1)
        relaxed = runner.valid_rollout(
            static, bare=True, n_steps=config["relaxation_steps"]
        )
        relaxed["garment_names"] = garments
        pickle_dump(dict(relaxed), out / "relaxation.pkl")
        initial = relaxed["pred"][-1]
        for i, key in enumerate(["prev_pos", "pos"]):
            v = initial + target[i] - target[0]
            v[pins] = target[i, pins]
            v, _ = projector.project(
                v,
                BodySurface(bodies[i], bf),
                fixed=pins,
                fixed_positions=target[i, pins],
            )
            c[key] = tensor(v)
        phase[0] = "motion"
        trajectory = runner.valid_rollout(sequence, bare=True, n_steps=steps)
    finally:
        handle.remove()
    trajectory["garment_names"] = garments
    if not np.array_equal(trajectory["obstacle"], bodies):
        raise RuntimeError("Rollout altered body motion")
    if not np.array_equal(trajectory["cloth_faces"], faces):
        raise RuntimeError("Rollout altered topology")
    if not np.allclose(trajectory["pred"][:, pins], target[:, pins], atol=3e-7):
        raise RuntimeError("Support targets were not preserved")
    (out / "contact_metrics.json").write_text(json.dumps(reports, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return trajectory
