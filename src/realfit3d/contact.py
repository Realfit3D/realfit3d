"""Validate contact settings without importing the simulation runtime."""

import math

DEFAULTS = dict(
    schema_version=1,
    relaxation_steps=30,
    lower_impulse_passes=2,
    max_waist_growth_ratio=1.06,
    waist_layer_radius_m=0.025,
    waist_band_below_m=0.08,
    seed=20261112,
    projector=dict(
        clearance=0.003,
        minimum_clearance=0.002,
        smoothness=12.0,
        max_displacement=0.035,
        near_radius=0.05,
        tolerance=2e-5,
    ),
)


def contact_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown contact configuration fields")
    result = {**DEFAULTS, **value}
    projector = value.get("projector", {})
    if not isinstance(projector, dict) or set(projector) - set(DEFAULTS["projector"]):
        raise ValueError("Unknown contact projector fields")
    result["projector"] = {**DEFAULTS["projector"], **projector}
    if result["schema_version"] != 1:
        raise ValueError("Contact schema_version must be 1")
    for key in ("relaxation_steps", "lower_impulse_passes", "seed"):
        if type(result[key]) is not int or result[key] < (
            1 if key == "relaxation_steps" else 0
        ):
            raise ValueError(f"Invalid contact setting: {key}")
    values = [
        result[k]
        for k in (
            "max_waist_growth_ratio",
            "waist_layer_radius_m",
            "waist_band_below_m",
        )
    ]
    values += list(result["projector"].values())
    if any(
        not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
        for x in values
    ):
        raise ValueError(
            "Contact distances and solver parameters must be finite and positive"
        )
    p = result["projector"]
    if (
        not p["minimum_clearance"] <= p["clearance"] < p["near_radius"]
        or result["max_waist_growth_ratio"] < 1
    ):
        raise ValueError("Invalid contact clearance/growth ranges")
    return result


def arguments(settings):
    import json

    if settings.get("contact_correction") is None:
        return []
    return [
        "--contact-config-json",
        json.dumps(contact_config(settings["contact_correction"])),
    ]
