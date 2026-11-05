"""Command boundary for the preserved handoff pipeline, without GPU imports."""

import json
import os
import subprocess
from pathlib import Path

from realfit3d.settings import runtime_environment
from realfit3d.assets import inspect_asset
from realfit3d.contracts import opacity_parameters
from realfit3d.fitting import repair_config

WORKERS = Path(__file__).resolve().parents[1] / "workers" / "contourcraft"
ENTRY = WORKERS.parent / 'runtime_entry.py'


class RuntimeUnavailable(RuntimeError):
    pass


class PreviewRuntime:
    def __init__(self, settings):
        self.settings = settings

    def path(self, key):
        value = self.settings.get(key)
        if not value:
            raise RuntimeUnavailable(f"GPU runtime is not configured: {key}")
        path = Path(value).expanduser()
        return path.absolute() if key.endswith('_python') else path.resolve()

    def check(self, full=False, motion=None):
        keys = ["ccraft_python", "contourcraft_root", "ccraft_data_root", "cmu_root", "body_model_root", "reference_motion"]
        if full:
            keys += ["gaugar_python", "gaugar_root", "checkpoint", "gg_data_root", "gg_output_root", "gg_aux_root"]
        for key in keys:
            path = self.path(key)
            if not path.exists():
                raise RuntimeUnavailable(f"Missing runtime path: {key}: {path}")
            if key.endswith("python") and not os.access(path, os.X_OK):
                raise RuntimeUnavailable(f"Python executable is not executable: {path}")
        if full:
            for filename in ["render_lean.py", "render_turntable.py", "inference.py"]:
                if not (self.path("gaugar_root") / filename).is_file():
                    raise RuntimeUnavailable(f"GG fork requires {filename}; initialize the pinned submodule")
            if motion and not self.motion_path(motion).is_file():
                raise RuntimeUnavailable(f"Missing motion: {motion}")

    def motion_path(self, motion):
        relative = Path(motion)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeUnavailable('Motion must be a relative catalog path')
        if relative.parts[:1] == ('CMU',):
            return self.path('cmu_root').joinpath(*relative.parts[1:])
        return self.path('cmu_root') / relative

    def check_assets(self, names):
        for name in names:
            report = inspect_asset(self.path('gg_output_root'), name)
            missing = [item['path'] for item in report['files'] if not item['exists']]
            if missing:
                raise RuntimeUnavailable('Missing garment render assets: ' + ', '.join(missing))

    def fit(self, directory, gender, height, weight):
        common = [str(self.path("ccraft_python"))]
        commands = [
            common + [str(WORKERS / "fit_body_from_measurements_fast.py"),
                      "--body-model-root", str(self.path("body_model_root")),
                      "--gender", gender, "--height", str(height), "--weight", str(weight),
                      "--output-betas", str(directory / "betas.pkl"), "--patience", "50"],
            common + [str(WORKERS / "render_body_preview.py"),
                      "--betas-path", str(directory / "betas.pkl"),
                      "--output-image", str(directory / "body_preview.png"),
                      "--body-model-root", str(self.path("body_model_root")),
                      "--reference-motion", str(self.path("reference_motion"))],
        ]
        for command in commands:
            subprocess.run(self.wrap_cc(command), cwd=self.path("contourcraft_root"), check=True,
                           capture_output=True, text=True, env=self.environment())
        for name in ["betas.pkl", "body_preview.png"]:
            if not (directory / name).is_file():
                raise RuntimeError(f"Worker did not produce {name}")

    def opacity_options(self, upper, lower, mode="direct", scale=1.0):
        options = opacity_parameters(mode, scale)
        arguments = ["--opacity-mode", options["opacity_mode"], "--opacity-scale", str(options["opacity_scale"])]
        if mode == "v3":
            for garment in (lower, upper):
                relative = garment.get("opacity_controller")
                if not relative:
                    raise ValueError(f"No V3 controller configured for {garment['name']}")
                root = self.path("gg_output_root") / garment["name"]
                path = (root / relative).resolve()
                if root.resolve() not in path.parents:
                    raise ValueError("V3 controller must be inside its garment asset directory")
                if not path.is_file():
                    raise RuntimeUnavailable(f"Missing V3 controller: {path}")
                arguments.extend(["--v3-controller", f"{garment['name']}={path}"])
        return arguments

    def command(self, body, upper, lower, motion, run_name, outfit, output,
                opacity_mode="direct", opacity_scale=1.0):
        command = [str(self.path("ccraft_python")), str(WORKERS / "run_outfit_preview_pipeline.py"),
                   "--garment-names", lower["name"], upper["name"],
                   "--source-genders", lower["source_gender"], upper["source_gender"],
                   "--gender", body["gender"], "--height", str(body["height"]),
                   "--weight", str(body["weight"]), "--precomputed-betas", str(body["betas_path"]),
                   "--motion", str(self.motion_path(motion)),
                   "--run-name", run_name, "--outfit-name", outfit, "--output-root", str(output),
                   "--pin-waistband-garments", lower["name"],
                   "--pin-waistband-if-snug-garments", lower["name"]]
        for key in ["contourcraft_root", "gaugar_root", "body_model_root", "checkpoint", "ccraft_python", "gaugar_python"]:
            command.extend(["--" + key.replace("_", "-"), str(self.path(key))])
        command.extend(self.opacity_options(upper, lower, opacity_mode, opacity_scale))
        repairs = repair_config(self.settings, [lower['name'], upper['name']], body['gender'])
        if repairs:
            command.extend(['--garment-repairs-json', json.dumps(repairs)])
        return self.wrap_cc(command)

    def wrap_cc(self, command):
        return [command[0], str(ENTRY), '--kind', 'cc', '--root', str(self.path('contourcraft_root')),
                '--script', command[1], '--', *command[2:]]

    def environment(self):
        return runtime_environment(self.settings, self.path('ccraft_python'))

    def launch(self, command, log_path):
        with log_path.open("w") as log:
            return subprocess.Popen(command, cwd=self.path("contourcraft_root"), stdout=log,
                                    stderr=subprocess.STDOUT, env=self.environment())


def read_status(info):
    try:
        payload = json.loads((info["dir"] / "status.json").read_text())
        if not isinstance(payload, dict):
            raise ValueError("Status must be an object")
    except (OSError, ValueError):
        payload = {"phase": "starting", "message": "Waiting for worker status"}
    code = info["process"].poll()
    if code is not None and (code != 0 or payload.get("phase") not in ("done", "error")):
        payload.update(phase="error", message=f"Worker exited ({code}); inspect the run log.")
    if payload.get("preview_ready"):
        payload["turntable_frames"] = len(list((info["dir"] / "turntable").glob("angle_*.png")))
    if payload.get("phase") == "full_rendering":
        payload["rendered_frames"] = len(list((info["dir"] / "frames/renders").glob("*.png")))
    return payload
