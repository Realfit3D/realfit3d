"""Render one batch job through the shared CC worker and pinned GG renderer."""

import shutil
import json
import subprocess
import time
from uuid import uuid4
from pathlib import Path
from realfit3d.adapters.preview import PreviewRuntime, WORKERS
from realfit3d.fitting import repair_config

from realfit3d.precompute.toolkit_common import (
    DISCARD_FRAMES, Job, ToolkitError, atomic_write_json, ensure_cache_copies,
    find_v3_controller, make_sidecar, motion_file, fail, steps_for_frames,
)


def _safe(s: str) -> str:
    return s.replace("+", "_")


def run_name(job: Job) -> str:
    return f"{job.gender}_{job.body_dir}_{job.video_stem}_f{job.frames}"


STAGE_NAMES = {
    "recovering_garments": "準備衣服", "untangling": "衣服纏繞修正(untangle)", "fitting_body": "擬合身體",
    "rebinding": "衣服綁定到身體", "full_simulating": "布料模擬", "full_rendering": "算圖與合成影片",
}


def stage_timings(log: Path) -> dict:
    """Seconds spent in each pipeline stage, from the `[status] <phase> (t=12.3s)` lines the pipeline
    prints (t = seconds since the pipeline started). Returns {} if the log has none."""
    import re
    marks = []
    try:
        for line in Path(log).read_text(errors="replace").splitlines():
            m = re.match(r"\[status\] (\w+) \(t=([\d.]+)s\)", line)
            if m:
                marks.append((m.group(1), float(m.group(2))))
    except OSError:
        return {}
    out = {}
    for (name, t0), (_, t1) in zip(marks, marks[1:]):
        if name in STAGE_NAMES:
            out[name] = round(out.get(name, 0.0) + (t1 - t0), 1)
    return out


def log_path(paths: dict, job: Job) -> Path:
    return Path(paths["work_root"]) / "logs" / _safe(job.outfit) / f"{run_name(job)}.log"


def betas_cache(paths: dict, job: Job) -> Path:
    return Path(paths["work_root"]) / "_betas" / f"{job.gender}_{job.body_dir}" / "betas.pkl"


def _pipeline_cmd(paths, catalog, job, run_dir_name, precomputed_betas):
    script = WORKERS / "run_outfit_preview_pipeline.py"
    if not script.is_file():
        fail(f"Missing packaged worker: {script}")
    source_genders = [catalog["garments"][g]["source_gender"] for g in job.garments]
    cmd = [str(paths["contourcraft_python"]), str(script),
           "--garment-names", *job.garments,
           "--outfit-name", job.work_name,
           "--height", f"{job.height / 100:.2f}",
           "--weight", str(job.weight),
           "--gender", job.gender,
           "--source-genders", *source_genders,
           "--motion", str(motion_file(paths, catalog, job.motion)),
           "--full-steps", str(steps_for_frames(job.frames)),
           "--discard-frames", str(DISCARD_FRAMES),
           "--fps", str(job.fps),
           "--opacity-scale", str(job.opacity),
           "--opacity-mode", job.opacity_mode,
           "--skip-preview",
           "--run-name", run_dir_name,
           "--contourcraft-root", str(paths["contourcraft_root"]),
           "--gaugar-root", str(paths["gaugar_root"]),
           "--ccraft-python", str(paths["contourcraft_python"]),
           "--gaugar-python", str(paths["gaugar_python"]),
           "--body-model-root", str(paths['body_model_root']),
           "--checkpoint", str(paths['checkpoint']),
           "--output-root", str(paths["work_root"])]
    if job.opacity_garments:
        cmd += ["--opacity-garments", *job.opacity_garments]
    if precomputed_betas is not None:
        cmd += ["--precomputed-betas", str(precomputed_betas)]
    if job.opacity_mode == "v3":
        for garment in job.effective_opacity_garments:
            controller = find_v3_controller(paths, garment)
            if controller is None:
                fail(
                    f"`{garment}` 找不到 V3 controller.pt。請放到 "
                    "data_root/garments/<衣服>/opacity_controller_v3/controller.pt，"
                    "或在 runtime configuration 的 v3_controllers 指定路徑。"
                )
            cmd += ["--v3-controller", f"{garment}={controller.resolve()}"]
    repairs = repair_config(paths, job.garments, job.gender)
    if repairs:
        cmd += ['--garment-repairs-json', json.dumps(repairs)]
    return PreviewRuntime(paths).wrap_cc(cmd)


def render_video(paths: dict, catalog: dict, job: Job, dest_video: Path, dest_sidecar: Path = None) -> dict:
    """Render `job` and put the mp4 at dest_video (and a sidecar json at dest_sidecar, if given).
    Returns the sidecar dict. Raises RenderFailed with diagnostics for the batch publisher."""
    work = Path(paths["work_root"])
    missing = ensure_cache_copies(paths, job.garments)
    if missing:
        error = ("資料包裡找不到這些檔案：" + "、".join(missing) +
                 "。請確認 package_root 中有完整衣物快取")
        raise RenderFailed(error, make_sidecar(job, "failed", error=error))
    rn = run_name(job) + '_' + uuid4().hex[:12]
    run_dir = work / "outfits" / job.work_name / "runs_preview" / rn
    lp = log_path(paths, job).with_name(rn + '.log')
    lp.parent.mkdir(parents=True, exist_ok=True)

    cached = betas_cache(paths, job)
    cmd = _pipeline_cmd(paths, catalog, job, rn, cached if cached.is_file() else None)
    t0 = time.perf_counter()
    try:
        with lp.open("w") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(paths["contourcraft_root"]), env=PreviewRuntime(paths).environment())
    except OSError as e:                       # e.g. the python path in runtime configuration can't be executed
        error = f"無法啟動 pipeline：{e}。請檢查 runtime configuration 內的 Python 路徑。"
        raise RenderFailed(error, make_sidecar(job, "failed", error=error))
    elapsed = time.perf_counter() - t0

    video = run_dir / "video.mp4"
    n_rendered = len(list((run_dir / "frames" / "renders").glob("*.png")))
    error = None
    if proc.returncode != 0:
        tail = lp.read_text(errors="replace").strip().splitlines()[-12:]
        error = f"pipeline 結束代碼 {proc.returncode}。log 最後幾行：\n" + "\n".join(tail)
    elif not video.is_file():
        error = "pipeline 沒有產生 video.mp4"
    elif n_rendered != job.frames:
        error = f"影片幀數不符：要 {job.frames} 幀，實際渲染 {n_rendered} 幀（steps_for_frames 的換算可能需要調整）"

    if error:
        sc = make_sidecar(job, "failed", elapsed_seconds=round(elapsed), error=error,
                          run_dir=str(run_dir), log_path=str(lp))
        raise RenderFailed(error, sc)

    betas_src = run_dir / "betas.pkl"
    if not cached.is_file() and betas_src.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(betas_src, cached)

    dest_video = Path(dest_video)
    dest_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_video.with_name(dest_video.name + ".tmp")
    shutil.copyfile(video, tmp)
    tmp.replace(dest_video)
    stages = stage_timings(lp)
    sc = make_sidecar(job, "done", elapsed_seconds=round(elapsed), stage_seconds=stages,
                      pipeline_fingerprint=paths.get('pipeline_fingerprint'), run_dir=str(run_dir), log_path=str(lp))
    if dest_sidecar is not None:
        atomic_write_json(dest_sidecar, sc)       # written last: sidecar says "done" only once the mp4 is in place
    # Keep trajectories and repair reports for inspection; cleanup is a separate operation.
    return sc


class RenderFailed(ToolkitError):
    def __init__(self, message, sidecar):
        super().__init__(message)
        self.sidecar = sidecar
