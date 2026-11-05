#!/usr/bin/env python3
"""Shared helpers for the batch video toolkit: paths.json / catalog.json /
job-file loading, job expansion, and the video-library path + sidecar-file
conventions. Imported by the other scripts; not run directly.

Library layout (path is a pure function of the job's parameters, so the same
video always lands at the same path no matter who renders it):

    <library_root>/<outfit>/<gender>/h<height>_w<weight:03d>/<motion>.mp4
    <library_root>/<outfit>/<gender>/h<height>_w<weight:03d>/<motion>.json   (sidecar)

The legacy/default opacity (direct, 1.0) keeps the paths above. Explicit
opacity variants append ``__opacity_<mode>_<value>`` to the motion name so
they never overwrite the opaque video.
"""

import itertools
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PATH_KEYS = ["contourcraft_root", "gaugar_root", "contourcraft_python", "gaugar_python",
             "data_root", "library_root", "work_root"]
GENDERS = ("male", "female")
OPACITY_MODES = ("direct", "optical_depth", "v3")

# Required files per garment inside the data package (relative to data_root/garments/<name>/).
GARMENT_FILES = [
    "stage1/template_uv.obj",
    "stage1/cameras.json",
    "stage3/ckpt/net.pt",
    "stage3/ckpt/info.txt",
    "cache/garment_dict.pkl",
    "cache/source_body_fit.pkl",
]

class ToolkitError(ValueError):
    """A problem the user can fix (bad config, missing file, bad job file)."""


def fail(msg: str):
    raise ToolkitError(msg)


# --------------------------------------------------------------------------- paths.json

def load_paths(path) -> dict:
    from realfit3d.settings import load_settings
    settings = load_settings(path)
    required = ("package_root", "library_root", "gg_output_root", "output_root",
                "ccraft_python", "gaugar_python", "contourcraft_root", "gaugar_root")
    missing = [key for key in required if not settings.get(key)]
    if missing:
        fail("Configure " + ", ".join(missing))
    paths = dict(settings)
    paths.update(data_root=Path(settings["package_root"]),
                 work_root=Path(settings["output_root"]),
                 contourcraft_python=Path(settings["ccraft_python"]))
    for key in PATH_KEYS:
        paths[key] = Path(paths[key])
    return paths


def gaugar_output_root(paths: dict) -> Path:
    return Path(paths["gg_output_root"])


# --------------------------------------------------------------------------- catalog.json

def load_catalog(data_root: Path) -> dict:
    p = Path(data_root) / "catalog.json"
    if not p.is_file():
        fail(f"資料包裡找不到 {p}。請確認資料包已完整下載到 package_root。")
    try:
        cat = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        fail(f"{p} 不是有效的 JSON：{e}")
    if not isinstance(cat.get("garments"), dict) or not isinstance(cat.get("motions"), dict):
        fail(f"{p} 格式不對：需要 `garments` 與 `motions` 兩個物件（見 docs/VIDEO_LIBRARY.md）。")
    return cat


def motion_file(paths: dict, catalog: dict, motion_id: str) -> Path:
    root = (Path(paths['data_root']) / 'motions').resolve()
    path = (root / catalog['motions'][motion_id]['file']).resolve()
    if root not in path.parents:
        fail('Motion path must stay inside package_root/motions')
    return path


# --------------------------------------------------------------------------- garment cache copies

def ensure_cache_copies(paths: dict, garments) -> list:
    """Restore missing writable cache copies; return missing package inputs."""
    work, data = Path(paths["work_root"]), Path(paths["data_root"]) / "garments"
    missing = []
    for g in garments:
        for name in ("garment_dict.pkl", "source_body_fit.pkl"):
            dst = work / g / name
            if dst.exists():
                continue
            src = data / g / "cache" / name
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.is_symlink():
                    dst.unlink()
                # Cache workers may pin/relax a garment; never write through to the package.
                import shutil
                shutil.copy2(src, dst)
            else:
                missing.append(f"{g}/cache/{name}")
    return missing


# --------------------------------------------------------------------------- jobs

AUTO_FRAMES = 0      # Job.frames == 0 means "run the whole motion"; resolve_auto_frames fills in the number

@dataclass(frozen=True)
class Job:
    garments: tuple
    gender: str
    height: int
    weight: int
    motion: str
    frames: int
    fps: int
    opacity: float = 1.0
    opacity_mode: str = "direct"
    opacity_garments: tuple = ()

    def __post_init__(self):
        from realfit3d.contracts import body_parameters, opacity_parameters
        for name in (*self.garments, self.motion):
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
                fail('Garment and motion IDs must be path-safe names')
        if len(self.garments) != 2 or len(set(self.garments)) != 2:
            fail('Select two distinct garments, lower then upper')
        body_parameters(self.height / 100, self.weight, self.gender)
        opacity_parameters(self.opacity_mode, self.opacity)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (self.height, self.weight, self.fps)):
            fail('Height, weight and fps must be positive integers')
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 0:
            fail('Frames must be auto or a positive integer')
        if set(self.opacity_garments) - set(self.garments):
            fail('Opacity targets must be selected garments')

    @property
    def outfit(self) -> str:
        return "+".join(self.garments)

    @property
    def work_name(self) -> str:
        """Name of this outfit's folders under work_root. Includes the gender: the untangled outfit is built for one
        gender, so male and female versions of the same outfit must not share a folder."""
        return f"{self.outfit}__{self.gender}"

    @property
    def body_dir(self) -> str:
        return f"h{self.height}_w{self.weight:03d}"

    @property
    def key(self) -> str:
        return f"{self.outfit}/{self.gender}/{self.body_dir}/{self.video_stem}"

    @property
    def video_stem(self) -> str:
        if self.opacity == 1.0 and self.opacity_mode == "direct":
            return self.motion
        value = f"{self.opacity:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        target = ""
        if self.opacity_garments and self.opacity_garments != self.garments:
            target = "__target_" + "+".join(self.opacity_garments)
        return f"{self.motion}__opacity_{self.opacity_mode}_{value}{target}"

    @property
    def effective_opacity_garments(self) -> tuple:
        return self.opacity_garments or self.garments

    def video_path(self, library: Path) -> Path:
        return Path(library) / self.outfit / self.gender / self.body_dir / f"{self.video_stem}.mp4"

    def sidecar_path(self, library: Path) -> Path:
        return self.video_path(library).with_suffix(".json")


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _positive_int(v, what, where):
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        fail(f"{where}：{what} 必須是正整數，但拿到 {v!r}")
    return v


def _opacity(v, what, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        fail(f"{where}：{what} 必須是 0 到 1 的數字，但拿到 {v!r}")
    value = float(v)
    if not 0.0 <= value <= 1.0:
        fail(f"{where}：{what} 必須介於 0 到 1，但拿到 {v!r}")
    return value


def expand_jobs(jobs_path) -> list:
    """Expand heights x weights x motions x opacities into individual jobs."""
    jobs_path = Path(jobs_path)
    if not jobs_path.is_file():
        fail(f"找不到工作單 {jobs_path}")
    try:
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        fail(f"{jobs_path} 不是有效的 JSON：{e}")
    if not isinstance(data, dict):
        fail('Jobs must be a JSON object')
    defaults = data.get("defaults", {})
    entries = data.get("jobs")
    if not isinstance(entries, list) or not entries:
        fail(f"{jobs_path} 需要一個非空的 `jobs` 清單。")
    jobs = []
    for i, e in enumerate(entries, 1):
        if not isinstance(e, dict):
            fail('Each job must be an object')
        where = f"{jobs_path.name} 第 {i} 個工作"
        for k in ("garments", "gender", "heights", "weights", "motions"):
            if k not in e:
                fail(f"{where} 缺少欄位 `{k}`")
        garments = tuple(_as_list(e["garments"]))
        if len(garments) < 2 or not all(isinstance(g, str) for g in garments):
            fail(f"{where}：`garments` 需要至少兩件衣服的名稱（下身在前、上身在後）")
        if e["gender"] not in GENDERS:
            fail(f"{where}：`gender` 必須是 male 或 female，但拿到 {e['gender']!r}")
        frames = e.get("frames", defaults.get("frames", "auto"))
        # "auto"（或不寫）= 跑完整個動作檔：幀數由動作的長度決定，見 resolve_auto_frames
        frames = AUTO_FRAMES if frames in ("auto", None) else _positive_int(frames, "frames", where)
        fps = _positive_int(e.get("fps", defaults.get("fps", 25)), "fps", where)
        opacity_mode = str(e.get("opacity_mode", defaults.get("opacity_mode", "direct")))
        if opacity_mode not in OPACITY_MODES:
            fail(f"{where}：`opacity_mode` 必須是 {', '.join(OPACITY_MODES)}，但拿到 {opacity_mode!r}")
        raw_opacities = e.get("opacities", defaults.get("opacities", [1.0]))
        opacities = [_opacity(v, "opacities", where) for v in _as_list(raw_opacities)]
        raw_targets = e.get("opacity_garments", defaults.get("opacity_garments", []))
        opacity_garments = tuple(str(g) for g in _as_list(raw_targets)) if raw_targets else ()
        unknown_targets = [g for g in opacity_garments if g not in garments]
        if unknown_targets:
            fail(f"{where}：`opacity_garments` 只能包含這套衣服裡的名稱；不在套裝內：{', '.join(unknown_targets)}")
        heights = [_positive_int(h, "heights", where) for h in _as_list(e["heights"])]
        weights = [_positive_int(w, "weights", where) for w in _as_list(e["weights"])]
        motions = [str(m) for m in _as_list(e["motions"])]
        for h, w, m, opacity in itertools.product(heights, weights, motions, opacities):
            jobs.append(Job(garments, e["gender"], h, w, m, frames, fps,
                            opacity, opacity_mode, opacity_garments))
    jobs = list(dict.fromkeys(jobs))                      # drop duplicates, keep order
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        fail('Jobs with different frames/fps or rounded opacity values share an output path')
    jobs.sort(key=lambda j: (j.outfit, j.gender, j.height, j.weight, j.motion,
                             j.opacity_mode, j.opacity, j.opacity_garments))
    return jobs


# ContourCraft simulates at 30 steps/second of motion (its `fps` setting); the CMU files are 120 fps,
# so every 4th frame is used. The pipeline drops the first DISCARD_FRAMES frames of the trajectory,
# and the trajectory has steps + 2 frames, so a video of N frames needs N + DISCARD_FRAMES - 2 steps.
SIM_FPS = 30
DISCARD_FRAMES = 10


def steps_for_frames(frames: int) -> int:
    return frames + DISCARD_FRAMES - 2


def max_frames_for_steps(n_steps: int) -> int:
    """The trajectory can never be longer than the motion itself (n_steps frames), and the first DISCARD_FRAMES
    are dropped. (A motion that is only just long enough produced 150 trajectory frames, not 152, in a test.)"""
    return n_steps - DISCARD_FRAMES


def motion_steps_available(paths: dict, motion_path: Path):
    """How many simulation steps this motion file can drive, or None if it cannot be read."""
    code = ("import sys, numpy as np; d = np.load(sys.argv[1], allow_pickle=True); "
            "skip = int(round(float(d['mocap_frame_rate']) / %d)); print(-(-len(d['trans']) // skip))" % SIM_FPS)
    try:
        r = subprocess.run([str(paths["contourcraft_python"]), "-c", code, str(motion_path)],
                           capture_output=True, text=True, timeout=120)
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def resolve_auto_frames(jobs, paths: dict, catalog: dict) -> list:
    """Replace frames=AUTO_FRAMES by what the motion file can drive: the whole motion = (steps - (DISCARD_FRAMES - 2)) frames."""
    steps = {}
    out = []
    for j in jobs:
        if j.frames != AUTO_FRAMES:
            out.append(j)
            continue
        if j.motion not in catalog["motions"]:
            out.append(j)                      # unknown motion: the catalog check reports it
            continue
        if j.motion not in steps:
            f = motion_file(paths, catalog, j.motion)
            steps[j.motion] = motion_steps_available(paths, f) if f.is_file() else None
        n = steps[j.motion]
        if n is None or max_frames_for_steps(n) < 1:
            fail(f"動作 `{j.motion}` 讀不出來或太短，無法決定影片幀數")
        out.append(Job(j.garments, j.gender, j.height, j.weight, j.motion,
                       max_frames_for_steps(n), j.fps, j.opacity, j.opacity_mode,
                       j.opacity_garments))
    return list(dict.fromkeys(out))


def motion_length_problems(jobs, paths: dict, catalog: dict) -> list:
    """A motion that is too short makes the simulation stop early and the video fail AFTER the
    long simulation has run -- so check it up front."""
    problems, avail = [], {}
    for m in sorted({j.motion for j in jobs if j.motion in catalog["motions"]}):
        f = motion_file(paths, catalog, m)
        if f.is_file():
            avail[m] = motion_steps_available(paths, f)
            if avail[m] is None:
                problems.append(f"動作 `{m}` 的檔案讀不出來（{f}），請確認它是正常的 .npz 動作檔")
    for j in jobs:
        n = avail.get(j.motion)
        if n is not None and j.frames > max_frames_for_steps(n):
            longest = max_frames_for_steps(n)
            problems.append(f"動作 `{j.motion}` 只夠模擬 {n} 步（約 {n / SIM_FPS:.1f} 秒的動作），"
                            f"跑不出 {j.frames} 幀的影片（需要 {steps_for_frames(j.frames)} 步）；這個動作最多可跑 {longest} 幀")
    return list(dict.fromkeys(problems))


def validate_jobs_against_catalog(jobs, paths: dict, catalog: dict) -> list:
    """Return a list of human-readable problems (empty = fine)."""
    problems = []
    for g in sorted({g for j in jobs for g in j.garments}):
        if g not in catalog["garments"]:
            problems.append(f"衣服 `{g}` 不在 catalog.json 裡")
    # Each outfit must be exactly one lower + one upper, lower first (innermost -> outermost),
    # because the website lets visitors pick one of each.
    for outfit in sorted({j.garments for j in jobs}):
        slots = [catalog["garments"].get(g, {}).get("slot") for g in outfit]
        if all(s is not None for s in slots) and slots != ["lower", "upper"]:
            problems.append(f"衣服組合 {'+'.join(outfit)} 必須是「一件下身 + 一件上身」（下身在前），"
                            f"但 catalog.json 裡的 slot 是 {slots}")
    # If catalog.json defines the height/weight ranges the website offers, videos must be rendered
    # at exactly those ranges' `value`s (the midpoints), or the website could never select them.
    for key, attr, unit in (("heights", "height", "cm"), ("weights", "weight", "kg")):
        if catalog.get(key):
            allowed = sorted({r["value"] for r in catalog[key]})
            for v in sorted({getattr(j, attr) for j in jobs} - set(allowed)):
                problems.append(f"{'身高' if attr == 'height' else '體重'} {v}{unit} 不是 catalog.json 定義的區間中間值"
                                f"（可用：{', '.join(map(str, allowed))}），網頁選不到這支影片")
    for m in sorted({j.motion for j in jobs}):
        if m not in catalog["motions"]:
            problems.append(f"動作 `{m}` 不在 catalog.json 裡")
        elif not motion_file(paths, catalog, m).is_file():
            problems.append(f"動作 `{m}` 的檔案不存在：{motion_file(paths, catalog, m)}")
    for j in jobs:
        if j.opacity_mode != "v3":
            continue
        for g in j.effective_opacity_garments:
            if find_v3_controller(paths, g) is None:
                problems.append(
                    f"{j.key} 要使用 V3 透明度，但 `{g}` 找不到 controller.pt；"
                    "請放到 data_root/garments/<衣服>/opacity_controller_v3/controller.pt，"
                    "或在 paths.json 的 v3_controllers 指定路徑"
                )
    return problems


def find_v3_controller(paths: dict, garment: str):
    """Return the first configured/installed V3 checkpoint for one garment."""
    candidates = []
    configured = paths.get("v3_controllers", {}).get(garment)
    if configured is not None:
        candidates.append(Path(configured))
    for folder in ('opacity_controller_v3', 'opacity_controller_v3_valid_cameras'):
        candidates.append(Path(paths['data_root']) / 'garments' / garment / folder / 'controller.pt')
    try:
        candidates.append(gaugar_output_root(paths) / garment / "opacity_controller_v3" / "controller.pt")
    except ToolkitError:
        pass
    return next((path for path in candidates if path.is_file()), None)


# --------------------------------------------------------------------------- sidecar files

def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_sidecar(path: Path):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def is_done(library: Path, job: Job) -> bool:
    """A video counts as done only if its mp4 exists and its sidecar says status=done
    with the same frames/fps the job asks for (a different length is a different video)."""
    sc = read_sidecar(job.sidecar_path(library))
    targets_match = (
        job.opacity == 1.0 and job.opacity_mode == "direct"
    ) or tuple(sc.get("opacity_garments", []) if sc else []) == job.opacity_garments
    return bool(sc and sc.get("status") == "done" and job.video_path(library).is_file()
                and sc.get("frames") == job.frames and sc.get("fps") == job.fps
                and float(sc.get("opacity", 1.0)) == job.opacity
                and sc.get("opacity_mode", "direct") == job.opacity_mode
                and targets_match)


def make_sidecar(job: Job, status: str, **extra) -> dict:
    d = dict(garments=list(job.garments), gender=job.gender, height=job.height, weight=job.weight,
             motion=job.motion, frames=job.frames, fps=job.fps,
             opacity=job.opacity, opacity_mode=job.opacity_mode,
             opacity_garments=list(job.opacity_garments),
             created_at=datetime.now().isoformat(timespec="seconds"),
             machine=socket.gethostname(), status=status)
    d.update(extra)
    return d
