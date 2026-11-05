#!/usr/bin/env python3
"""Two-phase preview pipeline for a multi-garment outfit.

Phase 1 fits the body, rebinds garments, simulates --preview-steps, and
renders the final preview frame from --turntable-angles cameras.
Phase 2 reuses the fitted body and rebound garments, simulates --full-steps,
renders the animation, and assembles video.mp4.

status.json records phase transitions and progress. Per-garment recover/relax
and per-outfit untangle results are cached under output-root; missing cache
entries are built there. Use a writable workspace for output-root.
The pipeline requires at least two --garment-names.

Output layout, under --output-root (default: gg_outputs_fixedsize/, shared
cache with the other fixedsize scripts):

    <output-root>/<garment-name>/                     per-garment cache
    <output-root>/_staging_garment_dicts/<outfit>/     untangled outfit cache
    <output-root>/outfits/<outfit-name>/runs_preview/<run-name>/
        status.json                     phase/progress marker, polled by a frontend
        betas.pkl / motion_with_body.npz         shared fit-body output (both phases)
        fixedsize_dicts/<outfit-name>/<name>.pkl  shared rebind output (both phases)
        preview.png                     Phase 1's angle_000.png, kept under this name too
        turntable/angle_NNN.png         Phase 1's turntable_frames camera angles of the settled pose
        preview_trajectory.pkl          Phase 1's short trajectory (debug/inspection only)
        frames/renders/*.png            Phase 2's frames (not meant to be shown one at a time --
                                         see status.json's total_frames for progress instead)
        trajectory.pkl                  Phase 2's full trajectory
        video.mp4                       Phase 2's final stitched output

Example:
    python run_outfit_preview_pipeline.py \
        --garment-names 00170_lower 00170_upper \
        --gender female --height 1.70 --weight 58 \
        --motion CMU/01/01_01_stageii.npz \
        --preview-steps 5 --full-steps 30
"""

import argparse
import glob
import json
import math
import os
import pickle
import shutil
import subprocess
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent

RENDER_ENV_EXTRA = dict(
    UCX_MEMTYPE_CACHE="n",
    UCX_TLS="tcp,cuda_copy",
    UCX_ERROR_SIGNALS="",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
)

# Retry failed subprocesses with a fixed delay.
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5


def run(cmd, **kwargs):
    env = {**os.environ, **kwargs.pop("extra_env", {})}
    if (env.get('REALFIT3D_CC_ROOT') and len(cmd) > 1
            and str(cmd[0]) == env.get('REALFIT3D_CC_PYTHON') and str(cmd[1]).endswith('.py')):
        cmd = [cmd[0], str(SCRIPT_DIR.parent / 'runtime_entry.py'), '--kind', 'cc',
               '--root', env['REALFIT3D_CC_ROOT'], '--script', str(cmd[1]), '--', *cmd[2:]]
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        print("+ " + " ".join(str(c) for c in cmd))
        try:
            subprocess.run(cmd, check=True, env=env, **kwargs)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                print(f"!! Command failed (attempt {attempt}/{RETRY_ATTEMPTS}, exit {exc.returncode}). "
                      f"Retrying in {RETRY_DELAY_SECONDS}s in case this was transient GPU flakiness...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"!! Command failed {RETRY_ATTEMPTS} times in a row, giving up.")
    raise last_exc


_PIPELINE_START = None  # set once at the top of run_pipeline(); read by write_status()


def write_status(run_dir: Path, phase: str, **extra) -> None:
    """Overwrites status.json. A frontend polls this file to know what to
    show -- see the module docstring's PHASES for the values `phase` takes.
    elapsed_seconds is seconds since this run_pipeline() call started, e.g.
    for citing "how long until the frontend can show something" (preview_ready)
    or "how long until the video is done" (done)."""
    elapsed = round(time.perf_counter() - _PIPELINE_START, 1) if _PIPELINE_START is not None else None
    payload = {"phase": phase, "updated_at": time.time(), "elapsed_seconds": elapsed, **extra}
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(run_dir / "status.json")  # atomic on POSIX -- no half-written reads
    print(f"[status] {phase} (t={elapsed}s)" + (f" -- {extra}" if extra else ""))


def find_source_trajectory(gaugar_root: Path, garment_name: str) -> Path:
    pattern = str(gaugar_root / "gaugar_trajectories" / "individual-garments" / f"{garment_name}_Take*.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trajectory found matching {pattern}. Pass --source-trajectories explicitly.")
    return Path(matches[0])


def is_relaxed(garment_dict_path: Path) -> bool:
    with garment_dict_path.open("rb") as f:
        return bool(pickle.load(f).get("recovery_metadata", {}).get("relaxed"))


def is_pinned(garment_dict_path: Path) -> bool:
    with garment_dict_path.open("rb") as f:
        return bool(pickle.load(f).get("recovery_metadata", {}).get("pinned"))


def ensure_garment(args, garment_dir: Path, garment_name: str, source_gender: str,
                    source_trajectory_override, pin_waistband: bool) -> Path:
    """Same cache-check logic as run_multigarment_pipeline_fixedsize.py's
    ensure_garment() -- duplicated, not imported, see module docstring."""
    garment_dict = garment_dir / "garment_dict.pkl"
    if garment_dict.exists() and is_relaxed(garment_dict):
        if pin_waistband and not is_pinned(garment_dict):
            raise FileExistsError(
                f"{garment_dict} was already relaxed WITHOUT being in --pin-waistband-garments. "
                f"Delete {garment_dict} and {garment_dir / 'relax_trajectory.pkl'} and rerun with "
                f"this garment listed there so it gets pinned before relaxing this time.")
        print(f"Reusing existing relaxed garment dict: {garment_dict}")
        return garment_dict

    garment_dir.mkdir(parents=True, exist_ok=True)
    source_fit = garment_dir / "source_body_fit.pkl"

    if not garment_dict.exists():
        source_trajectory = source_trajectory_override or find_source_trajectory(args.gaugar_root, garment_name)
        run([args.ccraft_python, str(SCRIPT_DIR / "recover_resize_garment_bootstrap.py"),
             "--contourcraft-root", str(args.contourcraft_root),
             "--body-model-root", str(args.body_model_root),
             "--trajectory", str(source_trajectory),
             "--garment-name", garment_name,
             "--output-fit", str(source_fit),
             "--output-garment-dict", str(garment_dict),
             "--gender", source_gender,
             "--frame", str(args.frame)])

    if pin_waistband and not is_pinned(garment_dict):
        run([args.ccraft_python, str(SCRIPT_DIR / "pin_garment_waistband.py"),
             "--contourcraft-root", str(args.contourcraft_root),
             "--garment-dict", str(garment_dict)])

    relax_traj = garment_dir / "relax_trajectory.pkl"
    if not relax_traj.exists():
        run([args.ccraft_python, str(SCRIPT_DIR / "relax_recovered_garment.py"),
             "--contourcraft-root", str(args.contourcraft_root),
             "--data-root", os.environ.get('REALFIT3D_CC_DATA_ROOT', str(args.contourcraft_root / "ccraft_data")),
             "--body-model-root", str(args.body_model_root),
             "--garment-dict", str(garment_dict),
             "--checkpoint", str(args.checkpoint),
             "--trajectory-output", str(relax_traj),
             "--steps", str(args.relax_steps),
             "--gender", source_gender])
    return garment_dict


def stage_into_flat_dir(staging_dir: Path, garment_name: str, garment_dict: Path) -> None:
    link = staging_dir / f"{garment_name}.pkl"
    target = garment_dict.resolve()
    if link.is_symlink() or link.exists():
        if link.resolve() == target:
            return
        link.unlink()
    link.symlink_to(target)


def ensure_outfit(args, staging_dir: Path, outfit_dir: Path) -> None:
    """Same cache-check logic as run_multigarment_pipeline_fixedsize.py's
    ensure_outfit() -- duplicated, not imported. In the intended usage this
    is always a cache hit (the catalog was precomputed offline); it still
    works standalone (builds it fresh) if not, for robustness."""
    manifest_path = outfit_dir / "manifest.json"
    expected = {
        "garment_names": list(args.garment_names),
        "outfit_name": args.outfit_name,
        "gender": args.gender,
        "model_type": "smplx",
        "n_epochs": args.untangle_epochs,
        "steps_per_stage": args.untangle_steps_per_stage,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest == expected:
            print(f"Reusing existing untangled outfit: {outfit_dir}")
            return
        raise FileExistsError(
            f"{outfit_dir} exists but was built with different settings.\n"
            f"  existing: {manifest}\n  requested: {expected}\n"
            f"Pass a different --outfit-name, or delete {outfit_dir} to rebuild.")

    untangle_cmd = [args.ccraft_python, str(SCRIPT_DIR / "combine_garments_untangle.py"),
         "--contourcraft-root", str(args.contourcraft_root),
         "--data-root", os.environ.get('REALFIT3D_CC_DATA_ROOT', str(args.contourcraft_root / "ccraft_data")),
         "--body-model-root", str(args.body_model_root),
         "--garment-dicts-dir", str(staging_dir),
         "--garment-names", *args.garment_names,
         "--outfit-name", args.outfit_name,
         "--checkpoint", str(args.checkpoint),
         "--gender", args.gender,
         "--model-type", "smplx",
         "--n-epochs", str(args.untangle_epochs),
         "--steps-per-stage", str(args.untangle_steps_per_stage),
         "--skip-safecheck-collision-count"]
    run(untangle_cmd)


def make_run_name(args) -> str:
    motion_stem = Path(args.motion).stem
    return f"{motion_stem}_h{int(round(args.height*100))}cm_w{int(round(args.weight))}kg"


def make_display_pose_sequence(motion_path: Path, output_path: Path) -> None:
    """Build a static display pose from the fitted motion, preserving betas.

    Three things happen, in order:

    1. Zero the ARTICULATED joint poses (fingers/jaw/eyes, and pose_body --
       see step 2 for what replaces it) so nothing carries over the source
       motion's mid-stride shape.
    2. pose_body: a literal zero is SMPL-X's raw T-pose -- confirmed via
       forward kinematics on the zero pose (elbow ~0.27m, wrist ~0.64m offset
       along the body's local left-right axis, near-zero on the other two
       axes). Swing both upper arms down to the sides using the shoulder
       joints (pose_body elements 45:51,
       left/right shoulder in SMPL-X's fixed 21-joint order), no elbow bend,
       landing each straight arm roughly beside the hip.
    3. root_orient/trans: deliberately NOT zeroed -- they're not "pose" in
       the T/display-pose sense, they're the rigid world placement (which way
       is up, where the pelvis sits on the ground). Confirmed empirically: a
       real CMU/AMASS file's root_orient is a large (~118 degree), near-
       constant-across-frames axis-angle correction that rotates SMPL-X's own
       raw canonical axes into the viewer's upright convention -- zeroing it
       left the body in SMPL-X's raw frame, which rendered lying flat on its
       back, not standing. Both are frozen at frame 0's value (repeated
       across all frames) so the display pose stands still rather than
       walking during the mini preview settle, then root_orient gets one
       more correction: a yaw so the body faces the camera (see below).
       Without it, frame 0's heading can point anywhere the source motion
       clip happens to start facing -- e.g. this project's own example clip
       renders as a side profile -- and a horizontal T-pose arm viewed
       nearly end-on is foreshortened.

    The yaw correction has to account for datasets/ccraft.py's swap_axes
    vertex permute: a fixed -90 degree rotation about the raw model's X axis
    that ccraft's dataloader applies to every vertex *after* root_orient
    (baked into the dataloader simulate_recovered_outfit.py uses, not
    something this script can pass a flag for -- see permute_axes there).
    render_lean.py's camera (inference.py's ait_render, cam=None branch) is
    fixed at target + (0,0,3) looking back at target -- it never turns to
    follow the body, so facing the camera means rotating the body, not the
    camera. Composing a world-space yaw here means conjugating it back
    through that permute before combining with root_orient. Confirmed
    empirically (forward kinematics against a real motion file, not
    assumed): post-permute, the raw body's local +Y is already the world "up"
    axis (matches the already-verified-standing render), and the raw body's
    local +Z (where the face points in the zero pose) is what needs to be
    rotated, about world "up", until it lines up with world +Z (the
    direction from the body to the camera).

    Not the cmu_npz_smplx_zeropos sequence loader's built-in T-pose generator,
    which hardcodes zero betas (an average body, not this run's requested
    height/weight)."""
    data = dict(np.load(motion_path, allow_pickle=True))  # matches fit_body_from_measurements_fast.py's own np.load call
    for key in ("pose_body", "pose_hand", "pose_jaw", "pose_eye"):
        if key in data:
            data[key] = np.zeros_like(data[key])

    if "pose_body" in data:
        # 90deg (horizontal T-pose -> straight down, touching the sides) let a
        # fitted sleeve/shoulder seam clip into the torso in this stiff, fully
        # -swung pose -- confirmed by comparing against the SAME outfit's real
        # walking-motion frames, which never hold the arm that tight against
        # the body and never showed the clipping. 78deg leaves the arm resting
        # naturally close to the body but with a few degrees of clearance from
        # the torso, avoiding that contact without looking like a different
        # pose (a couple degrees off vertical isn't visually distinct from
        # "arms at the sides").
        shoulder_swing = np.deg2rad(78.0)
        data["pose_body"][..., 45:48] = [0.0, 0.0, -shoulder_swing]  # left shoulder
        data["pose_body"][..., 48:51] = [0.0, 0.0, shoulder_swing]   # right shoulder, mirrored

    for key in ("root_orient", "trans"):
        if key in data and data[key].ndim >= 1 and data[key].shape[0] > 1:
            data[key] = np.tile(data[key][0:1], (data[key].shape[0], 1))

    if "root_orient" in data:
        # Fixed rotation ccraft's dataloader applies to every vertex, after
        # root_orient (see datasets/ccraft.py's permute_axes / swap_axes) --
        # in column-vector form this is a -90 degree rotation about the raw
        # model's X axis.
        swap_axes_rotation = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        root_matrix = Rotation.from_rotvec(data["root_orient"][0]).as_matrix()
        forward_final = swap_axes_rotation @ (root_matrix @ np.array([0.0, 0.0, 1.0]))
        yaw = -np.arctan2(forward_final[0], forward_final[2])
        yaw_world = Rotation.from_euler("y", yaw).as_matrix()
        # Conjugate the world-space yaw back through the fixed permute so it
        # can be folded into root_orient (which lives in the pre-permute raw
        # frame).
        yaw_raw = swap_axes_rotation.T @ yaw_world @ swap_axes_rotation
        new_root_matrix = yaw_raw @ root_matrix
        new_root_orient = Rotation.from_matrix(new_root_matrix).as_rotvec()
        data["root_orient"] = np.tile(new_root_orient[None, :], (data["root_orient"].shape[0], 1))

    np.savez(output_path, **data)


def rebind_all_garments(args, staging_dir: Path, motion_path: Path, source_fits: dict,
                         fixedsize_dicts_dir: Path) -> None:
    """Stage: fixed-size rebind, once per garment. Shared by both phases --
    call this ONCE, then point both the preview and full simulate at its
    output; rebinding depends only on height/weight/garments, not on how many
    steps you're about to simulate."""
    for garment_name, source_gender in zip(args.garment_names, args.source_genders):
        untangled_dict = staging_dir / args.outfit_name / f"{garment_name}.pkl"
        output_dict_path = fixedsize_dicts_dir / args.outfit_name / f"{garment_name}.pkl"
        prepare_cmd = [args.ccraft_python, str(SCRIPT_DIR / "contourcraft_fixed_size_upper.py"), "prepare",
             "--contourcraft-root", str(args.contourcraft_root),
             "--body-model-root", str(args.body_model_root),
             "--source-garment-dict", str(untangled_dict),
             "--source-betas", str(source_fits[garment_name]),
             "--target-sequence", str(motion_path),
             "--output-garment-dict", str(output_dict_path),
             "--model-type", "smplx",
             "--source-gender", source_gender,
             "--target-gender", args.gender]
        if garment_name in args.pin_waistband_if_snug_garments:
            prepare_cmd += ["--pin-waistband-if-snug", "--fit-ratio-threshold", str(args.fit_ratio_threshold)]
        run(prepare_cmd)
        config = args.garment_repairs_json.get(garment_name, {})
        if args.gender in config.get('genders', []):
            repaired = output_dict_path.with_suffix('.repaired.pkl')
            backup = output_dict_path.with_suffix('.before_repair.pkl')
            if backup.exists() or repaired.exists():
                raise FileExistsError('Use a new run directory for garment repair')
            run([args.ccraft_python, str(SCRIPT_DIR / 'repair_garment_lbs.py'),
                 '--input-garment-dict', str(output_dict_path),
                 '--reference-garment-dict', config['reference_dict'],
                 '--profile', config['profile'], '--body-model-root', str(args.body_model_root),
                 '--apply', '--output-garment-dict', str(repaired),
                 '--report', str(output_dict_path.with_suffix('.repair.json')),
                 '--preview', str(output_dict_path.with_suffix('.repair.png'))])
            shutil.copy2(output_dict_path, backup)
            repaired.replace(output_dict_path)


def simulate(args, fixedsize_dicts_dir: Path, motion_path: Path, n_steps: int, output_trajectory: Path) -> None:
    sim_cmd = [args.ccraft_python, str(SCRIPT_DIR / "simulate_recovered_outfit.py"),
         "--contourcraft-root", str(args.contourcraft_root),
         "--garment-dicts-dir", str(fixedsize_dicts_dir),
         "--outfit-name", args.outfit_name,
         "--garment-names", *args.garment_names,
         "--checkpoint", str(args.checkpoint),
         "--sequence", str(motion_path),
         "--sequence-loader", "cmu_npz_smplx",
         "--gender", args.gender,
         "--n-steps", str(n_steps),
         "--output-trajectory", str(output_trajectory),
         "--skip-safecheck-collision-count"]
    # Passthrough for simulate_recovered_outfit.py's own --body-collision-radius
    # (EXPERIMENTAL, see that script's help text) -- None means "don't pass
    # it, use the checkpoint's trained 3e-2m value", same as this script's
    # own prior default behavior. Not a fix for the "garment clips through
    # the body when it's much too long for the target's legs" limitation
    # (that needs real floor collision, out of scope) -- this only widens the
    # garment-vs-body clearance buffer, which may reduce how BADLY it clips
    # for a large mismatch, being tried as a cheap experiment before
    # investing in an actual garment-length-trimming preprocessing step.
    if args.body_collision_radius is not None:
        sim_cmd += ["--body-collision-radius", str(args.body_collision_radius)]
    run(sim_cmd)


def opacity_arguments(args):
    arguments = ["--opacity-mode", args.opacity_mode, "--opacity-scale", str(args.opacity_scale)]
    if args.opacity_garments:
        arguments += ['--opacity-garments', *args.opacity_garments]
    for mapping in args.v3_controller:
        arguments.extend(["--v3-controller", mapping])
    return arguments


def render(args, trajectory_path: Path, frames_dir: Path, start_from: int = 0) -> None:
    run([args.gaugar_python, "render_lean.py",
         "--traj_path", str(trajectory_path.resolve()),
         "--output_path", str(frames_dir.resolve()),
         "--start_from", str(start_from),
         "--bake-samples", str(args.bake_samples), *opacity_arguments(args)],
        cwd=str(args.gaugar_root), extra_env=RENDER_ENV_EXTRA)


def render_turntable(args, trajectory_path: Path, frame_idx: int, output_dir: Path, n_angles: int) -> None:
    """Phase 1's display pose is a single static frame -- render it from
    n_angles evenly-spaced camera azimuths instead of just one, so the
    frontend can show a slow turntable while Phase 2 runs. Uses
    render_turntable.py (new sibling of render_lean.py, same relationship it
    has to inference.py -- see that script's own docstring for why repeat
    angles of the SAME frame are much cheaper than n_angles separate full
    renders: the expensive mesh/texture bake is keyed on frame_idx and this
    output_dir, not on camera, and gets reused automatically starting from
    the second angle)."""
    run([args.gaugar_python, "render_turntable.py",
         "--traj_path", str(trajectory_path.resolve()),
         "--output_path", str(output_dir.resolve()),
         "--frame_idx", str(frame_idx),
         "--n-angles", str(n_angles),
         "--bake-samples", str(args.bake_samples), *opacity_arguments(args)],
        cwd=str(args.gaugar_root), extra_env=RENDER_ENV_EXTRA)


def stitch(args, renders_dir: Path, video_path: Path) -> None:
    stitch_script = (
        "import os\n"
        "from moviepy import ImageSequenceClip\n"
        f"folder = {str(renders_dir)!r}\n"
        "files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png'))\n"
        f"clip = ImageSequenceClip(files, fps={args.fps})\n"
        f"clip.write_videofile({str(video_path)!r}, codec='libx264', ffmpeg_params=['-movflags', '+faststart'], logger=None)\n"
    )
    run([args.gaugar_python, "-c", stitch_script])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opacity-mode", choices=["direct", "optical_depth", "v3"], default="direct")
    parser.add_argument("--opacity-scale", type=float, default=1.0)
    parser.add_argument("--v3-controller", action="append", default=[], metavar="GARMENT=PATH")
    parser.add_argument('--opacity-garments', nargs='+', default=[])
    parser.add_argument('--garment-repairs-json', type=json.loads, default={})
    parser.add_argument('--skip-preview', action='store_true')
    parser.add_argument("--garment-names", nargs="+", required=True,
        help="Innermost to outermost, 2+ garments (a fixed catalog combo). Each must match a "
             "Gaussian-Garments output_root/ folder.")
    parser.add_argument("--outfit-name", default=None, help="Default: garment names joined with '+'.")
    parser.add_argument("--height", type=float, required=True, help="target height in meters")
    parser.add_argument("--weight", type=float, required=True, help="target weight in kg")
    parser.add_argument("--precomputed-betas", type=Path, default=None,
        help="Skip fit_body_from_measurements_fast.py's ~20s body-shape optimization and reuse "
             "an already-fit betas.pkl instead (same dict format that script writes -- needs a "
             "'betas' key). --height/--weight are still required above (used for outfit-name/log "
             "purposes elsewhere) but no longer drive the actual body shape when this is set -- "
             "only --motion gets patched with these betas. Used by preview_server.py's two-stage "
             "flow: a visitor fits their body once on a separate page, then reuses it across "
             "however many outfits/motions they try.")
    parser.add_argument("--motion", type=Path, required=True, help="CMU/AMASS SMPL-X motion .npz")
    parser.add_argument("--gender", default="female")
    parser.add_argument("--source-genders", nargs="+", default=None,
        help="Per-garment gender for recovery, same order as --garment-names (default: --gender repeated).")
    parser.add_argument("--preview-steps", type=int, default=5,
        help="Phase 1: how many simulate steps before rendering the single preview frame. "
             "Small on purpose -- this is a quick settle, not the real animation.")
    parser.add_argument("--full-steps", type=int, default=60,
        help="Phase 2 simulation steps. With 60 steps and --discard-frames 10, "
             "the nominal output is 50 frames (about 2.1 seconds at 24 fps).")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--discard-frames", type=int, default=10,
        help="Phase 2's full animation: skip rendering this many initial frames (the real "
             "--motion's own unstable cloth-settling start, same reasoning as the other "
             "fixedsize scripts' --discard-frames). trajectory.pkl still holds all "
             "--full-steps frames; only the rendered video shrinks -- with the defaults "
             "(--full-steps 60, --discard-frames 10) that's 50 rendered frames, ~2.1s at "
             "24fps, so raise --full-steps if you want a longer resulting clip.")
    parser.add_argument("--bake-samples", type=int, default=4,
        help="Same meaning as the other fixedsize scripts' --bake-samples -- see their "
             "--help/READMEs for why 4 is the default (measured no visible difference vs 16).")
    parser.add_argument("--turntable-angles", type=int, default=12,
        help="Phase 1: how many evenly-spaced camera azimuths to render of the settled "
             "display pose (see render_turntable.py). Cheap per extra angle -- the expensive "
             "mesh/texture bake is shared across all of them, only camera-dependent shading + "
             "rasterization gets redone -- measured 7.3s total for 12 angles on a 2-garment "
             "outfit (vs. ~4.6s for just the first one), not the naive n_angles x 4.6s.")
    parser.add_argument("--pin-waistband-garments", nargs="+", default=[],
        help="Subset of --garment-names to pin before relax (Stage 1-2, shared cache). Only "
             "matters the first time a garment is recovered -- typically already baked into "
             "the precomputed catalog cache, listed here mainly for standalone/first-run use.")
    parser.add_argument("--pin-waistband-if-snug-garments", nargs="+", default=[],
        help="Subset of --garment-names to run the fit-aware waistband check for at rebind "
             "time -- see run_garment_pipeline_fixedsize_README.md's 褲頭要不要 pin 分節.")
    parser.add_argument("--fit-ratio-threshold", type=float, default=1.2)
    parser.add_argument("--body-collision-radius", type=float, default=None,
        help="Passthrough to simulate_recovered_outfit.py's own EXPERIMENTAL "
             "--body-collision-radius (see that script's --help for the full "
             "caveats). Default None = use the checkpoint's trained value.")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--relax-steps", type=int, default=30)
    parser.add_argument("--untangle-epochs", type=int, default=2)
    parser.add_argument("--untangle-steps-per-stage", type=int, default=30)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--source-trajectories", nargs="+", default=None)
    parser.add_argument("--contourcraft-root", type=Path, default=Path("/path/to/GG_projects/ContourCraft"))
    parser.add_argument("--gaugar-root", type=Path, default=Path("/path/to/GG_projects/Gaussian-Garments"))
    parser.add_argument("--body-model-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ccraft-python", default="/path/to/conda/envs/ccraft/bin/python")
    parser.add_argument("--gaugar-python", default="/path/to/conda/envs/gaugar-sm120/bin/python")
    parser.add_argument("--output-root", type=Path, default=None,
        help="Default: gg_outputs_fixedsize/ next to --contourcraft-root -- SAME as the other "
             "fixedsize scripts, deliberately, so the per-garment and per-outfit caches are shared.")
    args = parser.parse_args()
    if not isinstance(args.garment_repairs_json, dict):
        parser.error('--garment-repairs-json must be an object')
    if set(args.opacity_garments) - set(args.garment_names):
        parser.error('--opacity-garments must select trajectory garments')
    if not math.isfinite(args.opacity_scale) or not 0 <= args.opacity_scale <= 1:
        parser.error("--opacity-scale must be between 0 and 1")
    if args.v3_controller and args.opacity_mode != "v3":
        parser.error("--v3-controller requires --opacity-mode v3")

    if len(args.garment_names) < 2:
        raise ValueError("--garment-names needs at least 2 garments -- for a single garment this "
                          "two-phase treatment isn't needed, see run_garment_pipeline_fixedsize_fast.py.")
    if args.source_genders is None:
        args.source_genders = [args.gender] * len(args.garment_names)
    if args.source_trajectories is None:
        source_trajectory_overrides = [None] * len(args.garment_names)
    else:
        source_trajectory_overrides = [Path(p) for p in args.source_trajectories]

    if args.outfit_name is None:
        args.outfit_name = "+".join(args.garment_names)
    if args.body_model_root is None:
        args.body_model_root = args.contourcraft_root / "ccraft_data" / "aux_data" / "body_models"
    if args.checkpoint is None:
        args.checkpoint = args.contourcraft_root / "ccraft_data" / "trained_models" / "contourcraft.pth"
    if args.output_root is None:
        args.output_root = args.contourcraft_root.parent / "gg_outputs_fixedsize"
    if args.run_name is None:
        args.run_name = make_run_name(args)

    staging_dir = args.output_root / "_staging_garment_dicts"
    staging_dir.mkdir(parents=True, exist_ok=True)
    outfit_dir = args.output_root / "outfits" / args.outfit_name
    run_dir = outfit_dir / "runs_preview" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir} (pass --run-name to use a different name)")
    run_dir.mkdir(parents=True)

    try:
        run_pipeline(args, run_dir, staging_dir, source_trajectory_overrides)
    except BaseException as exc:
        # Without this, a mid-run crash leaves status.json stuck on whatever
        # phase was last written -- a polling frontend would just sit there
        # forever thinking it's still working instead of showing an error.
        write_status(run_dir, "error", message=str(exc), error=repr(exc))
        raise


def run_pipeline(args, run_dir: Path, staging_dir: Path, source_trajectory_overrides) -> None:
    global _PIPELINE_START
    _PIPELINE_START = time.perf_counter()
    write_status(run_dir, "recovering_garments", message="Preparing garments (usually cached, near-instant)")
    source_fits = {}
    for garment_name, source_gender, traj_override in zip(
            args.garment_names, args.source_genders, source_trajectory_overrides):
        pin_this = garment_name in args.pin_waistband_garments
        garment_dict = ensure_garment(
            args, args.output_root / garment_name, garment_name, source_gender, traj_override, pin_this)
        source_fits[garment_name] = args.output_root / garment_name / "source_body_fit.pkl"
        stage_into_flat_dir(staging_dir, garment_name, garment_dict)

    write_status(run_dir, "untangling", message="Untangling outfit (usually cached, near-instant)")
    ensure_outfit(args, staging_dir, staging_dir / args.outfit_name)

    betas_path = run_dir / "betas.pkl"
    motion_path = run_dir / "motion_with_body.npz"
    if args.precomputed_betas is not None:
        # Reuse an already-fit body (see --precomputed-betas' own help) --
        # skip fit_betas()'s ~20s optimization loop entirely, just do the
        # same patch fit_body_from_measurements_fast.py's patch_sequence_
        # betas() would (load --motion, overwrite its betas field, save) --
        # duplicated inline rather than importing that script as a module,
        # same "small helpers get duplicated, not imported" convention this
        # file already follows elsewhere (ensure_garment/ensure_outfit).
        write_status(run_dir, "fitting_body", message="Reusing your saved body")
        shutil.copyfile(args.precomputed_betas, betas_path)
        with betas_path.open("rb") as f:
            betas = pickle.load(f)["betas"]
        seq = dict(np.load(args.motion, allow_pickle=True))
        original = np.asarray(seq["betas"])
        padded = np.zeros_like(original) if original.ndim == 1 else np.zeros(original.shape[-1])
        padded[:10] = betas
        seq["betas"] = padded
        np.savez(motion_path, **seq)
    else:
        write_status(run_dir, "fitting_body", message=f"Fitting body to {args.height}m / {args.weight}kg")
        run([args.ccraft_python, str(SCRIPT_DIR / "fit_body_from_measurements_fast.py"),
             "--body-model-root", str(args.body_model_root),
             "--gender", args.gender,
             "--height", str(args.height),
             "--weight", str(args.weight),
             "--output-betas", str(betas_path),
             "--input-sequence", str(args.motion),
             "--output-sequence", str(motion_path),
             "--patience", "50"])

    write_status(run_dir, "rebinding", message="Fitting garments to your body")
    fixedsize_dicts_dir = run_dir / "fixedsize_dicts"
    rebind_all_garments(args, staging_dir, motion_path, source_fits, fixedsize_dicts_dir)

    if not args.skip_preview:
        # --- Phase 1: fast static preview ---
        write_status(run_dir, "preview_simulating", message="Generating quick preview")
        tpose_motion_path = run_dir / "tpose_motion.npz"
        make_display_pose_sequence(motion_path, tpose_motion_path)
        preview_trajectory = run_dir / "preview_trajectory.pkl"
        simulate(args, fixedsize_dicts_dir, tpose_motion_path, args.preview_steps, preview_trajectory)

        write_status(run_dir, "preview_rendering", message="Rendering preview turntable")
        turntable_dir = run_dir / "turntable"
        render_turntable(args, preview_trajectory, args.preview_steps, turntable_dir, args.turntable_angles)
        turntable_frames = sorted(turntable_dir.glob("angle_*.png"))
        if not turntable_frames:
            raise RuntimeError(f"Turntable render produced no frames in {turntable_dir}")
        # angle_000.png is theta=0, the same default camera every other render in
        # this project uses -- keep it as preview.png too, in case anything still
        # expects that filename.
        (run_dir / "preview.png").write_bytes(turntable_frames[0].read_bytes())

        write_status(run_dir, "preview_ready", message="Preview ready", preview_ready=True,
                     turntable_frames=len(turntable_frames))

    # --- Phase 2: full short animation, reusing Phase 1's fit/rebind above ---
    write_status(run_dir, "full_simulating", message=f"Generating full animation ({args.full_steps} steps)",
                 preview_ready=not args.skip_preview)
    trajectory_path = run_dir / "trajectory.pkl"
    simulate(args, fixedsize_dicts_dir, motion_path, args.full_steps, trajectory_path)

    # Derive progress from the produced trajectory after discarding warmup frames.
    with trajectory_path.open("rb") as f:
        total_frames = len(pickle.load(f)["pred"]) - args.discard_frames

    write_status(run_dir, "full_rendering", message="Rendering full animation",
                 preview_ready=not args.skip_preview, total_frames=total_frames)
    frames_dir = run_dir / "frames"
    render(args, trajectory_path, frames_dir, start_from=args.discard_frames)

    video_path = run_dir / "video.mp4"
    stitch(args, frames_dir / "renders", video_path)

    write_status(run_dir, "done", message="Done", preview_ready=not args.skip_preview, video_ready=True)
    print(f"\nDone. Preview: {run_dir / 'preview.png'}  Video: {video_path}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()  # writes status.json phase="error" itself on failure -- see run_pipeline()'s caller
    print(f"\nTotal execution time: {time.perf_counter() - start_time:.2f} seconds")
