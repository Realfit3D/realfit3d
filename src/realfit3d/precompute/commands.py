"""Plan batches, stage asset copies, and publish completed video-library entries."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from realfit3d.fitting import repair_config
from realfit3d.precompute.build_index import build, publish_site_files
from realfit3d.precompute.render_one_video import render_video, RenderFailed
from realfit3d.precompute.toolkit_common import (
    GARMENT_FILES, ToolkitError, atomic_write_json, expand_jobs, find_v3_controller,
    is_done, load_catalog, load_paths, motion_file, motion_length_problems,
    read_sidecar, resolve_auto_frames, validate_jobs_against_catalog,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_output_roots(paths):
    package = Path(paths['data_root']).resolve()
    for key in ('gg_output_root', 'work_root', 'library_root'):
        output = Path(paths[key]).resolve()
        if output == package or package in output.parents or output in package.parents:
            raise ToolkitError(f'{key} must not overlap the input package')


def package_files(paths, names):
    root = Path(paths['data_root']) / 'garments'
    result = []
    for name in names:
        for rel in GARMENT_FILES:
            source = root / name / rel
            if not source.is_file():
                raise ToolkitError(f'Missing packaged asset: {source}')
            result.append((name, rel, source))
    return result


def fingerprint(paths, jobs, files):
    """Invalidate cached videos/body fits when code, assets, or repair inputs change."""
    inputs = {source for _, _, source in files}
    inputs.add(Path(paths['data_root']) / 'catalog.json')
    inputs.update(motion_file(paths, load_catalog(paths['data_root']), j.motion) for j in jobs)
    inputs.update(Path(__file__).parents[1].rglob('*.py'))
    inputs.update((Path(paths['gaugar_root'])).glob('*.py'))
    for job in jobs:
        for cfg in repair_config(paths, job.garments, job.gender).values():
            inputs.update(Path(cfg[k]) for k in ('reference_dict', 'profile'))
        if job.opacity_mode == 'v3':
            inputs.update(find_v3_controller(paths, g) for g in job.effective_opacity_garments)
    inputs.add(Path(paths['checkpoint']))
    inputs.update(p for p in Path(paths['body_model_root']).rglob('*') if p.is_file())
    records = [(str(p), digest(p)) for p in sorted(inputs, key=str)]
    for key in ('gaugar_root', 'contourcraft_root'):
        result = subprocess.run(['git', '-C', str(paths[key]), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True)
        dirty = subprocess.run(['git', '-C', str(paths[key]), 'diff', 'HEAD', '--'],
                               capture_output=True, check=True)
        records.append((key, result.stdout.strip(), hashlib.sha256(dirty.stdout).hexdigest()))
    value = json.dumps([records, {k: str(v) for k, v in paths.items()}], sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def stage_assets(paths, files):
    """Copy only required assets; never install code or link writable caches to inputs."""
    gg = Path(paths['gg_output_root']).resolve()
    work = Path(paths['work_root']).resolve()
    package = Path(paths['data_root']).resolve()
    if gg == package or package in gg.parents or work == package or package in work.parents:
        raise ValueError('gg_output_root and output_root must be outside the input package')
    operations = []
    for name, rel, source in files:
        target = work / name / Path(rel).name if rel.startswith('cache/') else gg / name / rel
        root = work if rel.startswith('cache/') else gg
        if root not in target.resolve().parents or any(p.is_symlink() for p in (target, *target.parents) if p != root):
            raise ValueError(f'Refusing a symlinked destination: {target}')
        if target.exists():
            # Worker caches are writable derived state; model assets must match exactly.
            if not target.is_file() or (not rel.startswith('cache/') and digest(source) != digest(target)):
                raise ValueError(f'Destination differs from package: {target}; use a new output root')
        else:
            operations.append((source, target))
    for source, target in operations:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run_batch(config, jobs_file, execute=False):
    paths = load_paths(config)
    validate_output_roots(paths)
    catalog = load_catalog(paths['data_root'])
    jobs = resolve_auto_frames(expand_jobs(jobs_file), paths, catalog)
    problems = validate_jobs_against_catalog(jobs, paths, catalog)
    problems += motion_length_problems(jobs, paths, catalog)
    for job in jobs:
        repair_config(paths, job.garments, job.gender)
    for key in ('body_model_root', 'checkpoint', 'ccraft_data_root', 'cmu_root',
                'gg_data_root', 'gg_aux_root', 'contourcraft_python', 'gaugar_python',
                'gaugar_root', 'contourcraft_root'):
        if not paths.get(key) or not Path(paths[key]).exists():
            problems.append(f'Missing runtime path: {key}')
    if problems:
        raise ToolkitError('\n'.join(problems))
    files = package_files(paths, sorted({g for j in jobs for g in j.garments}))
    revision = fingerprint(paths, jobs, files)
    paths['pipeline_fingerprint'] = revision
    paths['work_root'] = Path(paths['work_root']) / 'precompute' / revision[:16]
    library = paths['library_root']
    todo = []
    for job in jobs:
        record = read_sidecar(job.sidecar_path(library)) or {}
        if not is_done(library, job) or record.get('pipeline_fingerprint') != revision:
            todo.append(job)
    print(json.dumps(dict(jobs=len(jobs), pending=len(todo), fingerprint=revision,
                          work_root=str(paths['work_root']), library_root=str(library),
                          outputs=[str(j.video_path(library)) for j in todo]), indent=2))
    if not execute or not todo:
        return 0
    paths['work_root'].mkdir(parents=True, exist_ok=True)
    library.mkdir(parents=True, exist_ok=True)
    with (library / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ToolkitError('Another batch is publishing this library') from exc
        stage_assets(paths, files)
        failures = []
        for job in todo:
            try:
                render_video(paths, catalog, job, job.video_path(library), job.sidecar_path(library))
            except RenderFailed as exc:
                failures.append(job.key)
                atomic_write_json(job.sidecar_path(library), exc.sidecar)
            finally:
                atomic_write_json(library / 'manifest.json', build(library))
        publish_site_files(library, paths['data_root'])
        print(json.dumps({'failed': failures}, indent=2))
        return 1 if failures else 0


def index_library(config, execute=False):
    paths = load_paths(config)
    validate_output_roots(paths)
    manifest = build(paths['library_root'])
    print(json.dumps({'videos': manifest['count'], 'library_root': str(paths['library_root'])}))
    if execute:
        paths['library_root'].mkdir(parents=True, exist_ok=True)
        with (paths['library_root'] / '.batch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_write_json(paths['library_root'] / 'manifest.json', build(paths['library_root']))
            publish_site_files(paths['library_root'], paths['data_root'])
    return 0
