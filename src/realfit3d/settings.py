import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_PATHS = {
    'gg_data_root': 'REALFIT3D_GG_DATA_ROOT',
    'gg_output_root': 'REALFIT3D_GG_OUTPUT_ROOT',
    'gg_aux_root': 'REALFIT3D_GG_AUX_ROOT',
    'ccraft_data_root': 'REALFIT3D_CC_DATA_ROOT',
    'cmu_root': 'REALFIT3D_CMU_ROOT',
    'actorshq_data_root': 'REALFIT3D_ACTORSHQ_DATA_ROOT',
    'contourcraft_root': 'REALFIT3D_CC_ROOT',
    'ccraft_python': 'REALFIT3D_CC_PYTHON',
}


def load_settings(filename=None):
    filename = filename or os.environ.get('REALFIT3D_CONFIG')
    settings = json.loads(Path(filename).read_text()) if filename else {}
    if not isinstance(settings, dict):
        raise ValueError('Settings must be a JSON object')
    # Resolve paths relative to the config file, not the caller's working directory.
    base = Path(filename).resolve().parent if filename else ROOT
    for key, value in settings.items():
        if key.endswith(('_root', '_python')) or key in ('checkpoint', 'reference_motion', 'catalog'):
            if value:
                path = Path(value).expanduser()
                # Resolving a venv's Python symlink bypasses that virtual environment.
                settings[key] = os.path.abspath(base / path) if key.endswith('_python') else str((base / path).resolve())
    for key, name in [('gaugar_root', 'Gaussian-Garments'), ('contourcraft_root', 'ContourCraft'),
                      ('actorshq_root', 'ActorsHQ-for-Gaussian-Garments')]:
        settings.setdefault(key, str(ROOT / 'upstream' / name))
    settings.setdefault('output_root', str(ROOT / 'storage'))
    if not isinstance(settings.get('garment_repairs', {}), dict):
        raise ValueError('garment_repairs must be an object')
    for cfg in settings.get('garment_repairs', {}).values():
        if not isinstance(cfg, dict):
            raise ValueError('Each garment repair must be an object')
        for key in ('reference_dict', 'profile'):
            if cfg.get(key):
                cfg[key] = str((base / Path(cfg[key]).expanduser()).resolve())
    settings['v3_controllers'] = {
        name: str((base / Path(path).expanduser()).resolve())
        for name, path in settings.get('v3_controllers', {}).items()
    }
    return settings


def runtime_environment(settings, python=None):
    env = os.environ.copy()
    for key, name in ENVIRONMENT_PATHS.items():
        if settings.get(key):
            path = Path(settings[key]).expanduser()
            env[name] = os.path.abspath(path) if key.endswith('_python') else str(path.resolve())
    if python:
        env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    return env
