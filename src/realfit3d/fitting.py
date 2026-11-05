"""Validate opt-in repair inputs before starting a GPU worker."""
import json
from pathlib import Path


def repair_config(settings, names, gender):
    configured = settings.get('garment_repairs', {})
    if not isinstance(configured, dict):
        raise ValueError('garment_repairs must be an object keyed by garment ID')
    result = {}
    for name in names:
        cfg = configured.get(name)
        if cfg is None:
            continue
        if (not isinstance(cfg, dict) or set(cfg) != {'reference_dict', 'profile', 'genders'}
                or not isinstance(cfg['genders'], list) or not cfg['genders']
                or any(g not in ('male', 'female') for g in cfg['genders'])):
            raise ValueError(f'Invalid repair configuration for {name}')
        if gender not in cfg['genders']:
            continue
        for key in ('reference_dict', 'profile'):
            if not isinstance(cfg[key], str) or not Path(cfg[key]).is_file():
                raise ValueError(f'Missing repair {key} for {name}: {cfg.get(key)}')
        profile = json.loads(Path(cfg['profile']).read_text())
        if not isinstance(profile, dict) or not profile.get('regions'):
            raise ValueError(f'Repair profile for {name} must select anatomical regions')
        result[name] = cfg
    return result
