"""Validate the optional calibrated-projection patch without GPU dependencies."""
import hashlib
import json
from pathlib import Path

MARKER = '.realfit3d-calibrated-projection.json'


def projection_marker(root):
    root = Path(root).resolve()
    marker = root / 'gs2mesh' / MARKER
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get('projection') != 'calibrated_pixel_centers' or not record.get('files_sha256'):
            raise ValueError('Invalid Stage 1 projection manifest; rerun the camera patch installer')
        for relative, expected in record['files_sha256'].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Invalid Stage 1 projection source: {relative}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Stage 1 projection source changed: {relative}; check and reapply the camera patch')
    return marker
