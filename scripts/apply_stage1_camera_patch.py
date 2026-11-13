"""Check or apply calibrated-principal-point support to pinned ActorsHQ/GS2Mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'patches/stage1-camera'
BASES = {'actorshq': '46651f55030407ea25463ed070eed9f7e837a9af',
         'gs2mesh': 'b08a68f2b896301472d25fe3f5287b5b1bf1b737'}
MARKER = '.realfit3d-calibrated-projection.json'


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)


def plan(root):
    result = []
    for name, directory, patch in (
        ('actorshq', root, BUNDLE / 'actorshq.patch'),
        ('gs2mesh', root / 'gs2mesh', BUNDLE / 'gs2mesh.patch'),
    ):
        head = git(directory, 'rev-parse', 'HEAD')
        if head.returncode or head.stdout.strip() != BASES[name]:
            raise ValueError(f'{name} requires base commit {BASES[name]}')
        forward = git(directory, 'apply', '--check', str(patch))
        if forward.returncode == 0:
            status = 'pending'
        elif git(directory, 'apply', '--reverse', '--check', str(patch)).returncode == 0:
            status = 'applied'
        else:
            raise ValueError(f'{name} patch conflicts with the checkout:\n{forward.stderr}')
        result.append((directory, patch, status))
    return result


def apply(root, steps):
    # Check both repositories before changing either one; never reset local edits.
    for directory, patch, status in steps:
        if status == 'pending':
            response = git(directory, 'apply', str(patch))
            if response.returncode:
                raise RuntimeError(response.stderr)
    files = {}
    for directory, patch, _ in steps:
        for line in patch.read_text().splitlines():
            if line.startswith('+++ b/'):
                path = directory / line[6:]
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker = {'base_commits': BASES, 'files_sha256': files, 'projection': 'calibrated_pixel_centers'}
    (root / 'gs2mesh' / MARKER).write_text(json.dumps(marker, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actorshq-root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Apply checked patches; default is dry run')
    args = parser.parse_args()
    root = args.actorshq_root.resolve()
    try:
        steps = plan(root)
        for directory, _, status in steps:
            print(f'{status}: {directory}')
        if args.apply:
            apply(root, steps)
            print('Applied and recorded source hashes.')
        else:
            print('Dry run: no files changed. Add --apply to install.')
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
