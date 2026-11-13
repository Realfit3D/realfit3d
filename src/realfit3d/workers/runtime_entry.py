"""Run a research script with process-local defaults, without editing upstream."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys


def configure(kind, root):
    if kind == 'cc':
        defaults = importlib.import_module('utils.defaults').DEFAULTS
        data = os.environ['REALFIT3D_CC_DATA_ROOT']
        defaults.update(data_root=data, aux_data=str(Path(data) / 'aux_data'),
                        project_dir=str(root), experiment_root=str(Path(data) / 'experiments'),
                        CMU_root=os.environ['REALFIT3D_CMU_ROOT'])
        from realfit3d.workers.contourcraft.registered_sequence import install
        install()
    elif kind == 'actors':
        defaults = importlib.import_module('defaults').DEFAULTS
        for key, environment in [('data_root', 'REALFIT3D_GG_DATA_ROOT'),
                                 ('output_root', 'REALFIT3D_GG_OUTPUT_ROOT'),
                                 ('aux_root', 'REALFIT3D_GG_AUX_ROOT'),
                                 ('AHQ_data_root', 'REALFIT3D_ACTORSHQ_DATA_ROOT')]:
            defaults[key] = Path(os.environ[environment])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['cc', 'actors'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    args, rest = parser.parse_known_args()
    root, script = args.root.resolve(), args.script.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(script.parent)]
    configure(args.kind, root)
    sys.argv = [str(script)] + (rest[1:] if rest[:1] == ['--'] else rest)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
