"""Shared local CLI. Plans are read-only unless --execute is provided."""
import argparse
import json
from pathlib import Path
import subprocess
from uuid import uuid4

from realfit3d.doctor import inspect
from realfit3d.adapters.preview import PreviewRuntime, RuntimeUnavailable
from realfit3d.contracts import body_parameters, select_outfit, outfit_name
from realfit3d.assets import inspect_asset
from realfit3d.pipelines.offline import STAGES, build_plan
from realfit3d.settings import ROOT, load_settings, runtime_environment

QUALITY = Path(__file__).resolve().parent / 'quality'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    commands = parser.add_subparsers(dest='action', required=True)
    library = commands.add_parser('library', help='Plan/render precomputed videos or rebuild their index')
    library.add_argument('operation', choices=['render', 'index'])
    library.add_argument('--jobs', type=Path)
    library.add_argument('--execute', action='store_true')
    doctor = commands.add_parser('doctor')
    doctor.add_argument('--probe', action='store_true', help='Check import specs in configured Python environments')
    assets = commands.add_parser('assets')
    assets.add_argument('names', nargs='+')
    assets.add_argument('--hash', action='store_true', help='Read and SHA256 each artifact, including weights')
    offline = commands.add_parser('offline')
    offline.add_argument('stage', choices=STAGES)
    offline.add_argument('--subject', required=True)
    offline.add_argument('--sequence', default='Sequence1')
    offline.add_argument('--gender', choices=['male', 'female'], default='male')
    offline.add_argument('--frame', type=int, default=0)
    offline.add_argument('--garment-type', choices=['upper', 'lower', 'dress'], default='upper')
    offline.add_argument('--prompt', default='shirt')
    offline.add_argument('--execute', action='store_true')
    tryon = commands.add_parser('tryon')
    tryon.add_argument('--betas', type=Path, required=True, help='Trusted body-fit betas.pkl matching the measurements')
    tryon.add_argument('--height', type=float, required=True)
    tryon.add_argument('--weight', type=float, required=True)
    tryon.add_argument('--gender', choices=['female', 'male'], required=True)
    tryon.add_argument('--upper', required=True)
    tryon.add_argument('--lower', required=True)
    tryon.add_argument('--motion', type=int, default=0)
    tryon.add_argument('--opacity-mode', choices=['direct', 'optical_depth', 'v3'], default='direct')
    tryon.add_argument('--opacity-scale', type=float, default=1.0)
    tryon.add_argument('--execute', action='store_true')
    quality = commands.add_parser('quality')
    quality.add_argument('--execute', action='store_true')
    quality.add_argument('tool', choices=sorted(p.stem for p in QUALITY.glob('*.py') if not p.name.startswith('_')))
    quality.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.action == 'library':
            from realfit3d.precompute.commands import run_batch, index_library
            if args.operation == 'index':
                return index_library(args.config, args.execute)
            if not args.jobs:
                raise ValueError('library render requires --jobs')
            return run_batch(args.config, args.jobs, args.execute)
        if args.action == 'doctor':
            report = inspect(settings, args.probe)
            print(json.dumps(report, indent=2))
            return 0 if report['paths_ready'] else 1
        if args.action == 'assets':
            reports = [inspect_asset(settings['gg_output_root'], name, args.hash) for name in args.names]
            print(json.dumps(reports, indent=2))
            return 0 if all(r['files_ready'] for r in reports) else 1
        if args.action == 'offline':
            plan = build_plan(settings, args.stage, args.subject, args.sequence, args.gender,
                              args.frame, args.garment_type, args.prompt).describe()
        elif args.action == 'tryon':
            body = body_parameters(args.height, args.weight, args.gender)
            body['betas_path'] = args.betas.resolve()
            catalog = json.loads(Path(settings.get('catalog', ROOT / 'configs/catalog.json')).read_text())
            upper, lower, motion = select_outfit(catalog, args.upper, args.lower, args.motion)
            runtime = PreviewRuntime(settings)
            run_id = 'cli_' + uuid4().hex
            name = outfit_name(upper, lower, args.gender)
            output = Path(settings['output_root'])
            command = runtime.command(body, upper, lower, motion, run_id, name, output,
                                      args.opacity_mode, args.opacity_scale)
            missing = [] if args.betas.is_file() else [str(args.betas)]
            plan = dict(stage='tryon', command=command, cwd=str(runtime.path('contourcraft_root')),
                        outputs=[str(output / 'outfits' / name / 'runs_preview' / run_id)], missing=missing,
                        notes=['Same command builder as the Web API; trusted betas must match gender and measurements.'])
            if args.execute:
                runtime.check(full=True, motion=motion)
                runtime.check_assets([lower['name'], upper['name']])
        else:
            python = settings.get('reconstruction_python')
            if not python:
                raise ValueError('Configure reconstruction_python for quality tools')
            rest = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
            plan = dict(stage='quality:' + args.tool, command=[python, str(QUALITY / (args.tool + '.py')), *rest],
                        cwd=str(QUALITY), required=[python], outputs=[],
                        notes=['Tool-specific outputs/overwrite semantics must be reviewed via --help.'],
                        missing=[] if Path(python).is_file() else [python])
        print(json.dumps(plan, indent=2))
        if not args.execute:
            return 0
        if plan['missing']:
            raise ValueError('Execution refused: prerequisite paths are missing')
        directory = Path(settings['output_root']) / 'offline_jobs' / uuid4().hex
        directory.mkdir(parents=True)
        record = dict(plan, phase='running', source_lock=json.loads((ROOT / 'upstream.lock.json').read_text()))
        manifest = directory / 'job.json'
        manifest.write_text(json.dumps(record, indent=2))
        try:
            with (directory / 'worker.log').open('w') as log:
                process = subprocess.run(plan['command'], cwd=plan['cwd'], stdout=log,
                                         stderr=subprocess.STDOUT,
                                         env=runtime_environment(settings, plan['command'][0]))
            code = process.returncode
        except BaseException as exc:
            record.update(phase='error', error=repr(exc))
            manifest.write_text(json.dumps(record, indent=2))
            raise
        missing_outputs = [p for p in plan['outputs'] if not Path(p).exists()]
        record.update(phase='done' if code == 0 and not missing_outputs else 'error',
                      exit_code=code, missing_outputs=missing_outputs)
        manifest.write_text(json.dumps(record, indent=2))
        print(f'Job record: {manifest}')
        return 0 if code == 0 and not missing_outputs else 1
    except (OSError, ValueError, KeyError, RuntimeUnavailable) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
