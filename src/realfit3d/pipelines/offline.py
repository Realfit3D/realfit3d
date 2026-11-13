from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

from realfit3d.adapters.preview import ENTRY, WORKERS
from realfit3d.adapters.stage1 import projection_marker

STAGES = ('prepare', 'initialize', 'reconstruct', 'repair', 'register-template',
          'register-sequence', 'appearance', 'opacity', 'convert', 'dynamics')


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Invalid subject/sequence identifier: {value!r}')
    return value


@dataclass
class Plan:
    stage: str
    command: list
    cwd: str
    required: list
    outputs: list
    notes: list

    def describe(self):
        result = asdict(self)
        result['missing'] = [p for p in self.required if not Path(p).exists()]
        return result


def build_plan(settings, stage, subject, sequence='Sequence1', gender='male',
               frame=0, garment_type='upper', prompt='shirt'):
    subject, sequence = identifier(subject), identifier(sequence)
    if stage not in STAGES or frame < 0 or gender not in ('male', 'female') or garment_type not in ('upper', 'lower', 'dress'):
        raise ValueError('Invalid offline stage parameters')

    def path(key):
        if not settings.get(key):
            raise ValueError(f'Missing setting: {key}')
        return Path(settings[key])

    output = path('gg_output_root') / subject
    data = path('gg_data_root') / subject / sequence
    required = []
    notes = ['Use a separate asset version for each repaired template; review masks and opening semantics.']
    if stage in ('prepare', 'initialize', 'reconstruct'):
        root, python = path('actorshq_root'), path('reconstruction_python')
        script = {'prepare': 'data_preparation.py', 'initialize': 'mesh_initialization.py',
                  'reconstruct': 'mesh_reconstruction.py'}[stage]
        args = ['--subject', subject, '--sequence', sequence]
        if stage == 'prepare':
            args += ['--masker_prompt', prompt, '--gender', gender]
            outputs = [str(data)]
        elif stage == 'initialize':
            args += ['--ff', str(frame)]
            outputs = [str(root / 'gs2mesh/data/custom' / f'{subject}_{sequence}')]
        else:
            args += ['--garment_type', garment_type, '--masker_prompt', prompt, '--masker_automask']
            projection = settings.get('stage1_projection', 'legacy')
            if projection not in ('legacy', 'calibrated'):
                raise ValueError('stage1_projection must be legacy or calibrated')
            if projection == 'calibrated':
                args.append('--GS_calibrated_principal_point')
                required.append(projection_marker(root))
                notes.append('Calibrated training and stereo projection; install patches/stage1-camera first and use a separate output root for comparisons.')
            required += [root / 'gs2mesh/data/custom' / f'{subject}_{sequence}']
            outputs = [str(output / 'stage1/template.obj')]
            notes.append('Upstream GS2Mesh uses its own automatic masks; quality whitelist is NOT automatically applied.')
        required += [path('actorshq_data_root') / subject / sequence, path('gg_aux_root'), root / 'gs2mesh/gs2mesh_utils']
        command = [str(python), str(ENTRY), '--kind', 'actors', '--root', str(root),
                   '--script', str(root / script), '--', *args]
    elif stage == 'repair':
        from realfit3d.settings import ROOT
        root, python = ROOT, path('reconstruction_python')
        cfg = settings.get('template_repairs', {}).get(subject)
        if not cfg:
            raise ValueError('Configure template_repairs for this subject')
        script = ROOT / 'src/realfit3d/quality/repair_template.py'
        args = ['--source', cfg['source'], '--recipe', cfg['recipe'], '--output', cfg['output']]
        for name in ('body', 'reference', 'body_model'):
            if cfg.get(name):
                args += ['--'+name.replace('_', '-'), cfg[name]]
                required.append(Path(cfg[name]))
        required += [Path(cfg['source']), Path(cfg['recipe'])]
        outputs = [str(Path(cfg['output'])/'template_uv.obj'), str(Path(cfg['output'])/'quality.json')]
        command = [str(python), str(script), *args]
        notes = ['Builds a separate candidate. Inspect quality.json and install a reviewed candidate into a new asset before registration.']
    elif stage == 'opacity':
        root, python = path('gaugar_root'), path('gaugar_python')
        command = [str(python), str(root / 'train_opacity_v3.py'),
                   '--subject', subject, '--sequence', sequence]
        required += [path('gg_data_root'), output / 'stage1/template_uv.obj', output / 'stage2' / sequence,
                     output / 'stage3/ckpt/net.pt', path('gg_aux_root')]
        input_cameras = data / 'cameras.json'
        stage2_cameras = output / 'stage2' / sequence / 'cameras.json'
        if input_cameras.is_file():
            required += [input_cameras]
        elif stage2_cameras.is_file() and isinstance(json.loads(stage2_cameras.read_text()), dict):
            required += [stage2_cameras]
        else:
            required += [output / 'stage1/cameras.json', stage2_cameras]
        outputs = [str(output / 'opacity_controller_v3/controller.pt')]
        notes = ['Trains V3 with frozen Stage 3 weights; requires Stage 2 meshes and AO/normal maps.',
                 'Image-free input uses Stage 1 full calibration and the Stage 2 camera list; the reader validates their consistency.',
                 'Default split: training frames 0-29, validation 30-39; existing controllers are never overwritten.']
    elif stage in ('register-template', 'register-sequence', 'appearance'):
        root, python = path('gaugar_root'), path('gaugar_python')
        script = 's3_appearance.py' if stage == 'appearance' else 's2_registration.py'
        args = ['-s', subject, '-so', subject, '--only_foreground_loss']
        if stage != 'appearance':
            args += ['-q', sequence]
        if stage == 'register-template':
            args += ['-tf', str(frame)]
        elif stage == 'register-sequence':
            args += ['-t', 'Template', '--start_from', str(frame)]
            required += [output / 'stage2/Template/template.obj']
        else:
            required += [output / 'stage2' / sequence]
        required += [data, output / 'stage1/template_uv.obj', output / 'stage1/point_cloud.ply', data / 'cameras.json', path('gg_aux_root')]
        outputs = [str(output / ('stage3' if stage == 'appearance' else 'stage2'))]
        command = [str(python), str(root / script), *args]
    else:
        root, python = path('contourcraft_root'), path('ccraft_python')
        required += [path('ccraft_data_root'), path('body_model_root'), path('checkpoint'), path('cmu_root')]
        if stage == 'convert':
            script = WORKERS / 'convert_training_assets.py'
            required += [output / 'stage2', output / 'stage1/template_uv.obj', data / 'smplx']
            args = ['--subject', subject, '--sequence', sequence, '--gender', gender, '--frame', str(frame),
                    '--gg-data-root', str(path('gg_data_root')), '--gg-output-root', str(path('gg_output_root')),
                    '--body-model-root', str(path('body_model_root')), '--checkpoint', str(path('checkpoint')),
                    '--garment-dicts', str(path('ccraft_data_root') / 'aux_data/garment_dicts/finetune'),
                    '--frame-rate', str(settings.get('training_frame_rate',30)),
                    '--validation-frames', str(settings.get('validation_frames',10)),
                    '--training-updates', str(settings.get('training_updates',1000)),
                    '--rollout-steps', str(settings.get('training_rollout_steps',8)),
                    '--checkpoint-every', str(settings.get('training_checkpoint_every',100)),
                    '--split-repeat', str(settings.get('training_split_repeat',100))]
            if garment_type == 'lower':
                support = output / 'stage1/repair_quality.json'
                required.append(support)
                args += ['--support-report', str(support)]
                notes.append('Lower-garment waist handles are installed before initial relaxation.')
            outputs = [str(output / 'stage4'), str(root / f'configs/finetune/{subject}.yaml')]
            notes.append('GauGarConverter writes a generated finetune YAML under the CC checkout; review before training.')
        else:
            script = root / 'train.py'
            args = [f'config=finetune/{subject}']
            required += [root / f'configs/finetune/{subject}.yaml', output / 'stage4']
            outputs = [str(path('ccraft_data_root') / 'trained_models/finetuning' / subject)]
        command = [str(python), str(ENTRY), '--kind', 'cc', '--root', str(root), '--script', str(script), '--', *args]
    required += [python, root, Path(command[command.index('--script') + 1]) if '--script' in command else Path(command[1])]
    return Plan(stage, command, str(root), [str(p) for p in required], outputs, notes)
