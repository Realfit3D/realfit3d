import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from realfit3d.adapters.preview import ENTRY, PreviewRuntime, RuntimeUnavailable
from realfit3d.assets import inspect_asset
from realfit3d.cli import main
from realfit3d.contracts import body_parameters
from realfit3d.pipelines.offline import STAGES, build_plan
from realfit3d.settings import ROOT, load_settings, runtime_environment


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = {k: str(self.root / k) for k in (
            'actorshq_root', 'reconstruction_python', 'gaugar_root', 'gaugar_python',
            'contourcraft_root', 'ccraft_python', 'body_model_root', 'gg_data_root',
            'gg_output_root', 'gg_aux_root', 'actorshq_data_root', 'ccraft_data_root',
            'cmu_root', 'checkpoint', 'output_root')}

    def test_settings_paths_are_config_relative(self):
        config = self.root / 'paths.json'
        config.write_text(json.dumps({'gg_data_root': 'data', 'ccraft_python': 'env/bin/python'}))
        settings = load_settings(config)
        self.assertEqual(settings['gg_data_root'], str(self.root / 'data'))
        self.assertEqual(settings['gaugar_root'], str(ROOT / 'upstream/Gaussian-Garments'))

    def test_environment_passes_all_roots_to_children(self):
        env = runtime_environment(self.settings, self.settings['ccraft_python'])
        self.assertEqual(env['REALFIT3D_CC_ROOT'], self.settings['contourcraft_root'])
        self.assertEqual(env['REALFIT3D_GG_OUTPUT_ROOT'], self.settings['gg_output_root'])
        self.assertEqual(env['REALFIT3D_CMU_ROOT'], self.settings['cmu_root'])

    def test_python_symlink_preserves_virtual_environment(self):
        interpreter = self.root / 'venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        config = self.root / 'paths.json'
        config.write_text(json.dumps(dict(ccraft_python='venv/bin/python')))
        settings = load_settings(config)
        self.assertEqual(settings['ccraft_python'], str(interpreter))
        self.assertEqual(PreviewRuntime(settings).path('ccraft_python'), interpreter)
        self.assertEqual(runtime_environment(settings)['REALFIT3D_CC_PYTHON'], str(interpreter))

    def test_all_offline_stages_are_plannable_without_gpu(self):
        self.settings['template_repairs'] = {'Actor08': {k: str(self.root/k) for k in ('source','recipe','output')}}
        for stage in STAGES:
            plan = build_plan(self.settings, stage, 'Actor08')
            self.assertTrue(plan.describe()['missing'])
            self.assertTrue(plan.command)
            self.assertTrue(plan.outputs)
        self.assertIn('--masker_automask', build_plan(self.settings, 'reconstruct', 'Actor08').command)

    def test_stage_subject_cannot_escape_paths_or_shell(self):
        for value in ('../actor', 'Actor;pwd', '', 'a/b'):
            with self.assertRaises(ValueError):
                build_plan(self.settings, 'prepare', value)

    def test_offline_is_dry_run_by_default(self):
        config = self.root / 'paths.json'
        config.write_text(json.dumps(self.settings))
        with patch('subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--config', str(config), 'offline', 'appearance', '--subject', 'Actor08']), 0)
        run.assert_not_called()
        self.assertFalse(Path(self.settings['output_root']).exists())

    def test_missing_prerequisites_refuse_execution(self):
        config = self.root / 'paths.json'
        config.write_text(json.dumps(self.settings))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--config', str(config), 'offline', 'appearance', '--subject', 'Actor08', '--execute'])
        self.assertFalse(Path(self.settings['output_root']).exists())

    def test_motion_uses_configured_cmu_root(self):
        runtime = PreviewRuntime(self.settings)
        self.assertEqual(runtime.motion_path('CMU/01/clip.npz'), self.root / 'cmu_root/01/clip.npz')
        with self.assertRaises(RuntimeUnavailable):
            runtime.motion_path('../secret')

    def test_runtime_bootstrap_applies_defaults_in_real_child(self):
        cc = self.root / 'cc'
        (cc / 'utils').mkdir(parents=True)
        (cc / 'utils/__init__.py').write_text('')
        original = 'DEFAULTS = {"data_root": "unconfigured"}\n'
        (cc / 'utils/defaults.py').write_text(original)
        script = self.root / 'child.py'
        script.write_text('import json\nfrom utils.defaults import DEFAULTS\nprint(json.dumps(DEFAULTS))\n')
        command = [sys.executable, str(ENTRY), '--kind', 'cc', '--root', str(cc), '--script', str(script)]
        response = subprocess.run(command, env=runtime_environment(self.settings), capture_output=True, text=True, check=True)
        result = json.loads(response.stdout)
        self.assertEqual(result['data_root'], self.settings['ccraft_data_root'])
        self.assertEqual(result['project_dir'], str(cc))
        self.assertEqual((cc / 'utils/defaults.py').read_text(), original)

    def test_assets_check_exact_ckpt_without_unpickling(self):
        for relative in ('stage1/template_uv.obj', 'stage1/cameras.json', 'stage3/ckpt/net.pt'):
            path = self.root / 'upper' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not a trusted model')
        report = inspect_asset(self.root, 'upper', hash_files=True)
        self.assertTrue(report['files_ready'])
        self.assertFalse(report['topology_verified'])
        self.assertTrue(all(len(f['sha256']) == 64 for f in report['files']))
        self.assertFalse(inspect_asset(self.root, 'missing')['files_ready'])

    def test_fork_rejects_bad_trajectory_before_model_loading(self):
        tree = ast.parse((ROOT / 'upstream/Gaussian-Garments/inference.py').read_text())
        simulation = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Simulation')
        constructor = next(node for node in simulation.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        code = compile(ast.Module(body=[constructor], type_ignores=[]), '<constructor>', 'exec')
        for trajectory in [dict(garment_names=[]), dict(garment_names=['../upper']),
                           dict(garment_names=['upper'], pred=[[1, 2]], obstacle=[[1]]),
                           dict(garment_names=['upper'], pred=[[1, 2, 3]], obstacle=[])]:
            namespace = dict(pickle=types.SimpleNamespace(load=lambda _: trajectory), open=lambda *a: None,
                             Path=Path, DEFAULTS=types.SimpleNamespace(output_root=str(self.root), stage1='stage1'),
                             read_obj=lambda _: dict(vertices=[1, 2, 3]))
            exec(code, namespace)
            with self.assertRaises(ValueError):
                namespace['__init__'](types.SimpleNamespace(), types.SimpleNamespace(traj_path='fake'))

    def test_body_contract_rejects_nonfinite(self):
        for value in ('nan', 'inf', '-inf'):
            with self.assertRaises(ValueError):
                body_parameters(value, 58, 'female')


if __name__ == '__main__':
    unittest.main()
