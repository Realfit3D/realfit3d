import ast
import json
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch

from apps.api.server import create_app
from realfit3d.adapters.preview import WORKERS
from realfit3d.fitting import repair_config
from realfit3d.precompute import commands
from realfit3d.precompute.build_index import build
from realfit3d.precompute.render_one_video import _pipeline_cmd, render_video, RenderFailed
from realfit3d.precompute.toolkit_common import (
    Job, ToolkitError, expand_jobs, load_paths, make_sidecar, is_done, motion_file,
)


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = Job(('pants', 'shirt'), 'male', 175, 75, 'walk', 20, 30, .5, 'v3', ('shirt',))
        self.settings = {k: str(self.root / k) for k in (
            'ccraft_python', 'gaugar_python', 'contourcraft_root', 'gaugar_root',
            'package_root', 'output_root', 'library_root', 'gg_output_root',
            'body_model_root', 'checkpoint', 'ccraft_data_root', 'cmu_root')}
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(self.settings))
        self.paths = load_paths(self.config)
        self.catalog = {'garments': {'pants': {'slot': 'lower', 'source_gender': 'male'},
                                     'shirt': {'slot': 'upper', 'source_gender': 'male'}},
                        'motions': {'walk': {'file': 'walk.npz'}}}

    def file(self, path, contents='fixture'):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def test_shared_worker_runtime_and_opacity_target(self):
        controller = self.file(self.paths['data_root'] / 'garments/shirt/opacity_controller_v3/controller.pt')
        ref, profile = self.file(self.root / 'ref.pkl'), self.file(self.root / 'profile.json', '{"regions":["left_arm"]}')
        self.paths['garment_repairs'] = {'shirt': dict(reference_dict=str(ref), profile=str(profile), genders=['male'])}
        command = _pipeline_cmd(self.paths, self.catalog, self.job, 'unique', None)
        self.assertIn(str(WORKERS.parent / 'runtime_entry.py'), command)
        self.assertIn(str(WORKERS / 'run_outfit_preview_pipeline.py'), command)
        self.assertIn('--skip-preview', command)
        self.assertIn(f'shirt={controller}', command)
        self.assertEqual(command[command.index('--opacity-garments') + 1], 'shirt')
        self.assertEqual(command[command.index('--full-steps') + 1], '28')
        self.assertIn('--garment-repairs-json', command)
        self.assertNotIn(str(Path(self.settings['contourcraft_root']) / 'run_outfit_preview_pipeline.py'), command)

    def test_repair_default_off_and_missing_reference_refused(self):
        self.assertEqual(repair_config({}, ['shirt'], 'male'), {})
        with self.assertRaises(ValueError):
            repair_config({'garment_repairs': {'shirt': {'genders': ['male']}}}, ['shirt'], 'male')

    def test_config_resolves_repair_paths_relative_to_config(self):
        self.settings['garment_repairs'] = {'shirt': dict(reference_dict='ref.pkl', profile='profile.json', genders=['male'])}
        self.config.write_text(json.dumps(self.settings))
        paths = load_paths(self.config)
        self.assertEqual(paths['garment_repairs']['shirt']['reference_dict'], str(self.root / 'ref.pkl'))

    def test_job_paths_cannot_escape_and_identity_includes_targets(self):
        with self.assertRaises(ValueError):
            Job(('pants', '../shirt'), 'male', 175, 75, 'walk', 20, 30)
        other = Job(self.job.garments, 'male', 175, 75, 'walk', 20, 30, .5, 'v3')
        self.assertNotEqual(self.job.key, other.key)
        self.catalog['motions']['walk']['file'] = '../../secret.npz'
        with self.assertRaises(ValueError):
            motion_file(self.paths, self.catalog, 'walk')

    def test_duplicate_output_with_different_frames_refused(self):
        job = dict(garments=['pants', 'shirt'], gender='male', heights=[175], weights=[75], motions=['walk'], frames=20)
        path = self.file(self.root / 'jobs.json', json.dumps({'jobs': [job, dict(job, frames=25)]}))
        with self.assertRaises(ToolkitError):
            expand_jobs(path)

    def test_asset_staging_copies_inputs_and_refuses_model_overwrite(self):
        src = self.file(self.paths['data_root'] / 'garments/shirt/cache/garment_dict.pkl')
        model = self.file(self.paths['data_root'] / 'garments/shirt/stage3/ckpt/net.pt')
        files = [('shirt', 'cache/garment_dict.pkl', src), ('shirt', 'stage3/ckpt/net.pt', model)]
        commands.stage_assets(self.paths, files)
        cache = self.paths['work_root'] / 'shirt/garment_dict.pkl'
        self.assertFalse(cache.is_symlink())
        cache.write_text('derived')
        self.assertEqual(src.read_text(), 'fixture')
        target = Path(self.paths['gg_output_root']) / 'shirt/stage3/ckpt/net.pt'
        target.write_text('other model')
        with self.assertRaises(ValueError):
            commands.stage_assets(self.paths, files)
        self.assertEqual(target.read_text(), 'other model')

    def test_asset_staging_rejects_symlink_cache(self):
        src = self.file(self.paths['data_root'] / 'garments/shirt/cache/garment_dict.pkl')
        target = self.paths['work_root'] / 'shirt/garment_dict.pkl'
        target.parent.mkdir(parents=True)
        target.symlink_to(src)
        with self.assertRaises(ValueError):
            commands.stage_assets(self.paths, [('shirt', 'cache/garment_dict.pkl', src)])

    def test_output_roots_cannot_overlap_package(self):
        commands.validate_output_roots(self.paths)
        for root in (self.paths['data_root'], self.paths['data_root'] / 'videos', self.root):
            with self.assertRaises(ToolkitError):
                commands.validate_output_roots(dict(self.paths, library_root=root))

    def test_catalog_change_invalidates_fingerprint(self):
        catalog_path = self.file(self.paths['data_root'] / 'catalog.json', json.dumps(self.catalog))
        self.file(self.paths['data_root'] / 'motions/walk.npz')
        self.file(self.paths['checkpoint'])
        job = Job(self.job.garments, 'male', 175, 75, 'walk', 20, 30)
        def git_result(*args, **kwargs):
            return type('Result', (), {'stdout': 'revision' if kwargs.get('text') else b''})()
        with patch.object(commands.subprocess, 'run', side_effect=git_result):
            before = commands.fingerprint(self.paths, [job], [])
            self.catalog['garments']['shirt']['source_gender'] = 'female'
            catalog_path.write_text(json.dumps(self.catalog))
            self.assertNotEqual(before, commands.fingerprint(self.paths, [job], []))

    def test_render_publishes_after_frame_validation_and_keeps_run(self):
        self.file(self.paths['data_root'] / 'garments/shirt/opacity_controller_v3/controller.pt')
        for name in self.job.garments:
            for rel in ('garment_dict.pkl', 'source_body_fit.pkl'):
                self.file(self.paths['data_root'] / 'garments' / name / 'cache' / rel)
        def fake(command, **kwargs):
            rn = command[command.index('--run-name') + 1]
            run = self.paths['work_root'] / 'outfits' / self.job.work_name / 'runs_preview' / rn
            self.file(run / 'video.mp4')
            for i in range(self.job.frames):
                self.file(run / 'frames/renders' / f'{i:05d}.png')
            self.file(run / 'betas.pkl')
            self.assertIn('REALFIT3D_CC_DATA_ROOT', kwargs['env'])
            return type('Result', (), {'returncode': 0})()
        lib = self.paths['library_root']
        with patch('realfit3d.precompute.render_one_video.subprocess.run', side_effect=fake):
            record = render_video(self.paths, self.catalog, self.job, self.job.video_path(lib), self.job.sidecar_path(lib))
        self.assertTrue(is_done(lib, self.job))
        self.assertTrue(Path(record['run_dir']).exists())
        self.assertEqual(build(lib)['count'], 1)

    def test_library_serving_restricts_private_artifacts_and_symlinks(self):
        root = Path(self.settings['library_root'])
        self.file(root / 'manifest.json', '{"videos":[]}')
        self.file(root / 'run/log.json', '{}')
        self.file(root / 'clip.mp4', 'video')
        (root / 'escape.mp4').symlink_to(self.file(self.root / 'secret.mp4'))
        client = create_app(self.settings).test_client()
        for url in ('/', '/library/', '/library/media/manifest.json', '/library/media/clip.mp4'):
            response = client.get(url)
            self.assertEqual(response.status_code, 200, url)
            response.close()
        for path in ('run/log.json', 'escape.mp4', '../secret.mp4'):
            self.assertEqual(client.get('/library/media/' + path).status_code, 403)
        response = client.get('/library/media/clip.mp4', headers={'Range': 'bytes=0-1'})
        self.assertEqual(response.status_code, 206)
        response.close()

    def test_worker_preserves_preview_default_and_repairs_before_simulation(self):
        source = (WORKERS / 'run_outfit_preview_pipeline.py').read_text()
        ast.parse(source)
        self.assertIn('if not args.skip_preview:', source)
        self.assertIn('preview_ready=not args.skip_preview', source)
        self.assertIn("'--apply', '--output-garment-dict'", source)
        self.assertIn("shutil.copy2(output_dict_path, backup)", source)

    def test_batch_plan_is_read_only_and_resume_checks_fingerprint(self):
        for key in ('gg_data_root', 'gg_aux_root'):
            self.settings[key] = str(self.root / key)
        for key, value in self.settings.items():
            if key in ('output_root', 'library_root', 'gg_output_root'):
                continue
            if key.endswith('_python') or key == 'checkpoint':
                self.file(value)
            else:
                Path(value).mkdir(parents=True, exist_ok=True)
        self.config.write_text(json.dumps(self.settings))
        self.file(self.paths['data_root'] / 'catalog.json', json.dumps(self.catalog))
        self.file(self.paths['data_root'] / 'motions/walk.npz')
        for name in self.job.garments:
            for rel in commands.GARMENT_FILES:
                self.file(self.paths['data_root'] / 'garments' / name / rel)
        jobs = self.file(self.root / 'jobs.json', json.dumps({'jobs': [dict(
            garments=['pants', 'shirt'], gender='male', heights=[175], weights=[75], motions=['walk'], frames=20)]}))
        output = io.StringIO()
        with patch.object(commands, 'fingerprint', return_value='revision'), patch.object(commands, 'motion_length_problems', return_value=[]):
            with redirect_stdout(output):
                self.assertEqual(commands.run_batch(self.config, jobs), 0)
        self.assertEqual(json.loads(output.getvalue())['pending'], 1)
        self.assertFalse(Path(self.settings['output_root']).exists())
        self.assertFalse(Path(self.settings['gg_output_root']).exists())
        self.assertFalse(Path(self.settings['library_root']).exists())
        job = expand_jobs(jobs)[0]
        self.file(job.video_path(self.paths['library_root']))
        self.file(job.sidecar_path(self.paths['library_root']), json.dumps(make_sidecar(job, 'done', pipeline_fingerprint='revision')))
        output = io.StringIO()
        with patch.object(commands, 'fingerprint', return_value='revision'), patch.object(commands, 'motion_length_problems', return_value=[]):
            with redirect_stdout(output):
                commands.run_batch(self.config, jobs)
        self.assertEqual(json.loads(output.getvalue())['pending'], 0)

    def test_failed_render_does_not_publish_video(self):
        lib = self.paths['library_root']
        class FailedProcess:
            returncode = 1
        with patch('realfit3d.precompute.render_one_video.ensure_cache_copies', return_value=[]), \
             patch('realfit3d.precompute.render_one_video._pipeline_cmd', return_value=['fake']), \
             patch('realfit3d.precompute.render_one_video.subprocess.run', return_value=FailedProcess()):
            with self.assertRaises(RenderFailed) as caught:
                render_video(self.paths, self.catalog, self.job, self.job.video_path(lib), self.job.sidecar_path(lib))
        self.assertFalse(self.job.video_path(lib).exists())
        self.assertFalse(self.job.sidecar_path(lib).exists())
        self.assertEqual(caught.exception.sidecar['status'], 'failed')
        self.assertTrue(Path(caught.exception.sidecar['log_path']).exists())


if __name__ == '__main__':
    unittest.main()
