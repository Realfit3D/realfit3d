import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from realfit3d.adapters.stage1 import projection_marker
from realfit3d.pipelines.offline import build_plan

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('camera_installer', ROOT/'scripts/apply_stage1_camera_patch.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class CameraPatchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'actors'
        self.bundle = Path(tmp.name)/'patches'
        self.bundle.mkdir()
        bases = {}
        for name, root in [('actorshq',self.root),('gs2mesh',self.root/'gs2mesh')]:
            root.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'sample.py').write_text('old = True\n')
            subprocess.run(['git','-C',str(root),'add','sample.py'],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.com',
                            'commit','-qm','fixture'],check=True)
            bases[name]=installer.git(root,'rev-parse','HEAD').stdout.strip()
            (self.bundle/f'{name}.patch').write_text(
                'diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n'
                '@@ -1 +1 @@\n-old = True\n+old = False\n')
        self.bundle_patch=patch.object(installer,'BUNDLE',self.bundle)
        self.base_patch=patch.object(installer,'BASES',bases)
        self.bundle_patch.start()
        self.base_patch.start()
        self.addCleanup(self.bundle_patch.stop)
        self.addCleanup(self.base_patch.stop)

    def test_dry_run_apply_idempotence_and_unrelated_files(self):
        (self.root/'notes.txt').write_text('preserve me')
        steps=installer.plan(self.root)
        self.assertTrue(all(s[2]=='pending' for s in steps))
        self.assertEqual((self.root/'sample.py').read_text(),'old = True\n')
        installer.apply(self.root,steps)
        self.assertTrue(all(s[2]=='applied' for s in installer.plan(self.root)))
        installer.apply(self.root,installer.plan(self.root))
        self.assertEqual((self.root/'notes.txt').read_text(),'preserve me')
        self.assertTrue(projection_marker(self.root).exists())

    def test_conflict_preflight_leaves_other_repo_unchanged(self):
        (self.root/'gs2mesh/sample.py').write_text('different = True\n')
        with self.assertRaises(ValueError):
            installer.plan(self.root)
        self.assertEqual((self.root/'sample.py').read_text(),'old = True\n')

    def test_stale_manifest_rejected(self):
        installer.apply(self.root,installer.plan(self.root))
        (self.root/'sample.py').write_text('changed again\n')
        with self.assertRaisesRegex(ValueError,'source changed'):
            projection_marker(self.root)

    def test_wrong_base_rejected(self):
        with patch.object(installer,'BASES',dict(actorshq='invalid',gs2mesh='invalid')):
            with self.assertRaisesRegex(ValueError,'base commit'):
                installer.plan(self.root)

    def test_cli_plan_opt_in_and_missing_manifest(self):
        settings={key:str(self.root/key) for key in ('actorshq_data_root','gg_data_root',
                  'gg_output_root','gg_aux_root','reconstruction_python')}
        settings['actorshq_root']=str(self.root)
        legacy=build_plan(settings,'reconstruct','Actor05')
        self.assertNotIn('--GS_calibrated_principal_point',legacy.command)
        settings['stage1_projection']='calibrated'
        calibrated=build_plan(settings,'reconstruct','Actor05')
        self.assertIn('--GS_calibrated_principal_point',calibrated.command)
        self.assertIn(str(self.root/'gs2mesh'/installer.MARKER),calibrated.describe()['missing'])
        settings['stage1_projection']='typo'
        with self.assertRaises(ValueError):
            build_plan(settings,'reconstruct','Actor05')


if __name__=='__main__':
    unittest.main()
