"""Check GitHub-facing guide links without network access."""

import configparser
import json
from pathlib import Path
import posixpath
import re
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def guide_links():
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            yield path.relative_to(ROOT), urlsplit(link)


class DocumentationLinkTests(unittest.TestCase):
    def test_relative_links_do_not_cross_submodule_boundaries(self):
        modules = configparser.ConfigParser(interpolation=None)
        modules.read(ROOT / '.gitmodules')
        paths = [modules[section]['path'] for section in modules.sections()]
        for document, link in guide_links():
            if link.scheme or link.netloc or not link.path:
                continue
            target = posixpath.normpath((document.parent / unquote(link.path)).as_posix())
            with self.subTest(document=str(document), link=link.geturl()):
                self.assertFalse(any(target.startswith(path + '/') for path in paths),
                                 'Link directly to the dependency repository on GitHub')

    def test_gg_guide_urls_use_fork_and_existing_files(self):
        lock = json.loads((ROOT / 'upstream.lock.json').read_text())
        gg = next(dep for dep in lock['dependencies'] if dep['name'] == 'Gaussian-Garments')
        prefix = gg['repository'].removesuffix('.git') + '/blob/main/'
        found = set()
        for document, link in guide_links():
            if 'REALFIT3D_CHANGES' not in link.path:
                continue
            with self.subTest(document=str(document), link=link.geturl()):
                url = link._replace(fragment='').geturl()
                self.assertTrue(url.startswith(prefix), 'Use the current GG fork URL')
                relative = unquote(url.removeprefix(prefix))
                self.assertTrue((ROOT / gg['path'] / relative).is_file())
                found.add(relative)
        self.assertEqual(found, {'REALFIT3D_CHANGES.md', 'REALFIT3D_CHANGES.zh-TW.md'})


if __name__ == '__main__':
    unittest.main()
