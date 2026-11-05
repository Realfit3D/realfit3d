"""UI-only playback check using explicitly supplied sample assets, without GPU work."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
from apps.api.server import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--screenshots', type=Path, required=True)
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='realfit-library-ui-') as tmp:
        root = Path(tmp)
        shutil.copy2(args.assets / 'videos/00170_05_08.mp4', root / 'opaque.mp4')
        shutil.copy2(args.assets / 'videos/00170_05_08.mp4', root / 'target.mp4')
        (root / 'thumbnails').mkdir()
        for name, source in [('pants', '00170_lower'), ('shirt', '00170_upper')]:
            shutil.copy2(args.assets / 'thumbnails' / (source + '.png'), root / 'thumbnails' / (name + '.png'))
        catalog = dict(garments=dict(pants=dict(slot='lower', label='Pants'), shirt=dict(slot='upper', label='Shirt')),
                       motions=dict(walk=dict(label='Walk')))
        common = dict(garments=['pants', 'shirt'], gender='male', height=175, weight=75, motion='walk', sample=True)
        manifest = dict(videos=[dict(common, path='opaque.mp4'),
                               dict(common, path='target.mp4', opacity=.5, opacity_mode='v3', opacity_garments=['shirt'])])
        (root / 'catalog.json').write_text(json.dumps(catalog))
        (root / 'manifest.json').write_text(json.dumps(manifest))
        server = make_server('127.0.0.1', 0, create_app(dict(library_root=str(root))))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, executable_path=os.environ.get('REALFIT3D_BROWSER'))
                for width, height in [(1440, 1000), (390, 844)]:
                    page = browser.new_page(viewport=dict(width=width, height=height))
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{server.server_port}/library/')
                    page.locator('#genders button', has_text='男性').click()
                    page.locator('#height').select_option('175')
                    page.locator('#weight').select_option('75')
                    page.screenshot(path=str(args.screenshots / f'body-{width}.png'), full_page=True)
                    page.locator('#go-try').click()
                    page.locator('#uppers .card').first.click()
                    page.locator('#lowers .card').first.click()
                    page.wait_for_function('document.querySelector("#video").readyState >= 2')
                    assert page.locator('#video').evaluate('(v) => v.videoWidth > 0 && v.videoHeight > 0')
                    assert page.locator('.card img').evaluate_all('(xs) => xs.every(x => x.complete && x.naturalWidth > 0)')
                    page.locator('#opacity').fill('1')
                    page.locator('#opacity').dispatch_event('change')
                    page.wait_for_function('document.querySelector("#video").src.includes("target.mp4")')
                    page.wait_for_function('document.querySelector("#video").readyState >= 2 && document.querySelector("#video").currentTime > 0.2')
                    page.wait_for_function('getComputedStyle(document.querySelector("#video")).opacity === "1"')
                    assert 'Shirt' in page.locator('#opacity-val').inner_text()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    page.screenshot(path=str(args.screenshots / f'playback-{width}.png'), full_page=True)
                    assert not errors, errors
                    page.close()
                    print(f'{width}x{height}: body selection, thumbnails, decoded video, target selection, overflow passed')
                browser.close()
        finally:
            server.shutdown()
            thread.join()


if __name__ == '__main__':
    main()
