# RealFit3D

A research virtual try-on integration built on
[Gaussian Garments](https://github.com/eth-ait/Gaussian-Garments),
ContourCraft, and ActorsHQ preprocessing.

The website, API, and workflow adapters live here. GG is included as a
**pinned submodule of our maintained fork**.
Detailed GG changes: [English](https://github.com/Realfit3D/Gaussian-Garments/blob/main/REALFIT3D_CHANGES.md) |
[Traditional Chinese](https://github.com/Realfit3D/Gaussian-Garments/blob/main/REALFIT3D_CHANGES.zh-TW.md).

## Status

The local web interface, CLI, and runtime adapters are implemented.
**End-to-end GPU try-on and training validation are pending.**
Opacity controls are connected to Web, CLI, and both GG rendering entries.
V3 uses a separate controller checkpoint for each garment.
Precomputed video playback is available at `/library/`; batch generation and
opt-in garment initialization repair share the existing runtime workers.

## Quick Start

Python 3.10+ and Git are required for the source checkout check.

```bash
git clone --recurse-submodules https://github.com/Realfit3D/realfit3d.git
cd realfit3d
python scripts/check_checkout.py
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m apps.api.server
```

Open http://127.0.0.1:5000. Without GPU runtime configuration, generation requests return 503.
Use `PORT=5001` if port 5000 is occupied.

Copy `configs/paths.example.json` to `configs/paths.local.json`, configure the
runtime/data paths, and set `REALFIT3D_CONFIG` to that file before starting GPU
work. GG, ContourCraft, and reconstruction need their own environments and assets;
the web `.venv` contains the API and interface dependencies.

**Local, single-user, single-process use only.** There is no authentication.
Do not expose the service publicly or run Web and CLI GPU jobs concurrently.
Do not execute upstream installation scripts without reviewing them.

## Documentation

| Topic | Guide |
|---|---|
| Precomputed videos, playback, and initialization repair | [English](docs/VIDEO_LIBRARY.md) / [Traditional Chinese](docs/VIDEO_LIBRARY.zh-TW.md) |
| GG fork changes, sources, and exclusions | [English](https://github.com/Realfit3D/Gaussian-Garments/blob/main/REALFIT3D_CHANGES.md) / [Traditional Chinese](https://github.com/Realfit3D/Gaussian-Garments/blob/main/REALFIT3D_CHANGES.zh-TW.md) |
| Web configuration, API, and deployment limits | [Web integration](docs/WEB.md) (Traditional Chinese) |
| Opacity training, asset configuration, and rendering | [English](docs/OPACITY.md) / [Traditional Chinese](docs/OPACITY.zh-TW.md) |
| Offline stages, CLI, prerequisites, and acceptance gaps | [Integration status](docs/INTEGRATION_STATUS.md) (Traditional Chinese) |
| Dependency sources and pinned versions | [Upstream dependencies](docs/UPSTREAM.md), [version lock](upstream.lock.json) |
| Fork/submodule development workflow | [Development](docs/DEVELOPMENT.md) (Traditional Chinese) |

For an existing checkout, `git submodule sync --recursive` refreshes dependency URLs.
Then `git submodule update --init --recursive` selects the
versions recorded by this project, not the latest upstream versions.

## Data and Validation

Datasets, body models, checkpoints, generated outputs, and private configuration
are not bundled. Obtain authorized assets separately and follow their terms.
Runtime configuration must point to these assets before generation.

Run CPU tests with `python -m unittest discover -s tests -v` in the configured
development environment. Source checkout checks do not load models or install CUDA.

Release licensing is under review. Dependency sources and terms are tracked in
[upstream notes](docs/UPSTREAM.md).
