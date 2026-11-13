"""Install a reviewed repair candidate into a new garment asset directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def install(candidate, destination, cameras=None, point_cloud=None):
    candidate, destination = Path(candidate), Path(destination)
    report = json.loads((candidate / "quality.json").read_text())
    if report.get("status") != "geometry_passed_pending_visual_review":
        raise ValueError("Candidate has not passed geometry and UV checks")
    names = ("template.obj", "template_uv.obj")
    for name in names:
        actual = hashlib.sha256((candidate / name).read_bytes()).hexdigest()
        if report.get("outputs_sha256", {}).get(name) != actual:
            raise ValueError(f"Candidate hash mismatch: {name}")
    extra = {
        name: Path(path)
        for name, path in [("cameras.json", cameras), ("point_cloud.ply", point_cloud)]
        if path is not None
    }
    for name, path in extra.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir(parents=True, exist_ok=False)
    stage1 = destination / "stage1"
    stage1.mkdir()
    for name in names:
        shutil.copy2(candidate / name, stage1 / name)
    for name, path in extra.items():
        shutil.copy2(path, stage1 / name)
    shutil.copy2(candidate / "quality.json", stage1 / "repair_quality.json")
    (stage1 / "asset_version.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                source_sha256=report["source_sha256"],
                recipe_sha256=report["recipe_sha256"],
                outputs_sha256=report["outputs_sha256"],
                initialization_sha256={
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in extra.items()
                },
                visual_review_accepted=True,
            ),
            indent=2,
        )
    )
    return stage1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="New GG output subject, not stage1 itself",
    )
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--point-cloud", type=Path, required=True)
    p.add_argument("--accept-visual-review", action="store_true")
    args = p.parse_args()
    if not args.accept_visual_review:
        p.error(
            "Review dimensions and opening semantics, then pass --accept-visual-review"
        )
    print(install(args.candidate, args.destination, args.cameras, args.point_cloud))


if __name__ == "__main__":
    main()
