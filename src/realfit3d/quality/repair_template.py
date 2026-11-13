"""Inspect or build a separate Stage 1.1 mesh and UV candidate."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--body", type=Path)
    p.add_argument(
        "--body-model",
        type=Path,
        help="SMPL-X NPZ skinning weights for body-supported prior",
    )
    p.add_argument(
        "--reference", type=Path, help="Original mesh used for hem dimensions"
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--inspect", action="store_true", help="Read-only geometry audit")
    args = p.parse_args()
    if not args.inspect and args.output is None:
        p.error("--output is required unless --inspect is selected")
    from realfit3d.repair.pipeline import run

    print(
        json.dumps(
            run(
                args.source,
                args.recipe,
                args.output,
                args.body,
                args.reference,
                args.inspect,
                args.body_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
