#!/usr/bin/env python3
"""Roll out an untangled outfit (multiple garment dicts) under new motion,
producing a trajectory.pkl that Gaussian-Garments' inference.py can render.

Sibling of simulate_recovered_garment.py for the multi-garment case: instead of
one --garment-dict, it takes an outfit produced by combine_garments_untangle.py.
--garment-dicts-dir/--outfit-name locate the untangled
<garment-dicts-dir>/<outfit-name>/<name>.pkl files, and --garment-names (same
innermost-to-outermost order used for untangling) becomes both the
comma-separated name ContourCraft's dataloader merges into one simulation graph,
and the garment_names list stamped onto the output trajectory so inference.py
knows which trained appearance checkpoint
(Gaussian-Garments/output_root/<name>/) to use for each garment.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contourcraft-root", type=Path, required=True)
    parser.add_argument("--garment-dicts-dir", type=Path, required=True,
        help="Same flat directory passed to combine_garments_untangle.py as --garment-dicts-dir.")
    parser.add_argument("--outfit-name", required=True)
    parser.add_argument("--garment-names", nargs="+", required=True,
        help="Same innermost-to-outermost order used for combine_garments_untangle.py.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True,
        help="Motion sequence file, e.g. a CMU/AMASS SMPL(-X) .npz.")
    parser.add_argument("--sequence-loader", default="cmu_npz_smplx",
        choices=["cmu_npz_smpl", "cmu_npz_smplx", "cmu_npz_smpl_zeropos", "cmu_npz_smplx_zeropos"])
    parser.add_argument("--gender", default="female")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--density", type=float, default=0.20022)
    parser.add_argument("--lame-mu", type=float, default=23600.0)
    parser.add_argument("--lame-lambda", type=float, default=44400.0)
    parser.add_argument("--bending-coeff", type=float, default=3.962e-05)
    parser.add_argument("--collision-radius", type=float, default=None,
        help="EXPERIMENTAL, off by default (checkpoint's trained value, 3e-2m, is used "
             "unless set). This is the max-distance threshold for creating a garment-garment "
             "world-edge candidate each step -- the actual driver of why multi-garment "
             "simulate costs far more than proportional to vertex count (measured ~12x "
             "per-step vs single-garment on a 2-garment outfit). Unlike the material params "
             "above, it is NOT domain-randomized during training (see configs/contourcraft."
             "yaml's material._min/_max vs model.ccraft.collision_radius, a bare constant), "
             "so shrinking it to cut cost is mechanically safe (plain float, not part of the "
             "loaded state_dict) but semantically untested -- risks the network missing real "
             "contacts (visible interpenetration) since it's off the trained distribution. "
             "Check the render for artifacts every time you change this.")
    parser.add_argument("--body-collision-radius", type=float, default=None,
        help="EXPERIMENTAL, same caveats as --collision-radius, for garment-body contacts "
             "instead of garment-garment.")
    parser.add_argument("--skip-safecheck-collision-count", action="store_true",
        help="Stub out find_close_faces (BVH tri-tri proximity query, official/"
             "unmodified) so runners/ccraft.py's per-step diagnostic 'ncoll' "
             "collision count always reads 0 instead of actually computing it. "
             "Neither this script nor simulate_recovered_garment.py ever reads "
             "that metric, so it has zero effect on the saved trajectory. "
             "Measured cost for a 2-garment outfit: this call took ~62x longer "
             "per call than for a single garment (775ms vs 12.5ms), ~25-28%% of "
             "total simulate wall time.")
    parser.add_argument('--body-model-root', type=Path)
    parser.add_argument('--contact-config-json', help='Bounded contact settings; two garments in lower,upper order')
    args = parser.parse_args()
    contact = None
    if args.contact_config_json:
        import json
        from realfit3d.contact import contact_config
        contact = contact_config(json.loads(args.contact_config_json))
        if len(args.garment_names) != 2 or args.sequence_loader != 'cmu_npz_smplx':
            parser.error('Contact rollout requires two garments and cmu_npz_smplx')
        import random
        import numpy as np
        import torch
        random.seed(contact['seed']); np.random.seed(contact['seed'])
        torch.manual_seed(contact['seed']); torch.cuda.manual_seed_all(contact['seed'])


    if args.output_trajectory.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_trajectory}")

    outfit_dir = args.garment_dicts_dir / args.outfit_name
    for name in args.garment_names:
        source = outfit_dir / f"{name}.pkl"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing untangled garment dict for '{name}': {source}. "
                f"Run combine_garments_untangle.py first.")

    sys.path.insert(0, str(args.contourcraft_root.resolve()))

    from utils.validation import (apply_material_params, load_runner_from_checkpoint,
                                   create_postcvpr_one_sequence_dataloader)
    from utils.arguments import load_params
    from utils.common import move2device
    from utils.io import pickle_dump

    material_dict = dict(density=args.density, lame_mu=args.lame_mu,
                         lame_lambda=args.lame_lambda, bending_coeff=args.bending_coeff)

    modules, experiment_config = load_params("contourcraft")
    experiment_config = apply_material_params(experiment_config, material_dict)
    if args.collision_radius is not None or args.body_collision_radius is not None:
        # See --collision-radius's help text: not part of the loaded state_dict (safe to
        # mutate here), but off the trained distribution (semantically unverified).
        model_name = list(experiment_config.model.keys())[0]
        if args.collision_radius is not None:
            experiment_config.model[model_name].collision_radius = args.collision_radius
        if args.body_collision_radius is not None:
            experiment_config.model[model_name].body_collision_radius = args.body_collision_radius
    _, runner = load_runner_from_checkpoint(args.checkpoint, modules, experiment_config)

    if args.skip_safecheck_collision_count:
        # Rebinds runners.utils.impulses' own reference to find_close_faces
        # (utils/selfcollisions.py, official/unmodified -- imported there via
        # `from ... import`, so the name lives in impulses' own module
        # namespace and rebinding it here redirects calls made from inside
        # that module, same trick used elsewhere in this project). Doesn't
        # touch any official file.
        import torch
        import runners.utils.impulses as impulses_module

        def _skip_close_faces(vertices, faces, threshold=0):
            return torch.zeros((0, 2), dtype=torch.long, device=vertices.device)

        impulses_module.find_close_faces = _skip_close_faces
        print("--skip-safecheck-collision-count: find_close_faces stubbed out "
              "(diagnostic 'ncoll' metric will read 0; no effect on the simulated trajectory)")

    # Physics/LBS side: each garment's untangled dict lives at <outfit-name>/<name>.pkl,
    # addressed relative to --garment-dicts-dir -- this is the format
    # utils.untanglement.save_new_garment_dicts documents ("garment_name for simulation").
    garment_name = ",".join(f"{args.outfit_name}/{n}" for n in args.garment_names)
    loader_options = {'body_model_root': str(args.body_model_root)} if args.body_model_root else {}
    dataloader = create_postcvpr_one_sequence_dataloader(
        args.sequence, garment_name, sequence_loader=args.sequence_loader,
        obstacle_dict_file=None, gender=args.gender,
        garment_dicts_dir=args.garment_dicts_dir, **loader_options)

    sequence = next(iter(dataloader))
    sequence = move2device(sequence, "cuda:0")

    print(f"Running rollout for {args.n_steps} steps ({', '.join(args.garment_names)})...")
    if contact is None:
        trajectories_dict = runner.valid_rollout(sequence, bare=True, n_steps=args.n_steps)
    else:
        from contact_rollout import rollout
        trajectories_dict = rollout(runner, sequence, args.n_steps, args.garment_names,
                                    args.output_trajectory.with_suffix('.contact'), contact)
    print("Rollout done.")

    # Appearance side: plain names, in the same order the dataloader concatenated
    # garment vertices in -- inference.py looks each one up under
    # Gaussian-Garments/output_root/<name>/ for its trained appearance checkpoint.
    trajectories_dict["garment_names"] = list(args.garment_names)

    pickle_dump(dict(trajectories_dict), args.output_trajectory)
    print(f"Saved trajectory to {args.output_trajectory}")


if __name__ == "__main__":
    main()
