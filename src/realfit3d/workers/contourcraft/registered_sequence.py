"""Register a process-local loader that preserves captured SMPL-X body poses."""

import importlib
import sys
import types


def loader_class():
    import numpy as np

    base = importlib.import_module(
        "datasets.sequence_loaders.cmu_npz_smplx"
    ).SequenceLoader

    class SequenceLoader(base):
        def convert_seq_to_hood_format(self, raw):
            fps = float(np.asarray(raw["mocap_frame_rate"]).item())
            if fps != self.mcfg.fps:
                raise ValueError(
                    "Registered body and garment motion must use the same frame rate"
                )
            result = super().convert_seq_to_hood_format(raw)
            return {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
            }

        def process_sequence(self, sequence):
            for key in ("separate_arms", "random_betas", "zero_betas"):
                if getattr(self.mcfg, key, False):
                    raise ValueError(f"Registered motion cannot use {key}")
            count = len(sequence["body_pose"])
            for key in ("betas", "expression"):
                if sequence[key].ndim == 1:
                    sequence[key] = np.repeat(sequence[key][None], count, axis=0)
                if len(sequence[key]) != count:
                    raise ValueError(f"Registered {key} and pose counts differ")
            return sequence

    return SequenceLoader


def install():
    name = "datasets.sequence_loaders.cmu_npz_smplx_preserve"
    module = types.ModuleType(name)

    def get_attribute(key):
        if key != "SequenceLoader":
            raise AttributeError(key)
        cls = loader_class()
        module.SequenceLoader = cls
        return cls

    module.__getattr__ = get_attribute
    sys.modules[name] = module
