import argparse
import os

import torch

from model import GDNLM, ModelArgs, model_export


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise ValueError(f"{path} does not contain model_args")
    model = GDNLM(ModelArgs(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained GDN checkpoint to run.c .bin format")
    parser.add_argument("--checkpoint", default=os.path.join("out", "ckpt.pt"))
    parser.add_argument("--output", default=os.path.join("out", "model.bin"))
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    model_export(model, args.output)
