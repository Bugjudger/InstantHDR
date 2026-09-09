"""Shared model loading for the InstantHDR release."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "InstantHDR.ckpt"


def checkpoint_path(path=None):
    path = path or os.environ.get("INSTANTHDR_CHECKPOINT") or ROOT / "checkpoints" / CHECKPOINT_NAME
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. See README.md for setup.")
    return path


def model_config():
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_config, ModelCfg

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=hdr", "model.encoder.pretrained_weights="])
    return load_typed_config(cfg.model, ModelCfg)


def load_model(checkpoint=None, device="cuda"):
    import torch
    from src.model.model.anysplat import AnySplat

    path = checkpoint_path(checkpoint)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("InstantHDR rendering requires an NVIDIA GPU and a working CUDA driver.")
    cfg = model_config()
    model = AnySplat(cfg.encoder, cfg.decoder)
    model.load_pretrained(path)
    return model.to(device)
