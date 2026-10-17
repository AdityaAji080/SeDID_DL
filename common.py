"""Small helpers shared by generate.py, invert.py and detect.py."""

import os
import re

import torch

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y")


def device_and_dtype():
    """CUDA + fp16 when available, otherwise CPU + fp32."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def native_image_size(pipeline) -> int:
    """Native resolution of a diffusers pipeline (e.g. 512 for SD1.5, 768 for SD2.1, 1024 for SDXL)."""
    denoiser = getattr(pipeline, "unet", None) or getattr(pipeline, "transformer", None)
    sample_size = getattr(denoiser.config, "sample_size", None)
    vae_scale = getattr(pipeline, "vae_scale_factor", 8)
    if isinstance(sample_size, (list, tuple)):
        sample_size = sample_size[0]
    return int(sample_size * vae_scale) if sample_size else 512


# ── File naming ───────────────────────────────────────────────────────────────
# Every image and latent in the pipeline is named "<5-digit index>_<prompt>.<ext>",
# e.g. "00012_a photo of a cat.png". invert.py recovers the prompt from this name,
# and detect.py pairs z_t / z_t' files by identical name.

def make_name(index: int, prompt: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|\s]+', " ", prompt).strip()[:180]
    return f"{index:05}_{safe}"


def prompt_from_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "_" not in stem or not stem.split("_", 1)[0].isdigit():
        raise ValueError(
            f"Cannot parse prompt from '{filename}'. Expected '<index>_<prompt>.<ext>', "
            f"e.g. '00012_a photo of a cat.png'."
        )
    return stem.split("_", 1)[1]
