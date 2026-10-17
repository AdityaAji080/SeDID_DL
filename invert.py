"""
Step 2 — compute SeDID stepwise latents by DDIM inversion + reconstruction.

For each image x0:
    z0              = VAE.encode(x0)                         (latent space, 4×H/8×W/8)
    z_1 … z_N       = DDIM inversion   z0  → noise           (forward, deterministic)
    z'_{N-1} … z'_0 = DDIM denoising   z_N → z0'             (reverse, deterministic)

For each requested step k the script saves the pair (z_{t_k}, z'_{t_k}), both at the
same noise level t_k. detect.py then scores ‖z_{t_k} − z'_{t_k}‖.

Output layout (ready for detect.py):
    <out_dir>/zt/step<k>/<index>_<prompt>.pth
    <out_dir>/zt_prime/step<k>/<index>_<prompt>.pth

Source images must be named "<index>_<prompt>.<ext>"; the prompt conditions the model
during inversion (generate.py already writes this format).

Usage:
    python invert.py data/real --out_dir latents/real --sampler_steps 6 --sedid_steps 2
    python invert.py data/fake --out_dir latents/fake --sampler_steps 6 --sedid_steps 2
    python invert.py data/real --model_id stabilityai/stable-diffusion-2-1 --sedid_steps 1,2,3
"""

import argparse
import inspect
import os
import types
from contextlib import nullcontext
from itertools import batched  # Python 3.12+

import torch
from diffusers import AutoPipelineForText2Image, schedulers
from PIL import Image
from tqdm import tqdm

from common import (DEFAULT_MODEL_ID, IMAGE_EXTS, device_and_dtype, make_name, native_image_size,
                    prompt_from_name, str2bool)
from inversion_utils import collect_intermediate_latents, inert_prepare_latents


# ── VAE helpers ───────────────────────────────────────────────────────────────

def _vae_dtype(pipeline, dtype):
    # SDXL's VAE overflows in fp16; diffusers marks such VAEs with force_upcast.
    return torch.float32 if getattr(pipeline.vae.config, "force_upcast", False) else dtype


def encode_images(pipeline, imgs, dtype):
    """PIL images → scaled latents, using the VAE posterior mode (deterministic)."""
    vae_dtype = _vae_dtype(pipeline, dtype)
    if pipeline.vae.dtype != vae_dtype:
        pipeline.vae.to(vae_dtype)
    x =pipeline.image_processor.preprocess(imgs).to(device=pipeline.device, dtype=vae_dtype)
    latents = pipeline.vae.encode(x).latent_dist.mode()
    shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
    return ((latents - shift) * pipeline.vae.config.scaling_factor).to(dtype)


def decode_latents(pipeline, latents, dtype):
    vae_dtype = _vae_dtype(pipeline, dtype)
    shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
    z = latents.to(vae_dtype) / pipeline.vae.config.scaling_factor + shift
    images = pipeline.vae.decode(z, return_dict=False)[0]
    return pipeline.image_processor.postprocess(images, output_type="pil")


# ── Scheduler helpers ─────────────────────────────────────────────────────────

def make_schedulers(orig_config):
    """DDIM inverse (image → noise) and DDIM (noise → image) built from the model's own config,
    so beta schedule, timestep spacing and prediction type (epsilon / v) are inherited."""
    if "FlowMatch" in orig_config.get("_class_name", ""):
        raise NotImplementedError(
            "Flow-matching models (SD3, FLUX, ...) are not supported by this script yet. "
            "inversion_utils.py contains experimental inverted flow-match schedules for them."
        )
    cfg = dict(orig_config, clip_sample=False)  # clipping x0 breaks exact invertibility
    return (schedulers.DDIMInverseScheduler.from_config(cfg),
            schedulers.DDIMScheduler.from_config(cfg))


def run(pipeline, scheduler, latents, prompts, args, collect):
    pipeline.scheduler = scheduler
    kwargs = dict(prompt=prompts, latents=latents, num_inference_steps=args.sampler_steps,
                  guidance_scale=args.cfg_scale, num_images_per_prompt=1, output_type="latent",
                  height=args.image_size, width=args.image_size)
    params = inspect.signature(pipeline.__call__).parameters
    if "eta" in params:
        kwargs["eta"] = 0.0  # deterministic DDIM — required for SeDID
    if "use_resolution_binning" in params:  # PixArt
        kwargs["use_resolution_binning"] = False
    if collect is not None:
        kwargs["callback_on_step_end"] = collect
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipeline(**kwargs).images


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDIM inversion/reconstruction latents for latent-space SeDID.")
    p.add_argument("src_img_dir", help="Directory of images named '<index>_<prompt>.<ext>'.")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model id or local path.")
    p.add_argument("--out_dir", default="latents", help="Root output directory.")
    p.add_argument("--image_size", default=None, type=int,
                   help="Resize inputs to this square size (default: model native resolution).")
    p.add_argument("--sampler_steps", default=6, type=int, help="Number of DDIM steps N.")
    p.add_argument("--sedid_steps", default=None, type=lambda s: [int(i) for i in s.split(",")],
                   help="Comma-separated step indices k in [0, N-2] to save as (z_t, z'_t) pairs. "
                        "Default: the middle step.")
    p.add_argument("--cfg_scale", default=1.0, type=float, help="Guidance scale (1.0 = none; keeps DDIM exact).")
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--save_images", default=False, type=str2bool, help="Also save reconstructed images.")
    p.add_argument("--save_endpt_latents", default=False, type=str2bool,
                   help="Also save z0, the inverted noise z_N and the reconstruction z0'.")
    p.add_argument("--save_all_latents", default=False, type=str2bool,
                   help="Also save full forward and reverse trajectories, one stacked tensor per image.")
    p.add_argument("--hub_token", default=None)
    args = p.parse_args()

    N = args.sampler_steps
    if N < 2:
        p.error("--sampler_steps must be >= 2.")
    if args.sedid_steps is None:
        args.sedid_steps = [(N - 2) // 2]
    bad = [k for k in args.sedid_steps if not 0 <= k <= N - 2]
    if bad:
        p.error(f"--sedid_steps {bad} out of range. Valid: 0..{N - 2}. "
                f"(Step {N - 1} is the fully inverted noise, which is the reverse pass's input, "
                f"so its reconstruction error is always 0.)")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device, dtype = device_and_dtype()

    pipeline = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=dtype, token=args.hub_token)
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if "callback_on_step_end" not in inspect.signature(pipeline.__call__).parameters:
        raise RuntimeError("This pipeline does not support callback_on_step_end; upgrade diffusers.")
    # Stop the txt2img pipeline from replacing our latents with fresh noise.
    pipeline.prepare_latents = types.MethodType(inert_prepare_latents, pipeline)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    inv_scheduler, ddim_scheduler = make_schedulers(pipeline.scheduler.config)
    args.image_size = args.image_size or native_image_size(pipeline)

    N = args.sampler_steps
    out = lambda *parts: os.path.join(args.out_dir, *parts)
    dirs = [out(kind, f"step{k}") for kind in ("zt", "zt_prime") for k in args.sedid_steps]
    if args.save_images:
        dirs.append(out("recon_images"))
    if args.save_endpt_latents:
        dirs += [out("z0"), out("zN"), out("z0_recon")]
    if args.save_all_latents:
        dirs += [out("fwd_all"), out("rev_all")]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    files = sorted(f for f in os.listdir(args.src_img_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"{len(files)} images | model={args.model_id} | size={args.image_size} | N={N} | "
          f"sedid_steps={args.sedid_steps}")

    autocast = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast:
        for batch in tqdm(list(batched(files, args.batch_size)), desc="Inverting"):
            prompts = [prompt_from_name(f) for f in batch]
            names = [os.path.splitext(f)[0] for f in batch]
            imgs = [Image.open(os.path.join(args.src_img_dir, f)).convert("RGB")
                    .resize((args.image_size, args.image_size), Image.BICUBIC) for f in batch]

            z0 = encode_images(pipeline, imgs, dtype)

            fwd_cb, _, fwd = collect_intermediate_latents()
            rev_cb, _, rev = collect_intermediate_latents()
            zN = run(pipeline, inv_scheduler, z0, prompts, args, fwd_cb)       # fwd[k] is at noise level t_k
            z0_recon = run(pipeline, ddim_scheduler, zN, prompts, args, rev_cb)  # rev[j] is at t_{N-2-j}

            for k in args.sedid_steps:
                zt, zt_prime = fwd[k], rev[N - 2 - k]  # same noise level t_k
                for i, name in enumerate(names):
                    torch.save(zt[i].clone(), out("zt", f"step{k}", name + ".pth"))
                    torch.save(zt_prime[i].clone(), out("zt_prime", f"step{k}", name + ".pth"))

            if args.save_images:
                for name, img in zip(names, decode_latents(pipeline, z0_recon, dtype)):
                    img.save(out("recon_images", name + ".png"))
            if args.save_endpt_latents:
                for i, name in enumerate(names):
                    torch.save(z0[i].cpu(), out("z0", name + ".pth"))
                    torch.save(zN[i].cpu(), out("zN", name + ".pth"))
                    torch.save(z0_recon[i].cpu(), out("z0_recon", name + ".pth"))
            if args.save_all_latents:
                fwd_all, rev_all = torch.stack(fwd, 1), torch.stack(rev, 1)  # (B, N, C, h, w)
                for i, name in enumerate(names):
                    torch.save(fwd_all[i].clone(), out("fwd_all", name + ".pth"))
                    torch.save(rev_all[i].clone(), out("rev_all", name + ".pth"))


if __name__ == "__main__":
    main()
