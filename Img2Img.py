"""
Usage: python Img2Img.py "outputs\gen_images" --save_all_latents true --save_specific_latent_steps 5
"""

import argparse
import inspect
import os
import types
from contextlib import nullcontext

import torch
from diffusers import AutoPipelineForText2Image, schedulers
from diffusers.pipelines.auto_pipeline import _get_task_class, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING
from itertools import batched
from PIL import Image
from torch import autocast
from tqdm import trange

from custom_img2img_utils import inert_prepare_latents, collect_intermediate_latents


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID   = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_SIZE = 512  # SD1.5 native resolution
VAE_SCALE  = 8    # SD VAE downscale factor, for documentation only


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_scheduler(pipeline, orig_sch_config, direction: str):
    """Swap the pipeline scheduler for DDIMInverse (fwd) or DDIM (rev)."""
    if direction == "fwd":
        pipeline.scheduler = schedulers.DDIMInverseScheduler.from_config(orig_sch_config)
    else:
        pipeline.scheduler = schedulers.DDIMScheduler.from_config(orig_sch_config)
    return pipeline


def encode_images(pipeline, imgs, torch_dtype, vae_sampling="argmax"):
    """Encode a list of PIL images to scaled latents using the VAE."""
    proc = pipeline.image_processor.preprocess(imgs).to(device=pipeline.device, dtype=torch_dtype)
    dist = pipeline.vae.encode(proc)
    latents = retrieve_latents(dist, generator=None, sample_mode=vae_sampling)
    latents = latents.to(dtype=torch_dtype)
    latents = pipeline.vae.config.scaling_factor * latents
    return latents


def decode_latents(pipeline, latents, torch_dtype):
    """Decode latents back to PIL images."""
    out = latents / pipeline.vae.config.scaling_factor
    images = pipeline.vae.decode(out, return_dict=False, generator=None)[0]
    return pipeline.image_processor.postprocess(images, output_type="pil")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate latents for SeDID using SD1.5 DDIM inversion.")
    parser.add_argument("src_img_dir", type=str, help="Directory of source images.")
    parser.add_argument("--out_dir", default="outputs/latents", type=str, help="Root output directory.")
    parser.add_argument("--sampler_steps", default=6, type=int, help="Number of DDIM steps.")
    parser.add_argument("--cfg_scale", default=1.0, type=float, help="CFG guidance scale (1.0 = no guidance).")
    parser.add_argument("--ddim_eta", default=0.0, type=float, help="DDIM eta (0 = deterministic).")
    parser.add_argument("--max_images", default=1, type=int, help="Images processed per batch.")
    parser.add_argument("--save_images", default=True, type=lambda x: x.lower() in ("1", "true", "yes"), help="Save VAE-decoded output images.")
    parser.add_argument("--save_endpt_latents", default=True, type=lambda x: x.lower() in ("1", "true", "yes"), help="Save init, noise, and final latents.")
    parser.add_argument("--save_all_latents", default=False, type=lambda x: x.lower() in ("1", "true", "yes"), help="Save all per-step latents for fwd and rev passes.")
    parser.add_argument("--save_specific_latent_steps", default=None, type=lambda x: [int(s) for s in x.split(",")], help="Comma-separated step indices to save (requires --save_all_latents).")
    parser.add_argument("--hub_token", default=None, type=str, help="HuggingFace read token (if required).")
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    args.src_img_dir = r"CIFAR-10_real_flat"

    if args.save_specific_latent_steps is not None:
        if not args.save_all_latents:
            print("Warning: --save_specific_latent_steps ignored because --save_all_latents is False.")
        else:
            invalid = [i for i in args.save_specific_latent_steps if i < 0 or i >= args.sampler_steps]
            if invalid:
                raise ValueError(f"save_specific_latent_steps {invalid} out of range for sampler_steps={args.sampler_steps}. Valid: [0, {args.sampler_steps-1}]")

    args.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ── Load pipeline ────────────────────────────────────────────────────────
    pipeline = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=args.torch_dtype,
        token=args.hub_token,
        safety_checker=None,
        requires_safety_checker=False,
    )
    orig_sch_config = pipeline.scheduler.config

    # Obtain retrieve_latents from the SD img2img pipeline module
    img2img_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, "StableDiffusionPipeline")
    global retrieve_latents
    retrieve_latents = getattr(inspect.getmodule(img2img_cls), "retrieve_latents")

    # Replace txt2img prepare_latents with a pass-through so our latents aren't overwritten
    pipeline.prepare_latents = types.MethodType(inert_prepare_latents, pipeline)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline.to(device)

    call_sig = inspect.signature(pipeline.__call__)

    # ── Output directories ───────────────────────────────────────────────────
    if args.save_images:
        os.makedirs(os.path.join(args.out_dir, "final_images"), exist_ok=True)
    if args.save_endpt_latents:
        for sub in ("init_latents", "noise_latents", "final_latents"):
            os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    if args.save_all_latents:
        for sub in ("fwd_all_latents", "rev_all_latents"):
            os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    # ── Image batching ───────────────────────────────────────────────────────
    img_files = sorted([f for f in os.listdir(args.src_img_dir) if os.path.isfile(os.path.join(args.src_img_dir, f))])
    img_file_batches = [list(b) for b in batched(img_files, n=args.max_images)]
    imgs_count = lats_count = 0

    # ── Main loop ────────────────────────────────────────────────────────────
    ctx = autocast("cuda") if torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), ctx:
        for img_file_batch in img_file_batches:

            init_imgs = [Image.open(os.path.join(args.src_img_dir, f)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)) for f in img_file_batch]

            # Derive prompts from filenames: "00001_a photo of a airplane.png" → "a photo of a airplane"
            prompts = [os.path.splitext(f)[0].split("_", 1)[1] for f in img_file_batch]

            for _ in trange(1, desc="Sampling"):

                # ── Encode images to latents ──────────────────────────────
                init_latents = encode_images(pipeline, init_imgs, args.torch_dtype)

                # ── Forward pass (DDIM inversion: image → noise) ──────────
                if args.save_all_latents:
                    fwd_modern_cb, fwd_legacy_cb, fwd_collected = collect_intermediate_latents()

                fwd_kwargs = dict(prompt=prompts, latents=init_latents, num_inference_steps=args.sampler_steps,
                                  guidance_scale=args.cfg_scale, num_images_per_prompt=1, output_type="latent")
                fwd_kwargs = {k: v for k, v in fwd_kwargs.items() if k in call_sig.parameters}
                fwd_kwargs.update(prompt=prompts, latents=init_latents, num_inference_steps=args.sampler_steps,
                                  guidance_scale=args.cfg_scale, num_images_per_prompt=1, output_type="latent")
                if "eta" in call_sig.parameters:
                    fwd_kwargs["eta"] = args.ddim_eta

                if args.save_all_latents:
                    if "callback_on_step_end" in call_sig.parameters:
                        fwd_kwargs["callback_on_step_end"] = fwd_modern_cb
                        fwd_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                    elif "callback" in call_sig.parameters:
                        fwd_kwargs["callback"] = fwd_legacy_cb
                        fwd_kwargs["callback_steps"] = 1

                pipeline = get_scheduler(pipeline, orig_sch_config, "fwd")
                noise_latents = pipeline(**fwd_kwargs).images  # shape: (B, 4, H/8, W/8)

                # ── Reverse pass (DDIM denoising: noise → image latent) ───
                if args.save_all_latents:
                    rev_modern_cb, rev_legacy_cb, rev_collected = collect_intermediate_latents()

                rev_kwargs = dict(prompt=prompts, latents=noise_latents, num_inference_steps=args.sampler_steps,
                                  guidance_scale=args.cfg_scale, num_images_per_prompt=1, output_type="latent")
                if "eta" in call_sig.parameters:
                    rev_kwargs["eta"] = args.ddim_eta

                if args.save_all_latents:
                    if "callback_on_step_end" in call_sig.parameters:
                        rev_kwargs["callback_on_step_end"] = rev_modern_cb
                        rev_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                    elif "callback" in call_sig.parameters:
                        rev_kwargs["callback"] = rev_legacy_cb
                        rev_kwargs["callback_steps"] = 1

                pipeline = get_scheduler(pipeline, orig_sch_config, "rev")
                final_latents = pipeline(**rev_kwargs).images

                # ── Decode final latents to images ────────────────────────
                x_samples = decode_latents(pipeline, final_latents, args.torch_dtype)

                # ── Save outputs ──────────────────────────────────────────
                if args.save_images:
                    for prompt, img in zip(prompts, x_samples):
                        img.save(os.path.join(args.out_dir, "final_images", f"{imgs_count:05}_{prompt}.png"))
                        imgs_count += 1

                if args.save_endpt_latents:
                    for prompt, il, nl, fl in zip(prompts, init_latents, noise_latents, final_latents):
                        tag = f"{lats_count:05}_{prompt}"
                        torch.save(il, os.path.join(args.out_dir, "init_latents",  tag + ".pth"))
                        torch.save(nl, os.path.join(args.out_dir, "noise_latents", tag + ".pth"))
                        torch.save(fl, os.path.join(args.out_dir, "final_latents", tag + ".pth"))

                if args.save_all_latents:
                    fwd_stacked = torch.stack(fwd_collected)
                    rev_stacked = torch.stack(rev_collected)
                    if args.save_specific_latent_steps is not None:
                        fwd_stacked = fwd_stacked[args.save_specific_latent_steps]
                        rev_indices = [len(rev_collected) - 1 - i for i in args.save_specific_latent_steps]
                        rev_stacked = rev_stacked[rev_indices]
                    for prompt, fwd_steps, rev_steps in zip(prompts, fwd_stacked.unbind(dim=1), rev_stacked.unbind(dim=1)):
                        tag = f"{lats_count:05}_{prompt}"
                        torch.save(fwd_steps, os.path.join(args.out_dir, "fwd_all_latents", tag + ".pth"))
                        torch.save(rev_steps, os.path.join(args.out_dir, "rev_all_latents", tag + ".pth"))

                if args.save_endpt_latents or args.save_all_latents:
                    lats_count += len(prompts)


if __name__ == "__main__":
    main()