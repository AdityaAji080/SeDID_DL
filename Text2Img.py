"""
Usage:
    python Text2Img.py --prompts_from_file prompts.txt
    python Text2Img.py --prompt_list "a photo of a cat" --prompt_list "a photo of a dog"
"""

import argparse
import inspect
import os
from contextlib import nullcontext

import torch
from diffusers import AutoPipelineForText2Image, schedulers
from itertools import batched
from torch import autocast
from tqdm import trange


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID   = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_SIZE = 512


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with SD1.5 for SeDID dataset.")
    parser.add_argument("--prompt_list", default=[], type=str, action="append", help="Prompt to generate. Can be specified multiple times.")
    parser.add_argument("--prompts_from_file", default=None, type=str,help="Path to a text file with one prompt per line.")
    parser.add_argument("--out_dir", default="outputs/gen_images", type=str,help="Directory to save generated images.")
    parser.add_argument("--sampler_steps", default=50, type=int, help="Number of sampling steps.")
    parser.add_argument("--cfg_scale", default=5.0, type=float, help="Classifier-free guidance scale.")
    parser.add_argument("--ddim_eta", default=0.0, type=float, help="DDIM eta (0.0 = deterministic).")
    parser.add_argument("--n_images_per_prompt", default=1, type=int,  help="Number of images to generate per prompt.")
    parser.add_argument("--n_gen_iter", default=1, type=int, help="Number of times to repeat the full prompt list generation.")
    parser.add_argument("--max_prompts", default=4, type=int, help="Number of prompts to process at once.")
    parser.add_argument("--sampler", default=None, choices=["pndm", "ddpm", "ddim", "dpm"],  help="Scheduler to use. None keeps the model's default scheduler.")
    parser.add_argument("--hub_token", default=None, type=str, help="HuggingFace read token (if required).")
    parser.add_argument("--seed", default=None, type=int, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    # Load prompts from file if specified (overrides --prompt_list)
    if args.prompts_from_file is not None:
        print(f"Reading prompts from {args.prompts_from_file}")
        with open(args.prompts_from_file, "r") as f:
            args.prompt_list = f.read().splitlines()

    assert len(args.prompt_list) > 0, "No prompts provided. Use --prompt_list or --prompts_from_file."

    # Chunk prompts into batches
    args.prompt_list = [list(b) for b in batched(args.prompt_list, n=args.max_prompts)]

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

    # ── Set scheduler ────────────────────────────────────────────────────────
    # None keeps the model's default scheduler (PNDM for SD1.5)
    config = pipeline.scheduler.config
    if args.sampler == "ddim":
        pipeline.scheduler = schedulers.DDIMScheduler.from_config(config)
    elif args.sampler == "pndm":
        pipeline.scheduler = schedulers.PNDMScheduler.from_config(config)
    elif args.sampler == "ddpm":
        pipeline.scheduler = schedulers.DDPMScheduler.from_config(config)
    elif args.sampler == "dpm":
        pipeline.scheduler = schedulers.DPMSolverMultistepScheduler.from_config(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline.to(device)

    call_sig = inspect.signature(pipeline.__call__)

    # ── Output directory ─────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    base_count = len([f for f in os.listdir(args.out_dir)
                      if os.path.isfile(os.path.join(args.out_dir, f))])

    # ── Generation loop ──────────────────────────────────────────────────────
    ctx = autocast("cuda") if torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), ctx:
        for _ in trange(args.n_gen_iter, desc="Sampling"):
            for prompt_batch in args.prompt_list:

                kwargs = dict(
                    prompt=prompt_batch,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    num_inference_steps=args.sampler_steps,
                    guidance_scale=args.cfg_scale,
                    num_images_per_prompt=args.n_images_per_prompt,
                )
                if "eta" in call_sig.parameters:
                    kwargs["eta"] = args.ddim_eta

                x_samples = pipeline(**kwargs).images

                # Expand prompts to match n_images_per_prompt
                expanded_prompts = [p for p in prompt_batch for _ in range(args.n_images_per_prompt)]

                for prompt, image in zip(expanded_prompts, x_samples):
                    filename = f"{prompt}_{base_count:05}.png"
                    image.save(os.path.join(args.out_dir, filename))
                    base_count += 1

    print("Generation complete.")


if __name__ == "__main__":
    main()