"""
Step 1 — generate the "fake" image set with a text-to-image diffusion model.

Images are saved as "<index>_<prompt>.png" so invert.py can recover each prompt.

Usage:
    python generate.py --prompts_from_file prompts.txt --out_dir data/fake
    python generate.py --prompt "a photo of a cat" --prompt "a photo of a dog"
    python generate.py --model_id stabilityai/stable-diffusion-2-1 --prompts_from_file prompts.txt
"""

import argparse
import os
from contextlib import nullcontext
from itertools import batched  # Python 3.12+

import torch
from diffusers import AutoPipelineForText2Image, schedulers
from tqdm import tqdm

from common import DEFAULT_MODEL_ID, device_and_dtype, make_name, native_image_size

SAMPLERS = {
    "ddim": schedulers.DDIMScheduler,
    "pndm": schedulers.PNDMScheduler,
    "ddpm": schedulers.DDPMScheduler,
    "dpm": schedulers.DPMSolverMultistepScheduler,
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with a diffusers text-to-image model.")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model id or local path.")
    p.add_argument("--prompt", dest="prompt_list", default=[], action="append",
                   help="Prompt to generate (repeatable).")
    p.add_argument("--prompts_from_file", default=None, help="Text file with one prompt per line.")
    p.add_argument("--out_dir", default="data/fake", help="Where to save generated images.")
    p.add_argument("--image_size", default=None, type=int, help="Output resolution (default: model native).")
    p.add_argument("--sampler_steps", default=50, type=int)
    p.add_argument("--cfg_scale", default=5.0, type=float, help="Classifier-free guidance scale.")
    p.add_argument("--ddim_eta", default=0.0, type=float, help="DDIM eta (only used by DDIM-style samplers).")
    p.add_argument("--n_images_per_prompt", default=1, type=int)
    p.add_argument("--n_gen_iter", default=1, type=int, help="Repeat the whole prompt list this many times.")
    p.add_argument("--batch_size", default=4, type=int, help="Prompts per forward pass.")
    p.add_argument("--sampler", default=None, choices=sorted(SAMPLERS),
                   help="Override the model's default scheduler.")
    p.add_argument("--hub_token", default=None, help="Hugging Face token, if the model is gated.")
    p.add_argument("--seed", default=None, type=int)
    args = p.parse_args()

    if args.prompts_from_file:
        with open(args.prompts_from_file, encoding="utf-8") as f:
            args.prompt_list = [line.strip() for line in f if line.strip()]
    if not args.prompt_list:
        p.error("No prompts given. Use --prompt or --prompts_from_file.")
    return args


def main():
    args = parse_args()
    device, dtype = device_and_dtype()
    generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed is not None else None

    pipeline = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=dtype, token=args.hub_token)
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if args.sampler:
        pipeline.scheduler = SAMPLERS[args.sampler].from_config(pipeline.scheduler.config)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    size = args.image_size or native_image_size(pipeline)
    os.makedirs(args.out_dir, exist_ok=True)
    # Continue numbering after any images already in out_dir.
    index = len([f for f in os.listdir(args.out_dir) if os.path.isfile(os.path.join(args.out_dir, f))])

    batches = [list(b) for b in batched(args.prompt_list, args.batch_size)] * args.n_gen_iter
    autocast = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast:
        for prompts in tqdm(batches, desc="Generating"):
            kwargs = dict(prompt=prompts, height=size, width=size, num_inference_steps=args.sampler_steps,
                          guidance_scale=args.cfg_scale, num_images_per_prompt=args.n_images_per_prompt,
                          generator=generator)
            if isinstance(pipeline.scheduler, schedulers.DDIMScheduler):
                kwargs["eta"] = args.ddim_eta
            images = pipeline(**kwargs).images

            # diffusers returns images grouped per prompt: p0,p0,...,p1,p1,...
            expanded = [p for p in prompts for _ in range(args.n_images_per_prompt)]
            for prompt, image in zip(expanded, images):
                image.save(os.path.join(args.out_dir, make_name(index, prompt) + ".png"))
                index += 1

    print(f"Done. {index} images in {args.out_dir}")


if __name__ == "__main__":
    main()
