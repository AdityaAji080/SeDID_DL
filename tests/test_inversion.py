"""
Sanity checks for invert.py, runnable on CPU with no model downloads.

A tiny randomly initialised Stable Diffusion pipeline is built on the fly. Its UNet is then
replaced by an "oracle" that predicts zero noise, which makes DDIM inversion exactly
invertible. So every (z_t, z'_t) pair that invert.py saves must match to float precision.
A pair taken from neighbouring noise levels would differ by O(1).

    pytest -q tests/
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def tiny_pipeline_dir(tmp_path_factory):
    from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    root = tmp_path_factory.mktemp("tiny_sd")
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=4)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(32, 64), latent_channels=4)
    text_encoder = CLIPTextModel(CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, pad_token_id=1, hidden_size=32, intermediate_size=37,
        num_attention_heads=4, num_hidden_layers=2, vocab_size=1000))
    vocab = {"<|startoftext|>": 0, "!": 1, "<|endoftext|>": 2}
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789"):
        vocab[c + "</w>"], vocab[c] = 3 + i, 40 + i
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"), model_max_length=16)
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              steps_offset=1, skip_prk_steps=True, set_alpha_to_one=False)
    StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                            scheduler=scheduler, safety_checker=None, feature_extractor=None,
                            requires_safety_checker=False).save_pretrained(root / "pipe")
    return str(root / "pipe")


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(d / f"{i:05}_a photo of a cat.png")
    return d


def _run_invert(monkeypatch, pipe_dir, image_dir, out_dir, *extra, oracle=True):
    import invert
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

    original = invert.AutoPipelineForText2Image.from_pretrained

    def load(*a, **k):
        pipe = original(*a, **k)
        if oracle:
            pipe.unet.forward = lambda x, *_, **__: UNet2DConditionOutput(sample=torch.zeros_like(x))
        return pipe

    monkeypatch.setattr(invert.AutoPipelineForText2Image, "from_pretrained", load)
    monkeypatch.setattr(sys, "argv", ["invert.py", str(image_dir), "--model_id", pipe_dir,
                                      "--out_dir", str(out_dir), "--image_size", "64", *extra])
    invert.main()


def test_pairs_are_at_the_same_noise_level(monkeypatch, tiny_pipeline_dir, image_dir, tmp_path):
    N = 6
    out = tmp_path / "lat"
    _run_invert(monkeypatch, tiny_pipeline_dir, image_dir, out,
                "--sampler_steps", str(N), "--sedid_steps", ",".join(map(str, range(N - 1))),
                "--batch_size", "2")
    for k in range(N - 1):
        files = sorted((out / "zt" / f"step{k}").glob("*.pth"))
        assert len(files) == 4
        for f in files:
            zt, zt_prime = torch.load(f), torch.load(out / "zt_prime" / f"step{k}" / f.name)
            assert zt.shape == zt_prime.shape and zt.shape[0] == 4  # (C, h, w), no batch dim
            assert (zt - zt_prime).norm() < 1e-4, f"step {k}: z_t and z'_t are not at the same noise level"


def test_trivial_final_step_is_rejected(monkeypatch, tiny_pipeline_dir, image_dir, tmp_path):
    with pytest.raises(SystemExit):
        _run_invert(monkeypatch, tiny_pipeline_dir, image_dir, tmp_path / "lat",
                    "--sampler_steps", "6", "--sedid_steps", "5")


def test_prompt_round_trip():
    from common import make_name, prompt_from_name
    assert prompt_from_name(make_name(12, "a photo of a cat") + ".png") == "a photo of a cat"
    assert prompt_from_name(make_name(3, "a/b: c?") + ".png") == "a b c"
    with pytest.raises(ValueError):
        prompt_from_name("a photo of a cat_00012.png")
