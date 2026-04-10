import torch
import numpy as np
from typing import List, Optional, Tuple, Union
import random

def collect_intermediate_latents():
    collected = []

    def modern_callback(pipe, step, timestep, callback_kwargs): # Function signature must match pipeline for compatibility.
        collected.append(callback_kwargs["latents"].detach().cpu()) # Detach and move to CPU to avoid GPU Vram from filling up during extended runs.
        return {}

    def legacy_callback(step, timestep, latents):
        collected.append(latents.detach().cpu())

    return modern_callback, legacy_callback, collected

def inert_prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        raise ValueError("Latent must be passed in when using deterministic inversion, but got None.")

    if hasattr(self, "_prepare_latent_image_ids") and hasattr(self, "_pack_latents"):

        # ASindhu Edit: FLUX reshapes latents into a 3D tensor and additionally expects latent_image_ids as an output
        if latents.ndim == 4:  # Input latents have 4 dimensions, packed latents have 3
            # Process latents normally as done by FLUX
            height = latents.shape[-2]
            width = latents.shape[-1]
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
            self._latent_image_ids_cache = latent_image_ids  # cache for rev pass
        elif latents.ndim == 3:
            # Latents already in correct form - maintain shape and reuse cached latent_ids from initial run
            latent_image_ids = self._latent_image_ids_cache
        else:
            raise ValueError(f"Unexpected number of dimensions for input latents - expected 3 or 4, got {latents.ndim}")

        return latents.to(device=device, dtype=dtype), latent_image_ids

    return latents.to(device=device, dtype=dtype)

def euler_discrete_inverted_set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary timesteps schedule. If `None`, timesteps will be generated
                based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps` and `sigmas`
                must be `None`, and `timestep_spacing` attribute will be ignored.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to support arbitrary timesteps schedule schedule. If `None`, timesteps and sigmas
                will be generated based on the relevant scheduler attributes. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`, and the timesteps will be generated based on the
                custom sigmas schedule.
        """

        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` should be set.")
        if num_inference_steps is None and timesteps is None and sigmas is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `timesteps` or `sigmas.")
        if num_inference_steps is not None and (timesteps is not None or sigmas is not None):
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps` or `sigmas`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_karras_sigmas = True`.")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_beta_sigmas = True`.")
        if (
            timesteps is not None
            and self.config.timestep_type == "continuous"
            and self.config.prediction_type == "v_prediction"
        ):
            raise ValueError(
                "Cannot set `timesteps` with `config.timestep_type = 'continuous'` and `config.prediction_type = 'v_prediction'`."
            )

        if num_inference_steps is None:
            num_inference_steps = len(timesteps) if timesteps is not None else len(sigmas) - 1
        self.num_inference_steps = num_inference_steps

        if sigmas is not None:
            log_sigmas = np.log(np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5))
            sigmas = np.array(sigmas).astype(np.float32)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas[:-1]])

        else:
            if timesteps is not None:
                timesteps = np.array(timesteps).astype(np.float32)
            else:
                # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://huggingface.co/papers/2305.08891
                if self.config.timestep_spacing == "linspace":
                    timesteps = np.linspace(
                        0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32
                    )[::-1].copy()
                elif self.config.timestep_spacing == "leading":
                    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (
                        (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                    )
                    timesteps += self.config.steps_offset
                elif self.config.timestep_spacing == "trailing":
                    step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (
                        (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
                    )
                    timesteps -= 1
                else:
                    raise ValueError(
                        f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                    )

            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            if self.config.interpolation_type == "linear":
                sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            elif self.config.interpolation_type == "log_linear":
                sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp().numpy()
            else:
                raise ValueError(
                    f"{self.config.interpolation_type} is not implemented. Please specify interpolation_type to either"
                    " 'linear' or 'log_linear'"
                )

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            if self.config.final_sigmas_type == "sigma_min":
                sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            elif self.config.final_sigmas_type == "zero":
                sigma_last = 0
            else:
                raise ValueError(
                    f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
                )

            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
            self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas[:-1]]).to(device=device)
        else:
            self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

        # ASindhu Edit: Flip timesteps and sigmas to force inverted schedule
        self.timesteps = torch.flip(self.timesteps, dims=(0,))
        self.sigmas = torch.flip(self.sigmas, dims=(0,))

def heun_discrete_inverted_set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        num_train_timesteps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            num_train_timesteps (`int`, *optional*):
                The number of diffusion steps used when training the model. If `None`, the default
                `num_train_timesteps` attribute is used.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, timesteps will be
                generated based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps`
                must be `None`, and `timestep_spacing` attribute will be ignored.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot use `timesteps` with `config.use_karras_sigmas = True`")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_beta_sigmas = True`.")

        num_inference_steps = num_inference_steps or len(timesteps)
        self.num_inference_steps = num_inference_steps
        num_train_timesteps = num_train_timesteps or self.config.num_train_timesteps

        if timesteps is not None:
            timesteps = np.array(timesteps, dtype=np.float32)
        else:
            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://huggingface.co/papers/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
            elif self.config.timestep_spacing == "leading":
                step_ratio = num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

        # JStyborski Edit: Add in reference to final_sigmas_type to prevent 0 as start
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        #sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        
        sigmas = torch.from_numpy(sigmas).to(device=device)
        # JStyborski Edit: Flip before repeat_interleave
        sigmas = torch.flip(sigmas, dims=(0,))
        self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

        timesteps = torch.from_numpy(timesteps)
        # JStyborski Edit: Flip before repeat_interleave
        timesteps = torch.flip(timesteps, dims=(0,))
        timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])

        self.timesteps = timesteps.to(device=device, dtype=torch.float32)

        # empty dt and derivative
        self.prev_derivative = None
        self.dt = None

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication


def flow_match_euler_discrete_inverted_set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`list[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`list[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps,
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting of
        #    "exponential" or "linear" type is applied
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)]).flip(0)
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

        # ASindhu Edit: Flip timesteps and sigmas to force inverted schedule
        self.timesteps = torch.flip(self.timesteps, dims=(0,))
        self.sigmas = torch.flip(self.sigmas, dims=(0,))

def flow_match_heun_discrete_inverted_set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
    """
    self.num_inference_steps = num_inference_steps

    timesteps = np.linspace(
        self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
    )

    sigmas = timesteps / self.config.num_train_timesteps
    sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

    timesteps = sigmas * self.config.num_train_timesteps
    # JStyborski Edit: Flip before repeat_interleave
    timesteps = torch.flip(timesteps, dims=(0,))
    timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
    self.timesteps = timesteps.to(device=device)

    # JStyborski Edit: Add in reference to final_sigmas_type to prevent 0 as start
    if self.config.final_sigmas_type == "sigma_min":
        sigma_last = self.sigma_min
    elif self.config.final_sigmas_type == "zero":
        sigma_last = 0
    sigmas = torch.cat([sigmas, sigma_last * torch.ones(1, device=sigmas.device)])
    #sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

    # JStyborski Edit: Flip before repeat_interleave
    sigmas = torch.flip(sigmas, dims=(0,))
    self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

    # empty dt and derivative
    self.prev_derivative = None
    self.dt = None

    self._step_index = None
    self._begin_index = None
