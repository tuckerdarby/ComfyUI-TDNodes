import torch
from tqdm.auto import trange, tqdm

import comfy.utils
import comfy.model_management
import comfy.conds
import latent_preview

from .tokenflow_utils import *
from .tokenflow_apply import wrap_model
from comfy.k_diffusion import sampling as k_diffusion_sampling

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def get_alpha_cumprod(sigma):
    return 1 / ((sigma * sigma) + 1)


class TokenFlow:
    def __init__(self, model, config, step_function, sigmas, extra_args=None, callback=None, noise_sampler=None):
        self.model = model
        self.config = config
        self.step_function = step_function
        self.extra_args = extra_args
        self.callback = callback
        self.noise_sampler = noise_sampler
        self.sigmas = sigmas
        self.timesteps = [self.model.model_sampling.timestep(
            sigma) for sigma in sigmas[:-1]]
        self.wrapped_model = wrap_model(self.model)

    @torch.no_grad()
    def denoise_step(self, x, sigma_index):
        sigma = self.sigmas[sigma_index]
        sigma_prev = self.sigmas[sigma_index+1]
        register_time(self.model.diffusion_model, self.timesteps[sigma_index])
        return self.step_function(
            self.wrapped_model,
            x,
            [sigma, sigma_prev],
            extra_args=self.extra_args,
            disable=True)

    def batched_denoise_step(self, x, sigma_index, indices):
        n_frames = len(x)
        batch_size = min(
            n_frames, self.config['batch_size'] if 'batch_size' in self.config else 8)
        denoised_latents = []
        pivotal_idx = torch.randint(
            batch_size, (n_frames//batch_size,)) + torch.arange(0, n_frames, batch_size)

        register_pivotal(self.model.diffusion_model, True)
        self.set_condhints_to_indices(self.extra_args['cond'], pivotal_idx)
        self.set_condhints_to_indices(self.extra_args['uncond'], pivotal_idx)
        self.denoise_step(x[pivotal_idx], sigma_index)
        register_pivotal(self.model.diffusion_model, False)
        for i, b in enumerate(range(0, len(x), batch_size)):
            self.set_condhints_to_indices(
                self.extra_args['cond'], torch.arange(b, b + batch_size))
            self.set_condhints_to_indices(
                self.extra_args['uncond'], torch.arange(b, b+batch_size))
            register_batch_idx(self.model.diffusion_model, i)
            denoised_latents.append(self.denoise_step(
                x[b:b + batch_size], sigma_index))
        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def sample_loop(self, x, indices):
        for i in tqdm(range(len(self.sigmas)-1), desc="Sampling"):
            x = self.batched_denoise_step(x, i, indices)
            self.callback(
                {'x': x, 'i': i, 'sigma': self.sigmas[i], 'denoised': x})
        return x

    def get_ddim_eps(self, latent, inversion_latents, inversion_sigma):
        alpha_prod_T = get_alpha_cumprod(inversion_sigma)
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (inversion_latents - mu_T * latent) / sigma_T
        return eps

    def set_condhints_to_indices(self, conds, indices):
        for cond in conds:
            if 'control' in cond:
                controlnet = cond['control']
                while controlnet is not None:
                    controlnet.cond_hint_original = controlnet.backup_hints[indices]
                    controlnet.cond_hint = None
                    controlnet = controlnet.previous_controlnet

    def set_backup_controlnet_hints(self, conds):
        for cond in conds:
            if 'control' in cond:
                controlnet = cond['control']
                while controlnet is not None:
                    setattr(controlnet, 'backup_hints',
                            controlnet.cond_hint_original)
                    controlnet = controlnet.previous_controlnet

    def reset_backup_controlnet_hints(self, conds):
        for cond in conds:
            if 'control' in cond:
                controlnet = cond['control']
                while controlnet is not None:
                    setattr(controlnet, 'backup_hints',
                            controlnet.cond_hint_original)
                    controlnet.cond_hint_original = controlnet.backup_hints
                    controlnet.backup_hints = None
                    controlnet.cond_hint = None
                    controlnet = controlnet.previous_controlnet

    def register_tokenflow(self, conv_injection_t, qk_injection_t):
        n_cond = len(self.extra_args['cond']) - 1
        n_uncond = len(self.extra_args['uncond'])
        qk_injection_timesteps = self.timesteps[:qk_injection_t] if qk_injection_t >= 0 else [
        ]
        conv_injection_timesteps = self.timesteps[:conv_injection_t] if conv_injection_t >= 0 else [
        ]
        register_extended_attention_pnp(
            self.model.diffusion_model, qk_injection_timesteps, n_cond, n_uncond)
        register_conv_injection(self.model.diffusion_model,
                                conv_injection_timesteps, n_cond, n_uncond)
        register_tokenflow_blocks(self.model.diffusion_model, n_cond, n_uncond)
        self.set_backup_controlnet_hints(self.extra_args['cond'])
        self.set_backup_controlnet_hints(self.extra_args['uncond'])

    def deregister_tokenflow(self):
        deregister_extended_attention_pnp(self.model.diffusion_model)
        deregister_conv_injection(self.model.diffusion_model)
        deregister_tokenflow_blocks(self.model.diffusion_model)
        self.reset_backup_controlnet_hints(self.extra_args['cond'])
        self.reset_backup_controlnet_hints(self.extra_args['uncond'])

    def process(self, latents, inversion_latents):
        pnp_f_t = int(self.config['steps'] * self.config['pnp_f_t'])
        pnp_attn_t = int(self.config['steps'] * self.config['pnp_attn_t'])
        self.register_tokenflow(conv_injection_t=pnp_f_t,
                                qk_injection_t=pnp_attn_t)
        try:
            inversion_sigma = self.model.model_sampling.sigmas[self.config['inversion_timestep']]
            noise = self.get_ddim_eps(
                latents, inversion_latents, inversion_sigma)
            # ddpm_add_noise(latents, self.sigmas[0], noise)
            noisy_latents = inversion_latents
            sampled_latents = self.sample_loop(
                noisy_latents, torch.arange(len(latents)))
        except Exception as e:
            self.deregister_tokenflow()
            raise e
        self.deregister_tokenflow()
        return sampled_latents


@torch.no_grad()
def ddpm_add_noise(original_samples, sigma, noise):
    # diffusers.add_noise ddpm
    alpha_cumprod = get_alpha_cumprod(sigma)

    sqrt_alpha_prod = alpha_cumprod ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = (sqrt_alpha_prod * original_samples +
                     sqrt_one_minus_alpha_prod * noise)
    return noisy_samples


def get_sampler_function(sampler_name):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(
            k_diffusion_sampling, "sample_{}".format(sampler_name))
    return sampler_function


class KSampler(comfy.samplers.KSampler):
    def sample(self,
               noise,
               positive,
               negative,
               cfg,
               latent_image,
               latent_inversion,
               config,
               callback):

        sigmas = self.sigmas

        if config['last_step'] is not None and config['last_step'] < (len(sigmas) - 1):
            sigmas = sigmas[:config['last_step'] + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if config['start_step'] is not None:
            if config['start_step'] < (len(sigmas) - 1):
                sigmas = sigmas[config['start_step']:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler_function = get_sampler_function(self.sampler)

        comfy.samplers.resolve_areas_and_cond_masks(
            positive, noise.shape[2], noise.shape[3], self.device)
        comfy.samplers.resolve_areas_and_cond_masks(
            negative, noise.shape[2], noise.shape[3], self.device)

        comfy.samplers.calculate_start_end_timesteps(self.model, positive)
        comfy.samplers.calculate_start_end_timesteps(self.model, negative)

        if latent_image is not None:
            latent_image = self.model.process_latent_in(latent_image)
            latent_inversion = self.model.process_latent_in(latent_inversion)

        if hasattr(self.model, 'extra_conds'):
            positive = comfy.samplers.encode_model_conds(
                self.model.extra_conds, positive, noise, self.device, "positive", latent_image=latent_image, denoise_mask=None, seed=config['seed'])
            negative = comfy.samplers.encode_model_conds(
                self.model.extra_conds, negative, noise, self.device, "negative", latent_image=latent_image, denoise_mask=None, seed=config['seed'])

        # make sure each cond area has an opposite one with the same area
        for c in positive:
            comfy.samplers.create_cond_with_same_area_if_none(positive, c)

        for c in negative:
            comfy.samplers.create_cond_with_same_area_if_none(negative, c)

        comfy.samplers.pre_run_control(self.model, negative + positive)

        comfy.samplers.apply_empty_x_to_equal_area(list(filter(lambda c: c.get(
            'control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
        comfy.samplers.apply_empty_x_to_equal_area(
            positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

        extra_args = {"cond": positive,
                      "uncond": negative,
                      "cond_scale": cfg,
                      "model_options": self.model_options,
                      "seed": config['seed']}

        config['steps'] = self.steps
        config['device'] = self.device

        def k_callback(x): return callback(
            x["i"], x["denoised"], x["x"], len(sigmas) - 1)
        tokenflow = TokenFlow(self.model, config, sampler_function,
                              sigmas, extra_args=extra_args, callback=k_callback)
        samples = tokenflow.process(latent_image, latent_inversion)

        return self.model.process_latent_out(samples.to(torch.float32))


def common_ksampler(model, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, latent_inversion, config):
    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype,
                        layout=latent_image.layout, device="cpu")
    callback = latent_preview.prepare_callback(model, steps)
    # comfy.sample.sample
    real_model, positive_copy, negative_copy, noise_mask, models = comfy.sample.prepare_sampling(
        model, noise.shape, positive, negative, None)

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)
    latent_inversion = latent_inversion.to(model.load_device)

    # comfy.samplers.KSampler
    sampler = KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name,
                       scheduler=scheduler, denoise=1.0, model_options=model.model_options)

    samples = sampler.sample(noise,
                             positive_copy,
                             negative_copy,
                             cfg=cfg,
                             latent_image=latent_image,
                             latent_inversion=latent_inversion,
                             config=config,
                             callback=callback)
    samples = samples.to(comfy.model_management.intermediate_device())
    del noise
    del latent_inversion
    del latent_image

    comfy.sample.cleanup_additional_models(models)
    comfy.sample.cleanup_additional_models(set(comfy.sample.get_models_from_cond(
        positive_copy, "control") + comfy.sample.get_models_from_cond(negative_copy, "control")))
    return samples


class KSamplerTF():
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                  "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "src_positive": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "latent_inversion": ("LATENT", ),
                 "inversion_timestep": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                 "batch_size": ("INT", {"default": 8, "min": 1, "max": 10000}),
                 "pnp_f_t": ("FLOAT", {"default": 0.80, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                 "pnp_attn_t": ("FLOAT", {"default": 0.50, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                 }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "samplers"

    @torch.no_grad()
    def sample(self,
               model,
               noise_seed,
               steps,
               cfg,
               sampler_name,
               scheduler,
               positive,
               negative,
               src_positive,
               latent_image,
               latent_inversion,
               inversion_timestep,
               start_at_step,
               end_at_step,
               batch_size,
               pnp_f_t,
               pnp_attn_t):

        config = {
            'batch_size': batch_size,
            'pnp_f_t': pnp_f_t,
            'pnp_attn_t': pnp_attn_t,
            'inversion_timestep': inversion_timestep,
            'start_step': start_at_step,
            'last_step': end_at_step,
            'seed': noise_seed,
            'n_cond': len(positive),
            'n_uncond': len(negative),
            'n_src_cond': len(src_positive)
        }

        for cond in src_positive:
            cond[1]['src_latent'] = True
        positive = positive[:] + src_positive[:]
        negative = negative[:]
        latent_image = latent_image['samples']
        latent_inversion = latent_inversion['samples']

        samples = common_ksampler(model, steps, cfg, sampler_name, scheduler,
                                  positive, negative, latent_image, latent_inversion, config)

        return ({'samples': samples}, )
