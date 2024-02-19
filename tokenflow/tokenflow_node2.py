import torch
from tqdm.auto import tqdm

import comfy.utils
import comfy.model_management
import comfy.conds

from .tokenflow_utils import *
from .tf_controlnet_utils import prepare_controlnet, reset_controlnet, set_controlnet_indexes, load_controlnet
from ..shared.custom_apply import wrap_model
from ..sampling.sampler_setup import common_ksampler
from ..sampling.sampling_config import SamplingConfig


class KSamplerTF:
  def __init__(self, model, config, step_function, sigmas, extra_args=None, callback=None, noise_sampler=None):
    self.model = model
    self.config = config
    self.step_function = step_function
    self.extra_args = extra_args
    self.callback = callback
    self.noise_sampler = noise_sampler
    self.sigmas = sigmas
    self.wrapped_model = wrap_model(self.model)
    self.extra_args['apply_options'] = {}
    self.timesteps = [self.model.model_sampling.timestep(
            sigma) for sigma in sigmas[:-1]]
    self.sampling_config = SamplingConfig(
      self.config['latent_shape'],
      self.sigmas,
      self.extra_args,
    )
    self.pivot_sampling_config = SamplingConfig(
      self.config['latent_shape'],
      self.sigmas,
      self.extra_args,
    )
    self.extra_args['apply_options']['instancediffusion'] = True

  @torch.no_grad()
  def denoise_step(self, x, step, is_pivotal):
    register_time(self.model.diffusion_model, self.timesteps[step])
    return self.step_function(
        self.wrapped_model,
        x.to(self.config['device']),
        [self.sigmas[step], self.sigmas[step + 1]],
        self.pivot_sampling_config if is_pivotal else self.sampling_config)

  def _set_controlnet_indexes(self, indexes):
    for cond in self.extra_args['cond']:
      set_controlnet_indexes(cond.get('control', None),
                             indexes)
    for cond in self.extra_args['uncond']:
      set_controlnet_indexes(cond.get('control', None),
                             indexes)
      
  def batched_denoise_step(self, x, sigma_index):
    n_frames = len(x)
    batch_size = min(
        n_frames, self.config['batch_size'] if 'batch_size' in self.config else 8)
    denoised_latents = []
    pivotal_idx = torch.randint(
        batch_size, (n_frames//batch_size,)) + torch.arange(0, n_frames, batch_size)

    register_pivotal(self.model.diffusion_model, True)
    self.pivot_sampling_config.prepare_run(sigma_index, pivotal_idx)
    self.extra_args['apply_options']['indexes'] = pivotal_idx
    self._set_controlnet_indexes(pivotal_idx)
    self.denoise_step(x[pivotal_idx], sigma_index, True)
    register_pivotal(self.model.diffusion_model, False)
    for i, b in enumerate(range(0, len(x), batch_size)):
        batch_indexes = torch.arange(b, b+batch_size)
        self.extra_args['apply_options']['indexes'] = batch_indexes
        self._set_controlnet_indexes(batch_indexes)
        register_batch_idx(self.model.diffusion_model, i)
        self.sampling_config.prepare_run(i, batch_indexes)
        denoised_latents.append(self.denoise_step(
            x[b:b + batch_size], sigma_index, False))
    denoised_latents = torch.cat(denoised_latents)
    return denoised_latents

  def sample_loop(self, x):
    for i in tqdm(range(len(self.sigmas)-1), desc="Sampling"):
      x = self.batched_denoise_step(x, i)
      self.callback(
          {'x': x, 'i': i, 'sigma': self.sigmas[i], 'denoised': x})
    return x
  
  def register_tokenflow(self):
    conv_injection_t = int(self.config['steps'] * self.config['pnp_f_t'])
    qk_injection_t = int(self.config['steps'] * self.config['pnp_attn_t'])
    n_cond = len(self.extra_args['cond']) - 1
    n_uncond = len(self.extra_args['uncond']) if self.config['cfg'] > 1.0 else 0
    qk_injection_timesteps = self.timesteps[:qk_injection_t] if qk_injection_t >= 0 else [
    ]
    conv_injection_timesteps = self.timesteps[:conv_injection_t] if conv_injection_t >= 0 else [
    ]
    register_extended_attention_pnp(
        self.model.diffusion_model, qk_injection_timesteps, n_cond, n_uncond)
    register_conv_injection(self.model.diffusion_model,
                            conv_injection_timesteps, n_cond, n_uncond)
    register_tokenflow_blocks(self.model.diffusion_model, n_cond, n_uncond)

  def deregister_tokenflow(self):
    deregister_extended_attention_pnp(self.model.diffusion_model)
    deregister_conv_injection(self.model.diffusion_model)
    deregister_tokenflow_blocks(self.model.diffusion_model)

  def _before_sampling(self):
    self.register_tokenflow()
    for cond in self.extra_args['cond']:
      prepare_controlnet(cond.get('control', None))
    for cond in self.extra_args['uncond']:
      prepare_controlnet(cond.get('control', None))

  def _after_sampling(self):
    self.deregister_tokenflow()
    for cond in self.extra_args['cond']:
      reset_controlnet(cond.get('control', None))
    for cond in self.extra_args['uncond']:
      reset_controlnet(cond.get('control', None))

  def process(self, latents):
    self._before_sampling()
    error = None
    try:
      samples = self.sample_loop(latents)
    except Exception as e:
      error = e
    self._after_sampling()
    if error:
      raise error
    return samples


class KSamplerTFNode():
  @classmethod
  def INPUT_TYPES(s):
    return {"required":
            {"model": ("MODEL",),
             "add_noise": (["enable", "disable"], ),
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
             add_noise,
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
        'add_noise': add_noise == 'enable',
        'batch_size': batch_size,
        'steps': steps,
        'start_step': start_at_step,
        'last_step': end_at_step,
        'seed': noise_seed,
        'n_cond': len(positive),
        'n_uncond': len(negative),
        'n_src_cond': len(src_positive),
        'device': model.load_device,
        'executor_class': KSamplerTF,
        'pnp_f_t': pnp_f_t,
        'pnp_attn_t': pnp_attn_t,
        'inversion_timestep': inversion_timestep,
        'latent_shape': latent_image['samples'].shape,
        'cfg': cfg,
    }
    for cond in src_positive:
      cond[1]['src_latent'] = True
    positive = positive[:] + src_positive[:]
    negative = negative[:]

    for cond in positive:
      load_controlnet(cond[1].get('control', None))
    for cond in negative:
      load_controlnet(cond[1].get('control', None))

    # latent_image = latent_image['samples']
    latent_image = latent_inversion['samples']  # switched for now

    samples = common_ksampler(model, steps, cfg, sampler_name, scheduler,
                              positive, negative, latent_image, config)

    return ({'samples': samples}, )
