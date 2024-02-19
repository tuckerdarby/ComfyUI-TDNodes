import torch

import comfy.utils
import comfy.model_management
import comfy.conds

from . import rave_utils as ru
from .rave_controlnet_utils import prepare_controlnet, reset_controlnet, set_controlnet_indexes, load_controlnet
from ..shared.custom_apply import wrap_model
from ..sampling.sampler_setup import common_ksampler
from .rave_sampling_config import RaveSamplingConfig


class KSamplerRAVE:
  def __init__(self, model, config, step_function, sigmas, extra_args=None, callback=None, noise_sampler=None):
    self.model = model
    self.config = config
    self.step_function = step_function
    self.extra_args = extra_args
    self.callback = callback
    self.noise_sampler = noise_sampler
    self.sigmas = sigmas
    self.wrapped_model = wrap_model(self.model)
    self.extra_args['apply_options'] = {
      'grid_size': self.config['grid_size']
    }
    self.sampling_config = RaveSamplingConfig(
      self.config['latent_shape'],
      self.sigmas,
      self.extra_args,
      self.config['grid_size']
    )
    

  @torch.no_grad()
  def denoise_step(self, x, step):
    return self.step_function(
        self.wrapped_model,
        x,
        [self.sigmas[step], self.sigmas[step + 1]],
        self.sampling_config)

  def _set_controlnet_indexes(self, indexes):
    for cond in self.extra_args['cond']:
      set_controlnet_indexes(cond.get('control', None),
                             indexes, self.config['grid_size'])
    for cond in self.extra_args['uncond']:
      set_controlnet_indexes(cond.get('control', None),
                             indexes, self.config['grid_size'])

  def sample_loop(self, x):
    length = len(x)
    grid_size = self.config['grid_size']
    batch_size = self.config['batch_size'] * grid_size * grid_size

    for step in range(self.config['steps'] - 1):
      latents_out = []
      indexes = ru.shuffle_indices(length)

      for i in range(length // batch_size + 1):
        start, end = (i * batch_size), min((i + 1) * batch_size, length)
        if start >= end:
          continue
        batch_indexes = torch.tensor(indexes[start:end])
        self.sampling_config.prepare_run(step, batch_indexes)
        self.extra_args['apply_options']['indexes'] = batch_indexes
        self._set_controlnet_indexes(batch_indexes)

        x_batch = ru.list_to_grid(
          x[batch_indexes], grid_size).to(self.config['device'])

        batch = self.denoise_step(x_batch, step)
        batch = ru.grid_to_list(batch, grid_size)

        latents_out.append(
          batch.to(comfy.model_management.intermediate_device()))

        self.callback(
          {'x': x_batch, 'i': step, 'sigma': self.sigmas[step], 'denoised': x})
      x = ru.shuffle_tensor_batch_other(torch.cat(latents_out),
                                        indexes, list(range(length)))

    return x

  def _before_sampling(self):
    for cond in self.extra_args['cond']:
      prepare_controlnet(cond.get('control', None), self.config['grid_size'])
    for cond in self.extra_args['uncond']:
      prepare_controlnet(cond.get('control', None), self.config['grid_size'])

  def _after_sampling(self):
    for cond in self.extra_args['cond']:
      reset_controlnet(cond.get('control', None))
    for cond in self.extra_args['uncond']:
      reset_controlnet(cond.get('control', None))

  def process(self, latents):
    self._before_sampling()
    error = None
    latents = ru.grid_to_list(latents, self.config['grid_size'])
    self.sampling_config.set_inversion_latents(latents)
    if self.config['noise_sampler_type'] == 'inversion':
      self.sampling_config.set_default_noise_sampler(self.sampling_config.inversion_noise_sampler, override=True)
    
    try:
      samples = self.sample_loop(latents)
    except Exception as e:
      error = e
    self._after_sampling()
    if error:
      raise error
    return samples


class KSamplerRaveNode():
  @classmethod
  def INPUT_TYPES(s):
    return {"required":
            {"model": ("MODEL",),
             "add_noise": (["disable", "enable"], ),
             "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
             "positive": ("CONDITIONING", ),
             "negative": ("CONDITIONING", ),
             "latent_image": ("LATENT", ),
             "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
             "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
             "batch_size": ("INT", {"default": 3, "min": 1, "max": 10000}),
             "grid_size": ("INT", {"default": 9, "min": 1, "max": 16}),
             "noise_sampler_type": (["default", "inversed", "none"],)
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
             latent_image,
             start_at_step,
             end_at_step,
             batch_size,
             grid_size,
             noise_sampler_type):

    config = {
        'add_noise': add_noise == 'enable',
        'batch_size': batch_size,
        'grid_size': grid_size,
        'steps': steps,
        'start_step': start_at_step,
        'last_step': end_at_step,
        'seed': noise_seed,
        'n_cond': len(positive),
        'n_uncond': len(negative),
        'device': model.load_device,
        'executor_class': KSamplerRAVE,
        'latent_shape': latent_image['samples'].shape,
        'noise_sampler_type': noise_sampler_type
    }

    positive = positive[:]
    negative = negative[:]

    for cond in positive:
      load_controlnet(cond[1].get('control', None), grid_size)
    for cond in negative:
      load_controlnet(cond[1].get('control', None), grid_size)

    latent_image = ru.list_to_grid(latent_image['samples'], grid_size)

    samples = common_ksampler(model, steps, cfg, sampler_name, scheduler,
                              positive, negative, latent_image, config)

    return ({'samples': samples}, )
