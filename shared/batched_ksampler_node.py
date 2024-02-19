import torch

import comfy.utils
import comfy.model_management
import comfy.conds

from .custom_apply import wrap_model
from .controlnet_utils import prepare_controlnet, reset_controlnet, set_controlnet_indexes
from ..sampling.sampler_setup import common_ksampler
from ..sampling.sampling_config import SamplingConfig
from ..sampling.plms_sampling import PLMSSampler, apply_mode_plms


class KSamplerBatched:
  def __init__(self, model, config, step_function, sigmas, extra_args=None, callback=None, noise_sampler=None):
    self.model =  model
    self.config = config
    self.step_function = step_function
    self.extra_args = extra_args
    self.callback = callback
    self.noise_sampler = noise_sampler
    self.sigmas = sigmas
    self.wrapped_model = wrap_model(self.model)
    self.sampling_config = SamplingConfig(
      self.config['latent_shape'],
      self.sigmas,
      self.extra_args,
    )
    if self.config['instance']:
      plms = PLMSSampler(model, self.config['device'])
      plms.make_schedule(self.sigmas)
      self.step_function = plms.plms_sampling
      self.extra_args['apply_options'] = {
        'model_function_wrapper': apply_mode_plms
      }


  @torch.no_grad()
  def denoise_step(self, x):
    return self.step_function(
        self.wrapped_model,
        x,
        self.sigmas,
        self.sampling_config)

  def _set_controlnet_indexes(self, indexes):
    for cond in self.extra_args['cond']:
      set_controlnet_indexes(cond.get('control', None), indexes)
    for cond in self.extra_args['uncond']:
      set_controlnet_indexes(cond.get('control', None), indexes)

  def sample_loop(self, x):
    latents_out = []
    batch_size = self.config['batch_size']
    for i in range(len(x) // batch_size + 1):
      start, end = (i * batch_size), min((i + 1) * batch_size, len(x))
      indexes = torch.tensor(range(start, end))
      self._set_controlnet_indexes(indexes)
      batch = self.denoise_step(
        x[start:end].to(self.config['device']))
      latents_out.append(
        batch.to(comfy.model_management.intermediate_device()))
      self.callback({'x': x, 'i': i, 'sigma': self.sigmas[i], 'denoised': x})
    return torch.cat(latents_out)

  def _before_sampling(self):
    for cond in self.extra_args['cond']:
      prepare_controlnet(cond.get('control', None))
    for cond in self.extra_args['uncond']:
      prepare_controlnet(cond.get('control', None))

  def _after_sampling(self):
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


class KSamplerBatchedNode():
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
             "latent_image": ("LATENT", ),
             "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
             "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
             "batch_size": ("INT", {"default": 8, "min": 1, "max": 10000}),
             "instance": (["enable", "disable"], ),
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
             instance):

    config = {
        'add_noise': add_noise == 'enable',
        'batch_size': batch_size,
        'start_step': start_at_step,
        'last_step': end_at_step,
        'seed': noise_seed,
        'n_cond': len(positive),
        'n_uncond': len(negative),
        'device': model.load_device,
        'executor_class': KSamplerBatched,
        'latent_shape': latent_image['samples'].shape,
        'cfg': cfg,
        'instance': instance == 'enable'
    }

    positive = positive[:]
    negative = negative[:]
    latent_image = latent_image['samples']

    samples = common_ksampler(model, steps, cfg, sampler_name, scheduler,
                              positive, negative, latent_image, config)

    return ({'samples': samples}, )
