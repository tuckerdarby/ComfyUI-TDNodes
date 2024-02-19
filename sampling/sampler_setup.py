import torch

import comfy.utils
import comfy.model_management
import comfy.conds
import latent_preview

from . import sampling as k_diffusion_sampling


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
             config,
             callback):

    sigmas = self.sigmas

    if config['last_step'] is not None and config['last_step'] < (len(sigmas) - 1):
      sigmas = sigmas[:config['last_step'] + 1]
      if True:
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
      latent_image += noise

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
    comfy.samplers.apply_empty_x_to_equal_area(
        positive, negative, 'gligen_video', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond": positive,
                  "uncond": negative,
                  "cond_scale": cfg,
                  "model_options": self.model_options,
                  "seed": config['seed']}

    config['steps'] = self.steps
    config['device'] = self.device

    def k_callback(x): return callback(
        x["i"], x["denoised"], x["x"], len(sigmas) - 1)
    ksampler_batched = config['executor_class'](self.model, config, sampler_function,
                                                sigmas, extra_args=extra_args, callback=k_callback)
    samples = ksampler_batched.process(latent_image)
    # return samples.to(torch.float32)
    return self.model.process_latent_out(samples.to(torch.float32))


def common_ksampler(model, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, config):
  if config['add_noise']:
    noise = comfy.sample.prepare_noise(latent_image, config['seed'])
  else:
    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype,
                        layout=latent_image.layout, device="cpu")
  callback = latent_preview.prepare_callback(
    model, len(latent_image) // config['batch_size'] + 1)
  # comfy.sample.sample
  real_model, positive_copy, negative_copy, noise_mask, models = comfy.sample.prepare_sampling(
      model, noise.shape, positive, negative, None)

  # noise = noise.to(model.load_device)
  # latent_image = latent_image.to(model.load_device)
  config['device'] = model.load_device

  # comfy.samplers.KSampler
  sampler = KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name,
                     scheduler=scheduler, denoise=1.0, model_options=model.model_options)

  samples = sampler.sample(noise,
                           positive_copy,
                           negative_copy,
                           cfg=cfg,
                           latent_image=latent_image,
                           config=config,
                           callback=callback)
  samples = samples.to(comfy.model_management.intermediate_device())

  comfy.sample.cleanup_additional_models(models)
  comfy.sample.cleanup_additional_models(set(comfy.sample.get_models_from_cond(
      positive_copy, "control") + comfy.sample.get_models_from_cond(negative_copy, "control")))
  return samples
