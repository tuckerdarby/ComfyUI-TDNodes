import torch
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import trange, tqdm
import random

import comfy.utils
import comfy.model_management
import comfy.conds
import latent_preview

import .rave_utils as ru


def apply_model_csd(model, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
      sigma = t
      xc = x  # self.model_sampling.calculate_input(sigma, x)
      if c_concat is not None:
          xc = torch.cat([xc] + [c_concat], dim=1)

      context = c_crossattn
      dtype = model.get_dtype()

      if model.manual_cast_dtype is not None:
          dtype = self.manual_cast_dtype

      xc = xc.to(dtype)
      t = model.model_sampling.timestep(t).float()
      context = context.to(dtype)
      extra_conds = {}
      for o in kwargs:
          extra = kwargs[o]
          if hasattr(extra, "dtype"):
              if extra.dtype != torch.int and extra.dtype != torch.long:
                  extra = extra.to(dtype)
          extra_conds[o] = extra

      model_output = model.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
      return model_output


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options, next_indices):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
            
        if p is None:
            continue
        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if comfy.samplers.can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = comfy.model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = comfy.samplers.cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            current_control = control
            while current_control is not None:
                orig_hint = current_control.cond_hint_original
                if len(orig_hint) < len(next_indices):
                    orig_hint = ru.grid_to_list(orig_hint, gs)
                orig_hint = ru.shuffle_tensor_batch(orig_hint, range(len(next_indices)), next_indices)
                orig_hint = ru.list_to_grid(orig_hint, gs)
                current_control.cond_hint_original = orig_hint
                current_control.cond_hint = None

                current_control = current_control.previous_controlnet

            c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

            current_control = control
            while current_control is not None:
                orig_hint = current_control.cond_hint_original
                orig_hint = ru.grid_to_list(orig_hint, gs)
                orig_hint = ru.shuffle_tensor_batch(orig_hint, next_indices, range(len(next_indices)))

                current_control.cond_hint_original = orig_hint
                current_control.cond_hint = None

                current_control = current_control.previous_controlnet
        

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            # output = model.apply_model_csd(input_x, timestep_, **c).chunk(batch_chunks)
            output = apply_model_csd(model, input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond

    # output = list(output)
    # output.reverse()
    # return output


def get_brownian_noise_sampler(x, sigmas, seed=None):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = comfy.k_diffusion.sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
    return noise_sampler


def get_alpha_cumprod(sigma):
    return 1 / ((sigma * sigma) + 1)


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

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


lr = 0.5
wd = 0.0
decay_iter = 20
decay_rate = 0.9
image_guidance_scale = 0.01
gs = 3

class CSDPredictor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sampler_euler_function(self, model, latents, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float('inf'), s_noise=1.):
        print('sampling with', lr, wd, decay_iter, decay_rate, image_guidance_scale)
        latents = latents.to('cpu')
        latents = ru.grid_to_list(latents, gs)

        src_latents = latents.clone().half().to('cuda')
        src_latents.requires_grad = False
        tgt_latents = src_latents.clone().detach().to('cuda')
        tgt_latents.requires_grad = True

        seed = extra_args['seed']
        cond = extra_args['cond']
        uncond = extra_args['uncond']
        src_cond = extra_args['src_cond']
        src_uncond = extra_args['src_uncond']
        model_options = extra_args['model_options']
        cond_scale = extra_args['cond_scale']

        optimizer = torch.optim.SGD([tgt_latents], lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_iter, gamma=decay_rate)
        noise_sampler = get_brownian_noise_sampler(src_latents, sigmas, seed=seed)


        batch_size = len(latents)
        indices = list(range(batch_size))
        print('cond_scale', cond_scale)

        for i in trange(len(sigmas) - 1):
            optimizer.zero_grad()
            # sigma = sigmas[random.randint(0, len(sigmas) - 1)]
            sigma = sigmas[i]

            with torch.no_grad():
                next_indices = ru.shuffle_indices(batch_size)
                s_in = src_latents.new_ones([src_latents.shape[0]])
                timestep = s_in * sigma
                
                # add noise
                # noise = torch.randn_like(src_latents)
                eta = 1.
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()

                # SRC
                src_latents_noisy = ddpm_add_noise(src_latents, sigma, noise)
                src_latents_noisy = ru.shuffle_tensor_batch(src_latents_noisy, indices, next_indices)
                src_latents_noisy = ru.list_to_grid(src_latents_noisy, gs)
                src_noise_pred = calc_cond_uncond_batch(model, src_cond, src_uncond, src_latents_noisy, timestep, model_options, next_indices)

                # TGT
                tgt_latents_noisy = ddpm_add_noise(tgt_latents, sigma, noise)
                tgt_latents_noisy = ru.shuffle_tensor_batch(tgt_latents_noisy, indices, next_indices)
                tgt_latents_noisy = ru.list_to_grid(tgt_latents_noisy, gs)
                tgt_noise_pred = calc_cond_uncond_batch(model, cond, uncond, tgt_latents_noisy, timestep, model_options, next_indices)

            # guidance
            src_cond_pred, src_uncond_pred = src_noise_pred
            src_noise_pred  = src_uncond_pred + (src_cond_pred - src_uncond_pred) * cond_scale
            tgt_cond_pred, tgt_uncond_pred = tgt_noise_pred
            tgt_noise_pred  = tgt_uncond_pred + (tgt_cond_pred - tgt_uncond_pred) * cond_scale

            noise = tgt_noise_pred - src_noise_pred

            w_t = (1-get_alpha_cumprod(sigma))
            with torch.cuda.amp.autocast():
                K, dK_dX = self.rbf_kernel(tgt_latents_noisy, tgt_latents_noisy, gamma=-1, ad=1)
                scores = torch.matmul(noise.transpose(0,3), K).transpose(0,3) + dK_dX
                grad = w_t * scores / K.size(0)

            tgt_latents.backward(gradient=grad, retain_graph=True)

            optimizer.step()
            scheduler.step()

            callback({ 'i': i, 'denoised': tgt_latents, 'sigma': sigma, 'sigma_hat': sigma, 'x': tgt_latents })

        return tgt_latents
            
    def rbf_kernel(self, X, Y, gamma=-1, ad=1):
        # X and Y should be tensors with shape (batch_size, num_channels, height, width)
        # gamma is a hyperparameter controlling the width of the RBF kernel

        # Reshape X and Y to have shape (batch_size, num_channels*height*width)
        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # Compute the pairwise squared Euclidean distances between the samples
        with torch.cuda.amp.autocast():
            dists = torch.cdist(X_flat, Y_flat, p=2)**2

        if gamma <0: # use median trick
            gamma = torch.median(dists)
            gamma = torch.sqrt(0.5 * gamma / np.log(dists.size(0) + 1))
            gamma = 1 / (2 * gamma**2)

        gamma = gamma * ad 
        # gamma = torch.max(gamma, torch.tensor(1e-3))
        # Compute the RBF kernel using the squared distances and gamma
        K = torch.exp(-gamma * dists)
        dK = -2 * gamma * K.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0))
        dK_dX = torch.sum(dK, dim=1)

        return K, dK_dX


class KSAMPLER(comfy.samplers.Sampler):
    def __init__(self, extra_options={}, inpaint_options={}):
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        csd = CSDPredictor()

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        if latent_image is not None:
            noise += latent_image

        samples = csd.sampler_euler_function(model, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=None)
        return samples


def from_KSampler_sample(model, noise, positive, negative, src_positive, src_negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    comfy.samplers.resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    comfy.samplers.resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    comfy.samplers.resolve_areas_and_cond_masks(src_positive, noise.shape[2], noise.shape[3], device)
    comfy.samplers.resolve_areas_and_cond_masks(src_negative, noise.shape[2], noise.shape[3], device)

    model_wrap = model

    comfy.samplers.calculate_start_end_timesteps(model, positive)
    comfy.samplers.calculate_start_end_timesteps(model, negative)
    
    comfy.samplers.calculate_start_end_timesteps(model, src_positive)
    comfy.samplers.calculate_start_end_timesteps(model, src_negative)

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = comfy.samplers.encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        negative = comfy.samplers.encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

        src_positive = comfy.samplers.encode_model_conds(model.extra_conds, src_positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        src_negative = comfy.samplers.encode_model_conds(model.extra_conds, src_negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        comfy.samplers.create_cond_with_same_area_if_none(negative, c)
    for c in src_positive:
        comfy.samplers.create_cond_with_same_area_if_none(src_positive, c)

    for c in negative:
        comfy.samplers.create_cond_with_same_area_if_none(negative, c)
    for c in src_negative:
        comfy.samplers.create_cond_with_same_area_if_none(src_negative, c)

    comfy.samplers.pre_run_control(model, negative + positive + src_negative + src_positive)

    comfy.samplers.apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    comfy.samplers.apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    comfy.samplers.apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, src_positive)), src_negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    comfy.samplers.apply_empty_x_to_equal_area(src_positive, src_negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "src_cond": src_positive, "src_uncond": src_negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    samples = sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))


class KSampler(comfy.samplers.KSampler):
    def sample(self, noise, positive, negative, src_positive, src_negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = KSAMPLER()

        return from_KSampler_sample(self.model, noise, positive, negative, src_positive, src_negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


# comfy.sample.sample
def sample_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, src_positive, src_negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    # positive.append(src_positive[0][:])
    # src_positive.append(src_positive[0][:])

    real_model, positive_copy, negative_copy, noise_mask, models = comfy.sample.prepare_sampling(model, noise.shape, positive, negative, noise_mask)
    _, src_positive_copy, src_negative_copy, _, _ = comfy.sample.prepare_sampling(model, noise.shape, src_positive, src_negative, noise_mask)

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)

    # comfy.samplers.KSampler
    sampler = KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, src_positive_copy, src_negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())

    comfy.sample.cleanup_additional_models(models)
    comfy.sample.cleanup_additional_models(set(comfy.sample.get_models_from_cond(positive_copy, "control") + comfy.sample.get_models_from_cond(negative_copy, "control")))
    comfy.sample.cleanup_additional_models(set(comfy.sample.get_models_from_cond(positive_copy, "control") + comfy.sample.get_models_from_cond(negative_copy, "control")))
    return samples


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, src_positive, src_negative, latent_image, denoise=1.0, disable_noise=True, start_step=None, last_step=None, force_full_denoise=False):
    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    callback = latent_preview.prepare_callback(model, steps)

    # comfy.sample.sample
    latents = sample_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, src_positive, src_negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=None, callback=callback, disable_pbar=True, seed=seed)
  
    return latents


class KSamplerCR():
  @classmethod
  def INPUT_TYPES(s):
      return {"required":
                  {"model": ("MODEL",),
                  "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                  "add_noise": ("BOOLEAN", {"default": False}),
                  "grid_size": ("INT", {"default": 3, "min": 2, "max": 8}),
                  "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                  "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.1}),
                  "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                  "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                  "positive": ("CONDITIONING", ),
                  "negative": ("CONDITIONING", ),
                  "src_positive": ("CONDITIONING", ),
                  "src_negative": ("CONDITIONING", ),
                  "latent_image": ("LATENT", ),
                  "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                  "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                  "learning_rate": ("FLOAT", {"default": 2.5, "min": 0.0, "round": 0.01}),
                  "weight_decay": ("FLOAT", {"default": 0.0, "min": 0.0, "round": 0.01 }),
                  "decay_iterations": ("INT", {"default": 20, "min": 0, "round": 0.01 }),
                  "decay_rates": ("FLOAT", {"default": 0.9, "min": 0.0, "round": 0.01 }),
                }
              }

  RETURN_TYPES = ("LATENT",)
  FUNCTION = "sample"

  CATEGORY = "CR"

  @torch.no_grad()
  def sample(self, model, noise_seed, add_noise, grid_size, steps, cfg, sampler_name, scheduler, positive, negative, src_positive, src_negative, latent_image, start_at_step, end_at_step, learning_rate, weight_decay, decay_iterations, decay_rates):
    global lr, wd, decay_iters, decay_rate, gs
    lr = learning_rate
    wd = weight_decay
    decay_iters = decay_iterations
    decay_rate = decay_rates
    gs = grid_size

    positive = positive[:]
    negative = negative[:]
    src_positive = src_positive[:]
    src_negative = src_negative[:]

    control_objs = []
    control_images = []

    for conditioning in [positive, negative, src_positive, src_negative]:
        for t in conditioning:
            if 'control' in t[1]:
                control = t[1]['control']
                control_objs.append(control)
                control_images.append(control.cond_hint_original)
                
                prev = control.previous_controlnet
                while prev != None:
                    control_objs.append(prev)
                    control_images.append(prev.cond_hint_original)
                    prev = prev.previous_controlnet


    for i in range(len(control_objs)):
        control_objs[i].set_cond_hint(list_to_grid(control_images[i], gs), control_objs[i].strength, control_objs[i].timestep_percent_range)

    latent_image = latent_image['samples']
    latent_image = ru.list_to_grid(latent_image, gs)
    samples = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, src_positive, src_negative, latent_image, denoise=1.0, disable_noise=True, start_step=start_at_step, last_step=end_at_step, force_full_denoise=False)

    for i in range(len(control_objs)):
        control_objs[i].set_cond_hint(control_images[i], control_objs[i].strength, control_objs[i].timestep_percent_range)

    return ({ 'samples': samples }, )

