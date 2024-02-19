from typing import Callable

import math
import torch
from torch import Tensor
from torch.nn.functional import group_norm
from einops import rearrange

import comfy.ldm.modules.attention as attention
from comfy.ldm.modules.diffusionmodules import openaimodel
import comfy.model_management as model_management
import comfy.samplers
import comfy.sample
import comfy.utils
from comfy.controlnet import ControlBase
import comfy.ops

from .context import ContextFuseMethod, ContextSchedules, get_context_weights, get_context_windows
from .sample_settings import IterationOptions, SampleSettings, SeedNoiseGeneration, prepare_mask_ad
from .utils_model import ModelTypeSD, wrap_function_to_inject_xformers_bug_info
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelGroup, MotionModelPatcher
from .motion_module_ad import AnimateDiffFormat, AnimateDiffInfo, AnimateDiffVersion, VanillaTemporalModule
from .logger import logger


##################################################################################
######################################################################
# Global variable to use to more conveniently hack variable access into samplers
class AnimateDiffHelper_GlobalState:
    def __init__(self):
        self.motion_models: MotionModelGroup = None
        self.params: InjectionParams = None
        self.sample_settings: SampleSettings = None
        self.reset()
    
    def initialize(self, model):
        # this function is to be run in sampling func
        if not self.initialized:
            self.initialized = True
            if self.motion_models is not None:
                self.motion_models.initialize_timesteps(model)
            if self.params.context_options is not None:
                self.params.context_options.initialize_timesteps(model)
            if self.sample_settings.custom_cfg is not None:
                self.sample_settings.custom_cfg.initialize_timesteps(model)

    def reset(self):
        self.initialized = False
        self.start_step: int = 0
        self.last_step: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        if self.motion_models is not None:
            del self.motion_models
            self.motion_models = None
        if self.params is not None:
            del self.params
            self.params = None
        if self.sample_settings is not None:
            del self.sample_settings
            self.sample_settings = None
    
    def update_with_inject_params(self, params: InjectionParams):
        self.params = params

    def is_using_sliding_context(self):
        return self.params is not None and self.params.is_using_sliding_context()
    
    def create_exposed_params(self):
        # This dict will be exposed to be used by other extensions
        # DO NOT change any of the key names
        # or I will find you 👁.👁
        return {
            "full_length": self.params.full_length,
            "context_length": self.params.context_options.context_length,
            "sub_idxs": self.params.sub_idxs,
        }

ADGS = AnimateDiffHelper_GlobalState()
######################################################################
##################################################################################


##################################################################################
#### Code Injection ##################################################

# refer to forward_timestep_embed in comfy/ldm/modules/diffusionmodules/openaimodel.py
def forward_timestep_embed_factory() -> Callable:
    def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
        for layer in ts:
            if isinstance(layer, openaimodel.VideoResBlock):
                x = layer(x, emb, num_video_frames, image_only_indicator)
            elif isinstance(layer, openaimodel.TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, VanillaTemporalModule):
                x = layer(x, context)
            elif isinstance(layer, attention.SpatialVideoTransformer):
                x = layer(x, context, time_context, num_video_frames, image_only_indicator, transformer_options)
                if "transformer_index" in transformer_options:
                    transformer_options["transformer_index"] += 1
                if "current_index" in transformer_options: # keep this for backward compat, for now
                    transformer_options["current_index"] += 1
            elif isinstance(layer, attention.SpatialTransformer):
                x = layer(x, context, transformer_options)
                if "transformer_index" in transformer_options:
                    transformer_options["transformer_index"] += 1
                if "current_index" in transformer_options:  # keep this for backward compat, for now
                    transformer_options["current_index"] += 1
            elif isinstance(layer, openaimodel.Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x
    return forward_timestep_embed


def unlimited_memory_required(*args, **kwargs):
    return 0


def groupnorm_mm_factory(params: InjectionParams, manual_cast=False):
    def groupnorm_mm_forward(self, input: Tensor) -> Tensor:
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        if not params.is_using_sliding_context():
            batched_conds = input.size(0)//params.full_length
        else:
            batched_conds = input.size(0)//params.context_options.context_length

        input = rearrange(input, "(b f) c h w -> b c f h w", b=batched_conds)
        if manual_cast:
            weight, bias = comfy.ops.cast_bias_weight(self, input)
        else:
            weight, bias = self.weight, self.bias
        input = group_norm(input, self.num_groups, weight, bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=batched_conds)
        return input
    return groupnorm_mm_forward


def get_additional_models_factory(orig_get_additional_models: Callable, motion_models: MotionModelGroup):
    def get_additional_models_with_motion(*args, **kwargs):
        models, inference_memory = orig_get_additional_models(*args, **kwargs)
        if motion_models is not None:
            for motion_model in motion_models.models:
                models.append(motion_model)
        # TODO: account for inference memory as well?
        return models, inference_memory
    return get_additional_models_with_motion
######################################################################
##################################################################################


def apply_params_to_motion_models(motion_models: MotionModelGroup, params: InjectionParams):
    params = params.clone()
    for context in params.context_options.contexts:
        if context.context_schedule == ContextSchedules.VIEW_AS_CONTEXT:
            context.context_length = params.full_length
    # TODO: check (and message) should be different based on use_on_equal_length setting
    if params.context_options.context_length:
        pass

    allow_equal = params.context_options.use_on_equal_length
    if params.context_options.context_length:
        enough_latents = params.full_length >= params.context_options.context_length if allow_equal else params.full_length > params.context_options.context_length
    else:
        enough_latents = False
    if params.context_options.context_length and enough_latents:
        logger.info(f"Sliding context window activated - latents passed in ({params.full_length}) greater than context_length {params.context_options.context_length}.")
    else:
        logger.info(f"Regular AnimateDiff activated - latents passed in ({params.full_length}) less or equal to context_length {params.context_options.context_length}.")
        params.reset_context()
    if motion_models is not None:
        # if no context_length, treat video length as intended AD frame window
        if not params.context_options.context_length:
            for motion_model in motion_models.models:
                if not motion_model.model.is_length_valid_for_encoding_max_len(params.full_length):
                    raise ValueError(f"Without a context window, AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames, but received {params.full_length} latents.")
            motion_models.set_video_length(params.full_length, params.full_length)
        # otherwise, treat context_length as intended AD frame window
        else:
            for motion_model in motion_models.models:
                view_options = params.context_options.view_options
                context_length = view_options.context_length if view_options else params.context_options.context_length
                if not motion_model.model.is_length_valid_for_encoding_max_len(context_length):
                    raise ValueError(f"AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames for a context window, but received context length of {params.context_options.context_length}.")
            motion_models.set_video_length(params.context_options.context_length, params.full_length)
        # inject model
        module_str = "modules" if len(motion_models.models) > 1 else "module"
        logger.info(f"Using motion {module_str} {motion_models.get_name_string(show_version=True)}.")
    return params


class FunctionInjectionHolder:
    def __init__(self):
        pass
    
    def inject_functions(self, model: ModelPatcherAndInjector, params: InjectionParams):
        # Save Original Functions
        self.orig_forward_timestep_embed = openaimodel.forward_timestep_embed # needed to account for VanillaTemporalModule
        self.orig_memory_required = model.model.memory_required # allows for "unlimited area hack" to prevent halving of conds/unconds
        self.orig_groupnorm_forward = torch.nn.GroupNorm.forward # used to normalize latents to remove "flickering" of colors/brightness between frames
        self.orig_groupnorm_manual_cast_forward = comfy.ops.manual_cast.GroupNorm.forward_comfy_cast_weights
        self.orig_sampling_function = comfy.samplers.sampling_function # used to support sliding context windows in samplers
        self.orig_prepare_mask = comfy.sample.prepare_mask
        self.orig_get_additional_models = comfy.sample.get_additional_models
        # Inject Functions
        openaimodel.forward_timestep_embed = forward_timestep_embed_factory()
        if params.unlimited_area_hack:
            model.model.memory_required = unlimited_memory_required
        if model.motion_models is not None:
            # only apply groupnorm hack if not [v3 or ([not Hotshot] and SD1.5 and v2 and apply_v2_properly)]
            info: AnimateDiffInfo = model.motion_models[0].model.mm_info
            if not (info.mm_version == AnimateDiffVersion.V3 or
                    (info.mm_format not in [AnimateDiffFormat.HOTSHOTXL] and info.sd_type == ModelTypeSD.SD1_5 and info.mm_version == AnimateDiffVersion.V2 and params.apply_v2_properly)):
                torch.nn.GroupNorm.forward = groupnorm_mm_factory(params)
                comfy.ops.manual_cast.GroupNorm.forward_comfy_cast_weights = groupnorm_mm_factory(params, manual_cast=True)
                # if mps device (Apple Silicon), disable batched conds to avoid black images with groupnorm hack
                try:
                    if model.load_device.type == "mps":
                        model.model.memory_required = unlimited_memory_required
                except Exception:
                    pass
            del info
        comfy.samplers.sampling_function = evolved_sampling_function
        comfy.sample.prepare_mask = prepare_mask_ad
        comfy.sample.get_additional_models = get_additional_models_factory(self.orig_get_additional_models, model.motion_models)

    def restore_functions(self, model: ModelPatcherAndInjector):
        # Restoration
        try:
            model.model.memory_required = self.orig_memory_required
            openaimodel.forward_timestep_embed = self.orig_forward_timestep_embed
            torch.nn.GroupNorm.forward = self.orig_groupnorm_forward
            comfy.ops.manual_cast.GroupNorm.forward_comfy_cast_weights = self.orig_groupnorm_manual_cast_forward
            comfy.samplers.sampling_function = self.orig_sampling_function
            comfy.sample.prepare_mask = self.orig_prepare_mask
            comfy.sample.get_additional_models = self.orig_get_additional_models
        except AttributeError:
            logger.error("Encountered AttributeError while attempting to restore functions - likely, an error occured while trying " + \
                         "to save original functions before injection, and a more specific error was thrown by ComfyUI.")


def motion_sample_factory(orig_comfy_sample: Callable, is_custom: bool=False) -> Callable:
    def motion_sample(model: ModelPatcherAndInjector, noise: Tensor, *args, **kwargs):
        # check if model is intended for injecting
        if type(model) != ModelPatcherAndInjector:
            return orig_comfy_sample(model, noise, *args, **kwargs)
        # otherwise, injection time
        latents = None
        cached_latents = None
        cached_noise = None
        function_injections = FunctionInjectionHolder()
        try:
            if model.sample_settings.custom_cfg is not None:
                model = model.sample_settings.custom_cfg.patch_model(model)
            # clone params from model
            params = model.motion_injection_params.clone()
            # get amount of latents passed in, and store in params
            latents: Tensor = args[-1]
            params.full_length = latents.size(0)
            # reset global state
            ADGS.reset()

            # apply custom noise, if needed
            disable_noise = kwargs.get("disable_noise") or False
            seed = kwargs["seed"]

            # apply params to motion model
            params = apply_params_to_motion_models(model.motion_models, params)

            # store and inject functions
            function_injections.inject_functions(model, params)

            # prepare noise_extra_args for noise generation purposes
            noise_extra_args = {"disable_noise": disable_noise}
            params.set_noise_extra_args(noise_extra_args)
            # if noise is not disabled, do noise stuff
            if not disable_noise:
                noise = model.sample_settings.prepare_noise(seed, latents, noise, extra_args=noise_extra_args, force_create_noise=False)

            # callback setup
            original_callback = kwargs.get("callback", None)
            def ad_callback(step, x0, x, total_steps):
                if original_callback is not None:
                    original_callback(step, x0, x, total_steps)
                # update GLOBALSTATE for next iteration
                ADGS.current_step = ADGS.start_step + step + 1
            kwargs["callback"] = ad_callback
            ADGS.motion_models = model.motion_models
            ADGS.sample_settings = model.sample_settings

            # apply adapt_denoise_steps
            args = list(args)
            if model.sample_settings.adapt_denoise_steps and not is_custom:
                # only applicable when denoise and steps are provided (from simple KSampler nodes)
                denoise = kwargs.get("denoise", None)
                steps = args[0]
                if denoise is not None and type(steps) == int:
                    args[0] = max(int(denoise * steps), 1)


            iter_opts = IterationOptions()
            if model.sample_settings is not None:
                iter_opts = model.sample_settings.iteration_opts
            iter_opts.initialize(latents)
            # cache initial noise and latents, if needed
            if iter_opts.cache_init_latents:
                cached_latents = latents.clone()
            if iter_opts.cache_init_noise:
                cached_noise = noise.clone()
            # prepare iter opts preprocess kwargs, if needed
            iter_kwargs = {}
            if iter_opts.need_sampler:
                # -5 for sampler_name (not custom) and sampler (custom)
                model_management.load_model_gpu(model)
                if is_custom:
                    iter_kwargs[IterationOptions.SAMPLER] = None #args[-5]
                else:
                    iter_kwargs[IterationOptions.SAMPLER] = comfy.samplers.KSampler(
                        model.model, steps=999, #steps=args[-7],
                        device=model.current_device, sampler=args[-5],
                        scheduler=args[-4], denoise=kwargs.get("denoise", None),
                        model_options=model.model_options)

            for curr_i in range(iter_opts.iterations):
                # handle GLOBALSTATE vars and step tally
                ADGS.update_with_inject_params(params)
                ADGS.start_step = kwargs.get("start_step") or 0
                ADGS.current_step = ADGS.start_step
                ADGS.last_step = kwargs.get("last_step") or 0
                if iter_opts.iterations > 1:
                    logger.info(f"Iteration {curr_i+1}/{iter_opts.iterations}")
                # perform any iter_opts preprocessing on latents
                latents, noise = iter_opts.preprocess_latents(curr_i=curr_i, model=model, latents=latents, noise=noise,
                                                              cached_latents=cached_latents, cached_noise=cached_noise,
                                                              seed=seed,
                                                              sample_settings=model.sample_settings, noise_extra_args=noise_extra_args,
                                                              **iter_kwargs)
                args[-1] = latents

                if model.motion_models is not None:
                    model.motion_models.pre_run(model)
                if model.sample_settings is not None:
                    model.sample_settings.pre_run(model)
                latents = wrap_function_to_inject_xformers_bug_info(orig_comfy_sample)(model, noise, *args, **kwargs)
            return latents
        finally:
            del latents
            del noise
            del cached_latents
            del cached_noise
            # reset global state
            ADGS.reset()
            # restore injected functions
            function_injections.restore_functions(model)
            del function_injections
    return motion_sample


def evolved_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options: dict={}, seed=None):
    ADGS.initialize(model)
    if ADGS.motion_models is not None:
        ADGS.motion_models.prepare_current_keyframe(t=timestep)
    if ADGS.params.context_options is not None:
        ADGS.params.context_options.prepare_current_context(t=timestep)
    if ADGS.sample_settings.custom_cfg is not None:
        ADGS.sample_settings.custom_cfg.prepare_current_keyframe(t=timestep)

    # never use cfg1 optimization if using custom_cfg (since can have timesteps and such)
    if ADGS.sample_settings.custom_cfg is None and math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    # add AD/evolved-sampling params to model_options (transformer_options)
    model_options = model_options.copy()
    if "tranformer_options" not in model_options:
        model_options["tranformer_options"] = {}
    model_options["transformer_options"]["ad_params"] = ADGS.create_exposed_params()

    if not ADGS.is_using_sliding_context():
        cond_pred, uncond_pred = comfy.samplers.calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)
    else:
        cond_pred, uncond_pred = sliding_calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result


# sliding_calc_cond_uncond_batch inspired by ashen's initial hack for 16-frame sliding context:
# https://github.com/comfyanonymous/ComfyUI/compare/master...ashen-sensored:ComfyUI:master
def sliding_calc_cond_uncond_batch(model, cond, uncond, x_in: Tensor, timestep, model_options):
    def prepare_control_objects(control: ControlBase, full_idxs: list[int]):
        if control.previous_controlnet is not None:
            prepare_control_objects(control.previous_controlnet, full_idxs)
        control.sub_idxs = full_idxs
        control.full_latent_length = ADGS.params.full_length
        control.context_length = ADGS.params.context_options.context_length
    
    def get_resized_cond(cond_in, full_idxs) -> list:
        # reuse or resize cond items to match context requirements
        resized_cond = []
        # cond object is a list containing a dict - outer list is irrelevant, so just loop through it
        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            # now we are in the inner dict - "pooled_output" is a tensor, "control" is a ControlBase object, "model_conds" is dictionary
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    if isinstance(cond_item, Tensor):
                        # check that tensor is the expected length - x.size(0)
                        if cond_item.size(0) == x_in.size(0):
                            # if so, it's subsetting time - tell controls the expected indeces so they can handle them
                            actual_cond_item = cond_item[full_idxs]
                            resized_actual_cond[key] = actual_cond_item
                        else:
                            resized_actual_cond[key] = cond_item
                    # look for control
                    elif key == "control":
                        control_item = cond_item
                        if hasattr(control_item, "sub_idxs"):
                            prepare_control_objects(control_item, full_idxs)
                        else:
                            raise ValueError(f"Control type {type(control_item).__name__} may not support required features for sliding context window; \
                                                use Control objects from Kosinkadink/ComfyUI-Advanced-ControlNet nodes, or make sure Advanced-ControlNet is updated.")
                        resized_actual_cond[key] = control_item
                        del control_item
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        # when in dictionary, look for tensors and CONDCrossAttn [comfy/conds.py] (has cond attr that is a tensor)
                        for cond_key, cond_value in new_cond_item.items():
                            if isinstance(cond_value, Tensor):
                                if cond_value.size(0) == x_in.size(0):
                                    new_cond_item[cond_key] = cond_value[full_idxs]
                            # if has cond that is a Tensor, check if needs to be subset
                            elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, Tensor):
                                if cond_value.cond.size(0) == x_in.size(0):
                                    new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond[full_idxs])
                        resized_actual_cond[key] = new_cond_item
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item  # just in case to prevent VRAM issues
            resized_cond.append(resized_actual_cond)
        return resized_cond

    # get context windows
    ADGS.params.context_options.step = ADGS.current_step
    context_windows = get_context_windows(ADGS.params.full_length, ADGS.params.context_options)
    # figure out how input is split
    batched_conds = x_in.size(0)//ADGS.params.full_length

    if ADGS.motion_models is not None:
        ADGS.motion_models.set_view_options(ADGS.params.context_options.view_options)

    # prepare final cond, uncond, and out_count
    cond_final = torch.zeros_like(x_in)
    uncond_final = torch.zeros_like(x_in)
    out_count_final = torch.zeros((x_in.shape[0], 1, 1, 1), device=x_in.device)
    bias_final = [0.0] * x_in.shape[0]

    # perform calc_cond_uncond_batch per context window
    for ctx_idxs in context_windows:
        ADGS.params.sub_idxs = ctx_idxs
        if ADGS.motion_models is not None:
            ADGS.motion_models.set_sub_idxs(ctx_idxs)
            ADGS.motion_models.set_video_length(len(ctx_idxs), ADGS.params.full_length)
        # update exposed params
        model_options["transformer_options"]["ad_params"]["sub_idxs"] = ctx_idxs
        model_options["transformer_options"]["ad_params"]["context_length"] = len(ctx_idxs)
        # account for all portions of input frames
        full_idxs = []
        for n in range(batched_conds):
            for ind in ctx_idxs:
                full_idxs.append((ADGS.params.full_length*n)+ind)
        # get subsections of x, timestep, cond, uncond, cond_concat
        sub_x = x_in[full_idxs]
        sub_timestep = timestep[full_idxs]
        sub_cond = get_resized_cond(cond, full_idxs) if cond is not None else None
        sub_uncond = get_resized_cond(uncond, full_idxs) if uncond is not None else None
        apply_options = { 'indexes': ctx_idxs, 'instancediffusion': True }
        sub_cond_out, sub_uncond_out = calc_cond_uncond_batch(model, sub_cond, sub_uncond, sub_x, sub_timestep, model_options, apply_options)
        # sub_cond_out, sub_uncond_out = comfy.samplers.calc_cond_uncond_batch(model, sub_cond, sub_uncond, sub_x, sub_timestep, model_options)

        if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
            full_length = ADGS.params.full_length
            for pos, idx in enumerate(ctx_idxs):
                # bias is the influence of a specific index in relation to the whole context window
                bias = 1 - abs(idx - (ctx_idxs[0] + ctx_idxs[-1]) / 2) / ((ctx_idxs[-1] - ctx_idxs[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                # take weighted average relative to total bias of current idx
                # and account for batched_conds
                for n in range(batched_conds):
                    bias_total = bias_final[(full_length*n)+idx]
                    prev_weight = (bias_total / (bias_total + bias))
                    new_weight = (bias / (bias_total + bias))
                    cond_final[(full_length*n)+idx] = cond_final[(full_length*n)+idx] * prev_weight + sub_cond_out[(full_length*n)+pos] * new_weight
                    uncond_final[(full_length*n)+idx] = uncond_final[(full_length*n)+idx] * prev_weight + sub_uncond_out[(full_length*n)+pos] * new_weight
                    bias_final[(full_length*n)+idx] = bias_total + bias
        else:
            # add conds and counts based on weights of fuse method
            weights = get_context_weights(len(ctx_idxs), ADGS.params.context_options.fuse_method) * batched_conds
            weights_tensor = torch.Tensor(weights).to(device=x_in.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            cond_final[full_idxs] += sub_cond_out * weights_tensor
            uncond_final[full_idxs] += sub_uncond_out * weights_tensor
            out_count_final[full_idxs] += weights_tensor

    if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
        # already normalized, so return as is
        del out_count_final
        return cond_final, uncond_final
    else:
        # normalize cond and uncond via division by context usage counts
        cond_final /= out_count_final
        uncond_final /= out_count_final
        del out_count_final
        return cond_final, uncond_final


import torch
import math
import collections

import comfy.utils
import comfy.model_management
import comfy.conds



def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
  sigma = t
  xc = self.model_sampling.calculate_input(sigma, x)
  if c_concat is not None:
    xc = torch.cat([xc] + [c_concat], dim=1)

  context = c_crossattn
  dtype = self.get_dtype()

  if self.manual_cast_dtype is not None:
    dtype = self.manual_cast_dtype

  xc = xc.to(dtype)
  t = self.model_sampling.timestep(t).float()
  context = context.to(dtype)
  extra_conds = {}
  for o in kwargs:
    extra = kwargs[o]
    if hasattr(extra, "dtype"):
      if extra.dtype != torch.int and extra.dtype != torch.long:
        extra = extra.to(dtype)
    extra_conds[o] = extra

  model_output = self.diffusion_model(
      xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
  return self.model_sampling.calculate_denoised(sigma, model_output, x)


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options, apply_options):
  src_cond = [c for c in cond if c.get('src_latent', False)]
  cond = [c for c in cond if not c.get('src_latent', False)]
  out_cond = torch.zeros_like(x_in)
  out_count = torch.ones_like(x_in) * 1e-37

  out_uncond = torch.zeros_like(x_in)
  out_uncond_count = torch.ones_like(x_in) * 1e-37

  COND = 0
  UNCOND = 1
  SRC = 2

  grouped = []
  isolated = []

  for x in src_cond:
    p = get_area_and_mult(x, x_in, timestep, apply_options)
    if p is not None:
      if p.isolated:
        isolated += [[(p, SRC)]]
      else:
        grouped += [(p, SRC)]

  for x in uncond:
    p = get_area_and_mult(x, x_in, timestep, apply_options)
    if p is not None:
      if p.isolated:
        isolated += [[(p, UNCOND)]]
      else:
        grouped += [(p, UNCOND)]

  for x in cond:
    p = get_area_and_mult(x, x_in, timestep, apply_options)
    if p is not None:
      if p.isolated:
        isolated += [[(p, COND)]]
      else:
        grouped += [(p, COND)]

  for to_run in [grouped, *isolated]:
    if len(to_run) == 0:
      continue
    input_x = []
    mult = []
    c = []
    cond_or_uncond = []
    area = []
    positions = []
    control = None
    patches = None
    for o in to_run:
      p = o[0]
      input_x.append(p.input_x)
      mult.append(p.mult)
      c.append(p.conditioning)
      area.append(p.area)
      cond_or_uncond.append(o[1])
      control = p.control
      patches = p.patches
      positions.append(p.positions)

    batch_chunks = len(cond_or_uncond)
    input_x = torch.cat(input_x)
    c = comfy.samplers.cond_cat(c)
    timestep_ = torch.cat([timestep] * batch_chunks)

    if control is not None:
      c['control'] = control.get_control(
          input_x, timestep_, c, len(cond_or_uncond))

    transformer_options = {}
    if 'transformer_options' in model_options:
      transformer_options = model_options['transformer_options'].copy()

    instance_models, instance_options = None, None
    if 'instancediffusion' in apply_options:
      if patches is None:
        patches = {}
      instance_models, instance_options = setup_instancing([o[0] for o in to_run], patches, apply_options)

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

    transformer_options["cond_or_uncond"] = [
        c if c < 2 else 0 for c in cond_or_uncond]
    transformer_options["sigmas"] = timestep

    c['transformer_options'] = transformer_options

    if 'model_function_wrapper' in apply_options:
      output = apply_options['model_function_wrapper'](model, input_x, timestep_,
                          **c).chunk(batch_chunks)
    elif 'model_function_wrapper' in model_options:
      output = model_options['model_function_wrapper'](model.apply_model, {
                                                      "input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
    else:
      output = apply_model(model, input_x, timestep_, **c).chunk(batch_chunks)
    del input_x

    for o in range(batch_chunks):
      # H, W, x, y
      if cond_or_uncond[o] == COND:
        out_cond[:, :, area[o][2]:area[o][0] + area[o][2], area[o]
                [3]:area[o][1] + area[o][3]] += output[o] * mult[o]
        out_count[:, :, area[o][2]:area[o][0] + area[o][2],
                  area[o][3]:area[o][1] + area[o][3]] += mult[o]
      elif cond_or_uncond[o] == UNCOND:
        out_uncond[:, :, area[o][2]:area[o][0] + area[o][2],
                  area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
        out_uncond_count[:, :, area[o][2]:area[o][0] + area[o]
                        [2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
    del mult

  if instance_models is not None:
      instance_models['position_net'] = instance_models['position_net'].to('cpu')
      fusers = list(instance_models['fusers_list'])
      for i in range(len(fusers)):
        fusers[i] = fusers[i].to('cpu')
      scaleu_nets = list(instance_models['scaleu_nets'])
      for idx in range(len(scaleu_nets)):
        scaleu_nets[idx] = scaleu_nets[idx].to('cpu')
  
  if instance_options is not None:
     del instance_options

  out_cond /= out_count
  del out_count
  out_uncond /= out_uncond_count
  del out_uncond_count
  return out_cond, out_uncond


def get_fuser_patch(fusers, instance_options):
  track = {'idx': 0}
  def fuser_patch(x, extra_options):
    track_idx = track['idx']
    total_attn = []
    fusers[track_idx%len(fusers)] = fusers[track_idx%len(fusers)].to('cuda')
    for idx in range(len(x)):
      idx_instance_options = { 
        'objs':  instance_options['objs'][idx].unsqueeze(0).to('cuda'), 
        'drop_box_mask': instance_options['drop_box_mask'],}
        # 'grounding_input': {'att_masks': instance_options['grounding_input']['att_masks'][idx].unsqueeze(0).to('cuda') }}
      attn = fusers[track_idx%len(fusers)](x[idx].unsqueeze(0), idx_instance_options)
      # if 'next_att_mask' in idx_instance_options['grounding_input']:
      #    instance_options['grounding_input']['att_masks'][idx] = idx_instance_options['grounding_input']['next_att_mask'].to('cpu')
      #    del instance_options['grounding_input']['next_att_mask']
      del idx_instance_options
      total_attn.append(attn.to('cpu').to(torch.float16))
    total_attn = torch.cat(total_attn)
    # instance_options['objs'] = instance_options['objs'].to('cuda')
    # instance_options['grounding_input']['att_masks'] = instance_options['grounding_input']['att_masks'].to('cuda')
    # total_attn = fusers[idx%len(fusers)].to('cuda')(x, { 'objs': instance_options['objs'].to('cuda'), 'drop_box_mask': instance_options['drop_box_mask'] })
    fusers[track_idx%len(fusers)].to('cpu')
    track['idx'] += 1
    return total_attn.to(torch.float16).to('cuda')
  
  return fuser_patch


def get_scaleu_patch(scaleu_nets):
  def scaleu_patch(h, hsp, transformer_options):
    _, idx = transformer_options['block']
    sk = scaleu_nets[idx].to('cuda')(h, hsp)
    scaleu_nets[idx].to('cpu')
    return sk
  
  return scaleu_patch


def setup_instancing(ps, patches, apply_options):
  indexes = apply_options['indexes']
  instance = None
  for p in ps:
    if p.instance is not None:
      instance = p.instance
      
  n_frames = instance['n_frames']
  grounding_tokenizer_input = instance['grounding_tokenizer_input']

  if 'middle_patch' not in patches:
    patches['middle_patch'] = []
  if 'output_block_patch' not in patches:
    patches['output_block_patch'] = []
  
  # objs, drop_box_mask = input['position_net']( grounding_input['boxes'], grounding_input['masks'], grounding_input['positive_embeddings'], grounding_input['scribbles'], grounding_input['polygons'], grounding_input['segs'], grounding_input['points'] )
  es = []
  for p in ps:
    if p.instance is not None:
      c = p.instance['embeddings'][0]
      e = {}
    else:
      c = grounding_tokenizer_input.get_null_input(n_frames)
      e = {}
    for key in c:
      e[key] = c[key][indexes]
    es.append(e)

  e = {}
  for e_ in es:
    for key in e_:
      if key not in e:
        e[key] = e_[key]
      else:
        e[key] = torch.cat([e[key], e_[key]])

  position_net = instance['instance_models']['position_net'].to('cuda')
  objs, drop_box_mask = position_net(e['boxes'].to('cuda'), e['masks'].to('cuda'), e['prompts'].to('cuda'), e['scribbles'].to('cuda'), e['polygons'].to('cuda'), e['segments'].to('cuda'), e['points'].to('cuda'))
  objs = objs.to('cpu')
  position_net = position_net.to('cpu')
  instance_options = { 'objs': objs, 'drop_box_mask': drop_box_mask } #  'grounding_input': {'att_masks': e['att_masks'].to('cpu') } }
  e = None
  del e
  
  patches['middle_patch'].append(get_fuser_patch(instance['instance_models']['fusers_list'], instance_options))
  patches['output_block_patch'].append(get_scaleu_patch(instance['instance_models']['scaleu_nets']))

  return instance['instance_models'], instance_options


def get_area_and_mult(conds, x_in, timestep_in, apply_options):
  area = (x_in.shape[2], x_in.shape[3], 0, 0)
  strength = 1.0

  if 'timestep_start' in conds:
    timestep_start = conds['timestep_start']
    if timestep_in[0] > timestep_start:
      return None
  if 'timestep_end' in conds:
    timestep_end = conds['timestep_end']
    if timestep_in[0] < timestep_end:
      return None
    
  isolated = True
  positions = None
  if 'positions' in conds:
    positions = conds['positions']
    # isolated = True
  if 'area' in conds:
    area = conds['area']
    isolated = True
  if 'strength' in conds:
    strength = conds['strength']

  input_x = x_in[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
  if 'mask' in conds:
    # Scale the mask to the size of the input
    # The mask should have been resized as we began the sampling process
    mask_strength = 1.0
    if "mask_strength" in conds:
      mask_strength = conds["mask_strength"]
    mask = conds['mask']
    assert (mask.shape[1] == x_in.shape[2])
    assert (mask.shape[2] == x_in.shape[3])
    mask = mask[:, area[2]:area[0] + area[2],
                area[3]:area[1] + area[3]] * mask_strength
    mask = mask.unsqueeze(1).repeat(
        input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
  else:
    mask = torch.ones_like(input_x)
  mult = mask * strength

  if 'mask' not in conds:
    rr = 8
    if area[2] != 0:
      for t in range(rr):
        mult[:, :, t:1 + t, :] *= ((1.0 / rr) * (t + 1))
    if (area[0] + area[2]) < x_in.shape[2]:
      for t in range(rr):
        mult[:, :, area[0] - 1 - t:area[0] -
             t, :] *= ((1.0 / rr) * (t + 1))
    if area[3] != 0:
      for t in range(rr):
        mult[:, :, :, t:1 + t] *= ((1.0 / rr) * (t + 1))
    if (area[1] + area[3]) < x_in.shape[3]:
      for t in range(rr):
        mult[:, :, :, area[1] - 1 - t:area[1] -
             t] *= ((1.0 / rr) * (t + 1))

  conditioning = {}
  model_conds = conds["model_conds"]
  for c in model_conds:
    conditioning[c] = model_conds[c].process_cond(
        batch_size=x_in.shape[0], device=x_in.device, area=area)

  control = conds.get('control', None)

  patches = {}
  if 'gligen' in conds:
    gligen = conds['gligen']
    gligen_type = gligen[0]
    gligen_model = gligen[1]
    if gligen_type == "position":
      gligen_patch = gligen_model.model.set_position(
          input_x.shape, gligen[2], input_x.device)
    else:
      gligen_patch = gligen_model.model.set_empty(
          input_x.shape, input_x.device)

    patches['middle_patch'] = [gligen_patch]
  
  instance= None
  if 'instance_diffusion' in conds:
    instance = conds['instance_diffusion']
    
  cond_obj = collections.namedtuple(
      'cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches', 'positions', 'isolated', 'instance'])
  return cond_obj(input_x, mult, conditioning, area, control, patches, positions, isolated, instance)


def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, apply_options={}):
  if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
    uncond_ = []
  else:
    uncond_ = uncond

  cond_pred, uncond_pred = calc_cond_uncond_batch(
      model, cond, uncond_, x, timestep, model_options, apply_options)

  if "sampler_cfg_function" in model_options:
    args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
            "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
    cfg_result = x - model_options["sampler_cfg_function"](args)
  else:
    cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

  for fn in model_options.get("sampler_post_cfg_function", []):
    args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
            "sigma": timestep, "model_options": model_options, "input": x}
    cfg_result = fn(args)

  return cfg_result

