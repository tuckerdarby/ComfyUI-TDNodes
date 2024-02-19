import torch
import math
import collections

import comfy.utils
import comfy.model_management
import comfy.conds

from .custom_gligen import get_gligen_video_patches


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

    if 'instancediffusion' in apply_options:
      if patches is None:
        patches = {}
      setup_instancing([o[0] for o in to_run], patches, apply_options)

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

  out_cond /= out_count
  del out_count
  out_uncond /= out_uncond_count
  del out_uncond_count
  return out_cond, out_uncond


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

  position_net = instance['instance_models']['position_net'].to('cuda')
  fusers = list(instance['instance_models']['fusers_list'])
  for i in range(len(fusers)):
    fusers[i] = fusers[i].to('cuda')
  scaleu_nets = list(instance['instance_models']['scaleu_nets'])
  for idx in range(len(scaleu_nets)):
    scaleu_nets[idx] = scaleu_nets[idx].to('cuda')
  
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
      e[key] = c[key][indexes].to('cuda')
    es.append(e)

  e = {}
  for e_ in es:
    for key in e_:
      if key not in e:
        e[key] = e_[key]
      else:
        e[key] = torch.cat([e[key], e_[key]])

  objs, drop_box_mask = position_net(e['boxes'], e['masks'], e['prompts'], e['scribbles'], e['polygons'], e['segments'], e['points'])
  position_net = position_net.to('cpu')
  transformer_options = { 'objs': objs, 'drop_box_mask': drop_box_mask, 'grounding_input': e }
  
  track = {'idx': 0}
  def fuser_patch(x, extra_options):
    idx = track['idx']
    attn = fusers[idx%len(fusers)](x, transformer_options)
    track['idx'] += 1
    return attn.to(torch.float16)
  
  patches['middle_patch'].append(fuser_patch)

  def scaleu_patch(h, hsp, transformer_options):
    _, idx = transformer_options['block']
    sk = scaleu_nets[idx](h, hsp)
    return sk

  patches['output_block_patch'].append(scaleu_patch)


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
    
  isolated = False
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

  gv_patches = get_gligen_video_patches(input_x, conds, apply_options, isolated)
  if gv_patches is not None:
    if 'middle_patch' not in patches:
      patches['middle_patch'] = []

    patches['middle_patch'] += gv_patches
  
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


class CFGNoisePredictor(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.inner_model = model

  def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None, apply_options={}):
    out = sampling_function(self.inner_model, x, timestep, uncond, cond,
                            cond_scale, model_options=model_options, seed=seed, apply_options=apply_options)
    return out

  def forward(self, *args, **kwargs):
    return self.apply_model(*args, **kwargs)


def wrap_model(model):
  return CFGNoisePredictor(model)
