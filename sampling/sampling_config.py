import torch
from collections import defaultdict


class SamplingConfig:
  def __init__(self, latent_shape, sigmas, extra_args, sampler=None, callback=None, disable=None, device='cuda') -> None:
    self.latent_shape = latent_shape
    self.sigmas = sigmas  # these should be all sigmas that will be stepped through -- not just for step
    self.extra_args = extra_args
    self.sampler = sampler
    self.callback_func = callback
    self.disable = disable
    self.indexes = None
    self.device = device
    self.cache = defaultdict(lambda: torch.zeros(self.latent_shape).to(self.device))
    self.value_cache = {}
    self.step = 0

  def prepare_run(self, step, indexes):
    self.step = step  # step used to index self.sigmas
    self.indexes = indexes  # Indexes can be set for batched runs

  def get_noise_from_cache(self, key):
    if key not in self.cache:
      return None
    if self.indexes is not None:
      return self.cache[key][self.indexes]
    return self.cache[key]

  def set_noise_to_cache(self, key, value):
    if self.indexes is not None:
      self.cache[key][self.indexes] = value
    else:
      self.cache[key] = value

  def get_value(self, key):
    return self.value_cache.get(key, None)

  def set_value(self, key, value):
    self.value_cache[key] = value

  def get_noise_sampler(self):
    return self.sampler

  def set_default_noise_sampler(self, sampler, override=False):
    if self.sampler is None or override:  # Samplers can choose their own default sampler if not already specified
      self.sampler = sampler
    return self.sampler

  def callback(self, *args, **kwargs):
    if self.callback_func:
      self.callback_func(*args, **kwargs)

  def reset(self):
    del self.cache
    self.cache = defaultdict(lambda: torch.zeros(self.latent_shape).to(self.device))