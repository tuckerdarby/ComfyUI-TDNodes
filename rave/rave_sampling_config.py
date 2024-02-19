from ..sampling.sampling_config import SamplingConfig
from . import rave_utils as ru


class RaveSamplingConfig(SamplingConfig):
  def __init__(self, latent_shape, sigmas, extra_args, grid_size, sampler=None, callback=None, disable=None, device='cuda') -> None:
    super().__init__(latent_shape, sigmas, extra_args, sampler, callback, disable, device)
    self.grid_size = grid_size
    self.inversion_latents = None

  def prepare_run(self, step, indexes, grid_size=None):
    super().prepare_run(step, indexes)
    if grid_size is not None:
      self.grid_size = grid_size

  def get_noise_from_cache(self, key):
    if key not in self.cache:
      return None
    frames = self.cache[key][self.indexes]
    return ru.list_to_grid(frames, self.grid_size)

  def set_noise_to_cache(self, key, value):
    if value is None:
      return
    frames = ru.grid_to_list(value, self.grid_size)
    self.cache[key][self.indexes] = frames

  def set_inversion_latents(self, inversion_latents):
    self.inversion_latents = inversion_latents

  def inversion_noise_sampler(self, sigma, sigma_next):
    return sigma*ru.list_to_grid(self.inversion_latents[self.indexes].to(self.device), self.grid_size)