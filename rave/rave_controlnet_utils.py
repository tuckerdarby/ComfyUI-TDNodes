from . import rave_utils as ru


BACKUP_INDEXES = 'presevered_hints'


def for_controlnets(controlnet, fn):
  while controlnet is not None:
    fn(controlnet)
    controlnet = controlnet.previous_controlnet


def load_controlnet(cn, grid_size):
  def prep(controlnet):
    # if hasattr(controlnet.cond_hint_original, 'condhint'):
    #   controlnet = controlnet.cond_hint_original
    if not hasattr(controlnet, BACKUP_INDEXES):
      setattr(controlnet, BACKUP_INDEXES,
              controlnet.cond_hint_original.to('cpu'))
    hint = ru.list_to_grid(
      controlnet.presevered_hints, grid_size)
    controlnet.cond_hint = None
    controlnet.set_cond_hint(
      hint, controlnet.strength, controlnet.timestep_percent_range)

  for_controlnets(cn, prep)


def prepare_controlnet(cn, grid_size):
  def prep(controlnet):
    hint = ru.grid_to_list(controlnet.cond_hint_original, grid_size)
    controlnet.presevered_hints = hint.to('cpu')

  for_controlnets(cn, prep)


def reset_controlnet(cn):
  def reset(controlnet):
    if hasattr(controlnet, BACKUP_INDEXES):
      hint = controlnet.presevered_hints
      controlnet.cond_hint = None
      controlnet.set_cond_hint(
          hint, controlnet.strength, controlnet.timestep_percent_range)
      delattr(controlnet, BACKUP_INDEXES)

    for_controlnets(cn, reset)


def set_controlnet_indexes(cn, indexes, grid_size):
  def set_indexes(controlnet):
    controlnet.cond_hint = None
    hint = ru.list_to_grid(
      controlnet.presevered_hints[indexes], grid_size)
    controlnet.set_cond_hint(
      hint, controlnet.strength, controlnet.timestep_percent_range)

  for_controlnets(cn, set_indexes)
