BACKUP_INDEXES = 'presevered_hints'


def for_controlnets(controlnet, fn):
  while controlnet is not None:
    fn(controlnet)
    controlnet = controlnet.previous_controlnet


def load_controlnet(cn):
  def prep(controlnet):
    if not hasattr(controlnet, BACKUP_INDEXES):
      setattr(controlnet, BACKUP_INDEXES,
              controlnet.cond_hint_original.to('cpu'))
    hint = controlnet.presevered_hints.to('cpu')
    controlnet.cond_hint = None
    controlnet.set_cond_hint(
      hint, controlnet.strength, controlnet.timestep_percent_range)

  for_controlnets(cn, prep)


def prepare_controlnet(cn):
  def prep(controlnet):
    hint = controlnet.cond_hint_original
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


def set_controlnet_indexes(cn, indexes):
  def set_indexes(controlnet):
    controlnet.cond_hint = None
    hint = controlnet.presevered_hints[indexes]
    controlnet.set_cond_hint(
      hint, controlnet.strength, controlnet.timestep_percent_range)

  for_controlnets(cn, set_indexes)
