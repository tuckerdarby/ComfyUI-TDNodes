
BACKUP_INDEXES = 'presevered_hints'


def for_controlnets(controlnet, fn):
  while controlnet is not None:
    fn(controlnet)
    controlnet = controlnet.previous_controlnet


def prepare_controlnet(cn):
  def prep(controlnet):
    if not hasattr(controlnet, BACKUP_INDEXES):
      setattr(controlnet, BACKUP_INDEXES,
              controlnet.cond_hint_original.to('cpu'))

  for_controlnets(cn, prep)


def reset_controlnet(cn):
  def reset(controlnet):
    if hasattr(controlnet, BACKUP_INDEXES):
      delattr(controlnet, BACKUP_INDEXES)

    for_controlnets(cn, reset)


def set_controlnet_indexes(cn, indexes):
  def set_indexes(controlnet):
    controlnet.cond_hint_original = controlnet.presevered_hints[indexes]
    controlnet.cond_hint = None

  for_controlnets(cn, set_indexes)
