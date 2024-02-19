import torch


def get_gligen_video_patches(input_x, conds, apply_options={}, isolated=False):
  if 'gligen_video' not in conds:
    return None
  indexes = apply_options['indexes']
  grid_size = apply_options['grid_size']

  patches = []

  gligen = conds['gligen_video']
  gligen_model = gligen[0].model
  cond_pos = gligen[1]
  if grid_size is None or grid_size == 1:
    gligen_patch = set_position(
      gligen_model, input_x.shape, cond_pos, indexes, input_x.device, isolated)
  elif grid_size > 1:
    gligen_patch = set_grid_position(
      gligen_model, input_x.shape, cond_pos, indexes, grid_size, input_x.device, isolated)
  patches += [gligen_patch]

  return patches


def set_position(model, latent_image_shape, cond_pos, indexes, device, isolated=False):
  batch, c, h, w = latent_image_shape
  # masks = (batch, object) ... 1.0 for not masked
  # boxes = (batch, object, (x1,y1,x2,y2))
  # conds = (batch, object, cond)
  masks = torch.zeros((batch, model.max_objs), device="cpu")
  boxes = torch.zeros((batch, model.max_objs, 4), device="cpu")
  conds = torch.zeros((batch, model.max_objs, model.key_dim), device="cpu")

  for idx_object, c in enumerate(cond_pos):
    cond_pooled = c['cond_pooled']
    positions = c['positions']
    for idx_batch, idx_position in enumerate(indexes):
      p = positions[idx_position]
      if p is None:
        continue
      masks[idx_batch][idx_object] = 1.0
      x1, y1, x2, y2, _, _ = p
      x1 = x1 // 8 / w
      x2 = x2 // 8 / w
      y1 = y1 // 8 / h
      y2 = y2 // 8 / h
      boxes[idx_batch][idx_object] = torch.tensor((x1, y1, x2, y2))
      conds[idx_batch][idx_object] = cond_pooled

  if not isolated:
    masks = torch.cat(
      [torch.zeros((batch, model.max_objs), device="cpu"), masks])
    boxes = torch.cat([torch.zeros(
      (batch, model.max_objs, 4), device="cpu"), boxes])
    conds = torch.cat([torch.zeros(
      (batch, model.max_objs, model.key_dim), device="cpu"), conds])
  model = model.to('cuda')
  model.position_net = model.position_net.to('cuda')

  return model._set_position(
      boxes.to(device),
      masks.to(device),
      conds.to(device))


def set_grid_position(model, latent_image_shape, cond_pos, indexes, grid_size, device, isolated=False):
  batch, c, h, w = latent_image_shape
  grid_count = grid_size * grid_size
  masks = torch.zeros((batch, model.max_objs), device="cpu")
  boxes = torch.zeros((batch, model.max_objs, 4), device="cpu")
  conds = torch.zeros((batch, model.max_objs, model.key_dim), device="cpu")

  for object_id, c in enumerate(cond_pos):
    cond_pooled = c['cond_pooled']
    positions = c['positions']
    for idx_frame in range(batch):
      for idx in range(grid_count):
        idx_grid_space = idx - (idx_frame * grid_count)
        idx_object = idx_grid_space + (object_id * grid_count)
        p = positions[indexes[idx + (idx_frame * grid_count)]]
        if p is None:
          continue
        column = idx // grid_size
        row = idx % grid_size
        x1, y1, x2, y2, space_w, space_h = p
        x1 = (x1 + column * space_w) // 8 / w
        x2 = (x2 + column * space_w) // 8 / w
        y1 = (y1 + row * space_h) // 8 / h
        y2 = (y2 + row * space_h) // 8 / h
        boxes[idx_frame][idx_object] = torch.tensor((x1, y1, x2, y2))
        conds[idx_frame][idx_object] = cond_pooled
        masks[idx_frame][idx_object] = 1.0

  if not isolated:
    masks = torch.cat(
      [torch.zeros((batch, model.max_objs), device="cpu"), masks])
    boxes = torch.cat([torch.zeros(
      (batch, model.max_objs, 4), device="cpu"), boxes])
    conds = torch.cat([torch.zeros(
      (batch, model.max_objs, model.key_dim), device="cpu"), conds])
  model = model.to('cuda')
  model.position_net = model.position_net.to('cuda')

  return model._set_position(
      boxes.to(device),
      masks.to(device),
      conds.to(device))
