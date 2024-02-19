from ..shared.text_utils import extract_prompts


class VideoTrackerPromptNode:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {"global_conditioning": ("CONDITIONING", ),
                         "clip": ("CLIP", ),
                         "tracking": ("TRACKING", ),
                         "gligen_textbox_model": ("GLIGEN", ),
                         "text": ("STRING", {"multiline": True}),
                         }}
  RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
  RETURN_NAMES = ("global", "regional")
  FUNCTION = "append"

  CATEGORY = "conditioning/gligen"

  def append(self, global_conditioning, clip, tracking, gligen_textbox_model, text):
    global_out = []
    regional_out = []
    prompt_pairs = extract_prompts(text)

    position_conds = []

    for tracker_id, class_id, prompt in prompt_pairs:
      cond, cond_pooled = clip.encode_from_tokens(
        clip.tokenize(prompt), return_pooled=True)
      if tracker_id != -1:
        position_cond = {'cond_pooled': cond_pooled, 'positions': 
                              tracking[class_id][tracker_id]}
        position_conds.append(position_cond)
        regional_out.append([cond, {"positions": position_cond["positions"], "pooled_output": cond_pooled, "gligen_video": (gligen_textbox_model, [position_cond])}])
      else:
        for tracker_id in tracking[class_id]:
          position_cond = {'cond_pooled': cond_pooled, 'positions': tracking[class_id][tracker_id]}
          position_conds.append(position_cond)
          regional_out.append([cond, {"positions": position_cond["positions"],"pooled_output": cond_pooled, "gligen_video": (gligen_textbox_model, [position_cond])}])

    for t in global_conditioning:
      n = [t[0], t[1].copy()]
      prev = []
      if "gligen_video" in n[1]:
        prev = n[1]['gligen_video'][2]

      n[1]['gligen_video'] = (gligen_textbox_model, prev + position_conds)
      global_out.append(n)

    return (global_out, regional_out)
