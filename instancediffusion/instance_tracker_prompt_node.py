from ..shared.text_utils import extract_prompts
from .instance_utils import prepare_embeddings


class InstanceTrackerPromptNode:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {"conditioning": ("CONDITIONING", ),
                         "unconditioning": ("CONDITIONING", ),
                         "clip": ("CLIP", ),
                         "tracking": ("TRACKING", ),
                         "instance_models": ("INSTANCE_MODELS", ),
                         "n_frames": ("INT", {"default": 1, "min": 1, "max": 10000}),
                         "img_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                         "img_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                         "alpha": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                         "text": ("STRING", {"multiline": True}),
                         }}
  RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
  RETURN_NAMES = ("positive", "negative")
  FUNCTION = "append"

  CATEGORY = "conditioning/instance"
  

  def append(self, conditioning, unconditioning, clip, tracking, instance_models, n_frames, img_width, img_height, alpha, text):
    prompt_pairs = extract_prompts(text)


    # prompts = [
    #   "a dragon", "a castle", "a knight", "a river bank"
    # ]
    # condz = []
    
    # for prompt in prompts:
    #   _, cond_pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
    #   position_cond = {'cond_pooled': cond_pooled, 'positions': {}}
    #   condz.append(position_cond)
    # embeddings = prepare_embeddings(condz, n_frames, img_width, img_height, True)  # TODO: use_attn_mask?
    grounding_tokenizer_input = instance_models['grounding_tokenizer_input']


    position_conds = []
    for tracker_id, class_id, prompt in prompt_pairs:
      _, cond_pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
      if tracker_id != -1:
        position_cond = {'cond_pooled': cond_pooled, 'positions': 
                              tracking[class_id][tracker_id]}
        position_conds.append(position_cond)
      else:
        for tracker_id in tracking[class_id]:
          position_cond = {'cond_pooled': cond_pooled, 'positions': tracking[class_id][tracker_id]}
          position_conds.append(position_cond)

    embeddings = prepare_embeddings(position_conds, n_frames, img_width, img_height, True)  # TODO: use_attn_mask?
    embeddings = grounding_tokenizer_input.prepare(embeddings, return_att_masks=True)
    cond_out = []
    for t in conditioning:
      n = [t[0], t[1].copy()]
      prev = []
      if "instance_diffusion" in n[1]:
        prev = n[1]['instance_diffusion']['embeddings']

      n[1]['instance_diffusion'] = {
        'instance_models': instance_models, 
        'alpha': alpha, 
        'embeddings': prev + [embeddings],
        'grounding_tokenizer_input': grounding_tokenizer_input,
        'n_frames': n_frames
        }
      cond_out.append(n)

    uncond_embeddings = grounding_tokenizer_input.get_null_input(n_frames, img_width//8, img_height//8)
    uncond_out = []
    for t in unconditioning:
      n = [t[0], t[1].copy()]
      prev = []
      if "instance_diffusion" in n[1]:
        prev = n[1]['instance_diffusion']['embeddings']

      n[1]['instance_diffusion'] = {
        'instance_models': instance_models, 
        'alpha': alpha, 
        'embeddings': prev + [uncond_embeddings],
        'grounding_tokenizer_input': grounding_tokenizer_input,
        'n_frames': n_frames
        }
      uncond_out.append(n)

    return (cond_out,uncond_out)
