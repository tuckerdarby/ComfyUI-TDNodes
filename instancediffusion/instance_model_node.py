import folder_paths
import torch
import os

from .modules.attention import GatedSelfAttentionDense
from .modules.scaleu import ScaleU
from .modules.text_grounding_net import UniFusion
from .modules.text_grounding_tokenizer_input import GroundingNetInput

folder_paths.folder_names_and_paths['InstanceDiffusion'] = os.path.join(folder_paths.models_dir, 'InstanceDiffusion')

def get_position_net_config():
    return {
        "params": {
          "in_dim": 768,
          "mid_dim": 3072,
          "out_dim": 768,
          "test_drop_boxes": False,
          "test_drop_masks": True,
          "test_drop_points": False,
          "test_drop_scribbles": True,
          "train_add_boxes": True,
          "train_add_masks": True,
          "train_add_points": True,
          "train_add_scribbles": True,
          "use_seperate_tokenizer": True,
        },
        "target": "ldm.modules.diffusionmodules.text_grounding_net.UniFusion"
    }


class InstanceDiffusionLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
      return {"required": {
                         "text": ("STRING", {"multiline": True}),
                         }}

    RETURN_TYPES = ("INSTANCE_MODELS",)
    FUNCTION = "load_instance_models"

    CATEGORY = "loaders"

    def load_instance_models(self, text):
        fusers_path = os.path.join(folder_paths.folder_names_and_paths['InstanceDiffusion'], 'fusers.ckpt')
        fusers_ckpt = torch.load(fusers_path)
        fusers_list = []
        for key in fusers_ckpt['input_blocks']:
            fusers_ckpt['input_blocks'][key]['params']['query_dim'] = fusers_ckpt['input_blocks'][key]['params']['n_heads'] * fusers_ckpt['input_blocks'][key]['params']['d_head']
            fuser = GatedSelfAttentionDense(**fusers_ckpt['input_blocks'][key]['params'])
            fuser.load_state_dict(fusers_ckpt['input_blocks'][key]['state'])
            fusers_list.append(fuser)

        
        fusers_ckpt['middle_block']['1']['params']['query_dim'] = fusers_ckpt['middle_block']['1']['params']['n_heads'] * fusers_ckpt['middle_block']['1']['params']['d_head']
        fuser = GatedSelfAttentionDense(**fusers_ckpt['middle_block']['1']['params'])
        fuser.load_state_dict(fusers_ckpt['middle_block']['1']['state'])
        fusers_list.append(fuser)

        for key in fusers_ckpt['output_blocks']:
            fusers_ckpt['output_blocks'][key]['params']['query_dim'] = fusers_ckpt['output_blocks'][key]['params']['n_heads'] * fusers_ckpt['output_blocks'][key]['params']['d_head']
            fuser = GatedSelfAttentionDense(**fusers_ckpt['output_blocks'][key]['params'])
            fuser.load_state_dict(fusers_ckpt['output_blocks'][key]['state'])
            fusers_list.append(fuser)

        scaleu_path = os.path.join(folder_paths.folder_names_and_paths['InstanceDiffusion'], "scaleu.ckpt")
        scaleu_ckpt = torch.load(scaleu_path)
        scaleu_nets = []
        for i in range(12):
          ckpt = scaleu_ckpt[f'{i}']
          scaleu = ScaleU(True, len(ckpt['scaleu_b']), len(ckpt['scaleu_s']))
          scaleu.load_state_dict(ckpt)
          scaleu_nets.append(scaleu)
        
        position_net = UniFusion(**get_position_net_config()['params'])
        position_net_path = os.path.join(folder_paths.folder_names_and_paths['InstanceDiffusion'], "position_net.ckpt")
        position_net.load_state_dict(torch.load(position_net_path))

        grounding_tokenizer = GroundingNetInput()
        models = {
            'scaleu_nets': scaleu_nets,
            'position_net': position_net,
            'fusers_list': fusers_list,
            'grounding_tokenizer_input': grounding_tokenizer
        }
        return (models,)