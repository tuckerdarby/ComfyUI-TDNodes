import torch

from .utils import isinstance_str
from .tokenflow_attention import sa_forward
from .tokenflow_block import make_tokenflow_block
from .tokenflow_conv import set_batch_to_head_dim, set_head_to_batch_dim, conv_forward


def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)


def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_time(model, t):
    conv_module = model.output_blocks[3*1+1][0]
    setattr(conv_module, 't', t)
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn2
            setattr(module, 't', t)
    
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.input_blocks[3*res+block+1][1].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.input_blocks[3*res+block+1][1].transformer_blocks[0].attn2
            setattr(module, 't', t)
            
    module = model.middle_block[1].transformer_blocks[0].attn1
    setattr(module, 't', t)
    
    module = model.middle_block[1].transformer_blocks[0].attn2
    setattr(module, 't', t)


def register_conv_injection(model, injection_schedule, n_cond, n_uncond):
    conv_module = model.output_blocks[3*1+1][0]
    conv_module.orig_forward = conv_module.forward
    conv_module.forward = conv_forward(conv_module, n_cond, n_uncond)
    setattr(conv_module, 'injection_schedule', injection_schedule)


def deregister_conv_injection(model):
    conv_module = model.output_blocks[3*1+1][0]
    conv_module.forward = conv_module.orig_forward


def register_extended_attention_pnp(model, injection_schedule, n_cond, n_uncond):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.orig_forward = module.attn1.forward
            module.attn1.forward = sa_forward(module.attn1, n_cond, n_uncond)
            module.attn1.head_to_batch_dim = set_head_to_batch_dim(module.attn1)
            module.attn1.batch_to_head_dim = set_batch_to_head_dim(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            module.orig_forward = module.forward
            module.forward = sa_forward(module)
            module.head_to_batch_dim = set_head_to_batch_dim(module)
            module.batch_to_head_dim = set_batch_to_head_dim(module)
            setattr(module, 'injection_schedule', injection_schedule)


def deregister_extended_attention_pnp(model):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = module.attn1.orig_forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            module.forward = module.orig_forward


def register_tokenflow_blocks(model: torch.nn.Module, n_cond, n_uncond):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.orig_forward = module.forward
            module.forward = make_tokenflow_block(module, n_cond, n_uncond)

    return model


def deregister_tokenflow_blocks(model: torch.nn.Module):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.forward = module.orig_forward

    return model