
def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

          

        # def forward(self, x, context=None, value=None, mask=None):
        def forward(x, context=None, transformer_options=None):
            encoder_hidden_states = context
            batch_size, sequence_length, dim = x.shape
            # good
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x

            # qkv are good
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            k_source = k[:n_frames]
            k_uncond = k[n_frames: 2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_source = v[:n_frames]
            v_uncond = v[n_frames:2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            # head_to_batch_dim -> needs to be injected
            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames: 2*n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)

            out_source = []
            out_uncond = []
            out_cond = []

            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)
            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

            # scale -> likely set to 1
            for j in range(h):
                sim_source_b = torch.bmm(q_src[:, j], k_src[:, j].transpose(-1, -2)) * self.scale
                sim_uncond_b = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                sim_cond = torch.bmm(q_cond[:, j], k_cond[:, j].transpose(-1, -2)) * self.scale

                out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[:, j]))
                out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[:, j]))
                out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[:, j]))

            out_source = torch.cat(out_source, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_cond = torch.cat(out_cond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)

            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    
    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            # -> add this
            module.attn1.head_to_batch_dim = head_to_batch_dim
            module.attn1.batch_to_head_dim = batch_to_head_dim

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
    # sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
    
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            # module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            module.head_to_batch_dim = head_to_batch_dim
            module.batch_to_head_dim = batch_to_head_dim
