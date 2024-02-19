import torch 

def set_head_to_batch_dim(self):
    def head_to_batch_dim(tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
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
    return head_to_batch_dim

def set_batch_to_head_dim(self):
    def batch_to_head_dim(tensor: torch.Tensor) -> torch.Tensor:
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
    return batch_to_head_dim

conv_call_count = 0
def conv_forward(self, n_cond=1, n_uncond=1):
    def _forward(x, emb):
        global conv_call_count
        conv_call_count += 1
        replicas = (1 + n_cond + n_uncond)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                h = h + emb_out
            h = self.out_layers(h)

        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            source_batch_size = int(h.shape[0] // replicas)
            for i in range(n_uncond):
                # inject unconditional
                h[source_batch_size:(2+i) * source_batch_size] = h[:source_batch_size]
            for i in range(n_cond):
                # inject conditional
                h[(1+n_uncond+i) * source_batch_size:(1+n_uncond+i+1) * source_batch_size] = h[:source_batch_size]

        return self.skip_connection(x) + h
    
    return _forward