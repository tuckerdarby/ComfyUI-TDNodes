import torch


def sa_forward(self, n_cond=1, n_uncond=1):
    # to_out is good
    to_out = self.to_out
    if type(to_out) is torch.nn.modules.container.ModuleList:
        to_out = self.to_out[0]
    else:
        to_out = self.to_out

    # to_out = self.to_out[0]

    def forward2(x, context=None, value=None, mask=None):
        encoder_hidden_states = context
        batch_size, sequence_length, dim = x.shape
        h = self.heads
        n_frames = batch_size // 3
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else x
        q = self.to_q(x)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        b, _, dim_head = q.shape
        dim_head //= h
        scale = dim_head ** -0.5

        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            # inject unconditional
            q[n_frames:2 * n_frames] = q[:n_frames]
            k[n_frames:2 * n_frames] = k[:n_frames]
            # inject conditional
            q[2 * n_frames:] = q[:n_frames]
            k[2 * n_frames:] = k[:n_frames]

        k_source = k[:n_frames]
        k_uncond = k[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
        k_cond = k[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

        v_source = v[:n_frames]
        v_uncond = v[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
        v_cond = v[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

        q_source = self.head_to_batch_dim(q[:n_frames])
        q_uncond = self.head_to_batch_dim(q[n_frames:2 * n_frames])
        q_cond = self.head_to_batch_dim(q[2 * n_frames:])
        k_source = self.head_to_batch_dim(k_source)
        k_uncond = self.head_to_batch_dim(k_uncond)
        k_cond = self.head_to_batch_dim(k_cond)
        v_source = self.head_to_batch_dim(v_source)
        v_uncond = self.head_to_batch_dim(v_uncond)
        v_cond = self.head_to_batch_dim(v_cond)

        q_src = q_source.view(n_frames, h, sequence_length, dim // h)
        k_src = k_source.view(n_frames, h, sequence_length, dim // h)
        v_src = v_source.view(n_frames, h, sequence_length, dim // h)
        q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
        k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
        v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
        q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
        k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
        v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

        out_source_all = []
        out_uncond_all = []
        out_cond_all = []
        
        single_batch = n_frames<=12
        b = n_frames if single_batch else 1

        for frame in range(0, n_frames, b):
            out_source = []
            out_uncond = []
            out_cond = []
            for j in range(h):
                sim_source_b = torch.bmm(q_src[frame: frame+ b, j], k_src[frame: frame+ b, j].transpose(-1, -2)) * scale
                sim_uncond_b = torch.bmm(q_uncond[frame: frame+ b, j], k_uncond[frame: frame+ b, j].transpose(-1, -2)) * scale
                sim_cond = torch.bmm(q_cond[frame: frame+ b, j], k_cond[frame: frame+ b, j].transpose(-1, -2)) * scale

                out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[frame: frame+ b, j]))
                out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame+ b, j]))
                out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame+ b, j]))

            out_source = torch.cat(out_source, dim=0)
            out_uncond = torch.cat(out_uncond, dim=0) 
            out_cond = torch.cat(out_cond, dim=0) 
            if single_batch:
                out_source = out_source.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_uncond = out_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_cond = out_cond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_source_all.append(out_source)
            out_uncond_all.append(out_uncond)
            out_cond_all.append(out_cond)
        
        out_source = torch.cat(out_source_all, dim=0)
        out_uncond = torch.cat(out_uncond_all, dim=0)
        out_cond = torch.cat(out_cond_all, dim=0)
            
        out = torch.cat([out_source, out_uncond, out_cond], dim=0)
        out = self.batch_to_head_dim(out)

        return to_out(out)

    def forward(x, context=None, value=None, mask=None):
        # x.shape (B*3, 135, 1280)
        encoder_hidden_states = context
        
        batch_size, sequence_length, dim = x.shape
        h = self.heads
        n_frames = batch_size // (1 + n_cond + n_uncond)  # RAVE: frames/grid ?
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else x

        q = self.to_q(x) # shape (B*3, 135, 1280)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        b, _, dim_head = q.shape
        dim_head //= h
        scale = dim_head ** -0.5

        # check if t is injected
        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t >= 999):
            for idx in range(n_cond + n_uncond):
              q[n_frames + (n_frames*idx):n_frames + n_frames*(idx+1)] = q[:n_frames]
              k[n_frames + (n_frames*idx):n_frames + n_frames*(idx+1)] = k[:n_frames]

        single_batch =  n_frames<=12  # RAVE: needs to change to account for grid?
        b = n_frames if single_batch else 1
        out = []

        for idx in range(1 + n_cond + n_uncond):
            start, end = n_frames*idx, n_frames*(idx+1)

            k_chunk = k[start:end]
            v_chunk = v[start:end]

            if idx > 0:
                k_chunk = k_chunk.reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
                v_chunk = v_chunk.reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            k_chunk =  self.head_to_batch_dim(k_chunk)
            v_chunk =  self.head_to_batch_dim(v_chunk)
            q_chunk =  self.head_to_batch_dim(q[start:end])
            
            repeat_count = n_frames if idx > 0 else 1

            k_chunk = k_chunk.view(n_frames, h, sequence_length * repeat_count, dim // h)
            v_chunk = v_chunk.view(n_frames, h, sequence_length * repeat_count, dim // h)
            q_chunk = q_chunk.view(n_frames, h, sequence_length, dim // h)
        
            for frame in range(0, n_frames, b):
                output_frame = []
                for j in range(h):
                    sim_b = torch.bmm(q_chunk[frame: frame + b, j], k_chunk[frame: frame+ b, j].transpose(-1, -2)) * scale
                    output_frame.append(torch.bmm(sim_b.softmax(dim=-1), v_chunk[frame: frame+ b, j]))

                output_frame = torch.cat(output_frame, dim=0)
                if single_batch:
                    output_frame = output_frame.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out.append(output_frame)
        
        out = torch.cat(out, dim=0)
        out = self.batch_to_head_dim(out)

        return to_out(out) # (B*3, 135, 1280)

    return forward