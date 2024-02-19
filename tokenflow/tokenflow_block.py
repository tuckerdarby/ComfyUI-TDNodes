import torch

from comfy.ldm.modules.attention import BasicTransformerBlock

from .utils import batch_cosine_sim


def make_tokenflow_block(self, n_cond=1, n_uncond=1):
    def forward(
        hidden_states,
        context=None,
        transformer_options=None,
    ) -> torch.Tensor:
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        replicas = (1 + n_cond + n_uncond)
        batch_size, sequence_length, dim = hidden_states.shape
        n_frames = batch_size // replicas
        hidden_states = hidden_states.view(replicas, n_frames, sequence_length, dim)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states.view(replicas, n_frames, sequence_length, dim)
        
        if self.pivotal_pass:
            self.pivot_hidden_states = norm_hidden_states  # shape = (3, 5, 2040, 320)
        else:
            idx1 = []
            idx2 = [] 
            batch_idxs = [self.batch_idx]
            if self.batch_idx > 0:
                batch_idxs.append(self.batch_idx - 1)
            
            sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim),
                                    self.pivot_hidden_states[0][batch_idxs].reshape(-1, dim))
            if len(batch_idxs) == 2:
                sim1, sim2 = sim.chunk(2, dim=1)
                # sim: n_frames * seq_len, len(batch_idxs) * seq_len
                idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
                idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
            else:
                idx1.append(sim.argmax(dim=-1))
            idx1 = torch.stack(idx1 * replicas, dim=0) # 3, n_frames * seq_len
            idx1 = idx1.squeeze(1)
            if len(batch_idxs) == 2:
                idx2 = torch.stack(idx2 * replicas, dim=0) # 3, n_frames * seq_len
                idx2 = idx2.squeeze(1)

        # 1. Self-Attention
        if self.pivotal_pass:
            self.attn_output = self.attn1(
                    norm_hidden_states.view(batch_size, sequence_length, dim),
                    context=context if self.disable_self_attn else None
                )
            self.kf_attn_output = self.attn_output 
        else:
            batch_kf_size, _, _ = self.kf_attn_output.shape  # (B*3, 2040, 320)
            # 3, n_frames, seq_len, dim --> 3, len(batch_idxs), seq_len, dim
            self.attn_output = self.kf_attn_output.view(replicas, batch_kf_size // replicas, sequence_length, dim)[:, batch_idxs]  



        # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
        if not self.pivotal_pass:
            if len(batch_idxs) == 2:
                attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))

                s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
                # distance from the pivot
                p1 = batch_idxs[0] * n_frames + n_frames // 2
                p2 = batch_idxs[1] * n_frames + n_frames // 2
                d1 = torch.abs(s - p1)
                d2 = torch.abs(s - p2)
                # weight
                w1 = d2 / (d1 + d2)
                w1 = torch.sigmoid(w1)
                
                w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(replicas, 1, sequence_length, dim)
                attn_output1 = attn_output1.view(replicas, n_frames, sequence_length, dim)
                attn_output2 = attn_output2.view(replicas, n_frames, sequence_length, dim)
                attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
            else:
                attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim)) # (3,1,2040,320) -> (3,16320,320)

            attn_output = attn_output.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim (24, 2040, 320)
        else:
            attn_output = self.attn_output
        hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
        hidden_states = attn_output + hidden_states
        if "middle_patch" in transformer_patches:
          patch = transformer_patches["middle_patch"]
          for p in patch:
              hidden_states = p(hidden_states, extra_options)


        # 2. Cross-Attention
        if self.attn2 is not None:
            if (hidden_states.dtype == torch.float32):
                # TODO: why is this needed?
                hidden_states = hidden_states.to(torch.float16)
            attn_output = self.attn2(
                self.norm2(hidden_states),
                context=context
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        
        skip = hidden_states if self.is_res else 0
        hidden_states = self.ff(norm_hidden_states) + hidden_states  # validate not skip

        return hidden_states
    
    return forward

    # def forward2(
    #     hidden_states,
    #     context=None,
    #     transformer_options=None,
    # ) -> torch.Tensor:
    #     # 3 = (1 + n_cond + n_uncond)
    #     batch_size, sequence_length, dim = hidden_states.shape
    #     n_frames = batch_size // 3
    #     hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

    #     norm_hidden_states = self.norm1(hidden_states)
    #     norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
        
    #     if self.pivotal_pass:
    #         self.pivot_hidden_states = norm_hidden_states
    #     else:
    #         idx1 = []
    #         idx2 = [] 
    #         batch_idxs = [self.batch_idx]
    #         if self.batch_idx > 0:
    #             batch_idxs.append(self.batch_idx - 1)
            
    #         sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim),
    #                                 self.pivot_hidden_states[0][batch_idxs].reshape(-1, dim))
    #         if len(batch_idxs) == 2:
    #             sim1, sim2 = sim.chunk(2, dim=1)
    #             # sim: n_frames * seq_len, len(batch_idxs) * seq_len
    #             idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
    #             idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
    #         else:
    #             idx1.append(sim.argmax(dim=-1))
    #         idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
    #         idx1 = idx1.squeeze(1)
    #         if len(batch_idxs) == 2:
    #             idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
    #             idx2 = idx2.squeeze(1)

    #     # 1. Self-Attention
    #     if self.pivotal_pass:
    #         self.attn_output = self.attn1(
    #                 norm_hidden_states.view(batch_size, sequence_length, dim),
    #                 context=context if self.disable_self_attn else None
    #             )
    #         self.kf_attn_output = self.attn_output 
    #     else:
    #         batch_kf_size, _, _ = self.kf_attn_output.shape
    #         # 3, n_frames, seq_len, dim --> 3, len(batch_idxs), seq_len, dim
    #         self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:, batch_idxs]  

    #     # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
    #     if not self.pivotal_pass:
    #         if len(batch_idxs) == 2:
    #             attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
    #             attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
    #             attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))

    #             s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
    #             # distance from the pivot
    #             p1 = batch_idxs[0] * n_frames + n_frames // 2
    #             p2 = batch_idxs[1] * n_frames + n_frames // 2
    #             d1 = torch.abs(s - p1)
    #             d2 = torch.abs(s - p2)
    #             # weight
    #             w1 = d2 / (d1 + d2)
    #             w1 = torch.sigmoid(w1)
                
    #             w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
    #             attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
    #             attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
    #             attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
    #         else:
    #             attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))

    #         attn_output = attn_output.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
    #     else:
    #         attn_output = self.attn_output
    #     hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
    #     hidden_states = attn_output + hidden_states

    #     # 2. Cross-Attention
    #     if self.attn2 is not None:
    #         if (hidden_states.dtype == torch.float32):
    #             # TODO: why is this needed?
    #             hidden_states = hidden_states.to(torch.float16)
    #         attn_output = self.attn2(
    #             self.norm2(hidden_states),
    #             context=context
    #         )
    #         hidden_states = attn_output + hidden_states

    #     # 3. Feed-forward
    #     norm_hidden_states = self.norm3(hidden_states)
        
    #     skip = hidden_states if self.is_res else 0
    #     hidden_states = self.ff(norm_hidden_states) + hidden_states  # validate not skip

    #     return hidden_states
    
    # return forward