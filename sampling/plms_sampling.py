import torch
import numpy as np
from tqdm import trange
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule, make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from .sampling_utils import alpha_generator


def get_alphacumprod(sigma):
    return 1 / ((sigma * sigma) + 1)


class PLMSSampler(object):
    def __init__(self, model, schedule="linear", device="cuda", add_noise=True):
        super().__init__()
        self.model = model
        self.device = device
        self.add_noise = add_noise
        self.ddpm_num_timesteps = 1000
        self.schedule = schedule
        sampling_settings = model.model_config.sampling_settings
        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)
        cosine_s=8e-3
        self.all_sigmas = model.model_sampling.sigmas
        self.betas = torch.tensor(make_beta_schedule(beta_schedule, 1000, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s))

    def set_alpha_scale(self, extra_args, alpha):
        for cond in (extra_args['cond'] + extra_args['uncond']):
            if 'instance_diffusion' in cond:
                for block in cond['instance_diffusion']['instance_models']['fusers']:
                    for idx in cond['instance_diffusion']['instance_models']['fusers'][block]:
                        cond['instance_diffusion']['instance_models']['fusers'][block][idx].scale = alpha

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, sigmas, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        ddim_num_steps = len(sigmas) - 1 # 50
        
        # Timesteps -- 50, should be 1 to 981
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        
        
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)
        acp = [get_alphacumprod(sigma) for sigma in self.all_sigmas]
        alphas_cumprod = torch.tensor(acp)
        
        alphas_cumprod_prev = torch.tensor([1.0] + acp[:-1])  # TODO find a better way here bud
        #1000 [] 0.0009, ... -- these are correct
        self.register_buffer('betas', to_torch(self.betas))
        #1000 [] 0.9992, 0.9983, 0.9974 -- direction is correct
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        try: # 1000 # yeah... 1.0 ... 0.9966 -- 
            self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        except:
            pass
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        
        self.register_buffer('ddim_sigmas', ddim_sigmas) # all zeros
        self.register_buffer('ddim_alphas', ddim_alphas) # 53 [] 0.9983, 0.9813, 0.9629 -- good
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev) # 53 [] 0.99915, 0.9983.. -- good
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        try:
            sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
                (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                            1 - self.alphas_cumprod / self.alphas_cumprod_prev))
            self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
        except:
            pass


    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None):
        self.make_schedule(ddim_num_steps=S)
        return self.plms_sampling(shape, input, uc, guidance_scale, mask=mask, x0=x0)


    @torch.no_grad()
    def plms_sampling(self, model, x, sigmas, config, eta=0.):
        # 

        # latent_shape, so batches
        b = x.shape[0]

        time_range = np.flip(self.ddim_timesteps) # [989, ...] -- good
        # just the number of steps -- odd way to get it
        total_steps = self.ddim_timesteps.shape[0] # 50

        old_eps = []

        # with 50 steps, the first 40 are 1, the rest are 0
        # if self.alpha_generator_func != None:
        alphas = alpha_generator(len(time_range), [0.8, 0.0, 0.2]) # 50 [] 40 ones, 10 zeros

        for run_idx in trange(len(sigmas) - 1, disable=config.disable):
            i = run_idx #config.step # 0....
            # print(config.step)
            # set alpha and restore first conv layer 
            # this function goes through all the Gated models and sets the 'scale' to alpha for that step
            self.set_alpha_scale(config.extra_args, alphas[i]) # 1 -- good
            # if  alphas[i] == 0:
            #     self.model.restore_first_conv_from_SD() # TODO does htis get hit?

            # run 
            # index is reveresed 
            index = total_steps - i - 1  # 49 good

            x, e_t = self.p_sample_plms(model, x, config, index=index,idx=i, old_eps=old_eps)
            
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

        return x


    @torch.no_grad()
    def p_sample_plms(self, model, x, config, index, idx, old_eps=None):
        # index = 52 -- should be 49
        # uc appears to be "uncond" to start
        # guidance_scale is cfg
        # old_eps = [] in first step
        b = x.shape[0]

        def get_x_prev_and_pred_x0(e_t, index):
            # this is the reversed index -- e.g. when i=0, index=49, with steps=50

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            # ddim_sqrt_one_minus_alphas = eta * sigmas ?
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt # + noise
            if self.add_noise:
              noise = sigma_t * torch.randn_like(x)
              x_prev += noise
            return x_prev, pred_x0
        
        s_in = x.new_ones([x.shape[0]])
        e_t = model(x, config.sigmas[idx] * s_in, **config.extra_args)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            x = x_prev
            e_t_next = model(x, config.sigmas[min(idx+1, len(config.sigmas))] * s_in, **config.extra_args)  # TODO +1 here
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, _ = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, e_t


def apply_mode_plms(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
  sigma = t
  xc = x
  # xc = self.model_sampling.calculate_input(sigma, x)
  if c_concat is not None:
    xc = torch.cat([xc] + [c_concat], dim=1)

  context = c_crossattn
  dtype = self.get_dtype()

  if self.manual_cast_dtype is not None:
    dtype = self.manual_cast_dtype

  xc = xc.to(dtype)
  t = self.model_sampling.timestep(t).float()
  context = context.to(dtype)
  extra_conds = {}
  for o in kwargs:
    extra = kwargs[o]
    if hasattr(extra, "dtype"):
      if extra.dtype != torch.int and extra.dtype != torch.long:
        extra = extra.to(dtype)
    extra_conds[o] = extra

  model_output = self.diffusion_model(
      xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
  return model_output