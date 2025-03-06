import einops
import torch

from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlNet(LatentDiffusion):
    """
    A 'UniControlNet' stripped down to "local" mode only (no global adaptor).
    """

    def __init__(self, mode, local_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We assume 'mode' must be 'local', since global or uni are not used.
        assert mode == 'local', "This modified version only supports 'local' mode."
        self.mode = mode

        # Instantiate the local adapter
        self.local_adapter = instantiate_from_config(local_control_config)
        # local_control_scales typically for applying scale factors at each U-Net level:
        self.local_control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        Overwrites base get_input. We only process:
          - the main image (for latents)
          - the text prompt
          - local conditions (no global)
        """
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # Prepare local conditions
        if len(batch['local_conditions']) != 0:
            local_conditions = batch['local_conditions']
            if bs is not None:
                local_conditions = local_conditions[:bs]
            local_conditions = local_conditions.to(self.device)
            # Re-arrange from (B,H,W,C) -> (B,C,H,W)
            local_conditions = einops.rearrange(local_conditions, 'b h w c -> b c h w')
            local_conditions = local_conditions.to(memory_format=torch.contiguous_format).float()
        else:
            local_conditions = torch.zeros(1,1,1,1).to(self.device).float()

        # Return the usual dictionary with cross-attn tokens + local control
        return x, dict(c_crossattn=[c], local_control=[local_conditions])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        The main forward pass of the diffusion model for one timestep.
        This version includes only local adapter logic.
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # Text embeddings
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # Local control
        local_control = torch.cat(cond['local_control'], 1)
        local_control = self.local_adapter(
            x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control
        )
        # Apply scaling
        local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]

        # Forward U-Net with local control
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        """
        For classifier-free guidance, returns embeddings for empty prompts.
        """
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0,
                   plot_denoise_rows=False, plot_diffusion_rows=False,
                   unconditional_guidance_scale=9.0, **kwargs):
        """
        Visualization / logging for training or validation steps.
        This is specialized for local-only.
        """
        use_ddim = ddim_steps is not None
        log = dict()

        # Grab data
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        # c_cat are local condition images
        c_cat = c["local_control"][0][:N]
        # c is text embeddings
        c = c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        # 1) Reconstruction of the input images
        log["reconstruction"] = self.decode_first_stage(z)
        # 2) Local condition images (scaled from [0,1] to [-1,1] for consistent logging)
        log["local_control"] = c_cat * 2.0 - 1.0
        # 3) Text prompt as an image
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        # Optionally log forward-diffusion noised images
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t_ = repeat(torch.tensor([t]), '1 -> b', b=n_row).to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t_, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        # Optionally sample new images from random noise
        if sample:
            cond_dict = {"local_control": [c_cat], "c_crossattn": [c]}
            samples, z_denoise_row = self.sample_log(
                cond=cond_dict, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        # Classifier-Free Guidance
        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            # We'll pass empty local conditions for unconditional
            uc_cat = torch.zeros_like(c_cat)
            uc_full = {"local_control": [uc_cat], "c_crossattn": [uc_cross]}
            cond_in = {"local_control": [c_cat], "c_crossattn": [c]}
            samples_cfg, _ = self.sample_log(
                cond=cond_in, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        """
        Simple DDIM-based sampling.
        """
        ddim_sampler = DDIMSampler(self)
        # For local-only, we can derive the height/width from the local_condition shape
        _, _, h, w = cond["local_control"][0].shape
        shape = (self.channels, h // 8, w // 8)

        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
        )
        return samples, intermediates

    def configure_optimizers(self):
        """
        Only optimize the local adapter (plus the final SD U-Net output blocks if not locked).
        """
        lr = self.learning_rate
        params = []

        # Always local only
        params += list(self.local_adapter.parameters())

        # If the main UNet is unlocked, also optimize those final layers
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        """
        Moves certain modules to/from GPU for low VRAM usage, if needed.
        """
        if is_diffusing:
            self.model = self.model.cuda()
            self.local_adapter = self.local_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.local_adapter = self.local_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
