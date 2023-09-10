from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm


class DiffusionInferencePipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, initial_noise, conditioner=None, n_inference_steps=1000):
        r"""
        Args:
        initial_noise (`torch.Tensor`):
            The initial noise to be denoised.
        conditioner (`torch.Tensor`, *optional*, defaults to `None`):
            The conditioning image.
        n_inference_steps (`int`, *optional*, defaults to 1000):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        """

        mel = initial_noise
        batch_size = mel.size(0)
        self.scheduler.set_timesteps(n_inference_steps)

        for t in tqdm(self.scheduler.timesteps, desc="Denoising", leave=False):
            timestep = torch.full(
                (batch_size, 1), t, device=mel.device, dtype=torch.long
            )

            # 1. predict noise model_output
            model_output = self.unet(mel, timestep, conditioner)

            # 2. denoise, compute previous step: x_t -> x_t-1
            mel = self.scheduler.step(model_output, t, mel).prev_sample

            # 3. clamp
            mel = mel.clamp(-1.0, 1.0)

        return mel
