import torch
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from models.svc.diffsvc.diffusion_inference_pipeline import DiffusionInferencePipeline


def diffsvc_inference(cfg, model, batch, inference_method="ddim"):
    conditioner = model[0](batch)

    # TODO: Support designated initial noise
    noise = torch.randn_like(batch["mel"])

    # TODO: DO NOT HARD CODE other arguments, need further improve config
    assert inference_method.lower() in ["ddpm", "ddim", "pndm"]
    if inference_method.lower() == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.model.diffusion.noise_schedule_factors[2]
        )
    elif inference_method.lower() == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=cfg.model.diffusion.noise_schedule_factors[2]
        )
    elif inference_method.lower() == "pndm":
        scheduler = PNDMScheduler(
            num_train_timesteps=cfg.model.diffusion.noise_schedule_factors[2]
        )
    else:
        raise NotImplementedError

    pipeline = DiffusionInferencePipeline(model[1], scheduler)
    res = pipeline(
        noise,
        conditioner,
        n_inference_steps=cfg.model.diffusion.noise_schedule_factors[2],
    )

    return res.cpu()
