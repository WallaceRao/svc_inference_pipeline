import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from collections import deque


denoise_fn = None
posterior_variance = None
posterior_log_variance_clipped = None
alphas_cumprod = None
sqrt_recip_alphas_cumprod = None
sqrt_recipm1_alphas_cumprod = None
posterior_mean_coef1 = None
posterior_mean_coef2 = None

noise_list = None

to_torch = partial(torch.tensor, dtype=torch.float32)


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predict_start_from_noise(x_t, t, noise):
    return (
        extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )


def q_posterior(x_start, x_t, t):
    posterior_mean = (
        extract(posterior_mean_coef1, t, x_t.shape) * x_start
        + extract(posterior_mean_coef2, t, x_t.shape) * x_t
    )
    # TODO: check global variable
    _posterior_variance = extract(posterior_variance, t, x_t.shape)
    _posterior_log_variance_clipped = extract(
        posterior_log_variance_clipped, t, x_t.shape
    )
    return posterior_mean, _posterior_variance, _posterior_log_variance_clipped


def p_mean_variance(x, t, cond, clip_denoised: bool):
    """
    Args:
        x: (B, 1, n_mels, T)
        diffusion module的默认输入: (B, T, n_mels)
        diffusion module的默认输出: (B, T, n_mels)

    """
    # print("input x:", x.shape)
    # print("input t:", t.shape)
    noise_pred, stats = denoise_fn(x.transpose(-1, -2).squeeze(1), cond, t.unsqueeze(1))
    x_recon = predict_start_from_noise(
        x, t=t, noise=noise_pred.transpose(-1, -2).unsqueeze(1)
    )

    if clip_denoised:
        x_recon.clamp_(-1.0, 1.0)

    model_mean, _posterior_variance, _posterior_log_variance = q_posterior(
        x_start=x_recon, x_t=x, t=t
    )
    return model_mean, _posterior_variance, _posterior_log_variance, stats


def p_sample(x, t, cond, clip_denoised=True, repeat_noise=False):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance, stats = p_mean_variance(
        x=x, t=t, cond=cond, clip_denoised=clip_denoised
    )
    noise = noise_like(x.shape, device, repeat_noise)
    # no noise when t == 0
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, stats


def p_sample_plms(x, t, interval, cond, clip_denoised=True, repeat_noise=False):
    """
    Use the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
    """

    def get_x_pred(x, noise_t, t):
        a_t = extract(alphas_cumprod, t, x.shape)
        a_prev = extract(
            alphas_cumprod,
            torch.max(t - interval, torch.zeros_like(t)),
            x.shape,
        )
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * (
            (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
            - 1
            / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
            * noise_t
        )
        x_pred = x + x_delta

        return x_pred

    noise_pred, stats = denoise_fn(x.transpose(-1, -2).squeeze(1), cond, t.unsqueeze(1))
    noise_pred = noise_pred.transpose(-1, -2).unsqueeze(1)

    if len(noise_list) == 0:
        x_pred = get_x_pred(x, noise_pred, t)
        # noise_pred_prev, stats = denoise_fn(x_pred, max(t - interval, 0), cond=cond)

        b = len(t)
        _t = torch.full(
            (b,), max(t[0] - interval, 0), device=t.device, dtype=torch.long
        )

        noise_pred_prev, stats = denoise_fn(
            x_pred.transpose(-1, -2).squeeze(1), cond, _t.unsqueeze(1)
        )
        noise_pred_prev = noise_pred_prev.transpose(-1, -2).unsqueeze(1)

        noise_pred_prime = (noise_pred + noise_pred_prev) / 2

    elif len(noise_list) == 1:
        noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
    elif len(noise_list) == 2:
        noise_pred_prime = (
            23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]
        ) / 12
    elif len(noise_list) >= 3:
        noise_pred_prime = (
            55 * noise_pred
            - 59 * noise_list[-1]
            + 37 * noise_list[-2]
            - 9 * noise_list[-3]
        ) / 24

    x_prev = get_x_pred(x, noise_pred_prime, t)
    noise_list.append(noise_pred)

    return x_prev, stats


def svc_model_inference(model, batch, cfg, fast_inference=False, speedup=10):
    # Conditioner: (N, frame_len, dim)

    with torch.no_grad():    
        cond = model[0](batch)
        b = cond.shape[0]
        device = cond.device

        # ============ Diffusion Setting ============
        training_noise_schedule = np.array(cfg.mapper.noise_schedule)
        inference_noise_schedule = training_noise_schedule
        t = len(inference_noise_schedule)
        print("Diffusion steps: ", t)

        # (#steps,)
        betas = inference_noise_schedule
        alphas = 1.0 - betas

        global alphas_cumprod
        # (#steps,)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # (#steps,)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        global sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod
        global posterior_mean_coef1, posterior_mean_coef2
        sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1.0 / alphas_cumprod)).to(device)
        sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1.0 / alphas_cumprod - 1)).to(device)
        posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        posterior_mean_coef2 = to_torch(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)

        global posterior_variance, posterior_log_variance_clipped
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))
        ).to(device)
        posterior_variance = to_torch(posterior_variance).to(device)

        alphas_cumprod = to_torch(alphas_cumprod).to(device)
        alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(device)

        # ============ Model ============
        global denoise_fn
        denoise_fn = model[1]

        # # x: (N, 1, n_mels, T)
        # # batch['y']: (N, T, n_mels)
        # x = torch.randn_like(batch["y"], device=device).transpose(-1, -2).unsqueeze(1)

        # TODO: decrease variance of init
        mean = 0
        std = 1 / 1.2
        x = (
            torch.normal(mean, std, size=batch["y"].shape, device=device)
            .transpose(-1, -2)
            .unsqueeze(1)
        )

        if fast_inference:
            global noise_list
            noise_list = deque(maxlen=4)

            iteration_interval = speedup
            for i in tqdm(
                reversed(range(0, t, iteration_interval)),
                desc="sample time step",
                total=t // iteration_interval,
            ):
                x, stats = p_sample_plms(
                    x,
                    torch.full((b,), i, device=device, dtype=torch.long),
                    iteration_interval,
                    cond,
                )

        else:
            for i in tqdm(reversed(range(0, t)), desc="sample time step", total=t):
                x, stats = p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
   
        # x: (N, 1, n_mels, T)
        mels_output = x.transpose(-1, -2).squeeze(1).squeeze(0)
        
        return  mels_output.T
