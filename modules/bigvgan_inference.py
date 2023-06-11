'''
Author: lmxue
Date: 2023-06-09 22:09:27
LastEditTime: 2023-06-11 01:20:25
LastEditors: lmxue
Description: 
FilePath: /worksapce/svc_inference_pipline/modules/bigvgan_inference.py
@Email: xueliumeng@gmail.com
'''
import torch
import numpy as np
import time
import torch.nn as nn
import math
from tqdm import tqdm
import torch.nn.functional as F


def vocoder_inference(cfg, model, mels, device, fast_inference=False):
    model.eval()

    with torch.no_grad():
        mels = mels.to(device)
        output = model.forward(mels)
    
    return output.squeeze(1).detach().cpu()


def synthesis_audios(model, mel, cfg, f0s=None, batch_size=None, fast_inference=False):
    # Get the device
    device = next(model.parameters()).device

    frame = mel.shape[-1]
    audio = vocoder_inference(
        cfg, model, mel.unsqueeze(0), device, fast_inference
    ).squeeze(0)
    fade_out = torch.linspace(
        1, 0, steps=20 * cfg.hop_length
    ).cpu()
    audio_length = frame * cfg.hop_length
    audio = audio[:audio_length]
    audio[-20 * cfg.hop_length :] *= fade_out

    return audio.cpu().detach().numpy()
