'''
Author: lmxue
Date: 2023-06-09 15:04:55
LastEditTime: 2023-06-11 07:38:42
LastEditors: lmxue
Description: 
FilePath: /worksapce/svc_inference_pipline/utils/acoustic_feature_extraction.py
@Email: xueliumeng@gmail.com
'''

import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.mel import extract_mel_features
from utils.f0 import get_f0_features_using_parselmouth



def get_target_f0_median(cfg):
    total_f0 = []
    with open(cfg.target_f0_file, "rb") as f:
        f0s = pickle.load(f)
    for f0 in f0s:
        total_f0 += f0.tolist()

    total_f0 = np.array(total_f0)
    voiced_position = np.where(total_f0 != 0)
    return np.median(total_f0[voiced_position])


def get_conversion_f0_factor(source_f0, target_median):
    """Align the median between source f0 and target f0

    Note: Here we use multiplication, whose factor is target_median/source_median

    Reference: Frequency and pitch interval
    http://blog.ccyg.studio/article/be12c2ee-d47c-4098-9782-ca76da3035e4/
    """
    voiced_position = np.where(source_f0 != 0)
    source_median = np.median(source_f0[voiced_position])

    factor = target_median / source_median
    return factor


def pitch_shift(raw_f0, cfg):
    target_f0_median = get_target_f0_median(cfg)
    factor = get_conversion_f0_factor(raw_f0, target_f0_median)
    
    return raw_f0 * factor
    
    


def acoutic_feature_extractor(wav_file, cfg):
    # mel: # [n_mel, T]                                
    audio, mel, energy = extract_mel_features(wav_file, cfg)
    f0, _ = get_f0_features_using_parselmouth(audio, mel.shape[-1], cfg)

    mel_norm_features = normalize_mel_channel(mel, cfg) # [n_mel, T]
    return mel.T, f0, energy


def load_mel_min_max(cfg):
    with open(cfg.min_mel_file, "rb") as f:
        mel_min = pickle.load(f)
    with open(cfg.max_mel_file, "rb") as f:
        mel_max = pickle.load(f)
        
    return mel_min, mel_max


def normalize_mel_channel(mel, cfg):
    ZERO = 1e-12
    mel_min, mel_max = load_mel_min_max(cfg)
    mel_min = np.expand_dims(mel_min, -1)
    mel_max = np.expand_dims(mel_max, -1)
    return (mel - mel_min) / (mel_max - mel_min + ZERO) * 2 - 1


def denormalize_mel_channel(mel, cfg):
    device = mel.device
    mel = mel.cpu().numpy()
    
    ZERO = 1e-12
    mel_min, mel_max = load_mel_min_max(cfg)
    
    # mel (frame_len, n_mels)
    mel_min = np.expand_dims(mel_min, -1)
    mel_max = np.expand_dims(mel_max, -1)
    mel_norm = (mel + 1) / 2 * (mel_max - mel_min + ZERO) + mel_min
    
    mel_norm = torch.as_tensor(mel_norm, device=device)
    
    return mel_norm

