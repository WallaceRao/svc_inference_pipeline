import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
import json
import glob
import time

from . import whisper_extractor as whisper
from torch.nn.utils.rnn import pad_sequence

def whisper_encoder(model, audio_paths):
    batch_mel = torch.zeros((1, 80, 3000), dtype=torch.float32, device=model.device)

    # (48000,)
    audio = whisper.load_audio(audio_paths)
    audio = whisper.pad_or_trim(audio)

    # (80, 3000)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    batch_mel[0] = mel

    with torch.no_grad():
        # (batch, 1500, 1024)
        features = model.embed_audio(batch_mel).squeeze(0)

    return features.cpu().numpy()


def get_mapped_whisper_features(
    raw_feats, mel, fast_mapping=True
):
    """
    Whisper: frameshift = 20ms (30s audio -> 1500 frames), hop_size = 480 in 24k
    # Ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/model.py#L136

    Now it's only used for mapping to bigvgan's mels (sr = 24k, hop_size = 256, frameshift ~= 10.7 ms)
    """
    source_hop = 480
    target_hop = 256

    factor = np.gcd(source_hop, target_hop)
    source_hop //= factor
    target_hop //= factor
    # print(
    #     "Mapping source's {} frames => target's {} frames".format(
    #         target_hop, source_hop
    #     )
    # )

    max_source_len = 1500

    target_len = mel.shape[0]
    # The max target_len is 2812
    target_len = min(target_len, max_source_len * source_hop // target_hop)

    # (1500, dim)
    width = raw_feats.shape[-1]

    if fast_mapping:
        source_len = target_len * target_hop // source_hop + 1
        raw_feats = raw_feats[:source_len]
    else:
        source_len = max_source_len

    # const ~= target_len * target_hop
    const = source_len * source_hop // target_hop * target_hop

    # (source_len * source_hop, dim)
    up_sampling_feats = np.repeat(raw_feats, source_hop, axis=0)
    # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
    down_sampling_feats = np.average(
        up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
    )
    assert len(down_sampling_feats) >= target_len

    # (target_len, dim)
    feat = down_sampling_feats[:target_len]

    return feat


def load_whisper_model(model_name, device):
    print("Loading Whisper '{}' model... ".format(model_name))
    
    model = whisper.load_model(model_name)
    if device == "cuda":
        model = model.cuda()

    model = model.eval()
    return model



def whisper_feature_extractor(wav_file, mel, cfg):
    model_name = cfg.whisper_model
    whisper_model = load_whisper_model(model_name, cfg.device)
    whisper_feature = whisper_encoder(whisper_model, wav_file)

    # Mapping to acoustic features' lengths
    whisper_feature_aligned = get_mapped_whisper_features(whisper_feature, mel)

    return whisper_feature_aligned