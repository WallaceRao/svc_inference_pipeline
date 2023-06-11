"""
Reference: https://github.com/svc-develop-team/so-vits-svc/blob/4.0/preprocess_hubert_f0.py
"""
import os
import json
import pickle
import librosa
import torch
import numpy as np
from fairseq import checkpoint_utils
from tqdm import tqdm


def load_hubert_model(ckpt_path, device):
    print("Load Hubert from {}".format(ckpt_path))

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path],
        suffix="",
    )
    model = models[0]

    if device == "cuda":
        model = model.cuda()
        
    model.eval()

    return model


def get_hubert_content(hmodel, wav_16k_tensor):
    feats = wav_16k_tensor
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(wav_16k_tensor.device),
        "padding_mask": padding_mask.to(wav_16k_tensor.device),
        "output_layer": 9,  # layer 9
    }
    with torch.no_grad():
        logits = hmodel.extract_features(**inputs)
        feats = hmodel.final_proj(logits[0])
    return feats.transpose(1, 2).squeeze(0)


def content_vector_encoder(model, audio_path, default_sampling_rate=16000):
    """
    # content vector default sr: 16000
    """
    wav16k, sr = librosa.load(audio_path, sr=default_sampling_rate)
    device = next(model.parameters()).device
    wav16k = torch.from_numpy(wav16k).to(device)

    # (256, frame_len)
    feats = get_hubert_content(model, wav_16k_tensor=wav16k)
    return feats.cpu().numpy()


def repeat_expand_2d(content, target_len):
    """
    content : [hubert_dim(256), src_len]
    target: [hubert_dim(256), target_len]
    """
    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(
        content.device
    )
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


def get_mapped_features(raw_feats, mel):
    """
    Content Vector: frameshift = 20ms, hop_size = 480 in 24k

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

    print('raw_feats.shape: ', raw_feats.shape)
    print('mel.shape: ', mel.shape)
    # mappping_feat: (mels_frame_len, n_mels)
    target_len = mel.shape[0]

    # (source_len, 256)
    source_len, width = raw_feats.shape

    # const ~= target_len * target_hop
    const = source_len * source_hop // target_hop * target_hop

    # (source_len * source_hop, dim)
    up_sampling_feats = np.repeat(raw_feats, source_hop, axis=0)
    # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
    down_sampling_feats = np.average(
        up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
    )

    err = abs(target_len - len(down_sampling_feats))
    if err > 3:
        print("mels:", mel.shape)
        print("raw content vector:", raw_feats.shape)
        print("up_sampling:", up_sampling_feats.shape)
        print("down_sampling_feats:", down_sampling_feats.shape)
        exit()
    if len(down_sampling_feats) < target_len:
        # (1, dim) -> (err, dim)
        end = down_sampling_feats[-1][None, :].repeat(err, axis=0)
        down_sampling_feats = np.concatenate([down_sampling_feats, end], axis=0)

    # (target_len, dim)
    feats = down_sampling_feats[:target_len]

    return feats



def contentVec_feature_extractor(wav_file, mel, cfg):
    model = load_hubert_model(cfg.contentVec_model, cfg.device)
    content_vector = content_vector_encoder(model, wav_file)

    content_vector_aligned = get_mapped_features(content_vector.T, mel) # [T, 256]
    
    return content_vector_aligned