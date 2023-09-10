import os
import numpy as np
import torch
import torchaudio


def save_feature(process_dir, feature_dir, item, feature, overrides=True):
    """Save features to path

    Args:
        process_dir (str): directory to store features
        feature_dir (_type_): directory to store one type of features (mel, energy, ...)
        item (str): uid
        feature (tensor): feature tensor
        overrides (bool, optional): whether to override existing files. Defaults to True.
    """
    process_dir = os.path.join(process_dir, feature_dir)
    os.makedirs(process_dir, exist_ok=True)
    out_path = os.path.join(process_dir, item + ".npy")

    if os.path.exists(out_path):
        if overrides:
            np.save(out_path, feature)
    else:
        np.save(out_path, feature)


def save_audio(path, waveform, fs, add_silence=False, turn_up=True, volume_peak=0.9):
    if turn_up:
        # continue to turn up to volume_peak
        ratio = volume_peak / max(waveform.max(), abs(waveform.min()))
        waveform = waveform * ratio

    if add_silence:
        silence_len = fs // 20
        silence = np.zeros((silence_len,), dtype=waveform.dtype)
        result = np.concatenate([silence, waveform, silence])
        waveform = result

    waveform = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if len(waveform.size()) == 1:
        waveform = waveform[None, :]
    torchaudio.save(path, waveform, fs, encoding="PCM_S", bits_per_sample=16)


def load_mel_extrema(cfg, dataset_name, split):
    dataset_dir = os.path.join(
        cfg.OUTPUT_PATH,
        "preprocess/{}_version".format(cfg.data.process_version),
        dataset_name,
    )
    # min_file = os.path.join(
    #     dataset_dir,
    #     "mel_min/{}".format(cfg.VOCODER.FS),
    #     "{}.pkl".format(split.split("_")[-1]),
    # )
    # max_file = os.path.join(
    #     dataset_dir,
    #     "mel_max/{}".format(cfg.VOCODER.FS),
    #     "{}.pkl".format(split.split("_")[-1]),
    # )
    # with open(min_file, "rb") as f:
    #     mel_min = pickle.load(f)
    # with open(max_file, "rb") as f:
    #     mel_max = pickle.load(f)

    min_file = os.path.join(
        dataset_dir,
        "mel_min_max",
        # "mel_min/{}".format(cfg.VOCODER.FS),
        split.split("_")[-1],
        "mel_min.npy",
    )
    max_file = os.path.join(
        dataset_dir,
        "mel_min_max",
        # "mel_max/{}".format(cfg.VOCODER.FS),
        split.split("_")[-1],
        "mel_max.npy",
    )
    mel_min = np.load(min_file)
    mel_max = np.load(max_file)
    return mel_min, mel_max
