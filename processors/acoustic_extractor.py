import os
import torch
import numpy as np

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from utils.io import save_feature
from utils.dsp import compress, audio_to_label
from utils.data_utils import phone_average_pitch

ZERO = 1e-12


def extract_utt_acoustic_features_parallel(metadata, dataset_output, cfg, n_workers=1):
    """Extract acoustic features from utterances using muliprocess

    Args:
        metadata (dict): dictionary that stores data in train.json and test.json files
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations
        n_workers (int, optional): num of processes to extract features in parallel. Defaults to 1.

    Returns:
        list: acoustic features
    """
    executor = ProcessPoolExecutor(max_workers=n_workers)
    results = []
    for item in metadata:
        results.append(
            executor.submit(
                partial(extract_utt_acoustic_features, dataset_output, cfg, item)
            )
        )

    return [result.result() for result in tqdm(results)]


def extract_utt_acoustic_features_serial(metadata, dataset_output, cfg):
    """Extract acoustic features from utterances (in single process)

    Args:
        metadata (dict): dictionary that stores data in train.json and test.json files
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations

    """
    for item in tqdm(metadata):
        extract_utt_acoustic_features(dataset_output, cfg, item)


def extract_utt_acoustic_features(dataset_output, cfg, utt):
    """Extract acoustic features from utterances (in single process)

    Args:
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations
        utt (dict): utterance info including dataset, singer, uid:{singer}_{song}_{index},
                    path to utternace, duration, utternace index

    """
    from utils import audio, f0, world

    uid = utt["Uid"]
    wav_path = utt["Path"]

    with torch.no_grad():
        # todo: first trim then load_audio_torch because content feature is extracted from the wav path rather than the loaded audio
        # if cfg.preprocess.trim_silence:
        # wav = audio.trim_silence(wav, cfg)
        # trimmed_wav_dir = os.path.join(dataset_output, cfg.preprocess.trimmed_wav_dir)
        # trimmed_wav_path = os.path.join(trimmed_wav_dir, uid + ".wav")
        # os.makedirs(trimmed_wav_dir, exist_ok=True)
        # # audio.save_wav(wav, trimmed_wav_path, cfg.preprocess.sample_rate)
        # save_audio(trimmed_wav_path, wav, cfg.preprocess.sample_rate)
        # wav_torch = torch.from_numpy(wav)

        # load audio data into tensor with sample rate 'fs'
        wav_torch, fs = audio.load_audio_torch(wav_path, cfg.preprocess.sample_rate)

        ### save wave
        # wav_dir = os.path.join(dataset_output, cfg.preprocess.wav_dir)
        # os.makedirs(wav_dir, exist_ok=True)
        # wav_path = os.path.join(wav_dir, uid + ".wav")
        # # audio.save_wav(wav, wav_path, cfg.preprocess.sample_rate)
        wav = wav_torch.cpu().numpy()
        # save_audio(wav_path, wav, fs)

        # extract features
        if cfg.preprocess.extract_mel:
            from utils.mel import extract_mel_features

            mel = extract_mel_features(wav_torch.unsqueeze(0), cfg.preprocess)
            save_feature(dataset_output, cfg.preprocess.mel_dir, uid, mel.cpu().numpy())

        if cfg.preprocess.extract_energy:
            if cfg.preprocess.extract_mel:
                energy = (mel.exp() ** 2).sum(0).sqrt().cpu().numpy()
            else:
                energy = audio.energy(wav, cfg.preprocess)
            save_feature(dataset_output, cfg.preprocess.energy_dir, uid, energy)

        if cfg.preprocess.extract_pitch:
            pitch = f0.get_f0(wav, cfg.preprocess)
            save_feature(dataset_output, cfg.preprocess.pitch_dir, uid, pitch)
            if cfg.preprocess.extract_uv:
                assert isinstance(pitch, np.ndarray)
                uv = pitch != 0
                save_feature(dataset_output, cfg.preprocess.uv_dir, uid, uv)

        if cfg.preprocess.extract_audio:
            save_feature(dataset_output, cfg.preprocess.audio_dir, uid, wav)

        if cfg.preprocess.extract_label:
            if cfg.preprocess.mu_law_norm:
                # compress audio
                wav = compress(wav, cfg.preprocess.bits)
            label = audio_to_label(wav, cfg.preprocess.bits)
            save_feature(dataset_output, cfg.preprocess.label_dir, uid, label)


def cal_normalized_mel(mel, dataset_name, cfg):
    mel_min, mel_max = load_mel_extrema(cfg, dataset_name)
    mel_norm = normalize_mel_channel(mel, mel_min, mel_max)
    return mel_norm


def cal_mel_min_max(dataset, output_path, cfg):
    dataset_output = os.path.join(output_path, dataset)

    metadata = []
    for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
        dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    tmp_mel_min = []
    tmp_mel_max = []
    for item in metadata:
        mel_path = os.path.join(
            dataset_output, cfg.preprocess.mel_dir, item["Uid"] + ".npy"
        )
        mel = np.load(mel_path)
        if mel.shape[0] != cfg.preprocess.n_mel:
            mel = mel.T
        assert mel.shape[0] == cfg.preprocess.n_mel

        tmp_mel_min.append(np.min(mel, axis=-1))
        tmp_mel_max.append(np.max(mel, axis=-1))

    mel_min = np.min(tmp_mel_min, axis=0)
    mel_max = np.max(tmp_mel_max, axis=0)

    ## save mel min max data
    mel_min_max_dir = os.path.join(dataset_output, cfg.preprocess.mel_min_max_stats_dir)
    os.makedirs(mel_min_max_dir, exist_ok=True)

    mel_min_path = os.path.join(mel_min_max_dir, "mel_min.npy")
    mel_max_path = os.path.join(mel_min_max_dir, "mel_max.npy")
    np.save(mel_min_path, mel_min)
    np.save(mel_max_path, mel_max)


# def normalize_mel(dataset, output_path, cfg):
#     types = ["train", "test"]
#     for dataset_type in types:
#         dataset_output = os.path.join(output_path, dataset)
#         dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
#         with open(dataset_file, "r") as f:
#             metadata = json.load(f)

#         tmp_mel_min = []
#         tmp_mel_max = []
#         for item in metadata:
#             mel_path = os.path.join(dataset_output,
#                                     cfg.preprocess.mel_dir,
#                                     item["Uid"]+'.npy')
#             mel = np.load(mel_path)
#             if mel.shape[0] != cfg.preprocess.n_mel:
#                 mel = mel.T
#             assert mel.shape[0] == cfg.preprocess.n_mel

#             tmp_mel_min.append(np.min(mel, axis=-1))
#             tmp_mel_max.append(np.max(mel, axis=-1))

#         mel_min = np.min(tmp_mel_min, axis=0)
#         mel_max = np.max(tmp_mel_max, axis=0)

#         ## save mel min max data
#         mel_min_max_dir = os.path.join(dataset_output,
#                                        cfg.preprocess.mel_min_max_stats_dir,
#                                        dataset_type)
#         os.makedirs(mel_min_max_dir, exist_ok=True)

#         mel_min_path = os.path.join(mel_min_max_dir, 'mel_min.npy')
#         mel_max_path = os.path.join(mel_min_max_dir, 'mel_max.npy')
#         np.save(mel_min_path, mel_min)
#         np.save(mel_max_path, mel_max)


#         for item in metadata:
#             mel_path = os.path.join(dataset_output,
#                                     cfg.preprocess.mel_dir,
#                                     item["Uid"]+'.npy')
#             mel = np.load(mel_path)
#             if mel.shape[0] != cfg.preprocess.n_mel:
#                 mel = mel.T
#             assert mel.shape[0] == cfg.preprocess.n_mel

#             mel_norm = normalize_mel_channel(mel, mel_min, mel_max)

#             mel_norm_save_dir = os.path.join(dataset_output,
#                                              cfg.preprocess.mel_min_max_norm_dir)
#             os.makedirs(mel_norm_save_dir, exist_ok=True)
#             mel_norm_save_path = os.path.join(mel_norm_save_dir,
#                                               item["Uid"]+'.npy')
#             # print(mel_norm_save_path)
#             np.save(mel_norm_save_path, mel_norm)


def denorm_for_pred_mels(cfg, dataset_name, split, pred):
    """
    Args:
        pred: a list whose every element is (frame_len, n_mels)
    Return:
        similar like pred
    """
    mel_min, mel_max = load_mel_extrema(cfg.preprocess, dataset_name)
    recovered_mels = [
        denormalize_mel_channel(mel.T, mel_min, mel_max).T for mel in pred
    ]

    return recovered_mels


def load_mel_extrema(cfg, dataset_name):
    data_dir = os.path.join(cfg.processed_dir, dataset_name, cfg.mel_min_max_stats_dir)

    min_file = os.path.join(data_dir, "mel_min.npy")
    max_file = os.path.join(data_dir, "mel_max.npy")

    mel_min = np.load(min_file)
    mel_max = np.load(max_file)

    return mel_min, mel_max


def denormalize_mel_channel(mel, mel_min, mel_max):
    mel_min = np.expand_dims(mel_min, -1)
    mel_max = np.expand_dims(mel_max, -1)
    return (mel + 1) / 2 * (mel_max - mel_min + ZERO) + mel_min


def normalize_mel_channel(mel, mel_min, mel_max):
    mel_min = np.expand_dims(mel_min, -1)
    mel_max = np.expand_dims(mel_max, -1)
    return (mel - mel_min) / (mel_max - mel_min + ZERO) * 2 - 1


def cal_pitch_statistics(dataset, output_path, cfg):
    # path of dataset
    dataset_dir = os.path.join(output_path, dataset)

    # load singers and ids
    singers = json.load(open(os.path.join(dataset_dir, "singers.json"), "r"))

    # combine train and test metadata
    metadata = []
    for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
        dataset_file = os.path.join(dataset_dir, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    # use different scalers for each singer
    pitch_scalers = [[] for _ in range(len(singers))]
    total_pitch_scalers = [[] for _ in range(len(singers))]

    for utt_info in metadata:
        utt = f'{utt_info["Dataset"]}_{utt_info["Uid"]}'
        singer = utt_info["Singer"]
        pitch_path = os.path.join(
            dataset_dir, cfg.preprocess.pitch_dir, utt_info["Uid"] + ".npy"
        )
        # total_pitch contains all pitch including unvoiced frames
        total_pitch = np.load(pitch_path)
        assert len(total_pitch) > 0
        # pitch contains only voiced frames
        pitch = total_pitch[total_pitch != 0]
        spkid = singers[f"{dataset}_{singer}"]

        # update pitch scalers
        pitch_scalers[spkid].extend(pitch.tolist())
        # update total pitch scalers
        total_pitch_scalers[spkid].extend(total_pitch.tolist())

    # save pitch statistics for each singer in dict
    sta_dict = {}
    for singer in singers:
        spkid = singers[singer]
        # voiced pitch statistics
        mean, std, min, max, median = (
            np.mean(pitch_scalers[spkid]),
            np.std(pitch_scalers[spkid]),
            np.min(pitch_scalers[spkid]),
            np.max(pitch_scalers[spkid]),
            np.median(pitch_scalers[spkid]),
        )

        # total pitch statistics
        mean_t, std_t, min_t, max_t, median_t = (
            np.mean(total_pitch_scalers[spkid]),
            np.std(total_pitch_scalers[spkid]),
            np.min(total_pitch_scalers[spkid]),
            np.max(total_pitch_scalers[spkid]),
            np.median(total_pitch_scalers[spkid]),
        )
        sta_dict[singer] = {
            "voiced_positions": {
                "mean": mean,
                "std": std,
                "median": median,
                "min": min,
                "max": max,
            },
            "total_positions": {
                "mean": mean_t,
                "std": std_t,
                "median": median_t,
                "min": min_t,
                "max": max_t,
            },
        }

    # save statistics
    save_dir = os.path.join(dataset_dir, cfg.preprocess.pitch_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "statistics.json"), "w") as f:
        json.dump(sta_dict, f, indent=4, ensure_ascii=False)
