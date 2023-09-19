import os
import torch
import glob
import pickle
import json
from tqdm import tqdm
import torchaudio
import librosa
import numpy as np

from models.vocoders.gan.generator import bigvgan, hifigan, melgan, nsfhifigan

from models.vocoders.flow.waveglow import waveglow

from models.vocoders.diffusion.diffwave import diffwave

from models.vocoders.autoregressive.wavenet import wavenet
from models.vocoders.autoregressive.wavernn import wavernn

from models.vocoders.autoregressive import autoregressive_vocoder_inference
from models.vocoders.diffusion import diffusion_vocoder_inference
from models.vocoders.flow import flow_vocoder_inference
from models.vocoders.gan import gan_vocoder_inference

# from models.vocoders.diffwave import diffwave, diffwave_inference
# from models.vocoders.wavernn import wavernn, wavernn_inference
# from models.vocoders.wavenet import wavenet, wavenet_inference
# from models.vocoders.nsfhifigan import nsfhifigan, nsfhifigan_inference
# from models.vocoders.bigvgan import bigvgan, bigvgan_inference

# from models.vocoders.world import world_inference
# from processors.acoustic_extractor import denorm_for_pred_mels

# from cuhkszsvc.engine.utils import (
#     extract_steps_num_of_vocoder,
#     find_checkpoint_of_vocoder,
# )
# from cuhkszsvc.preprocess import (
#     get_uids_and_wav_paths,
#     get_golden_samples_indexes,
#     get_specific_singer_indexes,
# )
from utils.io import save_audio, load_mel_extrema

# from cuhkszsvc.configs.config_parse import *

_VOCODERS = {
    "diffwave": diffwave.DiffWave,
    "wavernn": wavernn.WaveRNN,
    "wavenet": wavenet.WaveNet,
    "waveglow": waveglow.WaveGlow,
    "nsfhifigan": nsfhifigan.NSFHiFiGAN,
    "bigvgan": bigvgan.BigVGAN,
    "hifigan": hifigan.HiFiGAN,
    "melgan": melgan.MelGAN,
}

_VOCODER_INFER_FUNCS = {
    # "world": world_inference.synthesis_audios,
    # "wavernn": wavernn_inference.synthesis_audios,
    # "wavenet": wavenet_inference.synthesis_audios,
    # "diffwave": diffwave_inference.synthesis_audios,
    "nsfhifigan": gan_vocoder_inference.synthesis_audios,
    "bigvgan": gan_vocoder_inference.synthesis_audios,
    "melgan": gan_vocoder_inference.synthesis_audios,
    "hifigan": gan_vocoder_inference.synthesis_audios,
}


def load_nnvocoder(
    cfg,
    vocoder_name,
    weights_file=None,
    from_multi_gpu=False,
):
    # if not weights_file:
    #     if "RESUME_FILE" not in cfg.VOCODER:
    #         weights_file = find_checkpoint_of_vocoder(cfg.VOCODER.RESUME)
    #     else:
    #         weights_file = cfg.VOCODER.RESUME_FILE

    print("Loading Vocoder from Weights file: {}".format(weights_file))
    # cfg.VOCODER.WEIGHTS_FILE = weights_file

    model = _VOCODERS[vocoder_name](cfg)
    if vocoder_name in ["bigvgan", "hifigan", "melgan", "nsfhifigan"]:
        ckpt = torch.load(
            weights_file,
            map_location=torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )
        if from_multi_gpu:
            pretrained_generator_dict = ckpt["generator_state_dict"]
            generator_dict = model.state_dict()

            new_generator_dict = {
                k.split("module.")[-1]: v
                for k, v in pretrained_generator_dict.items()
                if (
                    k.split("module.")[-1] in generator_dict
                    and v.shape == generator_dict[k.split("module.")[-1]].shape
                )
            }

            generator_dict.update(new_generator_dict)

            model.load_state_dict(generator_dict)
        else:
            model.load_state_dict(ckpt["generator_state_dict"])
    else:
        model.load_state_dict(torch.load(weights_file)["state_dict"])

    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    return model


# def load_gt_mel_norm_features(
#     cfg, dataset_name, split, only_specific_singer, tag="", inference_vocoder=False
# ):
#     dataset_dir = os.path.join(
#         cfg.OUTPUT_PATH,
#         "preprocess",
#         "{}_version".format(cfg.PREPROCESS_VERSION),
#         dataset_name,
#     )

#     if inference_vocoder:
#         file = os.path.join(
#             dataset_dir,
#             "mels/{}".format(cfg.VOCODER.FS),
#             "{}{}.pkl".format(tag, split.split("_")[-1]),
#         )
#     else:
#         file = os.path.join(
#             dataset_dir,
#             "mel_norm/{}".format(cfg.VOCODER.FS),
#             "{}{}.pkl".format(tag, split.split("_")[-1]),
#         )

#     with open(file, "rb") as f:
#         # mel feat [(n_mels, frame_len), ...]
#         data = pickle.load(f)

#     if "golden" in split:
#         indexes = get_golden_samples_indexes(
#             dataset_name, dataset_dir, split=split.split("_")[-1]
#         )
#         data = [data[idx] for idx in indexes]
#     elif only_specific_singer is not None:
#         indexes = get_specific_singer_indexes(dataset_dir, only_specific_singer, split)
#         data = [data[idx] for idx in indexes]

#     return data


# def load_gt_mel_norm_features_of_bigdata(
#     cfg, dataset_name, split, tag="", inference_vocoder=False
# ):
#     assert dataset_name == "bigdata"
#     assert split != "golden_test"

#     dataset_dir = os.path.join(
#         cfg.OUTPUT_PATH,
#         "preprocess",
#         "{}_version".format(cfg.PREPROCESS_VERSION),
#         dataset_name,
#         cfg.BIGDATA_VERSION,
#     )
#     with open(os.path.join(dataset_dir, "{}.json".format(split))) as f:
#         utterances = json.load(f)

#     dataset_keys = list(set([utt["Dataset"] for utt in utterances]))
#     dataset_keys.sort()
#     mel_norm_features_dict = dict(
#         zip(
#             dataset_keys,
#             [
#                 load_gt_mel_norm_features(
#                     cfg, key, split, tag=tag, inference_vocoder=inference_vocoder
#                 )
#                 for key in dataset_keys
#             ],
#         )
#     )

#     data = [mel_norm_features_dict[utt["Dataset"]][utt["index"]] for utt in utterances]
#     return data


# def load_real_auxiliary_features(cfg, dataset_name, split):
#     dataset_dir = os.path.join(
#         cfg.OUTPUT_PATH,
#         "preprocess",
#         "{}_version".format(cfg.PREPROCESS_VERSION),
#         dataset_name,
#     )
#     split_dr = os.path.join(
#         cfg.OUTPUT_PATH,
#         "preprocess",
#         dataset_name,
#     )
#     file = os.path.join(
#         dataset_dir,
#         "{}/{}".format(cfg.AUXILIARY_FEATURES, cfg.VOCODER.FS),
#         "{}.pkl".format(split.split("_")[-1]),
#     )
#     with open(file, "rb") as f:
#         # f0 feat [(frame_len), ...]
#         data = pickle.load(f)

#     if "golden" in split:
#         indexes = get_golden_samples_indexes(
#             dataset_name, dataset_dir, split=split.split("_")[-1]
#         )
#         data = [data[idx] for idx in indexes]

#     return data


# def load_mel_extrema(cfg, dataset_name, split):
#     dataset_dir = os.path.join(
#         cfg.OUTPUT_PATH,
#         "preprocess/{}_version".format(cfg.PREPROCESS_VERSION),
#         dataset_name,
#     )
#     # min_file = os.path.join(
#     #     dataset_dir,
#     #     "mel_min/{}".format(cfg.VOCODER.FS),
#     #     "{}.pkl".format(split.split("_")[-1]),
#     # )
#     # max_file = os.path.join(
#     #     dataset_dir,
#     #     "mel_max/{}".format(cfg.VOCODER.FS),
#     #     "{}.pkl".format(split.split("_")[-1]),
#     # )
#     # with open(min_file, "rb") as f:
#     #     mel_min = pickle.load(f)
#     # with open(max_file, "rb") as f:
#     #     mel_max = pickle.load(f)

#     min_file = os.path.join(
#         dataset_dir,
#         "mel_min/{}".format(cfg.VOCODER.FS),
#         split.split("_")[-1],
#         "mel_min.npy"

#     )
#     max_file = os.path.join(
#         dataset_dir,
#         "mel_max/{}".format(cfg.VOCODER.FS),
#         split.split("_")[-1],
#         "mel_max.npy"
#     )
#     mel_min = np.load(min_file)
#     mel_max = np.load(max_file)
#     return mel_min, mel_max


def tensorize(data, device, n_samples):
    """
    data: a list of numpy array
    """
    assert type(data) == list
    if n_samples:
        data = data[:n_samples]
    data = [torch.as_tensor(x, device=device) for x in data]
    return data


# def test_vocoder(
#     cfg,
#     dataset_name,
#     split,
#     n_samples,
#     save_dir,
#     batch_size=None,
#     fast_inference=False,
#     multi_gpu=False,
# ):
#     vocoder_name = cfg.VOCODER if type(cfg.VOCODER) == str else cfg.VOCODER.NAME

#     print("\nSynthesis audios using {} vocoder...".format(vocoder_name))

#     # ====== Loading neural vocoder model ======
#     vocoder = load_nnvocoder(cfg, from_multi_gpu=multi_gpu)
#     device = next(vocoder.parameters()).device

#     mels_gt = tensorize(
#         load_gt_mel_norm_features(cfg, dataset_name, split, inference_vocoder=True),
#         device,
#         n_samples,
#     )
#     print("\n For ground truth mels, #sample = {}...".format(len(mels_gt)))

#     if vocoder_name == "nsfhifigan":
#         f0_gt = tensorize(
#             load_real_auxiliary_features(cfg, dataset_name, split), device, n_samples
#         )

#     if vocoder_name == "nsfhifigan":
#         audios_gt = _VOCODER_INFER_FUNCS[vocoder_name](
#             cfg,
#             vocoder,
#             mels_gt,
#             f0_gt,
#             batch_size=batch_size,
#             fast_inference=fast_inference,
#         )
#     else:
#         audios_gt = _VOCODER_INFER_FUNCS[vocoder_name](
#             cfg, vocoder, mels_gt, batch_size=batch_size, fast_inference=fast_inference
#         )

#     # ====== Save ======
#     # Sampling rate
#     sampling_rate = cfg.VOCODER.FS

#     sample_ids, sample_audios_paths = get_uids_and_wav_paths(cfg, dataset_name, split)
#     if not n_samples:
#         n_samples = len(sample_ids)

#     for i in tqdm(range(n_samples)):
#         uid, wave_file = sample_ids[i], sample_audios_paths[i]
#         if wave_file.split(".")[-1] == "wav":
#             real_y, real_sampling_rate = torchaudio.load(wave_file)
#         else:
#             real_y, real_sampling_rate = librosa.load(wave_file)
#             real_y = torch.from_numpy(real_y)

#         # save gt
#         gt_file = os.path.join(save_dir, "{}.wav".format(uid))
#         gt = torchaudio.functional.resample(
#             real_y, orig_freq=real_sampling_rate, new_freq=sampling_rate
#         )
#         save_audio(gt_file, gt, sampling_rate)

#         # save Vocoder synthesis gt
#         y_gt = audios_gt[i]
#         vocoder_gt_file = os.path.join(save_dir, "{}_{}.wav".format(uid, vocoder_name))
#         save_audio(vocoder_gt_file, y_gt, sampling_rate)


# def denorm_for_pred_mels(cfg, dataset_name, split, pred):
#     """
#     Args:
#         pred: a list whose every element is (frame_len, n_mels)
#     Return:
#         similar like pred
#     """
#     preprocess_version = cfg.data.process_version
#     if preprocess_version == "general":
#         return pred

#     # print(
#     #     "Recover from normalized mels, using {} version...\n".format(preprocess_version)
#     # )

#     if preprocess_version == "diffsinger":
#         # TODO
#         pass

#     if preprocess_version == "bigvgan":
#         mel_min, mel_max = load_mel_extrema(cfg, dataset_name, split)
#         recovered_mels = [
#             denormalize_mel_channel(mel.T, mel_min, mel_max).T for mel in pred
#         ]

#     return recovered_mels


def synthesis(
    vocoder_name,
    vocoder,
    cfg,
    vocoder_weight_file,
    # dataset_name,
    # split,
    n_samples,
    pred,
    f0s=None,
    # tag="",
    batch_size=64,
    fast_inference=False,
    # ground_truth_inference=False,
    # only_friendly=True,
):
    """
    pred:
        a list of numpy arrays. [(seq_len1, acoustic_features_dim), (seq_len2, acoustic_features_dim), ...]
    tag:
        some tags for saving name. Eg: loss level
    """
    # vocoder_name = cfg.VOCODER if type(cfg.VOCODER) == str else cfg.VOCODER.NAME

    print("Synthesis audios using {} vocoder...".format(vocoder_name))

    ###### TODO: World Vocoder Refactor ######
    # if vocoder_name == "world":
    #     world_inference.synthesis_audios(
    #         cfg, dataset_name, split, n_samples, pred, save_dir, tag
    #     )
    #     return

    # ====== Loading neural vocoder model ======
    #vocoder = load_nnvocoder(
    #    cfg, vocoder_name, weights_file=vocoder_weight_file, from_multi_gpu=True
    #)
    device = next(vocoder.parameters()).device

    # ====== Inference for predicted acoustic features ======
    # pred: (frame_len, n_mels) -> (n_mels, frame_len)
    mels_pred = tensorize([p.T for p in pred], device, n_samples)
    print("For predicted mels, #sample = {}...".format(len(mels_pred)))
    audios_pred = _VOCODER_INFER_FUNCS[vocoder_name](
        cfg,
        vocoder,
        mels_pred,
        f0s=f0s,
        batch_size=batch_size,
        fast_inference=fast_inference,
    )
    return audios_pred

    """
    
    if ground_truth_inference:
        mels_gt = load_gt_mel_norm_features(
            cfg, dataset_name, split, only_specific_singer
        )
        mels_gt = denorm_for_pred_mels(
            cfg, dataset_name, split, pred=[mel.T for mel in mels_gt]
        )
        mels_gt = tensorize([mel.T for mel in mels_gt], device, n_samples)

        print("For ground truth mels, #sample = {}...".format(len(mels_gt)))
        audios_gt = _VOCODER_INFER_FUNCS[vocoder_name](
            cfg,
            vocoder,
            mels_gt,
            f0s=f0s,
            batch_size=batch_size,
            fast_inference=fast_inference,
        )

    # ====== Save ======
    # Sampling rate
    sampling_rate = cfg.vocoder.fs

    # sample_ids, sample_audios_paths = get_uids_and_wav_paths(
    #     cfg, dataset_name, split, only_specific_singer=only_specific_singer
    # )
    # if n_samples is None:
    #     n_samples = len(sample_ids)

    for i in tqdm(range(n_samples)):
        # uid, wave_file = sample_ids[i], sample_audios_paths[i]
        y_pred = audios_pred[i]

        if ground_truth_inference:
            # save gt
            real_y, real_sampling_rate = torchaudio.load(wave_file)
            gt_file = os.path.join(save_dir, "{}.wav".format(uid))
            gt = torchaudio.functional.resample(
                real_y, orig_freq=real_sampling_rate, new_freq=sampling_rate
            )
            save_audio(gt_file, gt, sampling_rate)

            # save Vocoder synthesis gt
            y_gt = audios_gt[i]
            vocoder_gt_file = os.path.join(
                save_dir, "{}_{}.wav".format(uid, vocoder_name)
            )
            save_audio(vocoder_gt_file, y_gt, sampling_rate)

        # save Vocoder synthesis pred
        vocoder_pred_file = os.path.join(save_dir, "{}_pred{}.wav".format(uid, tag))
        
        save_audio(
            vocoder_pred_file,
            y_pred.numpy(),
            sampling_rate,
            add_silence=True,
            turn_up=True,
        )
        if not only_friendly:
            save_audio(
                vocoder_pred_file.replace(".wav", "_raw.wav"), y_pred, sampling_rate
            )

    print('Saving to {}.'.format(save_dir))
    """
