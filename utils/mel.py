import torch
import numpy as np

import torch.nn.functional as F
from utils.audio import load_audio_torch, load_audio_samples
from librosa.filters import mel as librosa_mel_fn

# from cuhkszsvc.extractor.acoustics.utils.audio import load_audio_torch


# def dynamic_range_compression(x, C=1, clip_val=1e-5):
#     return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


# def dynamic_range_decompression(x, C=1):
#     return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


# def dynamic_range_decompression_torch(x, C=1):
#     return torch.exp(x) / C



# def spectral_de_normalize_torch(magnitudes):
#     output = dynamic_range_decompression_torch(magnitudes)
#     return output


# class STFT:
#     def __init__(
#         self, fs, n_mels, n_fft, win_length, hop_length, fmin, fmax, clip_val=1e-5
#     ):
#         self.fs = fs
#         self.n_mels = n_mels
#         self.n_fft = n_fft
#         self.win_length = win_length
#         self.hop_length = hop_length
#         self.fmin = fmin
#         self.fmax = fmax
#         self.clip_val = clip_val

#         self.mel_basis = {}
#         self.hann_window = {}

#     def get_mel(self, y, keyshift=0, speed=1, center=False):
#         factor = 2 ** (keyshift / 12)
#         n_fft_new = int(np.round(self.n_fft * factor))
#         win_length_new = int(np.round(self.win_length * factor))
#         hop_length_new = int(np.round(self.hop_length * factor))

#         mel_basis_key = str(self.fmax) + "_" + str(y.device)
#         if mel_basis_key not in self.mel_basis:
#             mel = librosa_mel_fn(
#                 sr=self.fs,
#                 n_fft=self.n_fft,
#                 n_mels=self.n_mels,
#                 fmin=self.fmin,
#                 fmax=self.fmax,
#             )
#             self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

#         keyshift_key = str(keyshift) + "_" + str(y.device)
#         if keyshift_key not in self.hann_window:
#             self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
#                 y.device
#             )

#         y = torch.nn.functional.pad(
#             y.unsqueeze(1),
#             (
#                 (win_length_new - hop_length_new) // 2,
#                 (win_length_new - hop_length_new + 1) // 2,
#             ),
#             mode="reflect",
#         )
#         y = y.squeeze(1)

#         spec = torch.stft(
#             y,
#             n_fft_new,
#             hop_length=hop_length_new,
#             win_length=win_length_new,
#             window=self.hann_window[keyshift_key],
#             center=center,
#             pad_mode="reflect",
#             normalized=False,
#             onesided=True,
#             return_complex=False,
#         )

#         spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
#         if keyshift != 0:
#             size = self.n_fft // 2 + 1
#             resize = spec.size(1)
#             if resize < size:
#                 spec = F.pad(spec, (0, 0, 0, size - resize))
#             spec = spec[:, :size, :] * self.win_length / win_length_new

#         spec = torch.matmul(self.mel_basis[mel_basis_key], spec)

#         spec = dynamic_range_compression_torch(spec, clip_val=self.clip_val)

#         return spec

#     def __call__(self, wave_file):
#         audio, fs = load_audio_torch(wave_file, self.fs)
#         spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
#         return spect
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class STFT:
    def __init__(
        self, fs, n_mels, n_fft, win_length, hop_length, fmin, fmax, clip_val=1e-5
    ):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val

        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * factor))

        mel_basis_key = str(self.fmax) + "_" + str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.fs,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (win_length_new - hop_length_new) // 2,
                (win_length_new - hop_length_new + 1) // 2,
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * self.win_length / win_length_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)

        spec = dynamic_range_compression_torch(spec, clip_val=self.clip_val)

        return spec

    def __call__(self, wave_file):
        audio, fs = load_audio_torch(wave_file, self.fs)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect
    
    def __call__(self, samples, sample_rate):
        audio, fs = load_audio_samples(samples, sample_rate, tgt_fs)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect


mel_basis = {}
hann_window = {}


def extract_mel_features(
    y,
    cfg,
    center=False
    # n_fft, n_mel, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """Extract mel features

    Args:
        y (tensor): audio data in tensor
        cfg (dict): configuration in cfg.preprocess
        center (bool, optional): In STFT, whether t-th frame is centered at time t*hop_length. Defaults to False.

    Returns:
        tensor: a tensor containing the mel feature calculated based on STFT result
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if cfg.fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mel,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        mel_basis[str(cfg.fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(cfg.fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec.squeeze(0)
