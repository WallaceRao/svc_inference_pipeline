import torch
import librosa

from scipy.io.wavfile import read

import soundfile as sf
import numpy as np


def load_audio_samples(audio, src_fs, tgt_fs):
    """
    Args:
        wave_file
        fs
    Returns:
        audio(tensor)
        fs
    """
    # Set always_2d to be true to handle stereo audio
    sample_rate = src_fs
    # We only use monotonic audio
    print("input audio shape:", audio.shape)
    print("input audio:", audio)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        assert len(audio) > 2

    # Check the audio type - float, 8bit or 16bit
    if np.issubdtype(audio.dtype, np.integer):
        max_mag = -np.iinfo(audio.dtype).min
    else:
        max_mag = max(np.amax(audio), -np.amin(audio))
        max_mag = (
            (2**31) + 1
            if max_mag > (2**15)
            else ((2**15) + 1 if max_mag > 1.01 else 1.0)
        )
    print("max_mag:", max_mag)
    if max_mag == 0:
        max_mag = 1
    # Normalize the audio
    audio = torch.FloatTensor(audio.astype(np.float32)) / max_mag
    print("normalized audio shape:", audio.shape)
    print("audio:", audio)

    if (torch.isnan(audio) | torch.isinf(audio)).any():
        print("invalid audio")
        return [], sample_rate

    # Resample the audio to our target samplerate
    if tgt_fs is not None and tgt_fs != sample_rate:
        audio = torch.from_numpy(
            librosa.core.resample(audio.numpy(), orig_sr=sample_rate, target_sr=tgt_fs)
        )
        print("do resample from", sample_rate, " to ", tgt_fs)
        sample_rate = tgt_fs
    print("after resample, audio:", audio)
    return audio, tgt_fs

def load_audio_torch(wave_file, fs):
    """
    Args:
        wave_file
        fs
    Returns:
        audio(tensor)
        fs
    """
    # Set always_2d to be true to handle stereo audio
    sample_rate = None
    if wave_file.split(".")[-1] == "wav":
        audio, sample_rate = sf.read(wave_file, always_2d=True)
    else:
        audio, sample_rate = librosa.load(wave_file, sr=fs)

    # We only use monotonic audio
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        assert len(audio) > 2

    # Check the audio type - float, 8bit or 16bit
    if np.issubdtype(audio.dtype, np.integer):
        max_mag = -np.iinfo(audio.dtype).min
    else:
        max_mag = max(np.amax(audio), -np.amin(audio))
        max_mag = (
            (2**31) + 1
            if max_mag > (2**15)
            else ((2**15) + 1 if max_mag > 1.01 else 1.0)
        )

    # Normalize the audio
    audio = torch.FloatTensor(audio.astype(np.float32)) / max_mag

    if (torch.isnan(audio) | torch.isinf(audio)).any():
        return [], sample_rate or fs or 48000

    # Resample the audio to our target samplerate
    if fs is not None and fs != sample_rate:
        audio = torch.from_numpy(
            librosa.core.resample(audio.numpy(), orig_sr=sample_rate, target_sr=fs)
        )
        sample_rate = fs

    return audio, fs
