import torch
import librosa

from scipy.io.wavfile import read

import soundfile as sf
import numpy as np


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
