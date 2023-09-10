import torch
import numpy as np
from numpy import linalg as LA
import librosa
from scipy.io import wavfile
import soundfile as sf
import librosa.filters
import torchaudio


def load_audio_torch(wave_file, fs):
    """Load audio data into torch tensor

    Args:
        wave_file (str): path to wave file
        fs (int): sample rate

    Returns:
        audio (tensor): audio data in tensor
        fs (int): sample rate
    """
    # Set always_2d to be true to handle stereo audio
    sample_rate = None
    # if the file type is .wav
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


# def load_audio_torch(wave_file, fs, load_mode='torchaudio'):
#     """
#     Args:
#         wave_file
#         fs
#     Returns:
#         audio(tensor)
#         fs
#     """
#     # Set always_2d to be true to handle stereo audio
#     sample_rate = None

#     if load_mode == 'torchaudio':
#         audio, sample_rate = torchaudio.load(wave_file) # audio.shape: torch.Size([1, samples]), audio len: 2
#     elif load_mode == 'soundfile':
#         audio, sample_rate = sf.read(wave_file, always_2d=True) # audio.shape: (samples, 1), audio len: 2
#         audio = torch.from_numpy(audio).unsqueeze(0)
#     elif load_mode == 'librosa':
#         audio, sample_rate = librosa.load(wave_file, sr=fs) # audio.shape: (samples, 1), audio len: 1
#         audio = torch.from_numpy(audio).unsqueeze(0)

#     # Resample the audio to our target samplerate
#     if fs is not None and fs != sample_rate:
#         audio = torchaudio.functional.resample(
#             audio, orig_freq=sample_rate, new_freq=fs
#         )

#     audio = torch.clamp(audio[0], -1.0, 1.0) #  (samples,)

#     return audio, audio.cpu().numpy(), fs


# def extract_mel_features(
#     waveform,
#     hps,
# ):

#     with torch.no_grad():
#         mel = torchaudio.transforms.MelSpectrogram(
#             sample_rate=hps.sample_rate,
#             win_length=hps.win_size,
#             hop_length=hps.hop_size,
#             n_fft=hps.n_fft,
#             f_min=hps.fmin if hps.fmin != "None" else 20,
#             f_max=hps.fmax if hps.fmax != "None" else hps.sample_rate / 2,
#             n_mels=hps.n_mel,
#             power=1.0,
#             normalized=True,
#         )(waveform)
#         # amplitude_to_db
#         mel = 20 * torch.log10(torch.clamp(mel, min=1e-5)) - 20

#     return mel.cpu().numpy()


# def load_wav(wav_path, raw_sr, target_sr=16000, win_size=800, hop_size=200):
#     audio = librosa.core.load(wav_path, sr=raw_sr)[0]
#     if raw_sr != target_sr:
#         audio = librosa.core.resample(audio,
#                                       raw_sr,
#                                       target_sr,
#                                       res_type='kaiser_best')
#         target_length = (audio.size // hop_size +
#                          win_size // hop_size) * hop_size
#         pad_len = (target_length - audio.size) // 2
#         if audio.size % 2 == 0:
#             audio = np.pad(audio, (pad_len, pad_len), mode='reflect')
#         else:
#             audio = np.pad(audio, (pad_len, pad_len + 1), mode='reflect')
#     return audio


# def save_wav(wav, path, sample_rate, norm=False):
#     if norm:
#         wav *= 32767 / max(0.01, np.max(np.abs(wav)))
#         wavfile.write(path, sample_rate, wav.astype(np.int16))
#     else:
#         sf.write(path, wav, sample_rate)


def _stft(y, cfg):
    return librosa.stft(
        y=y, n_fft=cfg.n_fft, hop_length=cfg.hop_size, win_length=cfg.win_size
    )


def energy(wav, cfg):
    D = _stft(wav, cfg)
    magnitudes = np.abs(D).T  # [F, T]
    return LA.norm(magnitudes, axis=1)


# def trim_silence(wav, cfg):
#     '''
#     Trim leading and trailing silence
#     '''
#     # These params are separate and tunable per dataset.
#     unused_trimed, index = librosa.effects.trim(
#         wav,
#         top_db=cfg.preprocess.trim_top_db,
#         frame_length=cfg.preprocess.trim_fft_size,
#         hop_length=cfg.preprocess.trim_hop_size)

#     num_sil_samples = \
#         int(cfg.preprocess.num_silent_frames * cfg.data.hop_size)
#     start_idx = max(index[0] - num_sil_samples, 0)
#     stop_idx = min(index[1] + num_sil_samples, len(wav))
#     trimmed = wav[start_idx:stop_idx]

#     return trimmed
