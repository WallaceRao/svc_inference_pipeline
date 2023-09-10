import torch
import random

import numpy as np

from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import BaseDataset, BaseCollator
from processors.acoustic_extractor import cal_normalized_mel


class GANVocoderDataset(BaseDataset):
    def __init__(self, cfg, is_valid=False):
        super().__init__(cfg, is_valid)

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        if self.cfg.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.n_mel
            if self.cfg.use_min_max_norm_mel:
                mel = cal_normalized_mel(mel, utt_info["Dataset"], self.cfg)

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]

            if single_feature["target_len"] <= self.cfg.cut_mel_frame + 2:
                mel = F.pad(
                    mel,
                    (0, self.cfg.cut_mel_frame - mel.shape[-1], 0, 0),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0, mel.shape[-1] - self.cfg.cut_mel_frame - 2
                    )
                    end = start + self.cfg.cut_mel_frame
                    single_feature["start"] = start
                    single_feature["end"] = end
                mel = mel[:, single_feature["start"] : single_feature["end"]]
            single_feature["mel"] = mel

        if self.cfg.use_frame_pitch:
            frame_pitch = self.frame_utt2pitch[utt]
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)
            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )

            if single_feature["target_len"] <= self.cfg.cut_mel_frame + 2:
                aligned_frame_pitch = F.pad(
                    aligned_frame_pitch,
                    (
                        0,
                        self.cfg.cut_mel_frame - aligned_frame_pitch.shape[-1],
                    ),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0,
                        aligned_frame_pitch.shape[-1] - self.cfg.cut_mel_frame - 2,
                    )
                    end = start + self.cfg.cut_mel_frame
                    single_feature["start"] = start
                    single_feature["end"] = end
                aligned_frame_pitch = aligned_frame_pitch[
                    single_feature["start"] : single_feature["end"]
                ]
            single_feature["frame_pitch"] = aligned_frame_pitch

        if self.cfg.use_audio:
            audio = np.load(self.utt2audio_path[utt])

            assert "target_len" in single_feature.keys()

            if single_feature["target_len"] <= self.cfg.cut_mel_frame + 2:
                audio = F.pad(
                    audio,
                    (
                        0,
                        self.cfg.cut_mel_frame * self.cfg.hop_size - audio.shape[-1],
                    ),
                    mode="constant",
                )
            else:
                audio = audio[
                    single_feature["start"]
                    * self.cfg.hop_size : single_feature["end"]
                    * self.cfg.hop_size,
                ]
            single_feature["audio"] = audio

        return single_feature


class GANVocoderCollator(BaseCollator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, n_mels, frame]
        # frame_pitch: [b, frame]
        # audios: [b, frame * hop_size]

        for key in batch[0].keys():
            if key in ["target_len", "start", "end"]:
                continue
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
