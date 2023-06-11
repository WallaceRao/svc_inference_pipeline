'''
Author: lmxue
Date: 2023-06-09 18:45:16
LastEditTime: 2023-06-11 01:19:24
LastEditors: lmxue
Description: 
FilePath: /worksapce/svc_inference_pipline/utils/util.py
@Email: xueliumeng@gmail.com
'''

import json
import json5
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torchaudio


def save_audio(
    path, waveform, fs, add_silence=True, turn_up=True, volume_peak=0.9
):
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


def pack_data(data, device):
    packed_data = dict()
    for key, value in data.items():
        packed_data[key] = pad_sequence([torch.from_numpy(value)],
                                        batch_first=True,
                                        padding_value=0).to(device)
    return packed_data


def get_singer_id(cfg, singer_name):
    with open(cfg.singer_file, "r") as f:
        singer_lut = json.load(f)
    singer_id = np.array([singer_lut[singer_name] ], dtype=np.int32)  
                             
    return singer_id


def override_config(basic_config, new_config):
    for k, v in new_config.items():
        if type(v) == dict:
            if k not in basic_config.keys():
                basic_config[k] = {}
            basic_config[k] = override_config(basic_config[k], v)
        else:
            basic_config[k] = v
    return basic_config


def _load_config(config_fn):
    with open(config_fn, "r") as f:
        data = f.read()
    config_ = json5.loads(data)
    if 'basic_config' in config_:
        p_config_path = os.path.join(os.getenv("WORD_DIR"),
                                     config_["basic_config"])
        p_config_ = _load_config(p_config_path)
        config_ = override_config(p_config_, config_)
    return config_



def load_config(config_fn):
    config_ = _load_config(config_fn)
    hps = JsonHParams(**config_)
    return hps

def load_config(config_fn):
    config_ = _load_config(config_fn)
    hps = JsonHParams(**config_)
    return hps


class JsonHParams:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

