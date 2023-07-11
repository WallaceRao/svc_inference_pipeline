'''
Author: lmxue
Date: 2023-06-09 14:20:08
LastEditTime: 2023-07-11 18:51:48
LastEditors: lmxue
Description: 
FilePath: /svc_inference_pipeline/infer.py
@Email: xueliumeng@gmail.com
'''

import json
import numpy as np
import torch
from utils.acoustic_feature_extraction import acoutic_feature_extractor, denormalize_mel_channel, pitch_shift
from utils.whisper import whisper_feature_extractor
from utils.hubert import contentVec_feature_extractor
from utils.util import load_config, get_singer_id, pack_data, save_audio
from utils.load_models import svc_model_loader, vocoder_model_loader
from modules.diffsvcrepo_inference import svc_model_inference
from modules.bigvgan_inference import synthesis_audios
import time
import pickle


######################## IO Path ####################
wav_file = "./test_set/1100000814.wav"
singer_name = 'svcc_CDF1'
save_path = "./gen/1100000814_svcc_CDF1_fast_inference.wav"


######################## Load Configuration File ####################
config = "./config/config.json"
cfg = load_config(config)

if cfg.device == 'cuda':
    print('Using GPU...')
else:
    print("Using CPU...")
    

######################## Load Models ####################
print('Loading mapper and vocoder...')

svc_model = svc_model_loader(cfg)
vocoder_model = vocoder_model_loader(cfg)

start_time = time.time()

######################## Acoustic Features Extraction ####################
print('Extracting acoustic features...')

# mel, f0, energy
mel, f0, energy = acoutic_feature_extractor(wav_file, cfg)

# singer2id
singer = get_singer_id(cfg, singer_name)

# pitch shift for target singer
f0 = pitch_shift(f0, cfg)

######################## Content Features Extraction ####################
print('Extracting content features...')

whisper_feature = whisper_feature_extractor(wav_file, mel, cfg)
# contentVec_feature = contentVec_feature_extractor(wav_file, mel, cfg)

######################## Construct Data ####################
input_data = dict()
input_data["y"] = mel
input_data["melody"] = f0
input_data["loudness"] = energy
input_data["singer"] = singer
input_data["content_whisper"] = whisper_feature
model_input = pack_data(input_data, cfg.device)

######################## Converion ####################
print('Converion...')

y_pred = svc_model_inference(svc_model, model_input, cfg, fast_inference=True) # [n_mel, T]
y_pred = denormalize_mel_channel(y_pred, cfg)  # [n_mel, T]


######################## Waveform Reconstruction ####################
print('Generating waveform...')

audio = synthesis_audios(vocoder_model, y_pred, cfg)
end_time = time.time()
print('Using time: ', end_time - start_time)

save_audio(save_path, audio, cfg.fs)
print('Saving ', save_path)
