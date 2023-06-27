'''
Author: lmxue
Date: 2023-06-09 14:20:08
LastEditTime: 2023-06-11 08:07:56
LastEditors: lmxue
Description: 
FilePath: /worksapce/svc_inference_pipline/infer.py
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


import logging


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler('/mnt/workspace/yonghui/svc_inference_pipeline/logs/svc_server.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


######################## IO Path ####################
wav_file = "./test_set/1100000814.wav"
singer_name = 'svcc_CDF1'
save_path = "./gen/1100000814_svcc_CDF1.wav"


######################## Load Configuration File ####################
config = "./config/config.json"
cfg = load_config(config)

if cfg.device == 'cuda':
    logger.error('using GPU...')
else:
    logger.error('using CPU...')
    

######################## Load Models ####################
logger.debug('Loading mapper and vocoder...')

svc_model = svc_model_loader(cfg)
vocoder_model = vocoder_model_loader(cfg)

start_time = time.time()

######################## Acoustic Features Extraction ####################
logger.debug('Extracting acoustic features...')

# mel, f0, energy
mel, f0, energy = acoutic_feature_extractor(wav_file, cfg)

logger.debug('Extracted mel f0 and energy')
# singer2id
singer = get_singer_id(cfg, singer_name)

# pitch shift for target singer
f0 = pitch_shift(f0, cfg)

######################## Content Features Extraction ####################
logger.debug('Extracting content features...')

whisper_feature = whisper_feature_extractor(wav_file, mel, cfg)
logger.debug('Extracted whisper_feature')
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
logger.debug('inference svc model...')

y_pred = svc_model_inference(svc_model, model_input, cfg) # [n_mel, T]
y_pred = denormalize_mel_channel(y_pred, cfg)  # [n_mel, T]

logger.debug('generated converted mel')

######################## Waveform Reconstruction ####################
logger.debug('Generating waveform...')

audio = synthesis_audios(vocoder_model, y_pred, cfg)
logger.debug('synthesis waveform finished')
end_time = time.time()
logger.debug('Using time: ', end_time - start_time)

save_audio(save_path, audio, cfg.fs)
logger.debug('Saving %s', save_path)
