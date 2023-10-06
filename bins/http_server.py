import http.server as BaseHTTPServer
import socketserver as SocketServer
import urllib.parse as urlparse
import threading
import re
import argparse
import json
import soundfile as sf

import numpy as np
import torch

import time
import pickle
import base64

import scipy.io.wavfile

from utils.util import load_config

from models.svc.diffsvc.diffsvc_inference import DiffSVCInference
from models.svc.transformer.transformer_inference import TransformerInference
from models.vocoders.vocoder_inference import synthesis,load_nnvocoder
from processors.content_extractor import (
    ContentvecExtractor,
    WenetExtractor,
    WhisperExtractor,
)
from utils import f0
from utils.audio import load_audio_torch
from utils.data_utils import *
from utils.data_utils import pitch_shift_to_target
from utils.io import save_audio
from utils.mel import extract_mel_features
from utils.util import load_config
from inference_file import args2config, parse_vocoder, build_parser, do_convert, load_mono_audio

import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler('/workspace2/yonghui/log/svc_server.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


config = "./config/config.json"
cfg = load_config(config)

exp_name="zoulexiao_opencpop_DDPM_contentvec_conformer"
log_dir="/workspace2/lmxue/data/svc/"
acoustics_dir = log_dir + "/" + exp_name
vocoder_dir  = "/workspace2/lmxue/data/vocoder/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True"
target_singers = "opencpop_female1"
trans_key = "audo"
work_dir = "/workspace2/yonghui/svc_data/work_dir"
audio_dir = "/workspace2/yonghui/svc_data/audio_dir"
output_dir = "/workspace2/yonghui/svc_data/output_dir"
cfg = args2config(work_dir, audio_dir, output_dir)
# preload models
print("preload models")
print("preload contentvec extractor")
contentvec_extractor = ContentvecExtractor(cfg)
contentvec_extractor.load_model()
print("preload vocoder model")
vocoder_cfg, vocoder_ckpt = parse_vocoder(vocoder_dir)
vocoder_cfg.preprocess = cfg.preprocess
vocoder_cfg.preprocess.hop_length = vocoder_cfg.preprocess.hop_size
vocoder = load_nnvocoder(vocoder_cfg, "bigvgan", weights_file=vocoder_ckpt, from_multi_gpu=True)

print("preload acoustic model")
inference = None
args = build_parser().parse_args()
args.acoustics_dir = acoustics_dir
args.target_singers = target_singers
args.trans_key  = trans_key
args.audio_dir  = audio_dir
args.output_dir  = output_dir
args.checkpoint_file = None
args.checkpoint_dir = os.path.join(os.getenv("ACOUSTICS_DIR"), "checkpoints")
args.checkpoint_dir_of_vocoder = None
args.checkpoint_file_of_vocoder = None
args.inference_mode = "pndm"
if cfg.model_type == "Transformer":
    inference = TransformerInference(cfg, args)
elif cfg.model_type == "DiffSVC":
    inference = DiffSVCInference(cfg, args)
print("preload models finished")


def do_synthesis(wav_folder, sample_rate, singer_name):
    global contentvec_extractor
    global inference
    global vocoder
    global vocoder_cfg
    global vocoder_ckpt
    path =  wav_folder + "/audio_dir"
    # save samples to wav file
    # from scipy.io.wavfile import write
    # write(source_file, sample_rate, samples)
    output_path =  wav_folder + "/output_dir"
    os.makedirs(output_path, exist_ok=True)
    temp_path =  wav_folder + "/temp_dir"
    os.makedirs(temp_path, exist_ok=True)
    converted_file = output_path + "/1.wav"
    print("convert file from:", wav_folder, " to ", converted_file, ", use temp dir:", temp_path)
    do_convert(contentvec_extractor, inference, vocoder, vocoder_cfg, vocoder_ckpt, wav_folder, "opencpop_female1")
    audio, sr = load_mono_audio(converted_file)
    return audio, "OK"


class apiHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        print("data_string:", data_string)
        json_obj = json.loads(data_string)
        #print("received data:", data_string)
        err_msg = ""
        if "sample_rate" not in json_obj.keys():
            err_msg = "no sample rate provided"
        if "singer_name" not in json_obj.keys():
            err_msg = "no singer name provided"
        if "data" not in json_obj.keys():
            err_msg = "no data provided"
        if "format" not in json_obj.keys():
            err_msg = "no data format provided"
        data_format = json_obj["format"]
        if data_format != "mp3":
            err_msg = "only mp3 format is supported for now"
        if err_msg != "":
            response_str = json.dumps({"err_msg": err_msg})
            self.wfile.write(response_str.encode("UTF-8"))
            return

        data_str = json_obj["data"]
        decoded_binary = base64.b64decode(data_str)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        sample_rate = json_obj["sample_rate"]
        singer_name = json_obj["singer_name"]
        import time
        obj = time.gmtime(0)
        epoch = time.asctime(obj)
        curr_time = round(time.time()*1000)
        curr_time_str = str(curr_time)
        wav_folder = "/workspace2/yonghui/svc_data/" + curr_time_str
        os.makedirs(wav_folder, exist_ok=True)
        path =  wav_folder + "/audio_dir"
        os.makedirs(path, exist_ok=True)
        source_wav_file = path + "/1.wav"
        source_mp3_file = path + "/1.mp3"
        fout = open(source_mp3_file, 'wb')
        fout.write(decoded_binary)
        fout.close()
        # mp3 to wav
        sound = AudioSegment.from_file(source_mp3_file, format="mp3", sample_width=2, channels=1)
        samples = np.array(sound.get_array_of_samples())
        samples_normed = np.int16(samples)
        print("samples_normed:", samples_normed)
        print("samples_normed shape:", samples_normed.shape)

        #sound.export(source_wav_file, format="wav",  bitrate="384k", id3v2_version="4")
        #os.remove(source_mp3_file)
        #samples = np.frombuffer(decoded_binary, dtype=np.int16)
        from scipy.io.wavfile import write
        write(source_wav_file, sample_rate, samples_normed)
        os.remove(source_mp3_file)

        converted_audio, err_msg = do_synthesis(wav_folder, sample_rate, singer_name)
        converted_audio = converted_audio * 32768
        converted_audio = converted_audio.detach().numpy().astype(np.int16)
        pcm_bytes = converted_audio.tobytes()
        base64_str = base64.b64encode(pcm_bytes)  
        response_str = json.dumps({"sample_rate": "24000", "data": base64_str.decode(), "err_msg":err_msg})
        self.wfile.write(response_str.encode("UTF-8"))


class ThreadedServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
    def __init__(self, *args,**kwargs):
        self.screen_lock = threading.Lock()
        BaseHTTPServer.HTTPServer.__init__(self, *args, **kwargs)

    def safe_print(self,*args,**kwargs):
        try:
            self.screen_lock.acquire()
            print(*args,**kwargs)
        finally:
            self.screen_lock.release()


if __name__=="__main__":

    
    #Setup the server.
    server = ThreadedServer(("10.26.1.136", 8080), apiHandler)
 
    #start the server
    print('Server is Ready. http://%s:%s/<command>/<string>' % ("10.26.1.136", 8080))
    print('[?] - Remember: If you are going to call the api with wget, curl or something else from the bash prompt you need to escape the & with \& \n\n')
    
    while True:
        try:
            server.handle_request()
        except KeyboardInterrupt:
            break
        
    server.safe_print("Control-C hit: Exiting server...")
    server.safe_print("Web API Disabled...")
    server.safe_print("Server has stopped.")