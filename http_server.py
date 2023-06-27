import http.server as BaseHTTPServer
import socketserver as SocketServer
import urllib.parse as urlparse
import threading
import re
import argparse
import json

import json
import numpy as np
import torch
from utils.acoustic_feature_extraction import acoutic_feature_extractor_from_samples, denormalize_mel_channel, pitch_shift
from utils.whisper import whisper_feature_extractor, whisper_feature_extractor_samples, load_whisper_model
from utils.hubert import contentVec_feature_extractor
from utils.util import load_config, get_singer_id, pack_data, save_audio
from utils.load_models import svc_model_loader, vocoder_model_loader
from modules.diffsvcrepo_inference import svc_model_inference
from modules.bigvgan_inference import synthesis_audios
import time
import pickle
import base64


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler('/mnt/workspace/yonghui/svc_inference_pipeline/logs/svc_server.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


config = "./config/config.json"
cfg = load_config(config)

logger.debug('Loading models...')
svc_model = svc_model_loader(cfg)
vocoder_model = vocoder_model_loader(cfg)
whisper_model = load_whisper_model(cfg.whisper_model, cfg.device)
logger.debug('Loading models finished')

def do_synthesis(samples, sample_rate, singer_name):
    global svc_model
    global vocoder_model
    global cfg
    global whisper_model
    # mel, f0, energy
    start_time = time.time()
    err_msg = ""
    singer = get_singer_id(cfg, singer_name)

    if singer == -1:
        err_msg = "no singer name:" + singer_name
        return False, err_msg
    mel, f0, energy = acoutic_feature_extractor_from_samples(samples, sample_rate, cfg)

    logger.debug('Extracted mel f0 and energy')
    # singer2id
    # pitch shift for target singer
    f0 = pitch_shift(f0, cfg)

    ######################## Content Features Extraction ####################
    logger.debug('Extracting content features...')

    whisper_feature = whisper_feature_extractor_samples(whisper_model, samples, sample_rate, mel, cfg)
    logger.debug('Extracted whisper_feature')

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
    logger.debug('Using time: %f', end_time - start_time)
    return audio, err_msg

    # save_audio(save_path, audio, cfg.fs)
    # logger.debug('Saving %s', save_path)


class apiHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        json_obj = json.loads(data_string)
        #print("received data:", data_string)
        err_msg = ""
        if "sample_rate" not in json_obj.keys():
            err_msg = "no sample rate provided"
        if "singer_name" not in json_obj.keys():
            err_msg = "no singer name provided"
        if "data" not in json_obj.keys():
            err_msg = "no data provided"
      
        data_str = json_obj["data"]
        decoded_binary = base64.b64decode(data_str)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        if err_msg != "":
            response_str = json.dumps({"err_msg": err_msg})
            self.wfile.write(response_str.encode("UTF-8"))
            return
        sample_rate = json_obj["sample_rate"]
        singer_name = json_obj["singer_name"]
        samples = np.frombuffer(decoded_binary, dtype=np.int16)
        print("samples:", samples, "sample_rate:", sample_rate, "singer_name:", singer_name)
        converted_audio, err_msg = do_synthesis(samples, sample_rate, singer_name)
        converted_audio = converted_audio.astype(np.int16)
        print("converted_audio:", converted_audio)
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
    parser=argparse.ArgumentParser()
    parser.add_argument('-ip','--address',required=False,help='IP Address for the server to listen on.  Default is 127.0.0.1',default='127.0.0.1')
    parser.add_argument('port',type=int,help='You must provide a TCP Port to bind to')
    args = parser.parse_args()
    
    #Setup the server.
    server = ThreadedServer((args.address, args.port), apiHandler)
 
    #start the server
    print('Server is Ready. http://%s:%s/<command>/<string>' % (args.address, args.port))
    print('[?] - Remember: If you are going to call the api with wget, curl or something else from the bash prompt you need to escape the & with \& \n\n')
    
    while True:
        try:
            server.handle_request()
        except KeyboardInterrupt:
            break
        
    server.safe_print("Control-C hit: Exiting server...")
    server.safe_print("Web API Disabled...")
    server.safe_print("Server has stopped.")
