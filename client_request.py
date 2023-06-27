import asyncio
import websocket
import ssl
import numpy as np
import json
import websocket
import _thread as thread
import base64
 
import time
import datetime
import wave
import sys  
  
import http.client
import json

result_folder="result" 


i = 0
total_decoded_binary = []
total_size = 0
default_test_file="/mnt/workspace/yonghui/svc_inference_pipeline/0000.wav"


if __name__ == "__main__":
    f = open(default_test_file,"rb") 
    header = f.read(44)
    print("header len:", len(header))
    pcm = f.read()
    print("input samples count:", len(pcm) / 2)
    base64_str = base64.b64encode(pcm)  
    print("after encoding len:", len(base64_str))

    conn = http.client.HTTPConnection('127.0.0.1:8080')
    headers = {'Content-type': 'application/json'}
    foo = {'data': base64_str.decode(), "sample_rate": 48000, "singer_name":"svcc_CDF1"}
    json_data = json.dumps(foo)
    res = conn.request("POST", "convert", json_data)
    response = conn.getresponse()
    response_bytes = response.read()
    print("response len:", len(response_bytes))
    json_obj = json.loads(response_bytes)
    print("reponse keys:", json_obj.keys())
    print(response.status, response.reason)
    if "data" not in json_obj.keys():
        print("not data field returned")
    data_str = json_obj["data"]
    decoded_binary = base64.b64decode(data_str)
    samples = np.frombuffer(decoded_binary, dtype=np.int16)
    print("samples count:", len(samples))
    pcm_bytes = samples.tobytes()
    with wave.open("/mnt/workspace/yonghui/svc_inference_pipeline/converted.wav", "wb") as out_f:
        out_f.setnchannels(1)
        out_f.setsampwidth(2) # number of bytes
        out_f.setframerate(24000)
        out_f.writeframesraw(pcm_bytes)

