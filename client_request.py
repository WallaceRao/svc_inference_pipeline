import asyncio
import websocket
import ssl

import json
import websocket
import _thread as thread
import base64
 
import time
import datetime

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
    print("pcm len:", len(pcm))
    base64_str = base64.b64encode(pcm)  
    print("after encoding len:", len(base64_str))

    conn = http.client.HTTPConnection('127.0.0.1:8080')
    headers = {'Content-type': 'application/json'}
    foo = {'data': base64_str.decode(), "sample_rate": 24000, "singer_name":"svcc_CDF1"}
    json_data = json.dumps(foo)
    res = conn.request("POST", "convert", json_data)
    response = conn.getresponse()
    response_bytes = response.read()
    print("response len:", len(response_bytes))
    json_obj = json.loads(response_bytes)
    print("reponse keys:", json_obj.keys())
    print(response.status, response.reason)
