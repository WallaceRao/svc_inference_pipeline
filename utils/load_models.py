'''
Author: lmxue
Date: 2023-06-09 18:33:33
LastEditTime: 2023-06-11 01:24:22
LastEditors: lmxue
Description: 
FilePath: /worksapce/svc_inference_pipline/utils/load_models.py
@Email: xueliumeng@gmail.com
'''


import torch
from modules.encoder import EncoderFramework
from modules.diffsvc import DiffSVC
from modules.bigvgan import Generator

def init_mapper_model(cfg):
    encoder_framework = EncoderFramework(cfg)
    acoustic_mapper = DiffSVC(cfg)
    model = torch.nn.ModuleList([encoder_framework, acoustic_mapper])
    return model

def svc_model_loader(cfg):
    print('Load mapper model from ', cfg.svc_model_path)
    model = init_mapper_model(cfg.mapper)
    ckpt = torch.load(
        cfg.svc_model_path,
        map_location=torch.device(cfg.device)
    )
    pretrained_dict = ckpt["state_dict"]
    
    # delete "module."
    weights_dict = model.state_dict()
    updated_dict = {
        k.split("module.")[-1]: v
        for k, v in pretrained_dict.items()
        if (
            k.split("module.")[-1] in weights_dict
            and v.shape == weights_dict[k.split("module.")[-1]].shape
        )
    }
    weights_dict.update(updated_dict)
    model.load_state_dict(weights_dict)
        
    if cfg.device == "cuda":
        model = model.cuda()
        
    model = model.eval()
    return model


def vocoder_model_loader(cfg):
    print("Loading vocoder model from ", cfg.vocoder_model_path)
    model = Generator(cfg.vocoder)
    ckpt = torch.load(
        cfg.vocoder_model_path,
        map_location=torch.device(cfg.device)
    )
    
    pretrained_generator_dict = ckpt["generator_state_dict"]
    generator_dict = model.state_dict()

    new_generator_dict = {
        k.split("module.")[-1]: v
        for k, v in pretrained_generator_dict.items()
        if (
            k.split("module.")[-1] in generator_dict
            and v.shape == generator_dict[k.split("module.")[-1]].shape
        )
    }
    generator_dict.update(new_generator_dict)
    
    model.load_state_dict(generator_dict)
    
    if cfg.device == "cuda":
        model = model.cuda()
    model = model.eval()
    
    return model
