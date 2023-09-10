import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict

from models.base.base_inference import BaseInference, base_parser
from models.svc.diffsvc.diffsvc_dataset import DiffSVCTestCollator, DiffSVCTestDataset
from utils.util import save_config, load_model_config, load_config
from utils.io import save_audio
from modules.encoder.condition_encoder import ConditionEncoder
from modules.diffusion.dilated_cnn.dilated_cnn import DilatedCNN
from models.vocoders.vocoder_inference import synthesis
from processors.acoustic_extractor import denorm_for_pred_mels
from modules.diffusion.karras.karras_diffusion import KarrasDenoiser
from modules.diffusion.karras.random_utils import get_generator
from modules.diffusion.karras.karras_diffusion import karras_sample


class EDMInference(BaseInference):
    def __init__(self, cfg, args):
        BaseInference.__init__(self, cfg, args)
        self.args = args
        self.diffusion = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            distillation=False,
            weight_schedule="karras",
        )

    def create_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DilatedCNN(self.cfg.model.diffusion)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def load_model(self, state_dict):
        raw_dict = state_dict["model"]
        clean_dict = OrderedDict()
        for k, v in raw_dict.items():
            if k.startswith("module."):
                clean_dict[k[7:]] = v
            else:
                # drop diff.
                k_clean = k.replace("diff.", "")
                clean_dict[k_clean] = v

        self.model.load_state_dict(clean_dict)

    def build_criterion(self):
        criterion = nn.MSELoss(reduction="none")
        return criterion

    def inference(self, feature):
        return self.model.inference(feature)

    def build_test_dataset(self):
        return DiffSVCTestDataset, DiffSVCTestCollator

    def inference_each_batch(self, batch_data):
        cond = self.condition_encoder(batch_data)
        mel_output = batch_data["mel"]
        device = mel_output.device
        y_pred = karras_sample(
            self.diffusion,
            self.acoustic_mapper,
            mel_output.shape,
            steps=self.args.steps,
            condition=cond,
            device=device,
            clip_denoised=True,
            sampler=self.args.sampler,
            sigma_min=self.args.sigma_min,
            sigma_max=self.args.sigma_max,
            s_churn=self.args.s_churn,
            s_tmin=0.0,
            s_tmax=float("inf"),
            s_noise=1.0,
            generator=None,
            ts=None,
        )
        y_pred = y_pred.cpu().detach().numpy()
        x_lens = batch_data["target_len"]
        pred_res = [y_pred[i, :l] for i, l in enumerate(x_lens)]
        return pred_res, 0

    def build_save_dir(self, source_dataset, target_singer):
        save_dir = os.path.join(
            self.args.output_dir,
            "conversion_am_step-{}_infer_steps_{}".format(
                self.am_restore_step, self.args.steps
            ),
            self.args.vocoder_name,
            "{}_step-{}".format(self.vocoder_tag, self.vocoder_steps),
            "from_{}".format(source_dataset),
            "to_{}".format(target_singer),
        )

        # remove_and_create(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print("Saving to ", save_dir)

        return save_dir
