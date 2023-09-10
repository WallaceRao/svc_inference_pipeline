import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict

from models.base.base_inference import BaseInference, base_parser
from models.svc.diffsvc.diffsvc_dataset import (
    DiffSVCTestCollator as TransformerTestCollator,
    DiffSVCTestDataset as TransformerTestDataset,
)
from utils.util import save_config, load_model_config, load_config
from utils.io import save_audio
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.transformer.transformer import Transformer
from models.vocoders.vocoder_inference import synthesis
from processors.acoustic_extractor import denorm_for_pred_mels


class TransformerInference(BaseInference):
    def __init__(self, cfg, args):
        BaseInference.__init__(self, cfg, args)
        self.args = args

    def create_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = Transformer(self.cfg.model.transformer)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def load_model(self, state_dict):
        raw_dict = state_dict["model"]
        clean_dict = OrderedDict()
        for k, v in raw_dict.items():
            if k.startswith("module."):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v

        self.model.load_state_dict(clean_dict)

    def build_criterion(self):
        criterion = nn.MSELoss(reduction="none")
        return criterion

    def inference(self, feature):
        return self.model.inference(feature)

    def build_test_dataset(self):
        return TransformerTestDataset, TransformerTestCollator

    def inference_each_batch(self, batch_data):
        criterion = self.build_criterion()
        inference_stats = {}
        condition = self.condition_encoder(batch_data)
        y_pred, stat = self.acoustic_mapper(condition)
        inference_stats.update(stat)

        x_lens = batch_data["target_len"]
        pred_res = [y_pred[i, :l].cpu() for i, l in enumerate(x_lens)]

        return pred_res, inference_stats

    def synthesis_by_vocoder(self, pred):
        audios_pred = synthesis(
            self.args.vocoder_name,
            self.cfg.vocoder,
            self.checkpoint_file_of_vocoder,
            # self.args.source_dataset,
            # "test", #split,
            len(pred),  # n_samples,
            pred,
            # save_dir='.',
            # tag=trans_key_tag,
            # ground_truth_inference=False,
        )

        return audios_pred

    def infer_for_audio(self):
        # construct test_batch

        return None

    def build_save_dir(self, source_dataset, target_singer):
        save_dir = os.path.join(
            self.args.output_dir,
            "conversion_am_step-{}".format(self.am_restore_step),
            self.args.vocoder_name,
            "{}_step-{}".format(self.vocoder_tag, self.vocoder_steps),
            "from_{}".format(source_dataset),
            "to_{}".format(target_singer),
        )

        # remove_and_create(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print("Saving to ", save_dir)

        return save_dir
