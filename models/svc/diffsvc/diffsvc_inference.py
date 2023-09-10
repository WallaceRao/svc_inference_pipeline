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
from models.svc.diffsvc.diffsvc import DiffSVC
from models.svc.diffsvc import diffusion_inference

from models.vocoders.vocoder_inference import synthesis
from processors.acoustic_extractor import denorm_for_pred_mels


class DiffSVCInference(BaseInference):
    def __init__(self, cfg, args):
        BaseInference.__init__(self, cfg, args)
        self.args = args

    def create_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DiffSVC(self.cfg.model.diffusion)
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
        return DiffSVCTestDataset, DiffSVCTestCollator

    def inference_each_batch(self, batch_data):
        criterion = self.build_criterion()

        y_pred = diffusion_inference.diffsvc_inference(
            self.cfg, self.model, batch_data, self.args.inference_mode
        )

        x_lens = batch_data["target_len"]
        pred_res = [y_pred[i, :l] for i, l in enumerate(x_lens)]

        return pred_res, 0

    def synthesis_by_vocoder(self, pred):
        # TODO: Remove HARDCODE!!!
        self.cfg.preprocess.hop_length = self.cfg.preprocess.hop_size
        self.cfg.model.bigvgan = self.cfg.vocoder.vocoder
        audios_pred = synthesis(
            "bigvgan",
            self.cfg,
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
        raise NotImplementedError

    # def __call__(self, utt):
    #     feature = self.make_feature(utt)
    #     with torch.no_grad():
    #         start_time = time.time()
    #         pred_mel, stats = self.inference(feature)
    #         time_used = time.time() - start_time
    #         rtf = time_used / (pred_mel.shape[
    #                                2] * self.cfg.data.hop_size / self.cfg.data.sample_rate)
    #         print("Time used: {:.3f}, RTF: {:.4f}".format(time_used, rtf))
    #         self.avg_rtf.append(rtf)
    #     pred_mel = pred_mel.cpu().squeeze(0).numpy()
    #     mel_path = os.path.join(self.args.out_dir, utt)
    #     np.save(mel_path, pred_mel)
    #     return pred_mel

    def build_save_dir(self, source_dataset, target_singer):
        fast_inference = "_fast_infer" if self.args.fast_inference else ""
        save_dir = os.path.join(
            self.args.output_dir,
            "conversion_am_step-{}_{}{}".format(
                self.am_restore_step, self.args.inference_mode, fast_inference
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
