import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from models.base.base_trainer import BaseTrainer
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.transformer.transformer import Transformer
from utils.trainer_utils import check_nan
from models.svc.diffsvc.diffsvc_dataset import DiffSVCDataset as TransformerDataset
from models.svc.diffsvc.diffsvc_dataset import DiffSVCCollator as TransformerCollator


class TransformerTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)
        if not cfg.preprocess.use_frame_pitch:
            cfg.model.condition_encoder.input_melody_dim = 0

        if not cfg.preprocess.use_frame_energy:
            cfg.model.condition_encoder.input_loudness_dim = 0

        self.save_config_file()

    def build_dataset(self):
        return TransformerDataset, TransformerCollator

    def build_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = Transformer(self.cfg.model.transformer)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def build_optimizer(self):
        return Adam(self.model.parameters(), **self.cfg.train.adam)

    def build_scheduler(self):
        return ReduceLROnPlateau(self.optimizer, **self.cfg.train.lronPlateau)

    def build_criterion(self):
        return MSELoss(reduction="none")

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def train_step(self, data):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        mel = data["mel"]

        condition = self.condition_encoder(data)
        mel_pred, stat = self.acoustic_mapper(condition)
        train_stats.update(stat)

        loss = self.criterion(mel_pred, mel)
        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])
        check_nan(self.logger, loss, mel_pred, mel)

        train_losses["loss"] = loss
        total_loss += loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        return train_losses, train_stats, total_loss.item()

    @torch.no_grad()
    def eval_step(self, data, index):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        mel = data["mel"]

        device = mel.device

        condition = self.condition_encoder(data)
        mel_pred, stat = self.acoustic_mapper(condition)
        valid_stats.update(stat)

        loss = self.criterion(mel_pred, mel)
        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])
        check_nan(self.logger, loss, mel_pred, mel)

        valid_loss["loss"] = loss
        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss.item()

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)
