from models.base.base_trainer import BaseTrainer
from diffusers import DDPMScheduler
from models.svc.diffsvc.diffsvc_dataset import DiffSVCDataset, DiffSVCCollator
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.diffsvc.diffsvc import DiffSVC
from utils.trainer_utils import check_nan


class DiffSVCTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)

        # TODO: A more sophisticated design is needed，inherit more settings from cfg
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.model.diffusion.noise_schedule_factors[2]
        )

        # Keep legacy unchanged
        if not cfg.preprocess.use_frame_pitch:
            cfg.model.condition_encoder.input_melody_dim = 0
        if not cfg.preprocess.use_frame_energy:
            cfg.model.condition_encoder.input_loudness_dim = 0
        self.save_config_file()

    # Keep legacy unchanged
    def build_dataset(self):
        return DiffSVCDataset, DiffSVCCollator

    # Keep legacy unchanged
    def build_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg.train.adam)
        return optimizer

    # Keep legacy unchanged
    def build_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, **self.cfg.train.lronPlateau)
        return scheduler

    # Keep legacy unchanged
    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    # Keep legacy unchanged
    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    # Keep legacy unchanged
    def build_criterion(self):
        criterion = nn.MSELoss(reduction="none")
        return criterion

    # Keep legacy unchanged
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

    # Keep legacy unchanged
    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    # Keep legacy unchanged
    def build_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DiffSVC(self.cfg.model.diffusion)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def train_step(self, data):
        train_losses = {}
        total_loss = 0
        training_stats = {}

        mel_input = data["mel"]
        noise = torch.randn_like(mel_input).float()

        device = mel_input.device

        batch_size = mel_input.size(0)

        # TODO: UGLY!!!
        timesteps = torch.randint(
            0,
            self.cfg.model.diffusion.noise_schedule_factors[2],
            (batch_size,),
            device=device,
        ).long()

        noisy_mel = self.noise_scheduler.add_noise(mel_input, noise, timesteps)
        conditioner = self.condition_encoder(data)

        y_pred = self.acoustic_mapper(noisy_mel, timesteps[:, None], conditioner)
        # training_stats.update(stats)

        loss = self.criterion(y_pred, noise)
        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])
        check_nan(self.logger, loss, y_pred, noise)

        train_losses["loss"] = loss
        total_loss += loss

        # BP and Grad Updated
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        # if self.step % self.cfg.train.save_summary_steps == 0:
        #     track_value(self.logger, self.epoch, self.step, total_loss, y_pred, noise)

        return train_losses, training_stats, total_loss.item()

    @torch.no_grad()
    def eval_step(self, data, index):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        mel_input = data["mel"]
        noise = torch.randn_like(mel_input).float()

        device = mel_input.device

        batch_size = mel_input.size(0)

        # TODO: UGLY!!
        timesteps = torch.randint(
            0,
            self.cfg.model.diffusion.noise_schedule_factors[2],
            (batch_size,),
            device=device,
        ).long()

        noisy_mel = self.noise_scheduler.add_noise(mel_input, noise, timesteps)
        conditioner = self.condition_encoder(data)

        y_pred = self.acoustic_mapper(noisy_mel, timesteps[:, None], conditioner)
        # training_stats.update(stats)

        loss = self.criterion(y_pred, noise)
        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])
        check_nan(self.logger, loss, y_pred, noise)
        valid_loss["loss"] = loss

        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss.item()
