import argparse
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.trainer_utils import check_nan, track_value
from models.svc.diffsvc.diffsvc_dataset import DiffSVCDataset, DiffSVCCollator
from models.base.base_trainer import BaseTrainer
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.diffsvc.diffsvc import DiffSVC
from modules.diffusion.karras.karras_diffusion import KarrasDenoiser
from modules.diffusion.karras.sample import create_named_schedule_sampler
import functools
import copy
import time
from utils.util import (
    Logger,
    ValueWindow,
    remove_older_ckpt,
    set_all_random_seed,
    save_config,
)


class EDMTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)
        self.diffusion = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            distillation=False,
            weight_schedule="karras",
        )
        self.schedule_sampler = create_named_schedule_sampler(
            self.cfg.model.diffusion.schedule_sampler, self.diffusion
        )
        self.save_config_file()

    def build_dataset(self):
        return DiffSVCDataset, DiffSVCCollator

    def build_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg.train.adam)
        return optimizer

    def build_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, **self.cfg.train.lronPlateau)
        return scheduler

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def build_criterion(self):
        criterion = nn.MSELoss(reduction="none")
        return criterion

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
        cond = self.condition_encoder(data)
        self.optimizer.zero_grad()
        device = mel_input.device
        t, weights = self.schedule_sampler.sample(mel_input.shape[0], device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.acoustic_mapper,
            mel_input,
            t,
            condition=cond,
        )

        losses = compute_losses()

        loss = (losses["loss"] * weights).mean()

        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])

        train_losses["loss"] = loss
        total_loss += loss

        # BP and Grad Updated
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

        device = mel_input.device
        mel_input = data["mel"]
        cond = self.condition_encoder(data)
        self.optimizer.zero_grad()
        device = mel_input.device
        t, weights = self.schedule_sampler.sample(mel_input.shape[0], device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.acoustic_mapper,
            mel_input,
            t,
            condition=cond,
        )

        losses = compute_losses()
        # print(losses)
        # print(weights)
        loss = (losses["loss"] * weights).mean()
        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])
        # check_nan(self.logger, loss, y_pred, noise)
        valid_loss["loss"] = loss

        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss.item()
