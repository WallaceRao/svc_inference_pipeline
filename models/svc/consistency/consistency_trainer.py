import argparse
import numpy as np
from tqdm import tqdm
import os
import re
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.trainer_utils import check_nan, track_value
from models.svc.diffsvc.diffsvc_dataset import DiffSVCDataset, DiffSVCCollator
from models.base.base_trainer import BaseTrainer
from modules.encoder.condition_encoder import ConditionEncoder

# from models.svc.diffsvc.diffsvc import DiffSVC
from modules.diffusion.dilated_cnn.dilated_cnn import DilatedCNN
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
    find_checkpoint_of_mapper,
)


class ConsistencyTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)
        self.ema_scale_fn = self.create_ema_and_scales_fn(
            target_ema_mode="fixed",
            start_ema=0.95,
            scale_mode="fixed",
            start_scales=40,
            end_scales=None,
            total_steps=600000,
            distill_steps_per_iter=None,
        )
        self.diffusion = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            distillation=True,
            weight_schedule="karras",
        )
        self.teacher_diffusion = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            distillation=False,
            weight_schedule="karras",
        )

        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)
        self.teacher_model_path = cfg.model.teacher_model_path
        # model is the student model, target model is moving average update model
        self.teacher_acoustic_mapper = DilatedCNN(self.cfg.model.diffusion)
        self.teacher_model = torch.nn.ModuleList(
            [self.condition_encoder, self.teacher_acoustic_mapper]
        )
        state_dict = self.load_teacher_state_dict()
        self.load_teacher_model(state_dict)
        self.teacher_acoustic_mapper.cuda(self.args.local_rank)
        self.teacher_model.eval()
        for dst, src in zip(
            self.acoustic_mapper.parameters(), self.teacher_acoustic_mapper.parameters()
        ):
            dst.data.copy_(src.data)
        # self.target_condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.target_acoustic_mapper = DilatedCNN(self.cfg.model.diffusion)
        self.target_acoustic_mapper.requires_grad_(False)
        self.target_acoustic_mapper.cuda(self.args.local_rank)
        self.target_acoustic_mapper.train()
        self.save_config_file()

    def load_teacher_state_dict(self):
        self.checkpoint_file = self.teacher_model_path
        print("Load teacher acoustic model from {}".format(self.checkpoint_file))
        raw_state_dict = torch.load(self.checkpoint_file)  # , map_location=self.device)
        self.am_restore_step = re.findall(r"/step-(.+?)_loss", self.checkpoint_file)[0]
        return raw_state_dict

    def load_teacher_model(self, state_dict):
        raw_dict = state_dict["model"]
        clean_dict = OrderedDict()
        for k, v in raw_dict.items():
            if k.startswith("module."):
                clean_dict[k[7:]] = v
            elif k.startswith("0.content_vector_encoder"):
                # rename this key as "0.contentvec_encoder"
                clean_dict["0.contentvec_encoder" + k[24:]] = v
            else:
                clean_dict[k] = v
        self.teacher_model.load_state_dict(clean_dict)

    def create_ema_and_scales_fn(
        self,
        target_ema_mode,
        start_ema,
        scale_mode,
        start_scales,
        end_scales,
        total_steps,
        distill_steps_per_iter,
    ):
        def ema_and_scales_fn(step):
            if target_ema_mode == "fixed" and scale_mode == "fixed":
                target_ema = start_ema
                scales = start_scales
            elif target_ema_mode == "fixed" and scale_mode == "progressive":
                target_ema = start_ema
                scales = np.ceil(
                    np.sqrt(
                        (step / total_steps)
                        * ((end_scales + 1) ** 2 - start_scales**2)
                        + start_scales**2
                    )
                    - 1
                ).astype(np.int32)
                scales = np.maximum(scales, 1)
                scales = scales + 1

            elif target_ema_mode == "adaptive" and scale_mode == "progressive":
                scales = np.ceil(
                    np.sqrt(
                        (step / total_steps)
                        * ((end_scales + 1) ** 2 - start_scales**2)
                        + start_scales**2
                    )
                    - 1
                ).astype(np.int32)
                scales = np.maximum(scales, 1)
                c = -np.log(start_ema) * start_scales
                target_ema = np.exp(-c / scales)
                scales = scales + 1
            elif target_ema_mode == "fixed" and scale_mode == "progdist":
                distill_stage = step // distill_steps_per_iter
                scales = start_scales // (2**distill_stage)
                scales = np.maximum(scales, 2)

                sub_stage = np.maximum(
                    step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                    0,
                )
                sub_stage = sub_stage // (distill_steps_per_iter * 2)
                sub_scales = 2 // (2**sub_stage)
                sub_scales = np.maximum(sub_scales, 1)

                scales = np.where(scales == 2, sub_scales, scales)

                target_ema = 1.0
            else:
                raise NotImplementedError

            return float(target_ema), int(scales)

        return ema_and_scales_fn

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
        model = nn.ModuleList([self.condition_encoder, self.target_acoustic_mapper])
        state_dict = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def train_epoch(self):
        for i, batch_data in enumerate(self.data_loader["train"]):
            start_time = time.time()
            # Put the data to cuda device
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.cuda(self.args.local_rank)

            # Training step
            train_losses, train_stats, total_loss = self.train_step(batch_data)
            self.time_window.append(time.time() - start_time)

            if self.args.local_rank == 0 or not self.cfg.train.ddp:
                if self.step % self.args.stdout_interval == 0:
                    self.echo_log(train_losses, "Training")

                if self.step % self.cfg.train.save_summary_steps == 0:
                    self.logger.info(f"Save summary as step {self.step}")
                    self.write_summary(train_losses, train_stats)

                if (
                    self.step % self.cfg.train.save_checkpoints_steps == 0
                    and self.step != 0
                ):
                    saved_model_name = "target_step-{:07d}_loss-{:.4f}.pt".format(
                        self.step, total_loss
                    )
                    saved_model_path = os.path.join(
                        self.checkpoint_dir, saved_model_name
                    )
                    # only save target model
                    saved_state_dict = self.get_state_dict()
                    self.save_checkpoint(saved_state_dict, saved_model_path)
                    # ema_state_dicts = self.get_ema_state_dict()
                    # for rate, ema_state_dict in ema_state_dicts.items():
                    #     self.save_checkpoint(ema_state_dict, saved_model_path.replace(".pt", f"_{rate}_ema.pt"))
                    self.save_config_file()
                    # keep max n models
                    remove_older_ckpt(
                        saved_model_name,
                        self.checkpoint_dir,
                        max_to_keep=self.cfg.train.keep_checkpoint_max,
                    )

                if self.step != 0 and self.step % self.cfg.train.valid_interval == 0:
                    if isinstance(self.model, dict):
                        for key in self.model.keys():
                            self.model[key].eval()
                    else:
                        self.model.eval()
                    # Evaluate one epoch and get average loss
                    valid_losses, valid_stats = self.eval_epoch()
                    if isinstance(self.model, dict):
                        for key in self.model.keys():
                            self.model[key].train()
                    else:
                        self.model.train()
                    # Write validation losses to summary.
                    self.write_valid_summary(valid_losses, valid_stats)
            self.step += 1

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
        self.acoustic_mapper = DilatedCNN(self.cfg.model.diffusion)
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
        ema, num_scales = self.ema_scale_fn(self.step)
        compute_losses = functools.partial(
            self.diffusion.consistency_losses,
            self.acoustic_mapper,
            mel_input,
            num_scales,
            target_model=self.target_acoustic_mapper,
            teacher_model=self.teacher_acoustic_mapper,
            teacher_diffusion=self.teacher_diffusion,
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
        self._update_target_ema()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        # if self.step % self.cfg.train.save_summary_steps == 0:
        #     track_value(self.logger, self.epoch, self.step, total_loss, y_pred, noise)

        return train_losses, training_stats, total_loss.item()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.step)
        with torch.no_grad():
            self.update_ema(
                self.target_acoustic_mapper.parameters(),
                self.acoustic_mapper.parameters(),
                rate=target_ema,
            )

    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            self.update_ema(params, list(self.model.parameters()), rate=rate)

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
        ema, num_scales = self.ema_scale_fn(self.step)
        t, weights = self.schedule_sampler.sample(mel_input.shape[0], device)
        t = t.unsqueeze(-1)
        compute_losses = functools.partial(
            self.diffusion.consistency_losses,
            self.acoustic_mapper,
            mel_input,
            num_scales,
            target_model=self.target_acoustic_mapper,
            teacher_model=self.teacher_acoustic_mapper,
            teacher_diffusion=self.teacher_diffusion,
            condition=cond,
        )

        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()
        loss = torch.sum(loss * data["mask"]) / torch.sum(data["mask"])
        # check_nan(self.logger, loss, y_pred, noise)
        valid_loss["loss"] = loss

        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss.item()
