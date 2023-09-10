import os
import sys
import time
import torch
import itertools
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from librosa.filters import mel as librosa_mel_fn

from utils.util import (
    Logger,
    ValueWindow,
    remove_older_ckpt,
    set_all_random_seed,
    save_config,
)
from utils.mel import extract_mel_features
from models.base.base_trainer import BaseTrainer
from models.vocoders.gan.gan_vocoder_dataset import (
    GANVocoderDataset,
    GANVocoderCollator,
)

from models.vocoders.gan.generator.bigvgan import BigVGAN
from models.vocoders.gan.generator.hifigan import HiFiGAN
from models.vocoders.gan.generator.melgan import MelGAN
from models.vocoders.gan.generator.nsfhifigan import NSFHiFiGAN

from models.vocoders.gan.discriminator.mpd import MultiPeriodDiscriminator
from models.vocoders.gan.discriminator.mrd import MultiResolutionDiscriminator
from models.vocoders.gan.discriminator.mscqtd import MultiScaleCQTDiscriminator
from models.vocoders.gan.discriminator.msd import MultiScaleDiscriminator
from models.vocoders.gan.discriminator.msstftd import MultiScaleSTFTDiscriminator


class GANVocoderTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.log_dir = args.log_dir
        self.cfg = cfg

        self.checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if not cfg.train.ddp or args.local_rank == 0:
            self.sw = SummaryWriter(os.path.join(args.log_dir, "events"))
            self.logger = self.build_logger()
        self.time_window = ValueWindow(50)

        self.step = 0
        self.epoch = -1
        self.max_epochs = self.cfg.train.epochs
        self.max_steps = self.cfg.train.max_steps

        # set random seed & init distributed training
        set_all_random_seed(self.cfg.train.random_seed)
        if cfg.train.ddp:
            dist.init_process_group(backend="nccl")

        # setup data_loader
        self.data_loader = self.build_data_loader()

        # setup model & enable distributed training
        self.generator, self.discriminators = self.build_model()
        print(self.generator)
        print(self.discriminators)

        self.generator.cuda(self.args.local_rank)
        if cfg.train.ddp:
            self.generator = DistributedDataParallel(
                self.generator, device_ids=[self.args.local_rank]
            )

        for key, value in self.discriminators.items():
            value.cuda(self.args.local_rank)
            if key == "PQMF":
                continue
            if cfg.train.ddp:
                self.discriminators[key] = DistributedDataParallel(
                    value, device_ids=[self.args.local_rank]
                )

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.build_optimizer()
        self.generator_scheduler, self.discriminator_scheduler = self.build_scheduler()

        # create criterion
        self.criterions = self.build_criterion()
        for key, value in self.criterions.items():
            self.criterions[key].cuda(args.local_rank)

        # save config file
        self.config_save_path = os.path.join(self.checkpoint_dir, "args.json")

    def build_dataset(self):
        return GANVocoderDataset, GANVocoderCollator

    def build_model(self):
        if self.cfg.model.generator == "bigvgan":
            generator = BigVGAN(self.cfg)
        elif self.cfg.model.generator == "hifigan":
            generator = HiFiGAN(self.cfg)
        elif self.cfg.model.generator == "melgan":
            generator = MelGAN(self.cfg)
        elif self.cfg.model.generator == "nsfhifigan":
            generator = NSFHiFiGAN(self.cfg)
        else:
            raise NotImplementedError

        discriminators = dict()

        if "mpd" in self.cfg.model.discriminators:
            discriminators["mpd"] = MultiPeriodDiscriminator(self.cfg)
        if "mrd" in self.cfg.model.discriminators:
            discriminators["mrd"] = MultiResolutionDiscriminator(self.cfg)
        if "mscqtd" in self.cfg.model.discriminators:
            discriminators["mscqtd"] = MultiScaleCQTDiscriminator(self.cfg)
        if "msd" in self.cfg.model.discriminators:
            discriminators["msd"] = MultiScaleDiscriminator(self.cfg)
        if "msstftd" in self.cfg.model.discriminators:
            discriminators["msstftd"] = MultiScaleSTFTDiscriminator(
                self.cfg.model.msstftd.filters
            )

        return generator, discriminators

    def build_optimizer(self):
        optimizer_params_generator = [dict(params=self.generator.parameters())]
        generator_optimizer = AdamW(
            optimizer_params_generator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
        )

        optimizer_params_discriminator = []
        for discriminator in self.discriminators.keys():
            optimizer_params_discriminator.append(
                dict(params=self.discriminators[discriminator].parameters())
            )
        discriminator_optimizer = AdamW(
            optimizer_params_discriminator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
        )

        return generator_optimizer, discriminator_optimizer

    def build_scheduler(self):
        discriminator_scheduler = ExponentialLR(
            self.discriminator_optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch,
        )

        generator_scheduler = ExponentialLR(
            self.generator_optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch,
        )

        return generator_scheduler, discriminator_scheduler

    def build_criterion(self):
        class feature_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(feature_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, fmap_r, fmap_g):
                loss = 0

                if self.cfg.model.generator in ["hifigan", "nsfhifigan", "bigvgan"]:
                    for dr, dg in zip(fmap_r, fmap_g):
                        for rl, gl in zip(dr, dg):
                            loss += torch.mean(torch.abs(rl - gl))

                    loss = loss * 2
                elif self.cfg.model.generator in ["melgan"]:
                    for dr, dg in zip(fmap_r, fmap_g):
                        for rl, gl in zip(dr, dg):
                            loss += self.l1Loss(rl, gl)

                    loss = loss * 10
                elif self.cfg.model.generator in ["codec"]:
                    for dr, dg in zip(fmap_r, fmap_g):
                        for rl, gl in zip(dr, dg):
                            loss = loss + self.l1Loss(rl, gl) / torch.mean(
                                torch.abs(rl)
                            )

                    KL_scale = len(fmap_r) * len(fmap_r[0])

                    loss = 3 * loss / KL_scale
                else:
                    raise NotImplementedError

                return loss

        class discriminator_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(discriminator_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, disc_real_outputs, disc_generated_outputs):
                loss = 0
                r_losses = []
                g_losses = []

                if self.cfg.model.generator in ["hifigan", "nsfhifigan", "bigvgan"]:
                    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                        r_loss = torch.mean((1 - dr) ** 2)
                        g_loss = torch.mean(dg**2)
                        loss += r_loss + g_loss
                        r_losses.append(r_loss.item())
                        g_losses.append(g_loss.item())
                elif self.cfg.model.generator in ["melgan"]:
                    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                        r_loss = torch.mean(self.relu(1 - dr))
                        g_loss = torch.mean(self.relu(1 + dg))
                        loss = loss + r_loss + g_loss
                        r_losses.append(r_loss.item())
                        g_losses.append(g_loss.item())
                elif self.cfg.model.generator in ["codec"]:
                    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                        r_loss = torch.mean(self.relu(1 - dr))
                        g_loss = torch.mean(self.relu(1 + dg))
                        loss = loss + r_loss + g_loss
                        r_losses.append(r_loss.item())
                        g_losses.append(g_loss.item())

                    loss = loss / len(disc_real_outputs)
                else:
                    raise NotImplementedError

                return loss, r_losses, g_losses

        class generator_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(generator_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, disc_outputs):
                loss = 0
                gen_losses = []

                if self.cfg.model.generator in ["hifigan", "nsfhifigan", "bigvgan"]:
                    for dg in disc_outputs:
                        l = torch.mean((1 - dg) ** 2)
                        gen_losses.append(l)
                        loss += l
                elif self.cfg.model.generator in ["melgan"]:
                    for dg in disc_outputs:
                        l = -torch.mean(dg)
                        gen_losses.append(l)
                        loss += l
                elif self.cfg.model.generator in ["codec"]:
                    for dg in disc_outputs:
                        l = torch.mean(self.relu(1 - dg)) / len(disc_outputs)
                        gen_losses.append(l)
                        loss += l
                else:
                    raise NotImplementedError

                return loss, gen_losses

        class mel_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(mel_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, y_gt, y_pred):
                loss = 0

                if self.cfg.model.generator in [
                    "hifigan",
                    "nsfhifigan",
                    "bigvgan",
                    "melgan",
                    "codec",
                ]:
                    y_gt_mel = extract_mel_features(y_gt, self.cfg.preprocess)
                    y_pred_mel = extract_mel_features(
                        y_pred.squeeze(1), self.cfg.preprocess
                    )

                    loss = self.l1Loss(y_gt_mel, y_pred_mel) * 45
                else:
                    raise NotImplementedError

                return loss

        class wav_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(wav_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, y_gt, y_pred):
                loss = 0

                if self.cfg.model.generator in ["hifigan", "nsfhifigan", "bigvgan"]:
                    loss = self.l2Loss(y_gt, y_pred.squeeze(1)) * 100
                elif self.cfg.model.generator in ["melgan"]:
                    loss = self.l1Loss(y_gt, y_pred.squeeze(1)) / 10
                elif self.cfg.model.generator in ["codec"]:
                    loss = self.l1Loss(y_gt, y_pred.squeeze(1)) + self.l2Loss(
                        y_gt, y_pred.squeeze(1)
                    )
                    loss /= 10
                else:
                    raise NotImplementedError

                return loss

        criterions = dict()
        for key in self.cfg.train.criterions:
            if key == "feature":
                criterions["feature"] = feature_criterion(self.cfg)
            elif key == "discriminator":
                criterions["discriminator"] = discriminator_criterion(self.cfg)
            elif key == "generator":
                criterions["generator"] = generator_criterion(self.cfg)
            elif key == "mel":
                criterions["mel"] = mel_criterion(self.cfg)
            elif key == "wav":
                criterions["wav"] = wav_criterion(self.cfg)
            else:
                raise NotImplementedError

        return criterions

    def get_state_dict(self):
        state_dict = {
            "generator_state_dict": self.generator.state_dict(),
            "generator_optimizer_state_dict": self.generator_optimizer.state_dict(),
            "generator_scheduler_state_dict": self.generator_scheduler.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "discriminator_scheduler_state_dict": self.discriminator_scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        for key, value in self.discriminators.items():
            state_dict["{}_state_dict".format(key)] = value.state_dict()
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state_dict"]
        )
        self.generator_scheduler.load_state_dict(
            checkpoint["generator_scheduler_state_dict"]
        )

        for key, _ in self.discriminators.items():
            self.discriminators[key].load_state_dict(
                checkpoint["{}_state_dict".format(key)]
            )
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )
        self.discriminator_scheduler.load_state_dict(
            checkpoint["discriminator_scheduler_state_dict"]
        )

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

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
                    saved_model_name = "step-{:07d}_loss-{:.4f}.pt".format(
                        self.step, total_loss
                    )
                    saved_model_path = os.path.join(
                        self.checkpoint_dir, saved_model_name
                    )
                    saved_state_dict = self.get_state_dict()
                    self.save_checkpoint(saved_state_dict, saved_model_path)
                    self.save_config_file()
                    # keep max n models
                    remove_older_ckpt(
                        saved_model_name,
                        self.checkpoint_dir,
                        max_to_keep=self.cfg.train.keep_checkpoint_max,
                    )

                if self.step != 0 and self.step % self.cfg.train.valid_interval == 0:
                    self.generator.eval()
                    for key in self.discriminators.keys():
                        self.discriminators[key].eval()
                    # Evaluate one epoch and get average loss
                    valid_losses, valid_stats = self.eval_epoch()
                    self.generator.train()
                    for key in self.discriminators.keys():
                        self.discriminators[key].train()
                    # Write validation losses to summary.
                    self.write_valid_summary(valid_losses, valid_stats)
            self.step += 1

    def train_step(self, data):
        train_losses = {}
        total_loss = 0
        training_stats = {}

        generator_losses = {}
        generator_total_loss = 0
        discriminator_losses = {}
        discriminator_total_loss = 0

        mel_input = data["mel"]
        audio_gt = data["audio"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = data["frame_pitch"]

        device = mel_input.device

        if self.cfg.preprocess.use_frame_pitch:
            audio_pred = self.generator.forward(mel_input, pitch_input)
        else:
            audio_pred = self.generator.forward(mel_input)

        self.discriminator_optimizer.zero_grad()
        for key, _ in self.discriminators.items():
            y_r, y_g, _, _ = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred.detach()
            )
            (
                discriminator_losses["{}_discriminator_loss".format(key)],
                _,
                _,
            ) = self.criterions["discriminator"](y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_discriminator_loss".format(key)
            ]

        discriminator_total_loss.backward()
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        for key, _ in self.discriminators.items():
            y_r, y_g, f_r, f_g = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred
            )
            generator_losses["{}_feature".format(key)] = self.criterions["feature"](
                f_r, f_g
            )
            generator_losses["{}_generator".format(key)], _ = self.criterions[
                "generator"
            ](y_g)
            generator_total_loss += generator_losses["{}_feature".format(key)]
            generator_total_loss += generator_losses["{}_generator".format(key)]

        if "mel" in self.criterions.keys():
            generator_losses["mel"] = self.criterions["mel"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["mel"]

        if "wav" in self.criterions.keys():
            generator_losses["wav"] = self.criterions["wav"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["wav"]

        generator_total_loss.backward()
        self.generator_optimizer.step()

        total_loss = discriminator_total_loss + generator_total_loss
        train_losses.update(discriminator_losses)
        train_losses.update(generator_losses)

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        return train_losses, training_stats, total_loss.item()

    def eval_epoch(self):
        self.logger.info("Validation...")
        valid_losses = {}
        for i, batch_data in enumerate(self.data_loader["valid"]):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.cuda()
            valid_loss, valid_stats, total_valid_loss = self.eval_step(batch_data, i)
            for key in valid_loss:
                if key not in valid_losses:
                    valid_losses[key] = 0
                valid_losses[key] += valid_loss[key]

        # Add mel and audio to the Tensorboard

        # Average loss
        for key in valid_losses:
            valid_losses[key] /= i + 1
        self.echo_log(valid_losses, "Valid")
        return valid_losses, valid_stats

    def eval_step(self, data):
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        generator_losses = {}
        generator_total_loss = 0
        discriminator_losses = {}
        discriminator_total_loss = 0

        mel_input = data["mel"]
        audio_gt = data["audio"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = data["frame_pitch"]

        device = mel_input.device

        if self.cfg.preprocess.use_frame_pitch:
            audio_pred = self.generator.forward(mel_input, pitch_input)
        else:
            audio_pred = self.generator.forward(mel_input)

        for key, _ in self.discriminators.items():
            y_r, y_g, _, _ = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred
            )
            (
                discriminator_losses["{}_discriminator_loss".format(key)],
                _,
                _,
            ) = self.criterions["discriminator"](y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_discriminator_loss".format(key)
            ]

        for key, _ in self.discriminators.items():
            y_r, y_g, f_r, f_g = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred
            )
            generator_losses["{}_feature".format(key)] = self.criterions["feature"](
                f_r, f_g
            )
            generator_losses["{}_generator".format(key)], _ = self.criterions[
                "generator"
            ](y_g)
            generator_total_loss += generator_losses["{}_feature".format(key)]
            generator_total_loss += generator_losses["{}_generator".format(key)]

        if "mel" in self.criterions.keys():
            generator_losses["mel"] = self.criterions["mel"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["mel"]

        if "wav" in self.criterions.keys():
            generator_losses["wav"] = self.criterions["wav"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["wav"]

        total_loss = discriminator_total_loss + generator_total_loss
        valid_losses.update(discriminator_losses)
        valid_losses.update(generator_losses)

        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()

        return valid_losses, valid_stats, total_loss.item()
