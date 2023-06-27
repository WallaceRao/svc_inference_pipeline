import os
import collections
from collections import OrderedDict
import random
import json5
import numpy as np
import glob


try:
    from ruamel.yaml import YAML as yaml
except:
    from ruamel_yaml import YAML as yaml

import torch

from utils.hparam import HParams  # from tensorflow.contrib.training import HParams
import logging
from logging import handlers


def find_checkpoint_of_mapper(mapper_ckpt_dir):
    mapper_ckpts = glob.glob(os.path.join(mapper_ckpt_dir, "ckpts/*.pt"))

    # Select the max steps
    mapper_ckpts.sort()
    mapper_weights_file = mapper_ckpts[-1]
    return mapper_weights_file


def pad_f0_to_tensors(f0s, batched=None):
    # Initialize
    tensors = []

    if batched == None:
        # Get the max frame for padding
        size = -1
        for f0 in f0s:
            size = max(size, f0.shape[-1])

        tensor = torch.zeros(len(f0s), size)

        for i, f0 in enumerate(f0s):
            tensor[i, : f0.shape[-1]] = f0[:]

        tensors.append(tensor)
    else:
        start = 0
        while start + batched - 1 < len(f0s):
            end = start + batched - 1

            # Get the max frame for padding
            size = -1
            for i in range(start, end + 1):
                size = max(size, f0s[i].shape[-1])

            tensor = torch.zeros(batched, size)

            for i in range(start, end + 1):
                tensor[i - start, : f0s[i].shape[-1]] = f0s[i][:]

            tensors.append(tensor)

            start = start + batched

        if start != len(f0s):
            end = len(f0s)

            # Get the max frame for padding
            size = -1
            for i in range(start, end):
                size = max(size, f0s[i].shape[-1])

            tensor = torch.zeros(len(f0s) - start, size)

            for i in range(start, end):
                tensor[i - start, : f0s[i].shape[-1]] = f0s[i][:]

            tensors.append(tensor)

    return tensors


def pad_mels_to_tensors(mels, batched=None):
    """
    Args:
        mels: A list of mel-specs
    Returns:
        tensors: A list of tensors containing the batched mel-specs
        mel_frames: A list of tensors containing the frames of the original mel-specs
    """
    # Initialize
    tensors = []
    mel_frames = []

    # Split mel-specs into batches to avoid cuda memory exceed
    if batched == None:
        # Get the max frame for padding
        size = -1
        for mel in mels:
            size = max(size, mel.shape[-1])

        tensor = torch.zeros(len(mels), mels[0].shape[0], size)
        mel_frame = torch.zeros(len(mels), dtype=torch.int32)

        for i, mel in enumerate(mels):
            tensor[i, :, : mel.shape[-1]] = mel[:]
            mel_frame[i] = mel.shape[-1]

        tensors.append(tensor)
        mel_frames.append(mel_frame)
    else:
        start = 0
        while start + batched - 1 < len(mels):
            end = start + batched - 1

            # Get the max frame for padding
            size = -1
            for i in range(start, end + 1):
                size = max(size, mels[i].shape[-1])

            tensor = torch.zeros(batched, mels[0].shape[0], size)
            mel_frame = torch.zeros(batched, dtype=torch.int32)

            for i in range(start, end + 1):
                tensor[i - start, :, : mels[i].shape[-1]] = mels[i][:]
                mel_frame[i - start] = mels[i].shape[-1]

            tensors.append(tensor)
            mel_frames.append(mel_frame)

            start = start + batched

        if start != len(mels):
            end = len(mels)

            # Get the max frame for padding
            size = -1
            for i in range(start, end):
                size = max(size, mels[i].shape[-1])

            tensor = torch.zeros(len(mels) - start, mels[0].shape[0], size)
            mel_frame = torch.zeros(len(mels) - start, dtype=torch.int32)

            for i in range(start, end):
                tensor[i - start, :, : mels[i].shape[-1]] = mels[i][:]
                mel_frame[i - start] = mels[i].shape[-1]

            tensors.append(tensor)
            mel_frames.append(mel_frame)

    return tensors, mel_frames


def load_model_config(args):
    """Load model configurations (in args.json under checkpoint directory)

    Args:
        args (ArgumentParser): arguments to run bins/preprocess.py

    Returns:
        dict: dictionary that stores model configurations
    """
    if args.checkpoint_dir is None:
        assert args.checkpoint_file is not None
        checkpoint_dir = os.path.split(args.checkpoint_file)[0]
    else:
        checkpoint_dir = args.checkpoint_dir
    config_path = os.path.join(checkpoint_dir, "args.json")
    print("config_path: ", config_path)

    config = load_config(config_path)
    return config


def remove_and_create(dir):
    if os.path.exists(dir):
        os.system("rm -r {}".format(dir))
    os.makedirs(dir, exist_ok=True)


def has_existed(path):
    if os.path.exists(path):
        answer = input(
            "The path {} has existed. \nInput 'y' (or hit Enter) to skip it, and input 'n' to re-write it [y/n]\n".format(
                path
            )
        )
        if not answer == "n":
            return True

    return False


def remove_older_ckpt(saved_model_name, checkpoint_dir, max_to_keep=5):
    if os.path.exists(os.path.join(checkpoint_dir, "checkpoint")):
        with open(os.path.join(checkpoint_dir, "checkpoint"), "r") as f:
            ckpts = [x.strip() for x in f.readlines()]
    else:
        ckpts = []
    ckpts.append(saved_model_name)
    for item in ckpts[:-max_to_keep]:
        if os.path.exists(os.path.join(checkpoint_dir, item)):
            os.remove(os.path.join(checkpoint_dir, item))
    with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
        for item in ckpts[-max_to_keep:]:
            f.write("{}\n".format(item))


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def save_checkpoint(
    args,
    generator,
    g_optimizer,
    step,
    discriminator=None,
    d_optimizer=None,
    max_to_keep=5,
):
    saved_model_name = "model.ckpt-{}.pt".format(step)
    checkpoint_path = os.path.join(args.checkpoint_dir, saved_model_name)

    if discriminator and d_optimizer:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "global_step": step,
            },
            checkpoint_path,
        )
    else:
        torch.save(
            {
                "generator": generator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "global_step": step,
            },
            checkpoint_path,
        )

    print("Saved checkpoint: {}".format(checkpoint_path))

    if os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint")):
        with open(os.path.join(args.checkpoint_dir, "checkpoint"), "r") as f:
            ckpts = [x.strip() for x in f.readlines()]
    else:
        ckpts = []
    ckpts.append(saved_model_name)
    for item in ckpts[:-max_to_keep]:
        if os.path.exists(os.path.join(args.checkpoint_dir, item)):
            os.remove(os.path.join(args.checkpoint_dir, item))
    with open(os.path.join(args.checkpoint_dir, "checkpoint"), "w") as f:
        for item in ckpts[-max_to_keep:]:
            f.write("{}\n".format(item))


def attempt_to_restore(
    generator, g_optimizer, checkpoint_dir, discriminator=None, d_optimizer=None
):
    checkpoint_list = os.path.join(checkpoint_dir, "checkpoint")
    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readlines()[-1].strip()
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if generator:
            if not list(generator.state_dict().keys())[0].startswith("module."):
                raw_dict = checkpoint["generator"]
                clean_dict = OrderedDict()
                for k, v in raw_dict.items():
                    if k.startswith("module."):
                        clean_dict[k[7:]] = v
                    else:
                        clean_dict[k] = v
                generator.load_state_dict(clean_dict)
            else:
                generator.load_state_dict(checkpoint["generator"])
        if g_optimizer:
            g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        global_step = 100000
        if discriminator and "discriminator" in checkpoint.keys():
            discriminator.load_state_dict(checkpoint["discriminator"])
            global_step = checkpoint["global_step"]
            print("restore discriminator")
        if d_optimizer and "d_optimizer" in checkpoint.keys():
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
            print("restore d_optimizer...")
    else:
        global_step = 0
    return global_step


class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def apply_moving_average(model, ema):
    for name, param in model.named_parameters():
        if name in ema.shadow:
            ema.update(name, param.data)


def register_model_to_ema(model, ema):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

def get_singer_id(cfg, singer_name):
    with open(cfg.singer_file, "r") as f:
        singer_lut = json.load(f)
    singer_id = -1
    if singer_name in singer_lut.keys():
        singer_id = np.array([singer_lut[singer_name] ], dtype=np.int32)         
    return singer_id
    

class YParams(HParams):
    def __init__(self, yaml_file):
        if not os.path.exists(yaml_file):
            raise IOError("yaml file: {} is not existed".format(yaml_file))
        super().__init__()
        self.d = collections.OrderedDict()
        with open(yaml_file) as fp:
            for _, v in yaml().load(fp).items():
                for k1, v1 in v.items():
                    try:
                        if self.get(k1):
                            self.set_hparam(k1, v1)
                        else:
                            self.add_hparam(k1, v1)
                        self.d[k1] = v1
                    except Exception:
                        import traceback

                        print(traceback.format_exc())

    # @property
    def get_elements(self):
        return self.d.items()


def override_config(base_config, new_config):
    """Update new configurations in the original dict with the new dict

    Args:
        base_config (dict): original dict to be overridden
        new_config (dict): dict with new configurations

    Returns:
        dict: updated configuration dict
    """
    for k, v in new_config.items():
        if type(v) == dict:
            if k not in base_config.keys():
                base_config[k] = {}
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def get_lowercase_keys_config(cfg):
    """Change all keys in cfg to lower case

    Args:
        cfg (dict): dictionary that stores configurations

    Returns:
        dict: dictionary that stores configurations
    """
    updated_cfg = dict()
    for k, v in cfg.items():
        if type(v) == dict:
            v = get_lowercase_keys_config(v)
        updated_cfg[k.lower()] = v
    return updated_cfg


def _load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): whether changing keys to lower case. Defaults to False.

    Returns:
        dict: dictionary that stores configurations
    """
    with open(config_fn, "r") as f:
        data = f.read()
    config_ = json5.loads(data)
    if "base_config" in config_:
        # load configurations from new path
        p_config_path = os.path.join(os.getenv("WORK_DIR"), config_["base_config"])
        p_config_ = _load_config(p_config_path)
        config_ = override_config(p_config_, config_)
    if lowercase:
        # change keys in config_ to lower case
        config_ = get_lowercase_keys_config(config_)
    return config_


def load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): _description_. Defaults to False.

    Returns:
        JsonHParams: an object that stores configurations
    """
    config_ = _load_config(config_fn, lowercase=lowercase)
    # create an JsonHParams object with configuration dict
    cfg = JsonHParams(**config_)
    return cfg


def save_config(save_path, cfg):
    """Save configurations into a json file

    Args:
        save_path (str): path to save configurations
        cfg (dict): dictionary that stores configurations
    """
    with open(save_path, "w") as f:
        json5.dump(cfg, f, ensure_ascii=False, indent=4, quote_keys=True)


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class ValueWindow:
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1) :] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


class Logger(object):
    def __init__(
        self,
        filename,
        level="info",
        when="D",
        backCount=10,
        fmt="%(asctime)s : %(message)s",
    ):
        self.level_relations = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "crit": logging.CRITICAL,
        }
        if level == "debug":
            fmt = "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding="utf-8"
        )
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
        self.logger.info(
            "==========================New Starting Here=============================="
        )
