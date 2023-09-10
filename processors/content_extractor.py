import os
import torch
import torchaudio
import numpy as np
import yaml
import copy
from tqdm import tqdm
from torchaudio.compliance import kaldi
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from modules import whisper_extractor as whisper
from modules.wenet_extractor.utils.init_model import init_model
from modules.wenet_extractor.utils.checkpoint import load_checkpoint
from fairseq import checkpoint_utils

"""
    Extractor for content features
    1. whisper
    2. contentvec
    3. wenet

    Pipeline:
        in preprocess.py:
            call extract_utt_content_features() to extract content features for each utterance
            extract_utt_content_features() envelopes the following steps:
                1. load the model (whisper, contentvec, wenet)
                2. extract the content features
                3. save the content features into files
        in diffsvc_dataset.py:
            call offline_align() to align the content features to the given target length

"""

"""
    Extractor Usage:
        1. initialize an instance of extractor
            extractor = WhisperExtractor(cfg)
        2. load the specified model
            extractor.load_model()
        3. extract the content features
            extractor.extract_content(utt) for single utterance
            extractor.extract_content_batch(utts) for batch utterances
        4. save the content features
            extractor.save_feature(utt, content_feature) for single utterance
"""


class BaseExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.extractor_type = None
        self.model = None

    def offline_align(self, content, target_len):
        """
        args:
            content: (source_len, dim)
            target_len: target length
        return:
            mapped_feature: (target_len, dim)
        """
        target_hop = self.cfg.hop_size

        assert self.extractor_type in ["whisper", "contentvec", "wenet"]
        if self.extractor_type == "whisper":
            source_hop = (
                self.cfg.whisper_frameshift
                * self.cfg.whisper_downsample_rate
                * self.cfg.sample_rate
            )
        elif self.extractor_type == "contentvec":
            source_hop = self.cfg.contentvec_frameshift * self.cfg.sample_rate
        elif self.extractor_type == "wenet":
            source_hop = (
                self.cfg.wenet_frameshift
                * self.cfg.wenet_downsample_rate
                * self.cfg.sample_rate
            )
        source_hop = int(source_hop)
        factor = np.gcd(source_hop, target_hop)
        source_hop //= factor
        target_hop //= factor

        # (source_len, 256)
        _, width = content.shape
        # slice the content from padded feature
        source_len = min(target_len * target_hop // source_hop + 1, len(content))

        # const ~= target_len * target_hop
        const = source_len * source_hop // target_hop * target_hop

        # (source_len * source_hop, dim)
        up_sampling_feats = np.repeat(content, source_hop, axis=0)
        # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
        down_sampling_feats = np.average(
            up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
        )

        err = abs(target_len - len(down_sampling_feats))
        if err > 8:
            # err_log_dir is indeterminate
            err_log_dir = os.path.join(self.cfg.processed_dir, "align_max_err.log")
            try:
                with open(err_log_dir, "r") as f:
                    err_num = int(f.read())
            except:
                with open(err_log_dir, "w") as f:
                    f.write("0")
                err_num = 0
            if err > err_num:
                with open(err_log_dir, "w") as f:
                    f.write(str(err))

        if len(down_sampling_feats) < target_len:
            # (1, dim) -> (err, dim)
            end = down_sampling_feats[-1][None, :].repeat(err, axis=0)
            down_sampling_feats = np.concatenate([down_sampling_feats, end], axis=0)

        # (target_len, dim)
        mapped_feature = down_sampling_feats[:target_len]

        return mapped_feature

    def save_feature(self, utt, content_feature):
        """Save a single utternace to path {cfg.preprocess.processed_dir}

        Args:
            utt (dict): one item in metadata, containing information for one utterance
            content_feature (tensor): content feature of one utterance
        """
        uid = utt["Uid"]
        assert self.extractor_type != None
        out_dir = os.path.join(
            self.cfg.preprocess.processed_dir, utt["Dataset"], self.extractor_type
        )
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, uid + ".npy")
        np.save(save_path, content_feature.cpu().detach().numpy())


class WhisperExtractor(BaseExtractor):
    def __init__(self, config):
        super(WhisperExtractor, self).__init__(config)
        self.extractor_type = "whisper"

    def load_model(self):
        # load whisper checkpoint
        print("Loading Whisper Model...")
        model = whisper.load_model(self.cfg.preprocess.whisper_model)
        if torch.cuda.is_available():
            print("Using GPU...\n")
            model = model.cuda()
        else:
            print("Using CPU...\n")

        self.model = model.eval()

    def extract_content(self, utt):
        """Extract content features with whisper (audio tensor -> mel spect -> embedding)

        Args:
            utt (dict): dictionary containing metadata for one utterance

        Returns:
            tensor: utternace content feature
        """
        audio_path = utt["Path"]
        # extract whisper fetures
        audio = whisper.load_audio(str(audio_path))
        audio = torch.from_numpy(audio).to(self.model.device)
        audio = whisper.pad_or_trim(audio)

        # (80, 3000) -> (1, 80, 3000) to fit whipser.embed_audio() input
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device).unsqueeze(0)

        with torch.no_grad():
            # (1, 1500, 1024) -> (1500, 1024)
            feature = self.model.embed_audio(mel).squeeze(0)
            duration = utt["Duration"]
            frameshift = self.cfg.preprocess.whisper_frameshift  # 20ms
            num_frames = int(np.ceil((duration - frameshift) / frameshift)) + 1
            # only keep effective parts
            # (1500, 1024) -> (num_valid_frames, 1024)
            feature = feature[:num_frames, :]

        return feature

    def extract_content_batch(self, utts):
        batch_size = len(utts)
        batch_mel = torch.zeros(
            (batch_size, 80, 3000), dtype=torch.float32, device=self.model.device
        )

        for i, utt in enumerate(utts):
            # (48000,)
            audio = whisper.load_audio(utt["Path"])
            audio = whisper.pad_or_trim(audio)

            # (80, 3000)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            batch_mel[i] = mel

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.model.embed_audio(batch_mel)
        return features


class ContentvecExtractor(BaseExtractor):
    def __init__(self, cfg):
        super(ContentvecExtractor, self).__init__(cfg)
        self.extractor_type = "contentvec"

    def load_model(self):
        assert self.model == None
        # Load model
        ckpt_path = self.cfg.preprocess.contentvec_file
        print("Load Contentvec Model...")

        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path],
            suffix="",
        )
        model = models[0]
        model.eval()

        if torch.cuda.is_available():
            # print("Using GPU...\n")
            model = model.cuda()

        self.model = model

    def extract_content(self, utt):
        audio_path = utt["Path"]
        # extract contentvec features
        wav16k, sr = torchaudio.load(audio_path)
        device = next(self.model.parameters()).device
        if sr != 16000:
            wav16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
                wav16k
            )
        # double channel to single
        if wav16k.shape[0] == 2:
            wav16k = torch.mean(wav16k, dim=0, keepdim=True)
        assert wav16k.shape[0] == 1, "audio's channel is {}".format(wav16k.shape[0])
        # reshape into (1, len)
        feats = wav16k.view(1, -1)
        # generate a mask tensor with all "False" values
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(device),
            "padding_mask": padding_mask.to(device),
            "output_layer": 12,
        }

        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0]).squeeze(0)

        return feats

    def extract_content_batch(self, utts):
        device = next(self.model.parameters()).device
        # (batch, 1, max_len)
        feats_list = []
        # lens_list = []

        for idx, utt in enumerate(utts):
            audio_path = utt["Path"]
            # extract contentvec fetures
            wav16k, sr = torchaudio.load(audio_path)

            if sr != 16000:
                wav16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
                    wav16k
                )
            # double channel
            if wav16k.shape[0] == 2:
                wav16k = torch.mean(wav16k, dim=0, keepdim=True)
            assert wav16k.shape[0] == 1, "audio's channel is {}".format(wav16k.shape[0])
            wav16k = wav16k.squeeze()  # shape: (len)
            # lens_list.append(wav16k.shape[0])
            feats_list.append(wav16k)
        feats_tensor = pad_sequence(feats_list, batch_first=True).to(
            device
        )  # (batch, max_len)
        padding_mask = torch.eq(feats_tensor, torch.zeros_like(feats_tensor)).to(device)

        inputs = {
            "source": feats_tensor,
            "padding_mask": padding_mask,
            "output_layer": 12,  # change to layer 12
        }

        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0])
        return feats


class WenetExtractor(BaseExtractor):
    def __init__(self, config):
        super(WenetExtractor, self).__init__(config)
        self.extractor_type = "wenet"

    def load_model(self):
        wenet_cfg = self.cfg.preprocess.wenet_config
        wenet_model_path = self.cfg.preprocess.wenet_model_path
        # load Wenet config
        with open(wenet_cfg, "r") as w:
            wenet_configs = yaml.load(w, Loader=yaml.FullLoader)
        self.extract_conf = copy.deepcopy(wenet_configs["dataset_conf"])
        print("Loading Wenet Model...")
        self.model = init_model(wenet_configs)
        load_checkpoint(self.model, wenet_model_path)

        if torch.cuda.is_available():
            print("Using GPU...\n")
            self.model = self.model.cuda()
        else:
            print("Using CPU...\n")

        self.model = self.model.eval()

    def extract_content(self, utt):
        audio_path = utt["Path"]
        wav16k, sr = torchaudio.load(audio_path)
        device = next(self.model.parameters()).device
        if sr != 16000:
            wav16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
                wav16k
            )
        if wav16k.shape[0] == 2:
            wav16k = torch.mean(wav16k, dim=0, keepdim=True)
        assert wav16k.shape[0] == 1, "audio's channel is {}".format(wav16k.shape[0])

        # pad one frame to compensate for the frame cut off after feature extraction
        pad_tensor = torch.zeros(160, device=wav16k.device).unsqueeze(0)
        wav16k = torch.cat((wav16k, pad_tensor), dim=-1)

        # Extract fbank/mfcc features by kaldi
        assert self.extract_conf is not None, "load model first!"
        feats_type = self.extract_conf.get("feats_type", "fbank")
        assert feats_type in ["fbank", "mfcc"]
        wav16k *= 1 << 15
        if feats_type == "fbank":
            fbank_conf = self.extract_conf.get("fbank_conf", {})
            feat = kaldi.fbank(
                wav16k,
                sample_frequency=16000,
                num_mel_bins=fbank_conf["num_mel_bins"],
                frame_length=fbank_conf["frame_length"],
                frame_shift=fbank_conf["frame_shift"],
                dither=fbank_conf["dither"],
            )
        elif feats_type == "mfcc":
            mfcc_conf = self.extract_conf.get("mfcc", {})
            feat = kaldi.mfcc(
                wav16k,
                sample_frequency=16000,
                num_mel_bins=mfcc_conf["num_mel_bins"],
                frame_length=mfcc_conf["frame_length"],
                frame_shift=mfcc_conf["frame_shift"],
                dither=mfcc_conf["dither"],
                num_ceps=mfcc_conf.get("num_ceps", 40),
                high_freq=mfcc_conf.get("high_freq", 0.0),
                low_freq=mfcc_conf.get("low_freq", 20.0),
            )

        feat_lengths = torch.tensor([feat.shape[0]], dtype=torch.int32)
        # feat: (len, 80) -> (1, len, 80)
        feat = feat.unsqueeze(0).to(device)
        # feat_lengths: batch size
        feat_lengths = feat_lengths.to(device)

        assert feat.size(0) == 1
        feature = self.model.encoder_extractor(
            feat,
            feat_lengths,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
            simulate_streaming=False,
        )

        feature = feature.squeeze()
        return feature

    def extract_content_batch(self, utts):
        feats_list = []
        lengths_list = []
        device = next(self.model.parameters()).device
        # Extract fbank/mfcc features by kaldi
        assert self.extract_conf is not None, "load model first!"
        feats_type = self.extract_conf.get("feats_type", "fbank")
        assert feats_type in ["fbank", "mfcc"]

        for idx, utt in enumerate(utts):
            path = utt["Path"]
            wav16k, sr = torchaudio.load(path)
            if sr != 16000:
                wav16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
                    wav16k
                )
            if wav16k.shape[0] == 2:
                wav16k = torch.mean(wav16k, dim=0, keepdim=True)
            assert wav16k.shape[0] == 1, "audio's channel is {}".format(wav16k.shape[0])

            # pad one frame to compensate for the frame cut off after feature extraction
            pad_tensor = torch.zeros(160, device=wav16k.device).unsqueeze(0)
            wav16k = torch.cat((wav16k, pad_tensor), dim=-1)
            wav16k *= 1 << 15
            if feats_type == "fbank":
                fbank_conf = self.extract_conf.get("fbank_conf", {})
                feat = kaldi.fbank(
                    wav16k,
                    sample_frequency=16000,
                    num_mel_bins=fbank_conf["num_mel_bins"],
                    frame_length=fbank_conf["frame_length"],
                    frame_shift=fbank_conf["frame_shift"],
                    dither=fbank_conf["dither"],
                )
            elif feats_type == "mfcc":
                mfcc_conf = self.extract_conf.get("mfcc", {})
                feat = kaldi.mfcc(
                    wav16k,
                    sample_frequency=16000,
                    num_mel_bins=mfcc_conf["num_mel_bins"],
                    frame_length=mfcc_conf["frame_length"],
                    frame_shift=mfcc_conf["frame_shift"],
                    dither=mfcc_conf["dither"],
                    num_ceps=mfcc_conf.get("num_ceps", 40),
                    high_freq=mfcc_conf.get("high_freq", 0.0),
                    low_freq=mfcc_conf.get("low_freq", 20.0),
                )
            feats_list.append(feat)
            lengths_list.append(feat.shape[0])

        feats_lengths = torch.tensor(lengths_list, dtype=torch.int32).to(device)
        feats_tensor = pad_sequence(feats_list, batch_first=True).to(
            device
        )  # (batch, len, 80)
        assert len(utts) == feats_tensor.shape[0] == feats_lengths.shape[0]
        features = self.model.encoder_extractor(
            feats_tensor,
            feats_lengths,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
            simulate_streaming=False,
        )
        return features


def extract_utt_content_features(cfg, metadata):
    """Extract content features with models(whisper, contentvec, wenet)

    Args:
        cfg (dict): dictionary that stores configurations
        metadata (dict): dictionary that stores data in train.json and test.json files
    """
    if cfg.preprocess.extract_contentvec_feature:
        extractor = ContentvecExtractor(cfg)
        # load model from path in {cfg.preprocess.contentvec_file}
        extractor.load_model()
        batch_size = cfg.preprocess.contentvec_batch_size
        assert batch_size > 0
        # extract content in batch
        if batch_size > 1:
            with tqdm(total=len(metadata)) as pbar:
                start, end = 0, 0
                while end < len(metadata):
                    start = end
                    end = start + batch_size
                    if end > len(metadata):
                        num = len(metadata) - start
                    else:
                        num = batch_size
                    batch_content_features = extractor.extract_content_batch(
                        metadata[start:end]
                    )
                    for index, utt in enumerate(metadata[start:end]):
                        extractor.save_feature(utt, batch_content_features[index])
                    pbar.update(num)
        else:
            for utt in tqdm(metadata):
                content_feature = extractor.extract_content(utt)
                extractor.save_feature(utt, content_feature)

    if cfg.preprocess.extract_whisper_feature:
        extractor = WhisperExtractor(cfg)
        extractor.load_model()
        batch_size = cfg.preprocess.whisper_batch_size
        assert batch_size > 0
        if batch_size > 1:
            with tqdm(total=len(metadata)) as pbar:
                start, end = 0, 0
                while end < len(metadata):
                    start = end
                    end = start + batch_size
                    if end > len(metadata):
                        num = len(metadata) - start
                    else:
                        num = batch_size
                    batch_content_features = extractor.extract_content_batch(
                        metadata[start:end]
                    )
                    for index, utt in enumerate(metadata[start:end]):
                        extractor.save_feature(utt, batch_content_features[index])
                    pbar.update(num)
        else:
            for utt in tqdm(metadata):
                content_feature = extractor.extract_content(utt)
                extractor.save_feature(utt, content_feature)

    if cfg.preprocess.extract_wenet_feature:
        extractor = WenetExtractor(cfg)
        extractor.load_model()
        batch_size = cfg.preprocess.wenet_batch_size
        assert batch_size > 0
        if batch_size > 1:
            with tqdm(total=len(metadata)) as pbar:
                start, end = 0, 0
                while end < len(metadata):
                    start = end
                    end = start + batch_size
                    if end > len(metadata):
                        num = len(metadata) - start
                    else:
                        num = batch_size
                    batch_content_features = extractor.extract_content_batch(
                        metadata[start:end]
                    )
                    for index, utt in enumerate(metadata[start:end]):
                        extractor.save_feature(utt, batch_content_features[index])
                    pbar.update(num)
        else:
            for utt in tqdm(metadata):
                content_feature = extractor.extract_content(utt)
                extractor.save_feature(utt, content_feature)
