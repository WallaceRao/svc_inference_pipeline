import torch
import numpy as np
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from processors.acoustic_extractor import cal_normalized_mel


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """
        # control the effective range of the config

        assert isinstance(dataset, str)

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()

        self.data_root = processed_data_dir
        self.cfg = cfg

        if cfg.preprocess.use_spkid:
            self.spk2id_path = os.path.join(
                cfg.log_dir, cfg.exp_name, cfg.preprocess.spk2id
            )
            self.utt2spk_path = os.path.join(self.data_root, cfg.preprocess.utt2spk)
            """
            spk2id: {spk1: 0, spk2: 1, ...}
            utt2spk: {dataset_uid: spk1, ...}
            """
            self.spk2id, self.utt2spk = get_spk_map(self.spk2id_path, self.utt2spk_path)

        if cfg.preprocess.use_uv:
            self.utt2uv_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                self.utt2uv_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.uv_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_frame_pitch:
            self.utt2frame_pitch_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2frame_pitch_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.pitch_dir,
                    uid + ".npy",
                )

        elif cfg.preprocess.use_phone_pitch:
            self.load_phone_pitch()

        if cfg.preprocess.use_frame_energy:
            self.utt2frame_energy_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2frame_energy_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.energy_dir,
                    uid + ".npy",
                )

        elif cfg.preprocess.use_phone_energy:
            self.load_phone_energy()

        if cfg.preprocess.use_mel:
            self.utt2mel_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2mel_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.mel_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_audio:
            self.utt2audio_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2audio_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.audio_dir,
                    uid + ".npy",
                )
        elif cfg.preprocess.use_label:
            self.utt2label_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2label_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.label_dir,
                    uid + ".npy",
                )
        elif cfg.preprocess.use_one_hot:
            self.utt2one_hot_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2one_hot_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.one_hot_dir,
                    uid + ".npy",
                )

    def get_metadata(self):
        # metadata = list()
        # with open(self.metafile_path, encoding='utf-8') as f:
        #     lines = f.readlines()
        #     for x in lines:
        #         x = x.strip()
        #         metadata.append(x)

        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        if self.cfg.preprocess.use_spkid:
            single_feature["spk_id"] = np.array(
                [self.spk2id[self.utt2spk[utt]]], dtype=np.int32
            )

        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel  # [n_mels, T]
            if self.cfg.preprocess.use_min_max_norm_mel:
                # do mel norm
                mel = cal_normalized_mel(mel, utt_info["Dataset"], self.cfg.preprocess)

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]
            single_feature["mel"] = mel.T  # [T, n_mels]

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch_path = self.utt2frame_pitch_path[utt]
            frame_pitch = np.load(frame_pitch_path)
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)
            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )
            single_feature["frame_pitch"] = aligned_frame_pitch

            if self.cfg.preprocess.use_uv:
                frame_uv_path = self.utt2uv_path[utt]
                frame_uv = np.load(frame_uv_path)
                aligned_frame_uv = align_length(frame_uv, single_feature["target_len"])
                aligned_frame_uv = [
                    0 if frame_uv else 1 for frame_uv in aligned_frame_uv
                ]
                aligned_frame_uv = np.array(aligned_frame_uv)
                single_feature["frame_uv"] = aligned_frame_uv

        elif self.cfg.preprocess.use_phone_pitch:
            single_feature["phone_pitch"] = self.cal_phone_pitch()

        if self.cfg.preprocess.use_frame_energy:
            frame_energy_path = self.utt2frame_energy_path[utt]
            frame_energy = np.load(frame_energy_path)
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_energy)
            aligned_frame_energy = align_length(
                frame_energy, single_feature["target_len"]
            )
            single_feature["frame_energy"] = aligned_frame_energy

        elif self.cfg.preprocess.use_phone_energy:
            single_feature["phone_energy"] = self.cal_phone_energy()

        if self.cfg.preprocess.use_audio:
            audio = np.load(self.utt2audio_path[uid])
            single_feature["audio"] = audio

        return single_feature

    def __len__(self):
        return len(self.metadata)

    def load_phone_pitch(self):
        raise NotImplementedError

    def load_phone_energy(self):
        raise NotImplementedError

    def cal_phone_pitch(self):
        raise NotImplementedError

    def cal_phone_energy(self):
        raise NotImplementedError


class BaseCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, T, n_mels]
        # frame_pitch, frame_energy: [1, T]
        # target_len: [1]
        # spk_id: [b, 1]
        # mask: [b, T, 1]

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features


class BaseTestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, args):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BaseTestCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        raise NotImplementedError

    def __call__(self, batch):
        raise NotImplementedError
