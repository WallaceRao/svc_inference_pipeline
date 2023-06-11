import librosa
import numpy as np
import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(self, cfg, content_type):
        super().__init__()
        self.cfg = cfg

        self.input_dim = self.cfg.input_content_dim[content_type]
        self.output_dim = self.cfg.encoder_content_dim

        if self.input_dim != 0:
            self.nn = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        # x: (N, seq_len, input_dim) -> (N, seq_len, output_dim)
        return self.nn(x)


class MelodyEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = self.cfg.input_melody_dim
        self.output_dim = self.cfg.encoder_melody_dim
        self.n_bins = self.cfg.n_bins_melody

        if self.input_dim != 0:
            if self.n_bins == 0:
                # Not use quantization
                self.nn = nn.Linear(self.input_dim, self.output_dim)
            else:
                # C1: ~32, C2: ~65, C7: ~2093. There are 12*6=72 notes betwwen C1-C7
                self.f0_min = librosa.note_to_hz("C1")
                self.f0_max = librosa.note_to_hz("C7")
                if self.cfg.use_log_f0:
                    # ====== Mapping result ======
                    # 0 -> 0: unvoiced position
                    # (0, f0_min) -> 0: too low pitch
                    # f0_min -> 1
                    # f0_max -> n_bins - 2
                    # (f0_max, inf) -> n_bins - 1
                    self.melody_bins = nn.Parameter(
                        torch.exp(
                            torch.linspace(
                                # set f0_min - 0.1 but not f0_min (for modeling uv)
                                np.log(self.f0_min - 0.1),
                                np.log(self.f0_max),
                                self.n_bins - 1,
                            )
                        ),
                        requires_grad=False,
                    )

                self.nn = nn.Embedding(
                    num_embeddings=self.n_bins,
                    embedding_dim=self.output_dim,
                    padding_idx=None,
                )

    def forward(self, x):
        # x: (N, frame_len)
        if self.n_bins == 0:
            x = x.unsqueeze(-1)
        else:
            x = torch.bucketize(x, self.melody_bins)
        return self.nn(x)


class LoudnessEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = self.cfg.input_loudness_dim
        self.output_dim = self.cfg.encoder_loudness_dim
        self.n_bins = self.cfg.n_bins_loudness

        if self.input_dim != 0:
            if self.n_bins == 0:
                # Not use quantization
                self.nn = nn.Linear(self.input_dim, self.output_dim)
            else:
                # TODO: set trivially now
                self.loudness_min = 1e-30
                self.loudness_max = 1.5

                if self.cfg.use_log_loudness:
                    self.energy_bins = nn.Parameter(
                        torch.exp(
                            torch.linspace(
                                np.log(self.loudness_min),
                                np.log(self.loudness_max),
                                self.n_bins - 1,
                            )
                        ),
                        requires_grad=False,
                    )

                self.nn = nn.Embedding(
                    num_embeddings=self.n_bins,
                    embedding_dim=self.output_dim,
                    padding_idx=None,
                )

    def forward(self, x):
        # x: (N, frame_len)
        if self.n_bins == 0:
            x = x.unsqueeze(-1)
        else:
            x = torch.bucketize(x, self.energy_bins)
        return self.nn(x)


class SingerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = 1
        self.output_dim = self.cfg.encoder_singer_dim

        self.nn = nn.Embedding(
            num_embeddings=self.cfg.singer_table_size,
            embedding_dim=self.output_dim,
            padding_idx=None,
        )

    def forward(self, x):
        # x: (N, 1) -> (N, 1, output_dim)
        return self.nn(x)


class EncoderFramework(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.merge_mode = self.cfg.merge_mode

        modules_dict = dict()
        assert type(self.cfg.content_feature) == list
        for content_type in self.cfg.content_feature:
            modules_dict["content_{}".format(content_type)] = ContentEncoder(
                self.cfg, content_type
            )
        modules_dict["melody"] = MelodyEncoder(self.cfg)
        modules_dict["loudness"] = LoudnessEncoder(self.cfg)
        modules_dict["singer"] = SingerEncoder(self.cfg)

        self.registered_modules_dict = dict()
        encoder_output_dim = 0
        for k, encoder in modules_dict.items():
            input_dim = encoder.input_dim
            if input_dim != 0:
                self.registered_modules_dict[k] = encoder
                encoder_output_dim += encoder.output_dim
        # self.cfg.MAPPER.INPUT_DIM = encoder_output_dim

        self.registered_modules_dict = nn.ModuleDict(self.registered_modules_dict)

    def forward(self, x_dict):
        """
        Args:
            x_dict = {'content': (N, seq_len, input_dim1),
            'melody': (N, seq_len,), ... ,
            'singer': (N, ) }
        Returns:
            if merge_mode is concat, encoder_output is (N, seq_len, output_dim1 + output_dim2 + ... )
        """
        outputs = []

        
        for k, encoder in self.registered_modules_dict.items():
            # (N, seq_len, output_dim) if k is not "singer"
            
            out = encoder(x_dict[k])
            if k != "singer":
                outputs.append(out)
            else:
                # (N, 1, output_dim)
                singer_info = out

        # (N, 1, output_dim) -> (N, seq_len, output_dim)
        seq_len = outputs[0].shape[1]
        singer_info = singer_info.expand(-1, seq_len, -1)
        outputs.append(singer_info)

        encoder_output = None
        if self.merge_mode == "concat":
            encoder_output = torch.cat(outputs, dim=-1)
        if self.merge_mode == "add":
            # (#modules, N, seq_len, output_dim)
            outputs = torch.cat([out[None, :, :, :] for out in outputs], dim=0)
            # (N, seq_len, output_dim)
            encoder_output = torch.sum(outputs, dim=0)

        return encoder_output
