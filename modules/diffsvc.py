"""
    This is an unofficial implementation of the following paper
    Songxiang Liu, et al. DIFFSVC: A DIFFUSION PROBABILISTIC MODEL FOR SINGING VOICE CONVERSION.
    Access: https://doi.org/10.48550/arXiv.2105.13871
"""

"""
    Abbreviations used in this file:
        B: Batch size
        C: Residual channels
        L: Length of the data, refer to number of frames here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

Linear = nn.Linear


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class StepEncoder(nn.Module):
    def __init__(self, max_steps: int, FC_size: int):
        """
        @param max_steps: The max diffusion step would be used
        @param FC_size: The size of hidden layer in FC layers
        """
        super().__init__()

        self.max_steps = max_steps
        self.register_buffer(
            "embedding", self.build_embedding(max_steps), persistent=False
        )

        self.projection1 = Linear(128, FC_size)
        self.projection2 = Linear(FC_size, FC_size)

    def build_embedding(self, max_steps: int) -> torch.Tensor:
        """
        @param max_steps: The max diffusion step would be used
        @return: A lookup table to return corresponding step embedding
        """
        steps = torch.arange(max_steps).unsqueeze(1)  # [T, 1]
        dims = torch.arange(64).unsqueeze(0)  # [1, 64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T, 64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T, 128]
        return table  # [T, 128]

    def lerp_embedding(self, t: float) -> torch.Tensor:
        """
        @param t: Special Step
        @return: Corresponding step embedding with correction
        """
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def forward(self, cur_diffusion_step) -> torch.Tensor:
        """
        @param cur_diffusion_step: The current diffusion step # [b, 1]
        @return: The embedding of current step
        """
        # [1, 128]
        
        stats = {}
        
        
        if cur_diffusion_step.dtype in [  # torch.int64
            torch.int32,
            torch.int64,
        ]:
            x = self.embedding[cur_diffusion_step]
        else:
            x = self.lerp_embedding(cur_diffusion_step)

        stats['step_embedding'] = x
        
        x = self.projection1(x)  # [1, FC_size]
        x = F.silu(x)
        x = self.projection2(x)  # [1, FC_size]
        x = F.silu(x)

        stats['step_encoder_output'] = x
        
        return x, stats  # [1, FC_size]


class SpectrogramPreprocessor(nn.Module):
    """
    This class is used to preprocess of input mel-spectrogram (content features).
    Make the mel spectrogram from [Batch_size, n_frame, n_mel] into [Batch_size, n_frame, n_mel].
    """

    def __init__(self, n_mel: int, residual_channels: int):
        """
        @param n_mels: The mel binaries of mel-spectrogram
        @param residual_channels: The number of residual channels
        """
        super().__init__()
        self.projection = Conv1d(n_mel, residual_channels, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Input normalized mel-spectrogram

        Returns:
            mel-spectrogram after preprocess
        """

        # Note that we need to adopt [B, L, n_mel] -> [B, n_mel, L]
        x = torch.transpose(x, 1, 2)
        y = self.projection(x)  # [B, n_mel, L] -> [B, C, L]
        y = F.relu(y)

        return y  # [B, C, L]


class ResidualBlock(nn.Module):
    """
    This is class is used to build one residual block (or residual layer)
    """

    def __init__(
        self,
        conditioner_size: int,
        diffusion_FC_size: int,
        residual_channels: int,
        dilation: int,
        kernel_size=3,
    ):
        """
        Args:
            conditioner_size: Size of conditioner e.
            diffusion_FC_size: Full-connection size of hidden layer in step encoder.
            residual_channels: Residual channels, hyper-param
        """
        super().__init__()

        # The core conv here, [B, C, L] -> [B, 2C, L]
        if dilation == 1:
            # To stay same length, p = (k-1)/2
            if (kernel_size - 1) % 2 != 0:
                print("Wrong kernel size for Conv1d")
                exit()
            padding = int((kernel_size - 1) / 2)
            self.dilated_conv = Conv1d(
                in_channels=residual_channels,
                out_channels=2 * residual_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        else:
            # TODO: To stay same length, p = d(k-1) / 2
            assert kernel_size == 3
            self.dilated_conv = Conv1d(
                in_channels=residual_channels,
                out_channels=2 * residual_channels,
                kernel_size=kernel_size,
                padding=dilation,
                dilation=dilation,
            )

        # The mapping of diffusion, [1,128] -> [1, C]
        self.diffusion_projection = Linear(
            in_features=diffusion_FC_size, out_features=residual_channels
        )

        # The mapping of conditioner, [B, n_bin, L] -> [B, C, L]
        self.conditioner_projection = Conv1d(
            in_channels=conditioner_size,
            out_channels=2 * residual_channels,
            kernel_size=1,
        )

        # The output projection, [B, C, L] -> [B, 2C, L]
        self.output_projection = Conv1d(
            in_channels=residual_channels,
            out_channels=residual_channels * 2,
            kernel_size=1,
        )

    def forward(
        self, x: torch.Tensor, diffusion_step: torch.Tensor, conditioner: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Noisy mel-spectrogram inherited from previous residual layer: (B, C, L)
            diffusion_step: Diffusion step t: (1, FC_size)
            conditioner: Conditioner e (B, L, n_bin)

        Returns:
            [Noisy mel-spectrogram for next residual layer, Noisy mel-spectrogram for skip connection]
        """
        # (B, 1, C)
        diffusion_step = self.diffusion_projection(diffusion_step)
        # Suppose to add to every frame on the whole batch here
        # As size of x is [B, C, L],
        # size of diffusion_step is [1, C],
        # it should make B * L times adding here
        
        stats = {}
        y = x + torch.transpose(diffusion_step, 1, 2)
        stats["noise_step"] = y  # [B, 348, 732]

        conditioner = torch.transpose(
            conditioner, 1, 2
        )  # [B, L, n_bin] -> [B, n_bin, L]
        conditioner = self.conditioner_projection(
            conditioner
        )  # [B, n_bin, L] -> [B, 2C, L]
        y = self.dilated_conv(y) + conditioner  # [B, C, L] -> [B, 2C, L]
        stats["noise_step_condition"] = y   # [B, 768, 732]

        # Separation
        gate, filter = torch.chunk(y, 2, dim=1)  # [B, 2C, L] -> [B, C, L]
        # Elementwise Multiply
        y = torch.sigmoid(gate) * torch.tanh(filter)  # [B, C, L]

        y = self.output_projection(y)  # [B, C, L] -> [B, 2C, L]
        residual, skip = torch.chunk(y, 2, dim=1)  # [B, 2C, L] -> [B, C, L]

        return (x + residual) / sqrt(2.0), skip, stats # list([B, C, L], [B, C, L])


class DiffSVC(nn.Module):
    """
    Public class adopting DiffSVC, i.e. Diffusion Probabilistic Model, as mapper
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg

        self.cfg.noise_schedule = np.linspace(
            self.cfg.noise_schedule_factors[0],
            self.cfg.noise_schedule_factors[1],
            self.cfg.noise_schedule_factors[2],
        ).tolist()

        self.mel_preprocess = SpectrogramPreprocessor(
            self.cfg.n_mel, self.cfg.residual_channels
        )

        self.diffusion_embedding = StepEncoder(
            len(self.cfg.noise_schedule), self.cfg.diffusion_fc_size
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    self.cfg.conditioner_size,
                    self.cfg.diffusion_fc_size,
                    self.cfg.residual_channels,
                    2 ** (i % self.cfg.dilation_cycle_length),
                    kernel_size=self.cfg.residual_kernel_size,
                )
                for i in range(self.cfg.residual_layer_num)
            ]
        )
        self.skip_projection = Conv1d(
            self.cfg.residual_channels, self.cfg.residual_channels, 1
        )

        self.output_projection = Conv1d(
            self.cfg.residual_channels, self.cfg.n_mel, 1
        )

        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self, mel_spec: torch.Tensor, conditioner: torch.Tensor, diffusion_step
    ) -> torch.Tensor:
        """
        Args:
            mel_spec: Noisy mel-spectrogram input x
            conditioner: Conditioner e, i.e. acoustics features  
            diffusion_step: Diffusion step t

        Returns:
            Predicted Gaussian noise, i.e. y_pred
        """


        # [B, L, n_mel] -> [B, C, L]
        
        stats = dict()
        x = self.mel_preprocess(mel_spec)
        diffusion_step, step_stats = self.diffusion_embedding(diffusion_step)
        stats.update(step_stats)

        residual_layers_stats_list = list()
        skip = None
        for i in range(len(self.residual_layers)):
            x, skip_connection, layer_stats = self.residual_layers[i](x, diffusion_step, conditioner)
            residual_layers_stats_list.append(layer_stats)
            
            skip = skip_connection if skip is None else skip_connection + skip
        
        stats["residual_layers_stats_all_layers"] = residual_layers_stats_list
        
        x = skip / sqrt(len(self.residual_layers))

        x = self.skip_projection(x)  # [B, C, L] -> [B, 2C, L]
        x = F.relu(x)
        x = self.output_projection(x)  # [B, C, L] -> [B, n_mel, L]

        return torch.transpose(x, 1, 2), stats  # [B, L, n_mel]
