import math

from modules.encoder.position_encoder import PosEncoder
from modules.diffusion.dilated_cnn.residual_block import ResidualBlock
from modules.general.utils import *


class DilatedCNN(nn.Module):
    """
    Dilated CNN architecture with residual connections, default diffusion decoder.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_mel = cfg.n_mel
        self.n_layer = cfg.residual_layers
        self.n_channel = cfg.residual_channels
        self.kernel_size = cfg.residual_kernel_size
        self.dilation_cycle_length = cfg.dilation_cycle_length
        self.d_mlp = cfg.diffusion_fc_size
        self.d_context = cfg.conditioner_size
        self.register_embedding = cfg.use_register_embedding
        # if config has attribute noise_schedule_factors, use it, otherwise use default

        self.pos_encoder = PosEncoder(
            d_mlp=self.d_mlp,
            register_embedding=self.register_embedding,
        )

        self.input = nn.Sequential(
            Conv1d(
                self.n_mel,
                self.n_channel,
                1,
            ),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.n_channel,
                    kernel_size=self.kernel_size,
                    dilation=2 ** (i % self.dilation_cycle_length),
                    d_mlp=self.d_mlp,
                    d_context=self.d_context,
                )
                for i in range(self.n_layer)
            ]
        )

        self.out_proj = nn.Sequential(
            Conv1d(
                self.n_channel,
                self.n_channel,
                1,
            ),
            nn.ReLU(),
            Conv1d(
                self.n_channel,
                self.n_mel,
                1,
            ),
        )

        nn.init.zeros_(self.out_proj[2].weight)

    def forward(self, x, diffusion_step, context=None):
        """
        Args:
            x: Noisy mel-spectrogram [B x L x ``n_mel``]
            diffusion_step: Diffusion steps with the shape of [B x 1]
            context: Context with the shape of [B x L x ``d_context``], default to None.
        """
        h = self.input(x.transpose(1, 2))
        diff_emb = self.pos_encoder(diffusion_step)

        skip = None
        for i in range(self.n_layer):
            h, skip_connection = self.residual_blocks[i](h, diff_emb, context)
            skip = skip_connection if skip is None else skip_connection + skip

        out = skip / math.sqrt(self.n_layer)
        out = self.out_proj(out).transpose(1, 2)

        assert (
            out.size() == x.size()
        ), f"Output shape {out.size()} is not equal to input shape {x.size()}"
        return out
