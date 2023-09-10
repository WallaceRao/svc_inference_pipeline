import math
from modules.general.utils import *


class ResidualBlock(nn.Module):
    """
    Residual block with dilated convolution, main portion of ``DilatedCNN``
    """

    def __init__(
        self, channels=256, kernel_size=3, dilation=1, d_mlp=512, d_context=256
    ):
        """
        Args:
            channels: The number of channels of input and output, default to 256.
            kernel_size: The kernel size of dilated convolution, default to 3.
            dilation: The dilation rate of dilated convolution, default to 1.
            d_mlp: The dimension of hidden layer in MLP of diffusion step encoder, default to 512.
            d_context: The dimension of content encoder output, default to 256.
        """
        super().__init__()

        self.diff_proj = Linear(d_mlp, channels)

        self.dilated_conv = Conv1d(
            channels,
            channels * 2,
            kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )

        self.context_proj = Linear(d_context, channels * 2)

        self.out_proj = Conv1d(
            channels,
            channels * 2,
            1,
        )

    def forward(self, x, diff_emb, context=None):
        """
        Args:
            x: Latent noisy mel-spectrogram inherited from previous residual block with the shape of [B x C x L]
            diff_emb: Diffusion embeddings with the shape of [B x 1 x ``d_mlp``]
            context: Context with the shape of [B x L x ``d_context``], default to None.
        """
        diff_emb = self.diff_proj(diff_emb).transpose(1, 2)
        h = x + diff_emb

        h = self.dilated_conv(h)

        # Given context
        if context is not None:
            context = self.context_proj(context).transpose(1, 2)
            h = h + context

        h1, h2 = h.chunk(2, 1)
        h = torch.tanh(h1) * torch.sigmoid(h2)
        h = self.out_proj(h)
        res, skip = h.chunk(2, 1)

        return (res + x) / math.sqrt(2.0), skip
