import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import weight_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class MelGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.hop_length = np.prod(self.cfg.model.melgan.ratios)
        mult = int(2 ** len(self.cfg.model.melgan.ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(
                self.cfg.preprocess.n_mel,
                mult * self.cfg.model.melgan.ngf,
                kernel_size=7,
                padding=0,
            ),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(self.cfg.model.melgan.ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * self.cfg.model.melgan.ngf,
                    mult * self.cfg.model.melgan.ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(self.cfg.model.melgan.n_residual_layers):
                model += [
                    ResnetBlock(mult * self.cfg.model.melgan.ngf // 2, dilation=3**j)
                ]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(self.cfg.model.melgan.ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


# if __name__ == "__main__":
#     # cfg = Defau

#     model = Generator([8, 8, 2, 2], 128, 32, 3)

#     x = torch.randn(3, 128, 10)
#     print(x.shape)

#     y = model(x)
#     print(y.shape)
#     assert y.shape == torch.Size([3, 1, 2560])

#     pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)
