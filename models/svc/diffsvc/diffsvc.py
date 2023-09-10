from modules.diffusion.dilated_cnn.dilated_cnn import DilatedCNN
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


class DiffSVC(ModelMixin, ConfigMixin):
    def __init__(self, cfg):
        super().__init__()
        self.diff = DilatedCNN(cfg)

    def forward(self, x, t, c):
        return self.diff(x, t, c)
