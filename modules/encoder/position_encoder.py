import math
from modules.general.utils import *


class PosEncoder(nn.Module):
    """
    Encoder of positional embedding, generates PE and then feed into 2 full-connected layers with SiLU.
    """

    def __init__(
        self,
        d_emb_vec: int = 128,
        d_mlp: int = None,
        max_period: int = 10000,
        register_embedding: bool = True,
        max_steps: int = 1000,
    ):
        """
        Args:
            d_emb_vec: The dimension of raw embedding vectors, default to 128
            d_mlp: The dimension of hidden layer in MLP, default to ``d_emb_vec`` * 4
            max_period: controls the minimum frequency of the embeddings, default to 10000
            register_embedding: Whether to register the embedding table as buffer, default to True
            max_steps: The max step would be used, only applied when ``register_embedding`` is True, default to 1000
        """
        super().__init__()

        self.d_emb_vec = d_emb_vec
        self.d_mlp = self.d_emb_vec * 4 if d_mlp is None else d_mlp
        self.max_period = max_period
        self.register_embedding = register_embedding
        self.max_steps = max_steps

        if self.register_embedding:
            self.register_buffer("embedding", self._build_embedding(), persistent=False)

        self.out = nn.Sequential(
            Linear(self.d_emb_vec, self.d_mlp),
            nn.SiLU(),
            Linear(self.d_mlp, self.d_mlp),
            nn.SiLU(),
        )

    def forward(self, steps) -> torch.Tensor:
        """
        Args:
            steps: a [B x 1] 2D Tensor of N indices, one per batch element. These may be fractional.
        Returns:
            The embedding of steps after MLP with shape [B x 1 x d_mlp]
        """
        if self.register_embedding:
            # Extract buffered embeddings
            if steps.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                h = self.embedding[steps]
            else:
                h = self._lerp_embedding(steps)
        else:
            # Generate embeddings on the fly
            h = self._timestep_embedding(steps, self.d_emb_vec, self.max_period)
            h = h.unsqueeze(1)
        h = self.out(h)
        return h

    @staticmethod
    def _timestep_embedding(timesteps, dim, max_period=10000):
        """
        Create and return sinusoidal timestep embeddings directly.

        Args:
            timesteps: a [N x 1] 2D Tensor of N indices, one per batch element. These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings, default to 10000.
        Returns:
            an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            / half
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        )
        args = timesteps.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def _build_embedding(self) -> torch.Tensor:
        """
        Build the raw embedding table of steps
        """
        steps = torch.arange(self.max_steps)[:, None]  # [max_steps x 1]
        return self._timestep_embedding(steps, self.d_emb_vec, self.max_period)

    def _lerp_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: The steps with shape [B x 1]
        Returns:
            The raw embedding vectors of steps after linear interpolation
        """
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)
