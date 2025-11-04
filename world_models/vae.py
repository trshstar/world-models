from typing import Tuple

import torch
from torch import nn


class VAE(nn.Module):
    """Vision module (V): Convolutional beta-VAE scaffold per Section 2.1.

    Responsibilities:
    - encode(observation) -> latent z
    - decode(z) -> reconstruction
    - training step hooks for KL + reconstruction objective
    """

    def __init__(self, input_shape: Tuple[int, int, int], z_dim: int, beta: float = 1.0):
        super().__init__()
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.beta = beta
        # TODO: replace placeholders with actual CNN encoder/decoder from appendix
        self.encoder: nn.Module = nn.Identity()
        self.decoder: nn.Module = nn.Identity()
        self.fc_mu: nn.Module = nn.Identity()
        self.fc_logvar: nn.Module = nn.Identity()
        self.fc_decode: nn.Module = nn.Identity()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observation to latent code z.

        TODO: implement using CNN encoder. Return tensor with shape [B, z_dim].
        """
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct observation from latent code.

        TODO: implement using CNN decoder. Return tensor with shape [B, C, H, W].
        """
        raise NotImplementedError

    def loss(self, obs: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute beta-VAE loss.

        TODO: implement reconstruction + beta * KL terms.
        """
        raise NotImplementedError

    def train_step(self, batch: torch.Tensor) -> dict:
        """Single optimization step for VAE.

        Expects a batch of observations. Returns dict of metrics.
        """
        # TODO: forward, compute loss, backward, step optimizer
        raise NotImplementedError
