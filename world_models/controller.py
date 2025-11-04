from typing import Optional

import torch
from torch import nn


class Controller(nn.Module):
    """Controller module (C): policy over (z, h) -> action (Section 2.3).

    In the paper, C is often small (linear) and optimized with CMA-ES. Keep it
    flexible so you can experiment with other search strategies.
    """

    def __init__(self, z_dim: int, rnn_hidden_dim: int, action_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        # TODO: replace placeholder with linear/MLP policy as per chosen design
        self.policy: nn.Module = nn.Identity()

    def act(self, z: torch.Tensor, h: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Compute action from latent z and RNN hidden state h."""
        # TODO: implement forward pass; if stochastic, sample action
        raise NotImplementedError

    def parameters(self):
        """Torch-compatible parameters iterator for optimizers/ES."""
        return super().parameters()
