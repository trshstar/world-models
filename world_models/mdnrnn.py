from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class MDRNNState:
    """Container for RNN hidden state and optional cell state."""

    h: Optional[torch.Tensor] = None
    c: Optional[torch.Tensor] = None


class MDNRNN(nn.Module):
    """Memory module (M): MDN-RNN scaffold (Section 2.2 / Appendix C).

    Responsibilities:
    - roll latent sequence z_t and action a_t to predict z_{t+1} distribution
    - maintain recurrent state across steps
    - compute MDN loss (mixture of Gaussians over z)
    """

    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_mixtures: int = 5,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        # TODO: replace placeholders with actual LSTM + MDN heads per appendix
        self.rnn: nn.Module = nn.Identity()
        self.mdn_head: nn.Module = nn.Identity()

    def init_state(self, batch_size: int) -> MDRNNState:
        """Allocate zero/learned initial state."""
        # TODO: create framework tensors here
        return MDRNNState(h=None, c=None)

    def forward_step(self, z_t: torch.Tensor, a_t: torch.Tensor, state: MDRNNState) -> Tuple[torch.Tensor, MDRNNState]:
        """Advance one step: (z_t, a_t, state) -> MDN params for z_{t+1}.

        Returns (mdn_params, next_state).
        """
        # TODO: implement RNN + MDN head
        raise NotImplementedError

    def loss(self, mdn_params: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """Compute MDN negative log-likelihood of z_next under predicted mixture."""
        # TODO: implement MDN NLL
        raise NotImplementedError

    def train_step(self, batch) -> dict:
        """One optimization step on a batch of sequences.

        Batch expected to contain (z_t, a_t, z_{t+1}) sequences.
        """
        # TODO: unroll, accumulate loss, backprop, optimizer step
        raise NotImplementedError
