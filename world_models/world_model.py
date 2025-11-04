from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig


@dataclass
class WorldModel:
    """High-level orchestrator wiring V, M, and C (Section 2 overview).

    Coordinates the staged training protocol and imagined rollouts where the
    controller learns inside the latent world modeled by V and M.
    """

    V: Any
    M: Any
    C: Any
    cfg: TrainingConfig

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        return self.V.encode(obs)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.V.decode(z)

    def imagine_step(self, z_t: torch.Tensor, a_t: torch.Tensor, state) -> Tuple[Any, Any]:
        """Use M to roll forward one step in latent space."""
        mdn_params, next_state = self.M.forward_step(z_t, a_t, state)
        # TODO: sample z_{t+1} from MDN params
        raise NotImplementedError

    def train_vae(self, dataloader: DataLoader) -> Dict[str, float]:
        """Phase 1: Train V on reconstruction/regularization."""
        # TODO:
        # for epoch in range(self.cfg.vae_epochs):
        #     for obs_batch in dataloader:
        #         obs_batch = obs_batch.to(self.cfg.device)
        #         metrics = self.V.train_step(obs_batch)
        #         track metrics (e.g., wandb/tensorboard)
        raise NotImplementedError

    def train_mdnrnn(self, dataloader: DataLoader) -> Dict[str, float]:
        """Phase 2: Train M on latent sequences and actions."""
        # TODO:
        # for epoch in range(self.cfg.rnn_epochs):
        #     for obs_seq, action_seq in dataloader:
        #         z_seq = self.V.encode(obs_seq)
        #         batch = (z_seq[:, :-1], action_seq[:, :-1], z_seq[:, 1:])
        #         metrics = self.M.train_step(batch)
        #         log metrics
        raise NotImplementedError

    def train_controller(self, env: Any) -> Dict[str, float]:
        """Phase 3: Optimize C in the imagined environment.

        Typically done via ES on return obtained by rolling in M with C.
        """
        # TODO:
        # - define evaluation function that resets env, encodes obs -> z
        # - roll out using M to sample imagined trajectories
        # - feed (z, h) -> actions from Controller
        # - optimize controller parameters with CMA-ES / gradients
        raise NotImplementedError
