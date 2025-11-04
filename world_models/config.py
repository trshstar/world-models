from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Top-level knobs for training the World Model pipeline.

    Keep this lean; add specifics as you implement details from the paper.
    """

    # Data
    dataset_path: str = "./data"
    sequence_length: int = 100
    batch_size: int = 32

    # VAE
    z_dim: int = 32
    vae_beta: float = 1.0
    vae_lr: float = 1e-3

    # MDN-RNN (M)
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    num_mixture_components: int = 5
    rnn_lr: float = 1e-3

    # Controller (C)
    controller_hidden_dim: int = 64
    controller_lr: float = 1e-3

    # Training schedule
    vae_epochs: int = 10
    rnn_epochs: int = 10
    controller_epochs: int = 10

    # Misc
    device: str = "cpu"  # e.g., "cuda" when available
    seed: Optional[int] = 42

