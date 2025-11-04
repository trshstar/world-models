"""
World Models (2018) lightweight scaffolding.

This package provides the high-level class structure for a modular
reimplementation of the "World Models" paper by Ha and Schmidhuber (2018).

Intentionally minimal: only interfaces, wiring, and TODOs — no full logic.
"""

from .config import TrainingConfig
from .controller import Controller
from .mdnrnn import MDNRNN, MDRNNState
from .vae import VAE
from .world_model import WorldModel

__all__ = [
    "VAE",
    "MDNRNN",
    "MDRNNState",
    "Controller",
    "WorldModel",
    "TrainingConfig",
]
