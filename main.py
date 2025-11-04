"""Entry point wiring the World Models (2018) scaffold.

Fill in module internals following the paper once you dive into the appendix.
"""

from config import TrainingConfig
from vae import VAE
from mdnrnn import MDNRNN
from controller import Controller
from world_model import WorldModel


def main() -> None:
    # Minimal wiring example with placeholder dimensions.
    cfg = TrainingConfig()

    # Placeholder shapes/dims — adjust for your dataset/env.
    input_shape = (3, 64, 64)
    action_dim = 3

    V = VAE(input_shape=input_shape, z_dim=cfg.z_dim, beta=cfg.vae_beta)
    M = MDNRNN(z_dim=cfg.z_dim, action_dim=action_dim, hidden_dim=cfg.rnn_hidden_dim,
               num_layers=cfg.rnn_num_layers, num_mixtures=cfg.num_mixture_components)
    C = Controller(z_dim=cfg.z_dim, rnn_hidden_dim=cfg.rnn_hidden_dim, action_dim=action_dim,
                   hidden_dim=cfg.controller_hidden_dim)

    wm = WorldModel(V=V, M=M, C=C, cfg=cfg)

    # Print structure summary so you can start filling methods.
    print("WorldModel scaffold initialized:")
    print(f"- VAE: z_dim={cfg.z_dim}, input_shape={input_shape}")
    print(f"- MDNRNN: hidden={cfg.rnn_hidden_dim}, layers={cfg.rnn_num_layers}, mixtures={cfg.num_mixture_components}")
    print(f"- Controller: hidden={cfg.controller_hidden_dim}, action_dim={action_dim}")


if __name__ == "__main__":
    main()
