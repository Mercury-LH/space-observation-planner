"""Small supervised scorers. Scores order candidates; they never certify legality."""
from __future__ import annotations

import torch
from torch import nn


class CandidateMLP(nn.Module):
    def __init__(self, dynamic_dim: int = 0):
        super().__init__()
        if dynamic_dim < 0:
            raise ValueError('dynamic_dim must be nonnegative')
        self.dynamic_dim = dynamic_dim
        self.encoder = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.head = (nn.Sequential(nn.Linear(64 + dynamic_dim, 64), nn.ReLU(), nn.Linear(64, 1))
                     if dynamic_dim else nn.Linear(64, 1))

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(static)
        if self.dynamic_dim:
            if dynamic is None or dynamic.shape != (static.shape[0], self.dynamic_dim):
                raise ValueError('dynamic feature shape does not match checkpoint schema')
            encoded = torch.cat((encoded, dynamic), dim=1)
        elif dynamic is not None:
            raise ValueError('static model does not consume dynamic inputs')
        return self.head(encoded).squeeze(-1)
