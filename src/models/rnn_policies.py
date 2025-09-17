from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def fanin_init(tensor, fanin=None):
    if fanin is None:
        fanin = tensor.size(0)
    v = 1. / math.sqrt(fanin)
    return tensor.data.uniform_(-v, v)

class ActorLSTM(nn.Module):
    """
    LSTM(256) core + MLP(256,256) heads: total "3x256".
    For SAC: outputs mean & log_std for tanh-Gaussian policy.
    For DDPG: outputs deterministic tanh action (use same class; ignore log_std).
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256, mlp_hidden: int = 256, mlp_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        layers = []
        last = hidden_size
        for _ in range(mlp_layers):
            layers += [nn.Linear(last, mlp_hidden), nn.ReLU()]
            last = mlp_hidden
        self.mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(last, action_dim)
        self.log_std = nn.Linear(last, action_dim)

        # Initialization
        fanin_init(self.mu.weight)
        fanin_init(self.log_std.weight)

    def forward(self, obs_seq: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        obs_seq: (B, T, obs_dim)
        h: (h0,c0) each (1, B, hidden_size) or None
        Returns: mu, log_std, (hT,cT) where mu/log_std have shape (B,T,action_dim)
        """
        out, hT = self.lstm(obs_seq, h)
        z = self.mlp(out)
        mu = self.mu(z)
        log_std = torch.clamp(self.log_std(z), -5, 2)  # stabilize
        return mu, log_std, hT

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)

class CriticQ(nn.Module):
    """Simple MLP Q(s,a) with 3x256 total depth matching actor's capacity."""
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        dims = [obs_dim + action_dim] + [hidden]*layers + [1]
        net = []
        for i in range(len(dims)-2):
            net += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        net += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*net)
        for m in self.net:
            if isinstance(m, nn.Linear):
                fanin_init(m.weight)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)