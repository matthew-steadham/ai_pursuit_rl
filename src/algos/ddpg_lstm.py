from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..models.rnn_policies import ActorLSTM, CriticQ
from ..utils.replay import SequenceReplay

@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    noise_std: float = 0.2
    grad_clip: float = 1.0
    updates_per_step: int = 1

class DDPGAgent:
    def __init__(self, obs_dim, action_dim, device, cfg: DDPGConfig, hidden=256, mlp_hidden=256, mlp_layers=2):
        self.device = torch.device(device)
        self.cfg = cfg
        self.actor = ActorLSTM(obs_dim, action_dim, hidden, mlp_hidden, mlp_layers).to(self.device)
        self.actor_tgt = ActorLSTM(obs_dim, action_dim, hidden, mlp_hidden, mlp_layers).to(self.device)
        self.critic = CriticQ(obs_dim, action_dim).to(self.device)
        self.critic_tgt = CriticQ(obs_dim, action_dim).to(self.device)

        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, h=None):
        # Single-step inference using sequence length 1
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        mu, _, hT = self.actor(obs_t, h)
        a = torch.tanh(mu)  # deterministic
        return a.squeeze(0).squeeze(0).cpu().numpy(), hT

    def _soft_update(self, net, net_tgt, tau):
        with torch.no_grad():
            for p, p_tgt in zip(net.parameters(), net_tgt.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self, replay: SequenceReplay, batch_size: int, seq_len: int, gamma: float, tau: float, grad_clip: float):
        obs, act, rew, nxt, done = replay.sample(batch_size, seq_len)
        B, T, _ = obs.shape

        # Flatten time for critic (uses MLP without recurrence)
        obs_f = obs.reshape(B*T, -1)
        act_f = act.reshape(B*T, -1)
        nxt_f = nxt.reshape(B*T, -1)
        rew_f = rew.reshape(B*T)
        done_f = done.reshape(B*T)

        with torch.no_grad():
            # Target actor actions (deterministic tanh(mu))
            mu_nxt, _, _ = self.actor_tgt(nxt, None)
            a_nxt = torch.tanh(mu_nxt).reshape(B*T, -1)

            q_tgt = self.critic_tgt(nxt_f, a_nxt)
            y = rew_f + (1.0 - done_f) * gamma * q_tgt

        # Critic update
        q_pred = self.critic(obs_f, act_f)
        loss_c = F.mse_loss(q_pred, y)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        self.opt_critic.step()

        # Actor update: maximize Q(s, a(s))
        mu, _, _ = self.actor(obs, None)
        a = torch.tanh(mu).reshape(B*T, -1)
        q = self.critic(obs_f, a)
        loss_a = - q.mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        self.opt_actor.step()

        # Targets
        self._soft_update(self.actor, self.actor_tgt, tau)
        self._soft_update(self.critic, self.critic_tgt, tau)

        return dict(loss_actor=float(loss_a.item()), loss_critic=float(loss_c.item()))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict()), path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())