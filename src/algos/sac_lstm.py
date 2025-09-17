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
class SACConfig:
    gamma: float = 0.98
    tau: float = 0.02
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 5e-5
    target_entropy_scale: float = -1.5
    grad_clip: float = 1.0
    updates_per_step: float = 0.5

class SACLSTM:
    def __init__(self, obs_dim, action_dim, device, cfg: SACConfig, hidden=256, mlp_hidden=256, mlp_layers=2):
        self.device = torch.device(device)
        self.cfg = cfg
        self.action_dim = action_dim

        self.actor = ActorLSTM(obs_dim, action_dim, hidden, mlp_hidden, mlp_layers).to(self.device)
        self.q1 = CriticQ(obs_dim, action_dim).to(self.device)
        self.q2 = CriticQ(obs_dim, action_dim).to(self.device)
        self.q1_tgt = CriticQ(obs_dim, action_dim).to(self.device)
        self.q2_tgt = CriticQ(obs_dim, action_dim).to(self.device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        # Entropy temperature (auto)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = cfg.target_entropy_scale * action_dim

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False, h=None):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        mu, log_std, hT = self.actor(obs_t, h)
        if deterministic:
            a = torch.tanh(mu)
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            gauss = mu + std * eps
            a = torch.tanh(gauss)
        return a.squeeze(0).squeeze(0).cpu().numpy(), hT

    def _actor_sample(self, obs_seq):
        mu, log_std, _ = self.actor(obs_seq, None)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        gauss = mu + std * eps
        a = torch.tanh(gauss)

        # log-prob with tanh correction
        logp = -0.5 * (((gauss - mu) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        logp = logp.sum(dim=-1)
        # tanh correction
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        return a, logp

    def _soft_update(self, net, net_tgt, tau):
        with torch.no_grad():
            for p, p_tgt in zip(net.parameters(), net_tgt.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self, replay: SequenceReplay, batch_size: int, seq_len: int):
        cfg = self.cfg
        obs, act, rew, nxt, done = replay.sample(batch_size, seq_len)
        B, T, _ = obs.shape

        # Flatten for critics
        obs_f = obs.reshape(B*T, -1)
        act_f = act.reshape(B*T, -1)
        nxt_f = nxt.reshape(B*T, -1)
        rew_f = rew.reshape(B*T)
        done_f = done.reshape(B*T)

        with torch.no_grad():
            # Next action and entropy
            a2_seq, logp2_seq = self._actor_sample(nxt)  # (B,T,A), (B,T)
            a2 = a2_seq.reshape(B*T, -1)
            logp2 = logp2_seq.reshape(B*T)
            q1_t = self.q1_tgt(nxt_f, a2)
            q2_t = self.q2_tgt(nxt_f, a2)
            qmin = torch.min(q1_t, q2_t)
            alpha = self.log_alpha.exp()
            y = rew_f + (1.0 - done_f) * cfg.gamma * (qmin - alpha * logp2)

        # Critic losses
        q1 = self.q1(obs_f, act_f)
        q2 = self.q2(obs_f, act_f)
        loss_q1 = F.mse_loss(q1, y)
        loss_q2 = F.mse_loss(q2, y)
        self.opt_q1.zero_grad(set_to_none=True); loss_q1.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), cfg.grad_clip); self.opt_q1.step()
        self.opt_q2.zero_grad(set_to_none=True); loss_q2.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), cfg.grad_clip); self.opt_q2.step()

        # Actor loss
        a_seq, logp_seq = self._actor_sample(obs)
        a = a_seq.reshape(B*T, -1)
        logp = logp_seq.reshape(B*T)
        q1_pi = self.q1(obs_f, a)
        q2_pi = self.q2(obs_f, a)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        loss_actor = (alpha * logp - q_pi).mean()
        self.opt_actor.zero_grad(set_to_none=True); loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip); self.opt_actor.step()

        # Temperature loss
        loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(set_to_none=True); loss_alpha.backward(); self.opt_alpha.step()

        # Targets
        self._soft_update(self.q1, self.q1_tgt, cfg.tau)
        self._soft_update(self.q2, self.q2_tgt, cfg.tau)

        return dict(loss_actor=float(loss_actor.item()),
                    loss_q1=float(loss_q1.item()),
                    loss_q2=float(loss_q2.item()),
                    alpha=float(alpha.item()))