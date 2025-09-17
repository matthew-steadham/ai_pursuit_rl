from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch

@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool

class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []

    def add(self, tr: Transition):
        self.obs.append(tr.obs)
        self.actions.append(tr.action)
        self.rewards.append(tr.reward)
        self.next_obs.append(tr.next_obs)
        self.dones.append(tr.done)

    def length(self):
        return len(self.rewards)

class SequenceReplay:
    """
    Stores complete episodes then samples fixed-length sequences for truncated BPTT.
    """
    def __init__(self, capacity_episodes: int = 10000, obs_dim: int = 6, action_dim: int = 2, device: str = "cpu"):
        self.cap = capacity_episodes
        self.device = torch.device(device)
        self.episodes: List[EpisodeBuffer] = []

    def push_episode(self, ep: EpisodeBuffer):
        self.episodes.append(ep)
        if len(self.episodes) > self.cap:
            self.episodes.pop(0)

    def size(self):
        return sum(e.length() for e in self.episodes)

    def sample(self, batch_size: int, seq_len: int):
        """Return tensors with shape (B, T, *)"""
        if len(self.episodes) == 0:
            raise RuntimeError("Replay empty")
        B = batch_size

        obs_list = []
        act_list = []
        rew_list = []
        next_obs_list = []
        done_list = []

        for _ in range(B):
            ep = np.random.choice(self.episodes)
            if ep.length() < seq_len:
                # resample until long enough
                while ep.length() < seq_len:
                    ep = np.random.choice(self.episodes)
            start = np.random.randint(0, ep.length() - seq_len + 1)
            sl = slice(start, start+seq_len)
            obs_list.append(np.stack(ep.obs[sl], axis=0))
            act_list.append(np.stack(ep.actions[sl], axis=0))
            rew_list.append(np.array(ep.rewards[sl], dtype=np.float32))
            next_obs_list.append(np.stack(ep.next_obs[sl], axis=0))
            done_list.append(np.array(ep.dones[sl], dtype=np.float32))

        obs = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=self.device)
        act = torch.tensor(np.stack(act_list, axis=0), dtype=torch.float32, device=self.device)
        rew = torch.tensor(np.stack(rew_list, axis=0), dtype=torch.float32, device=self.device)
        nxt = torch.tensor(np.stack(next_obs_list, axis=0), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.stack(done_list, axis=0), dtype=torch.float32, device=self.device)
        return obs, act, rew, nxt, done
    