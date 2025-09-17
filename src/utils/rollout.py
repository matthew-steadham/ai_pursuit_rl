from typing import Callable
import numpy as np
from .replay import EpisodeBuffer, Transition

def collect_episode(env, policy_fn: Callable, device=None, noise_std: float = 0.0):
    """
    policy_fn(obs: np.ndarray) -> action (np.ndarray in [-1,1]^2)
    """
    ep = EpisodeBuffer()
    obs, _ = env.reset()
    done = False
    while not done:
        a = policy_fn(obs)
        a = np.asarray(a, dtype=np.float32)
        if noise_std > 0.0:
            a = np.clip(a + np.random.normal(0.0, noise_std, size=a.shape), -1.0, 1.0)
        next_obs, r, done, trunc, info = env.step(a)
        ep.add(Transition(obs=obs, action=a, reward=r, next_obs=next_obs, done=done or trunc))
        obs = next_obs
    return ep, info