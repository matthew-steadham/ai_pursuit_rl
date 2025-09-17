import numpy as np
from src.envs.pursuit2d import Pursuit2DEnv

def test_env_runs():
    env = Pursuit2DEnv()
    obs,_ = env.reset()
    for _ in range(10):
        a = np.random.uniform(-1,1,size=(2,))
        obs, r, done, trunc, info = env.step(a)
        if done or trunc: break
    assert obs.shape[0] == 6