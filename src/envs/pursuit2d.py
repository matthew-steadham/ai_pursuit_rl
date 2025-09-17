import math
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Pursuit2DEnv(gym.Env):
    """
    2D pursuit–evasion environment.
    Agent ("interceptor") must intercept a moving target subject to thrust limits.
    Observation: noisy relative state.
    Action: 2D thrust vector in [-1, 1] scaled by max_accel.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        dt: float = 0.05,
        max_steps: int = 600,
        arena_size: float = 2000.0,
        capture_radius: float = 2.0,
        mass: float = 1.0,
        max_accel: float = 8.0,
        max_speed: float = 20.0,
        sensor_noise_std: float = 0.05,
        process_noise_std: float = 0.05,
        target_speed: float = 8.0,
        target_turn_std: float = 0.2,
        init_min_sep: float = 200.0,
        init_max_sep: float =200.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.capture_radius = capture_radius
        self.mass = mass
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.sensor_noise_std = sensor_noise_std
        self.process_noise_std = process_noise_std
        self.target_speed = target_speed
        self.target_turn_std = target_turn_std
        self.init_min_sep = init_min_sep
        self.init_max_sep = init_max_sep

        # state variables
        self._reset_state()

        # Observation = [rel_px, rel_py, rel_vx, rel_vy, self_vx, self_vy]
        high = np.array([self.arena_size]*2 + [self.max_speed]*4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action: thrust in [-1,1]^2 -> scaled to max_accel
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.render_mode = "none"

    def _reset_state(self):
        self.t = 0
        self.step_count = 0

        # Sample initial positions with min/max separation
        angle = self.rng.uniform(0, 2*np.pi)
        dist = self.rng.uniform(self.init_min_sep, self.init_max_sep)
        self.target_pos = self.rng.uniform(-self.arena_size*0.5, self.arena_size*0.5, size=2)
        self.self_pos = self.target_pos + dist*np.array([math.cos(angle), math.sin(angle)], dtype=float)

        # Initial velocities
        self.self_vel = self.rng.normal(0, 0.1, size=2)
        th = self.rng.uniform(0, 2*np.pi)
        self.target_vel = self.target_speed*np.array([math.cos(th), math.sin(th)], dtype=float)

        self.done = False

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _observe(self) -> np.ndarray:
        rel_pos = self.target_pos - self.self_pos
        rel_vel = self.target_vel - self.self_vel
        obs = np.concatenate([rel_pos, rel_vel, self.self_vel], dtype=np.float32)

        # sensor noise (absolute std per component)
        noise = self.rng.normal(0.0, self.sensor_noise_std, size=obs.shape).astype(np.float32)
        return (obs + noise).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        return self._observe(), {}

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Step called after episode done; call reset().")

        # Scale action to acceleration
        a_cmd = np.clip(action, self.action_space.low, self.action_space.high)
        accel = a_cmd * self.max_accel

        # Process noise on dynamics (acceleration perturbation)
        if self.process_noise_std > 0.0:
            accel = accel + self.rng.normal(0.0, self.process_noise_std, size=2)

        # Update interceptor dynamics
        self.self_vel = self.self_vel + accel * self.dt
        speed = np.linalg.norm(self.self_vel)
        if speed > self.max_speed:
            self.self_vel = self.self_vel * (self.max_speed / (speed + 1e-8))
        self.self_pos = self.self_pos + self.self_vel * self.dt

        # Update target with random-walk heading
        turn = self.rng.normal(0.0, self.target_turn_std)
        v = self.target_vel
        vmag = np.linalg.norm(v) + 1e-8
        heading = math.atan2(v[1], v[0]) + turn
        self.target_vel = self.target_speed * np.array([math.cos(heading), math.sin(heading)], dtype=float)
        self.target_pos = self.target_pos + self.target_vel * self.dt

        # Compute reward
        rel_pos = self.target_pos - self.self_pos
        rel_vel = self.target_vel - self.self_vel
        dist = np.linalg.norm(rel_pos)
        closing_speed = - np.dot(rel_pos, rel_vel) / (dist + 1e-6)  # LOS closing rate
        control_pen = 0.01 * float(np.linalg.norm(a_cmd))

        reward = -0.02*dist + 0.05*closing_speed - control_pen - 0.001  # time penalty

        if dist < 12.0:
            # Pull toward zero distance, up to +0.5 when on top
            reward += 0.5 * (12.0 - dist) / 12.0

            # Penalize sideways (tangential) motion around the LOS so it commits to closure
            r = self.target_pos - self.self_pos
            r_norm = np.linalg.norm(r) + 1e-8
            r_hat = r / r_norm
            v_rel = (self.target_vel - self.self_vel)
            closing_speed = float(-np.dot(r_hat, v_rel))                # along LOS
            v_tan = v_rel - (np.dot(v_rel, r_hat) * r_hat)              # perpendicular to LOS
            tan_mag = float(np.linalg.norm(v_tan))
            reward -= 0.02 * tan_mag

        # Terminations
        captured = dist <= self.capture_radius
        out_of_bounds = (
            np.any(np.abs(self.self_pos) > self.arena_size) or
            np.any(np.abs(self.target_pos) > self.arena_size)
        )

        if captured:
            reward += 75.0
            self.done = True
        elif out_of_bounds:
            reward -= 25.0
            self.done = True
        else:
            self.step_count += 1
            self.done = self.step_count >= self.max_steps

        obs = self._observe()
        info = {"captured": captured, "out_of_bounds": out_of_bounds}
        return obs, reward, self.done, False, info

    def render(self):
        pass