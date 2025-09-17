# 2D Pursuit–Evasion RL

**Purpose:** Safe, research project to train an autonomous **interceptor drone** to capture a moving target in a 2‑D plane with **thrust limits**, **sensor noise**, and **process noise** under **partial observability**. Architectures: **LSTM (3×256)** actor, MLP critics; Algorithms: **DDPG** and **SAC** from scratch (PyTorch).

## Quickstart (VS Code)

```bash
# 1) Python 3.10+ recommended
python -V

# 2) Create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 4) Train (DDPG with LSTM policy)
python train_ddpg.py --steps 150000

# 5) Train (SAC with LSTM policy)
python train_sac.py --steps 150000

# 6) Evaluate / visualize a trained checkpoint
python eval_policy.py --algo ddpg --ckpt runs/ddpg/latest.pt --render
python eval_policy.py --algo sac  --ckpt runs/sac/latest.pt  --render
```

## Project Structure

```
ai_pursuit_rl/
  configs/
    default.yaml
  src/
    envs/pursuit2d.py          # Gymnasium-style env with thrust limits + noise
    models/rnn_policies.py     # LSTM(3x256) actor; MLP critics
    algos/ddpg_lstm.py         # DDPG trainer with sequence sampling
    algos/sac_lstm.py          # SAC trainer with sequence sampling + entropy tuning
    utils/replay.py            # Episode/sequence replay buffer
    utils/rollout.py           # Data collection loops
    vis/plotting.py            # Trajectory visualization
  train_ddpg.py
  train_sac.py
  eval_policy.py
  requirements.txt
  README.md
```

## Key Design Choices

- **LSTM 3×256 policy:** LSTM core (hidden=256) + two 256‑unit MLP heads → total “3×256”. Policy outputs are tanh‑squashed to the env’s thrust bounds.
- **Partial observability:** The agent observes relative position/velocity with noise; the LSTM captures temporal context.
- **Sequence replay:** Off‑policy updates use truncated BPTT on fixed‑length sequences sampled from replay.
- **Thrust limits:** Action ∈ [−1, +1]^2 scaled to max acceleration; velocity is clipped to `max_speed`.
- **Noise:** Sensor noise (obs) + process noise (dynamics) are configurable.
- **SAC details:** Twin critics, target networks, reparameterization, automatic entropy temperature.
- **DDPG details:** Target networks, Polyak averaging, action noise (Gaussian), gradient clipping.
- **Reproducibility:** Global seeding, deterministic PyTorch where feasible.

## Safety & Scope

This code focuses on **non‑weaponized** pursuit–evasion. Applying it to weapons or weapon guidance is **not permitted**. Keep it in robotics, autonomy, or game AI contexts.

## Hyperparameter Tips

- Increase `hidden_size`, `seq_len`, or batch size for more memory use by the policy.
- If learning stalls, reduce sensor/process noise or increase reward shaping for closing velocity.
- SAC is more stable than DDPG; start with SAC for baseline performance.

## License

MIT