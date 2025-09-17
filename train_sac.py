import os, yaml, argparse, json
import numpy as np
import torch

from src.envs.pursuit2d import Pursuit2DEnv
from src.utils.replay import SequenceReplay
from src.utils.rollout import collect_episode
from src.algos.sac_lstm import SACLSTM, SACConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--run_dir", type=str, default="runs/sac")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    np.random.seed(seed); torch.manual_seed(seed)

    env_cfg = cfg["env"]
    env = Pursuit2DEnv(seed=seed, **env_cfg)

    device = cfg["device"]
    obs_dim = cfg["model"]["obs_dim"]
    action_dim = cfg["model"]["action_dim"]

    sac_cfg = SACConfig(**cfg["sac"])
    agent = SACLSTM(obs_dim, action_dim, device, sac_cfg,
                    hidden=cfg["model"]["hidden_size"],
                    mlp_hidden=cfg["model"]["mlp_hidden"],
                    mlp_layers=cfg["model"]["mlp_layers"])

    replay = SequenceReplay(capacity_episodes=cfg["replay"]["capacity"],
                            obs_dim=obs_dim, action_dim=action_dim, device=device)

    os.makedirs(args.run_dir, exist_ok=True)
    log_path = os.path.join(args.run_dir, "log.jsonl")
    ckpt_path = os.path.join(args.run_dir, "latest.pt")

    # Warmup with random policy
    min_init_episodes = cfg["replay"]["min_init_episodes"]
    print(f"Collecting {min_init_episodes} warmup episodes (random policy) ...")
    for _ in range(min_init_episodes):
        ep, info = collect_episode(
            env,
            policy_fn=lambda ob: np.random.uniform(-1, 1, size=(action_dim,)).astype(np.float32)
        )
        replay.push_episode(ep)

    total_steps = 0
    best_score = -1e9

    with open(log_path, "w") as logf:
        while total_steps < args.steps:
            def pol(obs):
                a, _ = agent.act(obs, deterministic=False)
                return a
            ep, info = collect_episode(env, pol, noise_std=0.0)
            replay.push_episode(ep)
            ep_return = float(sum(ep.rewards))
            total_steps += ep.length()

            # Updates
            updates = int(sac_cfg.updates_per_step * ep.length())
            for _ in range(updates):
                stats = agent.update(replay,
                                     batch_size=cfg["replay"]["batch_size"],
                                     seq_len=cfg["replay"]["seq_len"])
            # Logging
            rec = {"steps": int(total_steps), "episode_return": ep_return}
            rec.update(stats)
            print(rec)
            logf.write(json.dumps(rec) + "\\n"); logf.flush()

            # Checkpoint
            if ep_return > best_score:
                best_score = ep_return
                torch.save(dict(actor=agent.actor.state_dict(),
                                q1=agent.q1.state_dict(),
                                q2=agent.q2.state_dict(),
                                log_alpha=float(agent.log_alpha.item())),
                           ckpt_path)

    print("Training complete. Saved:", ckpt_path)

if __name__ == "__main__":
    main()