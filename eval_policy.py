import argparse, yaml, torch, numpy as np
from src.envs.pursuit2d import Pursuit2DEnv
from src.algos.ddpg_lstm import DDPGAgent, DDPGConfig
from src.algos.sac_lstm import SACLSTM, SACConfig
from src.vis.plotting import plot_trajectories

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--algo", type=str, choices=["ddpg","sac"], required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env = Pursuit2DEnv(seed=cfg["seed"], **cfg["env"])
    device = cfg["device"]
    obs_dim = cfg["model"]["obs_dim"]
    action_dim = cfg["model"]["action_dim"]

    if args.algo == "ddpg":
        agent = DDPGAgent(obs_dim, action_dim, device, DDPGConfig(**cfg["ddpg"]),
                          hidden=cfg["model"]["hidden_size"],
                          mlp_hidden=cfg["model"]["mlp_hidden"],
                          mlp_layers=cfg["model"]["mlp_layers"])
        agent.load(args.ckpt, map_location=device)
        def pol(obs): a,_ = agent.act(obs); return a
    else:
        agent = SACLSTM(obs_dim, action_dim, device, SACConfig(**cfg["sac"]),
                        hidden=cfg["model"]["hidden_size"],
                        mlp_hidden=cfg["model"]["mlp_hidden"],
                        mlp_layers=cfg["model"]["mlp_layers"])
        ckpt = torch.load(args.ckpt, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        def pol(obs): a,_ = agent.act(obs, deterministic=True); return a

    trajs = []
    for ep in range(args.episodes):
        obs,_ = env.reset()
        done = False
        S = [env.self_pos.copy()]
        T = [env.target_pos.copy()]
        while not done:
            a = agent.act(obs, deterministic=args.deterministic)[0]
            obs, r, done, trunc, info = env.step(a)
            S.append(env.self_pos.copy())
            T.append(env.target_pos.copy())
        trajs.append({"self": np.stack(S,0), "target": np.stack(T,0)})
    captures = 0
    steps_list = []
    final_dists = []
    for tr in trajs:
        steps_list.append(len(tr["self"]) - 1)
    # Re-run quick sims to grab capture flags & final dists
    cap2 = 0; dists = []
    for _ in range(args.episodes):
        obs,_ = env.reset()
        done = False
        while not done:
            a = agent.act(obs, deterministic=args.deterministic)[0]
            obs, r, done, trunc, info = env.step(a)
        cap2 += int(info.get("captured", False))
        dists.append(np.linalg.norm(env.target_pos - env.self_pos))
    print(f"capture_rate: {cap2}/{args.episodes} ({cap2/args.episodes:.1%})")
    print(f"avg_steps: {np.mean(steps_list):.1f}")
    print(f"median_final_distance: {np.median(dists):.2f}")

    tot = getattr(args, "episodes", 100)
    caps = touts = oobs = 0
    final_d, min_d = [], []

    for _ in range(tot):
        obs, _ = env.reset()
        done = False
        dmin = 1e9
        while not done:
            a = agent.act(obs, deterministic=True)[0]
            obs, r, done, trunc, info = env.step(a)
            # compute distance using env state (adapt names if yours differ)
            d = float(np.linalg.norm(env.target_pos - env.self_pos))
            if d < dmin:
                dmin = d
        if info.get("captured", False):
            caps += 1
        elif info.get("oob", False) or info.get("terminated_reason", "") == "oob":
            oobs += 1
        else:
            touts += 1
        final_d.append(d)
        min_d.append(dmin)

    print(f"[EVAL] capture_rate={caps}/{tot} ({caps/tot:.1%}), "f"timeouts={touts}, oob={oobs}, "f"median_final_d={np.median(final_d):.2f}, median_min_d={np.median(min_d):.2f}")

    if args.render:
        plot_trajectories(trajs, arena_size=cfg["env"]["arena_size"], capture_radius=cfg["env"]["capture_radius"])

if __name__ == "__main__":
    main()