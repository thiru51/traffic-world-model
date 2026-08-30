import sys
import time

import numpy as np

from twm.config import config_from_args, config_to_dict
from twm.data.buffer import EpisodeStore
from twm.envs.scripted_policy import NoisyLaneFollower
from twm.envs.traffic_env import TrafficSceneEnv


def collect(cfg):
    env = TrafficSceneEnv(cfg.env)
    rng = np.random.default_rng(cfg.data.seed)
    policy = NoisyLaneFollower(rng, noise=cfg.data.action_noise)
    store = EpisodeStore(cfg.data.dir)

    print(f"obs shape {env.obs_shape}, action dim {env.action_dim}")
    t0 = time.time()
    stats = []
    for ep in range(cfg.data.episodes):
        seed = cfg.env.start_seed + (ep % cfg.env.num_scenarios)
        obs, _ = env.reset(seed=seed)
        policy.reset()

        obs_buf = [obs]
        act_buf = [np.zeros(env.action_dim, np.float32)]
        rew_buf = [0.0]
        cont_buf = [1.0]
        first_buf = [1.0]
        # Ground-truth ego state is not fed to the model; it is stored so rollout
        # fidelity can later be checked in metric space rather than only in pixels.
        ego_buf = [env.ego_state()]
        ep_return, crashed = 0.0, False

        for _ in range(cfg.data.max_steps_per_episode):
            action = policy(env.agent)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action.astype(np.float32))
            rew_buf.append(reward)
            cont_buf.append(0.0 if terminated else 1.0)
            first_buf.append(0.0)
            ego_buf.append(env.ego_state())
            ep_return += reward
            if terminated or truncated:
                crashed = bool(info.get("crash", False) or info.get("crash_vehicle", False))
                break

        store.write(
            ep,
            obs=np.stack(obs_buf).astype(np.uint8),
            action=np.stack(act_buf).astype(np.float32),
            reward=np.array(rew_buf, np.float32),
            cont=np.array(cont_buf, np.float32),
            is_first=np.array(first_buf, np.float32),
            ego_pos=np.stack([e["position"] for e in ego_buf]).astype(np.float32),
            ego_heading=np.array([e["heading"] for e in ego_buf], np.float32),
            ego_speed=np.array([e["speed"] for e in ego_buf], np.float32),
        )
        stats.append(
            {"episode": ep, "seed": seed, "steps": len(obs_buf), "return": ep_return, "crash": crashed}
        )
        print(
            f"ep {ep:>3} seed {seed:>3} steps {len(obs_buf):>4} return {ep_return:8.2f} "
            f"crash {crashed}",
            flush=True,
        )

    elapsed = time.time() - t0
    total_steps = int(sum(s["steps"] for s in stats))
    meta = {
        "obs_shape": list(env.obs_shape),
        "action_dim": env.action_dim,
        "episodes": len(stats),
        "total_steps": total_steps,
        "mean_episode_steps": total_steps / max(len(stats), 1),
        "mean_return": float(np.mean([s["return"] for s in stats])),
        "crash_rate": float(np.mean([s["crash"] for s in stats])),
        "collect_seconds": round(elapsed, 1),
        "config": config_to_dict(cfg),
        "per_episode": stats,
    }
    store.write_meta(meta)
    env.close()
    print(
        f"\ncollected {len(stats)} episodes / {total_steps} transitions in {elapsed:.1f}s "
        f"-> {cfg.data.dir}"
    )
    return meta


def main():
    def add_args(p):
        p.add_argument("--episodes", type=int, default=None)
        p.add_argument("--data-dir", default=None)

    cfg, args = config_from_args(sys.argv[1:], "collect scripted driving episodes", add_args)
    if args.episodes is not None:
        cfg.data.episodes = args.episodes
    if args.data_dir is not None:
        cfg.data.dir = args.data_dir
    collect(cfg)


if __name__ == "__main__":
    main()
