"""Imagine-then-act: score candidate manoeuvres inside the world model before committing
a real environment step.

At every real step the agent proposes a set of candidate action sequences, rolls each of
them `horizon` steps forward through the RSSM prior (no simulator calls, no pixels
decoded), scores them with the reward head, the continue head and the critic, and only
then executes the first action of the winner.

The candidate set is the integration seam. `manoeuvre_primitives` below is a stand-in for
whatever proposes manoeuvres upstream; see the "Integration point" section of the README
for how a negotiation policy would be dropped in here instead.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from twm.config import config_from_args
from twm.envs.traffic_env import TrafficSceneEnv
from twm.models.actor_critic import ImaginationActorCritic
from twm.models.world_model import WorldModel
from twm.utils.device import resolve_device, setup_backends
from twm.utils.run import device_info, seed_everything, write_json

# (steering, throttle) held constant for the whole imagined horizon. Deliberately coarse:
# these stand for "manoeuvres a higher-level planner might propose", not a control basis.
MANOEUVRES = {
    "hold": (0.0, 0.3),
    "coast": (0.0, 0.0),
    "brake": (0.0, -0.6),
    "accelerate": (0.0, 0.7),
    "nudge_left": (-0.25, 0.3),
    "nudge_right": (0.25, 0.3),
    "turn_left": (-0.6, 0.15),
    "turn_right": (0.6, 0.15),
}


def manoeuvre_primitives(horizon, device):
    names = list(MANOEUVRES)
    seq = torch.tensor([MANOEUVRES[n] for n in names], dtype=torch.float32, device=device)
    return names, seq[None].expand(horizon, -1, -1).clone()


class Prescreener:
    def __init__(self, wm, ac, horizon, gamma, n_actor_samples=16):
        self.wm = wm
        self.ac = ac
        self.horizon = horizon
        self.gamma = gamma
        self.n_actor_samples = n_actor_samples

    @torch.no_grad()
    def candidates(self, state, device):
        names, prims = manoeuvre_primitives(self.horizon, device)
        if self.n_actor_samples == 0:
            return names, prims
        feat = self.wm.rssm.to_feat(state)
        rolled = self._actor_sequences(state, feat, device)
        return names + [f"actor_{i}" for i in range(rolled.shape[1])], torch.cat([prims, rolled], 1)

    @torch.no_grad()
    def _actor_sequences(self, state, feat, device):
        n = self.n_actor_samples
        s = {k: v.expand(n, *v.shape[1:]).contiguous() for k, v in state.items()}
        seqs = []
        for _ in range(self.horizon):
            a = self.ac.actor.act(self.wm.rssm.to_feat(s), sample=True)
            seqs.append(a)
            s = self.wm.rssm.img_step(s, a)
        return torch.stack(seqs, 0)

    @torch.no_grad()
    def score(self, state, action_seqs):
        """action_seqs: [H, N, A]. Returns [N] imagined discounted value."""
        n = action_seqs.shape[1]
        s = {k: v.expand(n, *v.shape[1:]).contiguous() for k, v in state.items()}
        total = torch.zeros(n, device=action_seqs.device)
        alive = torch.ones(n, device=action_seqs.device)
        disc = 1.0
        feat = None
        for t in range(self.horizon):
            s = self.wm.rssm.img_step(s, action_seqs[t])
            feat = self.wm.rssm.to_feat(s)
            reward = self.wm.reward_mean(feat)
            total = total + disc * alive * reward
            # Multiplying by the predicted survival probability is how a crash the model
            # foresees suppresses everything the candidate would have earned afterwards.
            alive = alive * self.wm.cont_prob(feat)
            disc = disc * self.gamma
        total = total + disc * alive * self.ac.critic.value(feat)
        return total

    @torch.no_grad()
    def choose(self, state, device):
        names, seqs = self.candidates(state, device)
        scores = self.score(state, seqs)
        best = int(scores.argmax())
        return seqs[0, best].clone(), names[best], scores


@torch.no_grad()
def run_episodes(cfg, ckpt_path, episodes=5, max_steps=250, mode="prescreen", n_actor_samples=16):
    device = resolve_device(cfg.train.device)
    setup_backends(device, cfg.train.tf32, cfg.train.cudnn_benchmark)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm = WorldModel(tuple(ckpt["obs_shape"]), ckpt["action_dim"], cfg.model).to(device).eval()
    wm.load_state_dict(ckpt["wm"])
    ac = ImaginationActorCritic(wm.rssm.feat_dim, ckpt["action_dim"], cfg.ac).to(device).eval()
    ac.load_state_dict(ckpt["ac"])

    env = TrafficSceneEnv(cfg.env)
    screener = Prescreener(wm, ac, cfg.ac.horizon, cfg.ac.gamma, n_actor_samples)

    rows, screen_times = [], []
    for ep in range(episodes):
        seed = cfg.env.start_seed + ep
        obs, _ = env.reset(seed=seed)
        state = wm.rssm.initial(1, device)
        prev_action = torch.zeros(1, ckpt["action_dim"], device=device)
        is_first = torch.ones(1, device=device)
        ret, steps, changed, picks = 0.0, 0, 0, {}

        for _ in range(max_steps):
            x = torch.from_numpy(obs).to(device)[None].permute(0, 3, 1, 2).float().div_(255).sub_(0.5)
            embed = wm.encoder(x)
            state, _ = wm.rssm.obs_step(state, prev_action, embed, is_first)
            is_first = torch.zeros(1, device=device)

            feat = wm.rssm.to_feat(state)
            greedy = ac.actor.act(feat, sample=False)
            if mode == "prescreen":
                t0 = time.perf_counter()
                action, name, _ = screener.choose(state, device)
                screen_times.append(time.perf_counter() - t0)
                picks[name] = picks.get(name, 0) + 1
                changed += int(not torch.allclose(action, greedy[0], atol=0.05))
                action = action[None]
            else:
                action = greedy

            prev_action = action
            obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
            ret += reward
            steps += 1
            if terminated or truncated:
                break

        rows.append(
            {
                "episode": ep,
                "seed": seed,
                "return": ret,
                "steps": steps,
                "crash": bool(info.get("crash", False) or info.get("crash_vehicle", False)),
                "arrive_dest": bool(info.get("arrive_dest", False)),
                "action_changed_frac": changed / max(steps, 1),
                "candidate_picks": picks,
            }
        )
        print(f"[{mode}] ep {ep} seed {seed} return {ret:8.2f} steps {steps:4d} "
              f"changed {changed / max(steps, 1):.2f}", flush=True)

    env.close()
    return {
        "mode": mode,
        "episodes": rows,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "crash_rate": float(np.mean([r["crash"] for r in rows])),
        "mean_action_changed_frac": float(np.mean([r["action_changed_frac"] for r in rows])),
        "n_candidates": len(MANOEUVRES) + (n_actor_samples if mode == "prescreen" else 0),
        "imagination_horizon": cfg.ac.horizon,
        "mean_prescreen_seconds": float(np.mean(screen_times)) if screen_times else None,
        "device": device_info(),
    }


def main():
    def add_args(p):
        p.add_argument("--episodes", type=int, default=5)
        p.add_argument("--max-steps", type=int, default=250)
        p.add_argument("--actor-samples", type=int, default=16)
        p.add_argument("--checkpoint", default=None)

    cfg, args = config_from_args(sys.argv[1:], "imagine-then-act action pre-screening", add_args)
    seed_everything(cfg.train.seed)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg.train.out_dir) / "latest.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found; run `pixi run train` first")

    kw = dict(episodes=args.episodes, max_steps=args.max_steps, n_actor_samples=args.actor_samples)
    screened = run_episodes(cfg, ckpt, mode="prescreen", **kw)
    direct = run_episodes(cfg, ckpt, mode="actor_only", **kw)
    out = {"prescreen": screened, "actor_only": direct}
    write_json(Path(cfg.train.out_dir) / "prescreen.json", out)

    print("\n=== imagine-then-act ===")
    for k in ("mean_return", "mean_steps", "crash_rate", "mean_prescreen_seconds"):
        print(f"{k}: prescreen={screened[k]}  actor_only={direct[k]}")
    print(f"action changed by pre-screening: {screened['mean_action_changed_frac']:.2%} of steps")


if __name__ == "__main__":
    main()
