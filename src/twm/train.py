import sys
import time
from pathlib import Path

import numpy as np
import torch

from twm.config import config_to_dict, load_config, parse_cli_overrides
from twm.data.buffer import SequenceSampler
from twm.models.actor_critic import ImaginationActorCritic
from twm.models.world_model import WorldModel
from twm.utils.run import (
    JsonlLogger,
    count_params,
    cuda_mem_report,
    device_info,
    seed_everything,
    write_json,
)

HOLDOUT_EPISODES = 6


def build(cfg, obs_shape, action_dim, device):
    wm = WorldModel(obs_shape, action_dim, cfg.model).to(device)
    ac = ImaginationActorCritic(wm.rssm.feat_dim, action_dim, cfg.ac).to(device)
    return wm, ac


def flatten_states(post):
    """[B, T, ...] posterior states -> [B*T, ...] imagination start points."""
    return {k: v.reshape(-1, *v.shape[2:]).detach() for k, v in post.items()}


def train(cfg):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = SequenceSampler(
        cfg.data.dir, cfg.train.seq_len, device=str(device), holdout=HOLDOUT_EPISODES
    )
    obs_shape = (data.obs_shape[2], data.obs_shape[0], data.obs_shape[1])
    wm, ac = build(cfg, obs_shape, data.action_dim, device)

    wm_total, wm_train = count_params(wm)
    ac_total, ac_train = count_params(ac)
    print(f"device: {device_info()}")
    print(f"obs {obs_shape}  action_dim {data.action_dim}  train episodes {len(data.episodes)}")
    print(f"world model params: {wm_total/1e6:.2f}M ({wm_train/1e6:.2f}M trainable)")
    print(f"actor-critic params: {ac_total/1e6:.2f}M ({ac_train/1e6:.2f}M trainable)")

    wm_opt = torch.optim.Adam(
        wm.parameters(), lr=cfg.train.lr, eps=cfg.train.eps, weight_decay=cfg.train.weight_decay
    )
    ac_params = list(ac.actor.parameters()) + list(ac.critic.parameters())
    ac_opt = torch.optim.Adam(ac_params, lr=cfg.train.ac_lr, eps=cfg.train.eps)

    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out / "metrics.jsonl")
    write_json(out / "config.json", config_to_dict(cfg))

    # bf16 rather than fp16: Ada supports it natively and it has fp32's exponent range,
    # so the KL and two-hot log-softmax terms do not need a GradScaler to stay finite.
    amp_dtype = torch.bfloat16
    use_amp = cfg.train.amp and device.type == "cuda"
    rng = np.random.default_rng(cfg.train.seed)

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    t0 = time.time()
    step_times = []
    peak_smi = 0

    for step in range(1, cfg.train.steps + 1):
        step_t0 = time.time()
        batch = data.sample(cfg.train.batch_size, rng)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            wm_loss, post, metrics = wm.loss(batch)
        wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        wm_gn = torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.train.grad_clip)
        wm_opt.step()
        metrics["wm/grad_norm"] = wm_gn.detach()

        if cfg.train.train_actor_critic:
            start = flatten_states(post)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                actor_loss, critic_loss, ac_metrics = ac.losses(wm, start)
            ac_opt.zero_grad(set_to_none=True)
            (actor_loss + critic_loss).backward()
            ac_gn = torch.nn.utils.clip_grad_norm_(ac_params, cfg.train.ac_grad_clip)
            ac_opt.step()
            ac.update_slow()
            ac_metrics["ac/grad_norm"] = ac_gn.detach()
            metrics.update(ac_metrics)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - step_t0)

        if step % cfg.train.log_every == 0 or step == 1:
            mem = cuda_mem_report()
            if mem.get("nvidia_smi_used_mib"):
                peak_smi = max(peak_smi, mem["nvidia_smi_used_mib"])
            metrics.update(mem)
            metrics["sec_per_step"] = float(np.mean(step_times[-cfg.train.log_every :]))
            logger.log(step, metrics)

        if step % cfg.train.ckpt_every == 0 or step == cfg.train.steps:
            torch.save(
                {
                    "step": step,
                    "wm": wm.state_dict(),
                    "ac": ac.state_dict(),
                    "config": config_to_dict(cfg),
                    "obs_shape": obs_shape,
                    "action_dim": data.action_dim,
                },
                out / "latest.pt",
            )

    elapsed = time.time() - t0
    summary = {
        "steps": cfg.train.steps,
        "wall_clock_seconds": round(elapsed, 1),
        "sec_per_step_mean": round(float(np.mean(step_times)), 4),
        "sec_per_step_median": round(float(np.median(step_times)), 4),
        "batch_size": cfg.train.batch_size,
        "seq_len": cfg.train.seq_len,
        "transitions_per_step": cfg.train.batch_size * cfg.train.seq_len,
        "world_model_params": wm_total,
        "actor_critic_params": ac_total,
        "peak_nvidia_smi_used_mib": peak_smi,
        "device": device_info(),
        "amp_dtype": "bfloat16" if use_amp else "fp32",
        "train_episodes": len(data.episodes),
        "holdout_episodes": len(data.eval_episodes),
        "dataset_transitions": data.total_steps,
    }
    summary.update(cuda_mem_report())
    write_json(out / "train_summary.json", summary)
    logger.close()
    print("\n=== training summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return summary


def main():
    cfg = load_config(overrides=parse_cli_overrides(sys.argv[1:]))
    train(cfg)


if __name__ == "__main__":
    main()
