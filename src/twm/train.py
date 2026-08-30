import sys
import time
from pathlib import Path

import numpy as np
import torch

from twm.config import config_from_args, config_to_dict
from twm.data.buffer import SequenceSampler
from twm.models.actor_critic import ImaginationActorCritic
from twm.models.world_model import WorldModel
from twm.utils.device import (
    auto_batch_size,
    autocast,
    make_grad_scaler,
    maybe_compile,
    pick_amp_dtype,
    resolve_device,
    runtime_report,
    setup_backends,
)
from twm.utils.run import (
    JsonlLogger,
    count_params,
    cuda_mem_report,
    seed_everything,
    write_json,
)

HOLDOUT_EPISODES = 6


def build(cfg, obs_shape, action_dim, device):
    wm = WorldModel(obs_shape, action_dim, cfg.model).to(device)
    if device.type == "cuda":
        wm.to_channels_last()
    ac = ImaginationActorCritic(wm.rssm.feat_dim, action_dim, cfg.ac).to(device)
    return wm, ac


def flatten_states(post):
    """[B, T, ...] posterior states -> [B*T, ...] imagination start points."""
    return {k: v.reshape(-1, *v.shape[2:]).detach() for k, v in post.items()}


def train(cfg):
    device = resolve_device(cfg.train.device)
    backends = setup_backends(device, tf32=cfg.train.tf32, benchmark=cfg.train.cudnn_benchmark)
    seed_everything(cfg.train.seed)

    amp_dtype = pick_amp_dtype(device)
    use_amp = bool(cfg.train.amp) and amp_dtype is not None
    # GradScaler exists only on the fp16 path. bf16 carries fp32's exponent range, so
    # scaling it is at best a no-op and at worst hides a real overflow.
    scaler = make_grad_scaler(device, amp_dtype, use_amp)

    if cfg.train.batch_size <= 0:
        cfg.train.batch_size = auto_batch_size(device, cfg.train.seq_len, cfg.model.size)
        print(f"[batch] auto-selected batch_size={cfg.train.batch_size} from free VRAM")

    rt = runtime_report(device, amp_dtype, use_amp, backends)
    print("[runtime] " + "  ".join(f"{k}={v}" for k, v in rt.items()))

    data = SequenceSampler(
        cfg.data.dir, cfg.train.seq_len, device=str(device), holdout=HOLDOUT_EPISODES
    )
    obs_shape = (data.obs_shape[2], data.obs_shape[0], data.obs_shape[1])
    wm, ac = build(cfg, obs_shape, data.action_dim, device)

    wm_total, wm_train = count_params(wm)
    ac_total, ac_train = count_params(ac)
    print(f"obs {obs_shape}  action_dim {data.action_dim}  model size '{cfg.model.size}'")
    print(f"world model params: {wm_total/1e6:.2f}M ({wm_train/1e6:.2f}M trainable)")
    print(f"actor-critic params: {ac_total/1e6:.2f}M ({ac_train/1e6:.2f}M trainable)")

    wm_opt = torch.optim.Adam(
        wm.parameters(), lr=cfg.train.lr, eps=cfg.train.eps, weight_decay=cfg.train.weight_decay
    )
    ac_params = list(ac.actor.parameters()) + list(ac.critic.parameters())
    ac_opt = torch.optim.Adam(ac_params, lr=cfg.train.ac_lr, eps=cfg.train.eps)

    step_fn = maybe_compile(wm.loss, cfg.train.compile, "world model loss")

    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out / "metrics.jsonl")
    write_json(out / "config.json", config_to_dict(cfg))

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.train.seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    step_times = []
    peak_smi = 0

    for step in range(1, cfg.train.steps + 1):
        step_t0 = time.time()
        batch = data.sample(cfg.train.batch_size, generator=gen)

        with autocast(device, amp_dtype, use_amp):
            wm_loss, post, metrics = step_fn(batch)
        wm_opt.zero_grad(set_to_none=True)
        scaler.scale(wm_loss).backward()
        # Unscale before clipping, otherwise the clip threshold is applied to scaled
        # gradients and means nothing. A no-op when the scaler is disabled (bf16/fp32).
        scaler.unscale_(wm_opt)
        wm_gn = torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.train.grad_clip)
        scaler.step(wm_opt)
        metrics["wm/grad_norm"] = wm_gn.detach()

        if cfg.train.train_actor_critic:
            # Imagination runs through the RSSM, so this backward also writes gradients
            # into the world model. Harmless: only ac_params are stepped, and wm's grads
            # are zeroed at the top of the next iteration before they are ever used.
            start = flatten_states(post)
            with autocast(device, amp_dtype, use_amp):
                actor_loss, critic_loss, ac_metrics = ac.losses(wm, start)
            ac_opt.zero_grad(set_to_none=True)
            scaler.scale(actor_loss + critic_loss).backward()
            scaler.unscale_(ac_opt)
            ac_gn = torch.nn.utils.clip_grad_norm_(ac_params, cfg.train.ac_grad_clip)
            scaler.step(ac_opt)
            ac.update_slow()
            ac_metrics["ac/grad_norm"] = ac_gn.detach()
            metrics.update(ac_metrics)

        # One update() per iteration no matter how many optimisers were stepped; calling
        # it twice halves the fp16 loss scale on every step until it underflows.
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.time() - step_t0)

        if step % cfg.train.log_every == 0 or step == 1:
            mem = cuda_mem_report()
            if mem.get("nvidia_smi_used_mib"):
                peak_smi = max(peak_smi, mem["nvidia_smi_used_mib"])
            metrics.update(mem)
            recent = step_times[-cfg.train.log_every :]
            sec = float(np.mean(recent))
            metrics["sec_per_step"] = sec
            metrics["steps_per_sec"] = 1.0 / sec
            metrics["transitions_per_sec"] = cfg.train.batch_size * cfg.train.seq_len / sec
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
    # The first few steps include cudnn.benchmark picking kernels and the caching
    # allocator warming up, so they are dropped from the throughput headline.
    warm = step_times[10:] or step_times
    summary = {
        "steps": cfg.train.steps,
        "wall_clock_seconds": round(elapsed, 1),
        "sec_per_step_mean": round(float(np.mean(warm)), 4),
        "sec_per_step_median": round(float(np.median(warm)), 4),
        "steps_per_sec": round(1.0 / float(np.mean(warm)), 3),
        "transitions_per_sec": round(
            cfg.train.batch_size * cfg.train.seq_len / float(np.mean(warm)), 1
        ),
        "batch_size": cfg.train.batch_size,
        "seq_len": cfg.train.seq_len,
        "transitions_per_step": cfg.train.batch_size * cfg.train.seq_len,
        "model_size": cfg.model.size,
        "world_model_params": wm_total,
        "actor_critic_params": ac_total,
        "peak_nvidia_smi_used_mib": peak_smi,
        "runtime": rt,
        "data_storage": data.storage,
        "dataset_mib": round(data.nbytes / 2**20, 1),
        "train_episodes": data.n_episodes,
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
    cfg, _ = config_from_args(sys.argv[1:], "train the world model and imagination actor-critic")
    train(cfg)


if __name__ == "__main__":
    main()
