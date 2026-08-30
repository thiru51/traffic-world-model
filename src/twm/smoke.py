"""Shape/plumbing check on random tensors. Runs in seconds and needs no dataset, so it
is the first thing to run after touching the model code or moving to a new machine."""

import sys

import torch

from twm.config import config_from_args
from twm.models.actor_critic import ImaginationActorCritic
from twm.models.world_model import WorldModel
from twm.utils.device import (
    autocast,
    make_grad_scaler,
    pick_amp_dtype,
    resolve_device,
    runtime_report,
    setup_backends,
)
from twm.utils.run import count_params, cuda_mem_report


def main():
    cfg, _ = config_from_args(sys.argv[1:], "shape and precision smoke test on random tensors")
    device = resolve_device(cfg.train.device)
    backends = setup_backends(device, cfg.train.tf32, cfg.train.cudnn_benchmark)
    amp_dtype = pick_amp_dtype(device)
    use_amp = bool(cfg.train.amp) and amp_dtype is not None
    scaler = make_grad_scaler(device, amp_dtype, use_amp)
    print("[runtime] " + "  ".join(f"{k}={v}" for k, v in runtime_report(
        device, amp_dtype, use_amp, backends).items()))

    channels, res = 5, cfg.env.resolution
    obs_shape = (channels, res, res)
    action_dim = 2
    b = max(2, min(4, cfg.train.batch_size or 4))
    t = cfg.train.seq_len

    wm = WorldModel(obs_shape, action_dim, cfg.model).to(device)
    if device.type == "cuda":
        wm.to_channels_last()
    ac = ImaginationActorCritic(wm.rssm.feat_dim, action_dim, cfg.ac).to(device)
    print(f"model size         : {cfg.model.size}")
    print(f"world model params : {count_params(wm)[0] / 1e6:.2f}M")
    print(f"actor-critic params: {count_params(ac)[0] / 1e6:.2f}M")
    print(f"feature dim        : {wm.rssm.feat_dim}")

    batch = {
        "obs": torch.rand(b, t, *obs_shape, device=device) - 0.5,
        "action": torch.rand(b, t, action_dim, device=device) * 2 - 1,
        "reward": torch.randn(b, t, device=device),
        "cont": torch.ones(b, t, device=device),
        "is_first": torch.zeros(b, t, device=device),
    }
    batch["is_first"][:, 0] = 1.0

    with autocast(device, amp_dtype, use_amp):
        loss, post, metrics = wm.loss(batch)
    # Every loss term must come back fp32 even under autocast; if one of the fp32()
    # guards in the model is ever dropped this assert is what catches it.
    assert loss.dtype == torch.float32, f"world model loss came back as {loss.dtype}"
    scaler.scale(loss).backward()
    print("world model loss:", float(loss))
    for k, v in metrics.items():
        print(f"  {k}: {float(v):.4f}")

    start = {k: v.reshape(-1, *v.shape[2:]).detach() for k, v in post.items()}
    with autocast(device, amp_dtype, use_amp):
        actor_loss, critic_loss, ac_metrics = ac.losses(wm, start)
    assert actor_loss.dtype == torch.float32 and critic_loss.dtype == torch.float32
    scaler.scale(actor_loss + critic_loss).backward()
    print("actor loss:", float(actor_loss), "critic loss:", float(critic_loss))
    for k, v in ac_metrics.items():
        print(f"  {k}: {float(v):.4f}")

    for name, t_ in [("wm loss", loss), ("actor", actor_loss), ("critic", critic_loss)]:
        assert torch.isfinite(t_), f"{name} is not finite"
    print("memory:", cuda_mem_report())
    print("smoke test OK")


if __name__ == "__main__":
    main()
