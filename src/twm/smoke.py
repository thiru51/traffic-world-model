"""Shape/plumbing check on random tensors. Runs in seconds and needs no dataset, so it
is the first thing to run after touching the model code or moving to a new machine."""

import sys

import torch

from twm.config import config_to_dict, load_config, parse_cli_overrides
from twm.models.actor_critic import ImaginationActorCritic
from twm.models.world_model import WorldModel
from twm.utils.run import count_params, cuda_mem_report, device_info


def main():
    cfg = load_config(overrides=parse_cli_overrides(sys.argv[1:]))
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print("device:", device_info())

    channels, res = 5, cfg.env.resolution
    obs_shape = (channels, res, res)
    action_dim = 2
    b, t = 4, cfg.train.seq_len

    wm = WorldModel(obs_shape, action_dim, cfg.model).to(device)
    ac = ImaginationActorCritic(wm.rssm.feat_dim, action_dim, cfg.ac).to(device)
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

    amp = cfg.train.amp and device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        loss, post, metrics = wm.loss(batch)
    loss.backward()
    print("world model loss:", float(loss))
    for k, v in metrics.items():
        print(f"  {k}: {float(v):.4f}")

    start = {k: v.reshape(-1, *v.shape[2:]).detach() for k, v in post.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        actor_loss, critic_loss, ac_metrics = ac.losses(wm, start)
    (actor_loss + critic_loss).backward()
    print("actor loss:", float(actor_loss), "critic loss:", float(critic_loss))
    for k, v in ac_metrics.items():
        print(f"  {k}: {float(v):.4f}")

    print("memory:", cuda_mem_report())
    print("smoke test OK")


if __name__ == "__main__":
    main()
