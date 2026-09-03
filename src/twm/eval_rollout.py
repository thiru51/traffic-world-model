"""Open-loop fidelity check: how far can the learned model imagine before it drifts?

Protocol per evaluation window:
  1. Feed `context` real frames through the posterior so the recurrent state is warm.
  2. Roll the *prior* forward `horizon` steps using the actions that were really taken,
     but with no further observations. This is what "imagination" means at act time.
  3. Decode and compare against the real frames.

Three curves are reported so the imagination number means something:
  imagined   - the model rolling its own prior forward (what we care about)
  persistence- repeat the last observed frame for the whole horizon (the trivial
               baseline any model must beat, otherwise it has just learned to copy)
  posterior  - reconstruction with every real frame available (an upper bound on what
               the encoder/decoder pair can represent at all)
"""

import sys
from pathlib import Path

import numpy as np
import torch

from twm.config import config_from_args
from twm.data.buffer import SequenceSampler
from twm.models.world_model import WorldModel
from twm.utils.device import resolve_device, setup_backends
from twm.utils.run import device_info, seed_everything, write_json

CONTEXT = 5
HORIZON = 15
HOLDOUT_EPISODES = 6


def psnr(mse):
    # Images live in [-0.5, 0.5], so peak-to-peak is 1.0 and the usual 20*log10(MAX/RMSE)
    # collapses to -10*log10(mse).
    return -10.0 * np.log10(np.maximum(mse, 1e-12))


# Channel 0 is the static road network. It is identical in every frame of an episode, so
# including it rewards a model for copying the background and says nothing about whether it
# predicted where the traffic went. It is also the channel that breaks the threshold: its
# maximum raw value is 126/255, which centres to -0.006 and never clears thresh=0.0, while
# an undertrained decoder does spray stray pixels above it -- inflating the union and
# collapsing IoU. Both problems disappear by scoring the traffic channels only.
TRAFFIC_CHANNELS = slice(1, None)


def occupancy_iou(pred, target, thresh=0.0, channels=TRAFFIC_CHANNELS):
    """IoU of the thresholded frames over the moving-traffic channels. Pixels are centred on
    zero, so >0 means 'brighter than mid grey', which is where the drawn vehicles are."""
    pred = pred[..., channels, :, :]
    target = target[..., channels, :, :]
    p = pred > thresh
    t = target > thresh
    inter = (p & t).sum(dim=(-3, -2, -1)).float()
    union = (p | t).sum(dim=(-3, -2, -1)).float().clamp(min=1.0)
    return inter / union


def centroid_error_px(pred, target, thresh=0.0, channels=TRAFFIC_CHANNELS):
    """Pixel-space centroid displacement of the occupied region, per frame. Traffic channels
    only, for the same reason as occupancy_iou."""
    pred = pred[..., channels, :, :]
    target = target[..., channels, :, :]
    h, w = pred.shape[-2:]
    ys = torch.arange(h, device=pred.device).float()
    xs = torch.arange(w, device=pred.device).float()

    def centroid(x):
        m = (x > thresh).float().sum(dim=-3)
        total = m.sum(dim=(-2, -1)).clamp(min=1e-6)
        cy = (m.sum(-1) * ys).sum(-1) / total
        cx = (m.sum(-2) * xs).sum(-1) / total
        return torch.stack([cy, cx], -1), total

    cp, mp = centroid(pred)
    ct, mt = centroid(target)
    valid = (mp > 1.0) & (mt > 1.0)
    err = (cp - ct).norm(dim=-1)
    return err, valid


@torch.no_grad()
def evaluate(cfg, ckpt_path, n_windows=64, context=CONTEXT, horizon=HORIZON, save_frames=True):
    device = resolve_device(cfg.train.device)
    setup_backends(device, cfg.train.tf32, cfg.train.cudnn_benchmark)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    data = SequenceSampler(
        cfg.data.dir, context + horizon, device=str(device), holdout=HOLDOUT_EPISODES
    )
    episodes = data.eval_episodes or data.episodes
    held_out = bool(data.eval_episodes)

    wm = WorldModel(tuple(ckpt["obs_shape"]), ckpt["action_dim"], cfg.model).to(device)
    wm.load_state_dict(ckpt["wm"])
    wm.eval()

    rng = np.random.default_rng(cfg.train.seed + 1)
    window = context + horizon
    picks = []
    for _ in range(n_windows):
        e = int(rng.integers(len(episodes)))
        n = episodes[e]["obs"].shape[0]
        if n < window:
            continue
        picks.append((e, int(rng.integers(0, n - window + 1))))

    mse_imag, mse_persist, mse_post = [], [], []
    iou_imag, iou_persist = [], []
    cent_imag, cent_persist, cent_valid = [], [], []
    rew_err = []
    strip = None

    for e, s in picks:
        batch = _window_batch(episodes[e], s, window, device)
        obs, action = batch["obs"], batch["action"]

        embed = wm.encoder(obs[:, :context])
        post, _ = wm.rssm.observe(embed, action[:, :context], batch["is_first"][:, :context])
        state = {k: v[:, -1] for k, v in post.items()}

        feats = []
        for t in range(horizon):
            state = wm.rssm.img_step(state, action[:, context + t])
            feats.append(wm.rssm.to_feat(state))
        feats = torch.stack(feats, 1)
        imagined = wm.decoder(feats)

        target = obs[:, context : context + horizon]
        last_seen = obs[:, context - 1 : context].expand_as(target)

        full_embed = wm.encoder(obs)
        full_post, _ = wm.rssm.observe(full_embed, action, batch["is_first"])
        recon = wm.decoder(wm.rssm.to_feat(full_post))[:, context : context + horizon]

        per_step = lambda x, y: ((x - y) ** 2).mean(dim=(-3, -2, -1))[0].cpu().numpy()
        mse_imag.append(per_step(imagined, target))
        mse_persist.append(per_step(last_seen, target))
        mse_post.append(per_step(recon, target))
        iou_imag.append(occupancy_iou(imagined, target)[0].cpu().numpy())
        iou_persist.append(occupancy_iou(last_seen, target)[0].cpu().numpy())
        ci, vi = centroid_error_px(imagined, target)
        cp, _ = centroid_error_px(last_seen, target)
        cent_imag.append(ci[0].cpu().numpy())
        cent_persist.append(cp[0].cpu().numpy())
        cent_valid.append(vi[0].cpu().numpy())

        pred_r = wm.reward_mean(feats)[0]
        rew_err.append((pred_r - batch["reward"][0, context : context + horizon]).abs().cpu().numpy())

        if save_frames and strip is None:
            strip = (target[0].cpu(), imagined[0].cpu(), recon[0].cpu())

    stack = lambda x: np.stack(x, 0)
    cv = stack(cent_valid).astype(bool)
    ci, cp = stack(cent_imag), stack(cent_persist)
    results = {
        "checkpoint": str(ckpt_path),
        "train_step_of_checkpoint": int(ckpt.get("step", -1)),
        "windows_evaluated": len(picks),
        "context_frames": context,
        "imagination_horizon": horizon,
        "evaluated_on_held_out_episodes": held_out,
        "held_out_episode_count": len(data.eval_episodes),
        "device": device_info(),
        "per_step": {
            "imagined_mse": stack(mse_imag).mean(0).tolist(),
            "persistence_mse": stack(mse_persist).mean(0).tolist(),
            "posterior_mse": stack(mse_post).mean(0).tolist(),
            "imagined_iou": stack(iou_imag).mean(0).tolist(),
            "persistence_iou": stack(iou_persist).mean(0).tolist(),
            "imagined_centroid_err_px": _masked_mean(ci, cv).tolist(),
            "persistence_centroid_err_px": _masked_mean(cp, cv).tolist(),
            "reward_abs_err": stack(rew_err).mean(0).tolist(),
        },
        "summary": {
            "imagined_mse_mean": float(stack(mse_imag).mean()),
            "persistence_mse_mean": float(stack(mse_persist).mean()),
            "posterior_mse_mean": float(stack(mse_post).mean()),
            "imagined_psnr_db_mean": float(psnr(stack(mse_imag).mean())),
            "persistence_psnr_db_mean": float(psnr(stack(mse_persist).mean())),
            "posterior_psnr_db_mean": float(psnr(stack(mse_post).mean())),
            "imagined_mse_step15": float(stack(mse_imag)[:, -1].mean()),
            "persistence_mse_step15": float(stack(mse_persist)[:, -1].mean()),
            "imagined_iou_mean": float(stack(iou_imag).mean()),
            "persistence_iou_mean": float(stack(iou_persist).mean()),
            "imagined_centroid_err_px_mean": float(ci[cv].mean()) if cv.any() else None,
            "persistence_centroid_err_px_mean": float(cp[cv].mean()) if cv.any() else None,
            "reward_abs_err_mean": float(stack(rew_err).mean()),
        },
    }

    # Write beside the checkpoint that was actually scored, not beside cfg.train.out_dir.
    # Those differ whenever --checkpoint points at another run, and taking the config path
    # silently drops run 2's results into run 1's directory, overwriting them.
    out_dir = Path(ckpt_path).parent if ckpt_path else Path(cfg.train.out_dir)
    write_json(out_dir / "rollout_fidelity.json", results)
    if strip is not None:
        _save_strip(strip, out_dir / "rollout_comparison.png")
    return results


def _masked_mean(values, mask):
    out = np.full(values.shape[1], np.nan)
    for t in range(values.shape[1]):
        col = values[:, t][mask[:, t]]
        if col.size:
            out[t] = col.mean()
    return out


def _window_batch(ep, start, length, device):
    sl = slice(start, start + length)
    obs = torch.from_numpy(ep["obs"][sl][None]).to(device)
    obs = obs.permute(0, 1, 4, 2, 3).float().div_(255.0).sub_(0.5)
    to = lambda k: torch.from_numpy(ep[k][sl][None]).to(device).float()
    return {
        "obs": obs,
        "action": to("action"),
        "reward": to("reward"),
        "cont": to("cont"),
        "is_first": to("is_first"),
    }


def _save_strip(strip, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target, imagined, recon = strip
    rows = [("real", target), ("imagined (open loop)", imagined), ("posterior recon", recon)]
    steps = target.shape[0]
    show = list(range(0, steps, max(1, steps // 8)))[:8]
    fig, axes = plt.subplots(len(rows), len(show), figsize=(1.5 * len(show), 1.6 * len(rows)))
    for r, (name, seq) in enumerate(rows):
        for c, t in enumerate(show):
            ax = axes[r, c]
            # Collapse the multi-channel top-down stack to a single greyscale image so
            # the three rows are visually comparable regardless of channel count.
            img = (seq[t] + 0.5).clamp(0, 1).mean(0).numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t+{t+1}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    def add_args(p):
        p.add_argument("--windows", type=int, default=64, help="evaluation windows to average")
        p.add_argument("--context", type=int, default=CONTEXT)
        p.add_argument("--horizon", type=int, default=HORIZON)
        p.add_argument("--checkpoint", default=None)

    cfg, args = config_from_args(sys.argv[1:], "open-loop imagination fidelity", add_args)
    seed_everything(cfg.train.seed)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg.train.out_dir) / "latest.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found; run `pixi run train` first")
    res = evaluate(cfg, ckpt, n_windows=args.windows, context=args.context, horizon=args.horizon)
    print("\n=== open-loop rollout fidelity ===")
    for k, v in res["summary"].items():
        print(f"{k}: {v}")
    print(f"\nper-step imagined MSE: {[round(x, 5) for x in res['per_step']['imagined_mse']]}")
    print(f"per-step persistence MSE: {[round(x, 5) for x in res['per_step']['persistence_mse']]}")
    print(f"\nwrote {Path(ckpt).parent / 'rollout_fidelity.json'}")


if __name__ == "__main__":
    main()
