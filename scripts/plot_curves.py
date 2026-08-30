"""Turn a run's metrics.jsonl into the loss-curve figure committed under results/.

    python scripts/plot_curves.py checkpoints/run1 results/loss_curves.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [
    ("wm/loss", "world model total loss", "log"),
    ("wm/image", "image reconstruction (nats/frame)", "log"),
    ("wm/recon_mse", "reconstruction MSE", "log"),
    ("wm/kl_dyn", "dynamics KL (nats)", "linear"),
    ("wm/reward", "reward head NLL", "linear"),
    ("ac/critic_loss", "critic loss", "linear"),
]


def load(run_dir):
    rows = [json.loads(l) for l in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if l]
    # A rerun appends to the same file, so keep only the last pass through each step.
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    return [seen[k] for k in sorted(seen)]


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/run1")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/loss_curves.png")
    rows = load(run_dir)
    summary = {}
    sp = run_dir / "train_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, title, scale) in zip(axes.flat, PANELS):
        xs = [r["step"] for r in rows if key in r]
        ys = [r[key] for r in rows if key in r]
        if not xs:
            ax.set_visible(False)
            continue
        ax.plot(xs, ys, lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("gradient step", fontsize=8)
        if scale == "log" and min(ys) > 0:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)

    bits = [
        f"{summary.get('model_size', '?')} ({summary.get('world_model_params', 0)/1e6:.1f}M params)",
        f"batch {summary.get('batch_size', '?')} x seq {summary.get('seq_len', '?')}",
        f"{summary.get('steps_per_sec', '?')} steps/s",
        f"peak {summary.get('torch_peak_alloc_mib', 0):.0f} MiB torch alloc",
        f"{summary.get('wall_clock_seconds', '?')}s wall clock",
        summary.get("runtime", {}).get("gpu_name", ""),
    ]
    fig.suptitle(
        "traffic-world-model training (short, deliberately not converged)\n" + "  |  ".join(
            b for b in bits if b),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out} from {len(rows)} logged points")


if __name__ == "__main__":
    main()
