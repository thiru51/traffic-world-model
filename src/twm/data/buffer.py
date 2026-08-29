import json
from pathlib import Path

import numpy as np
import torch


class EpisodeStore:
    """Flat directory of per-episode .npz files.

    Observations stay uint8 on disk and are only converted to float on the GPU, which is
    what keeps a 50-step x 16-sequence batch inside a few hundred MB.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, index, **arrays):
        np.savez_compressed(self.root / f"ep_{index:05d}.npz", **arrays)

    def write_meta(self, meta):
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))

    def read_meta(self):
        return json.loads((self.root / "meta.json").read_text())

    def files(self):
        return sorted(self.root.glob("ep_*.npz"))


class SequenceSampler:
    """Uniform sampler over (episode, start offset) pairs, kept fully in RAM.

    The dataset is small enough by design that loading it once beats streaming from
    disk every step; on a laptop the disk is the first thing that starves the GPU.
    """

    KEYS = ("obs", "action", "reward", "cont", "is_first")

    def __init__(self, root, seq_len, device="cuda", holdout=0):
        self.seq_len = seq_len
        self.device = device
        store = EpisodeStore(root)
        files = store.files()
        if not files:
            raise FileNotFoundError(f"no episodes in {root}; run `pixi run collect-data` first")
        self.meta = store.read_meta() if (Path(root) / "meta.json").exists() else {}
        self.eval_episodes = []
        self.episodes = []
        for i, f in enumerate(files):
            d = np.load(f)
            ep = {k: d[k] for k in self.KEYS}
            if ep["obs"].shape[0] < seq_len + 1:
                continue
            (self.eval_episodes if i < holdout else self.episodes).append(ep)
        if not self.episodes:
            raise ValueError(f"every episode in {root} is shorter than seq_len={seq_len}")
        self.lengths = np.array([e["obs"].shape[0] for e in self.episodes])
        self.total_steps = int(self.lengths.sum())
        self.obs_shape = self.episodes[0]["obs"].shape[1:]
        self.action_dim = self.episodes[0]["action"].shape[1]

    def sample(self, batch_size, rng):
        # Weight by length so every transition is equally likely, not every episode.
        valid = self.lengths - self.seq_len
        probs = valid / valid.sum()
        idx = rng.choice(len(self.episodes), size=batch_size, p=probs)
        out = {k: [] for k in self.KEYS}
        for i in idx:
            ep = self.episodes[i]
            start = rng.integers(0, self.lengths[i] - self.seq_len + 1)
            sl = slice(start, start + self.seq_len)
            for k in self.KEYS:
                out[k].append(ep[k][sl])
        return self._to_torch({k: np.stack(v, 0) for k, v in out.items()})

    def episode_slice(self, ep, start, length, eval_set=False):
        source = self.eval_episodes if eval_set else self.episodes
        e = source[ep]
        sl = slice(start, start + length)
        return self._to_torch({k: e[k][sl][None] for k in self.KEYS})

    def _to_torch(self, batch):
        dev = self.device
        obs = torch.from_numpy(batch["obs"]).to(dev, non_blocking=True)
        # NHWC uint8 -> NCHW float centred on zero; DreamerV3 predicts pixels in
        # [-0.5, 0.5] with a unit-variance Gaussian, so the target must match.
        obs = obs.permute(0, 1, 4, 2, 3).float().div_(255.0).sub_(0.5)
        return {
            "obs": obs,
            "action": torch.from_numpy(batch["action"]).to(dev).float(),
            "reward": torch.from_numpy(batch["reward"]).to(dev).float(),
            "cont": torch.from_numpy(batch["cont"]).to(dev).float(),
            "is_first": torch.from_numpy(batch["is_first"]).to(dev).float(),
        }
