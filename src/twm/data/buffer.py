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
        # Zero-length files are what an interrupted collect run leaves behind; skipping
        # them here beats a confusing np.load failure much later.
        return sorted(p for p in self.root.glob("ep_*.npz") if p.stat().st_size > 0)


class SequenceSampler:
    """Uniform sampler over (episode, start offset) pairs.

    Every episode is concatenated into one flat uint8 tensor and, when it fits, parked in
    VRAM. Sampling is then a single gather on device: no per-item numpy slicing, no
    host-to-device copy, no synchronisation point in the middle of the step loop. On a
    laptop the CPU side of batch assembly is the first thing that starves the GPU, and
    the dataset here is a few hundred MB by design, so this trade is nearly free.

    If the dataset would take more than `gpu_fraction` of free VRAM the tensors stay in
    pinned host memory instead and each batch is copied with non_blocking=True, which at
    least overlaps the transfer with the previous step's compute.
    """

    KEYS = ("obs", "action", "reward", "cont", "is_first")

    def __init__(self, root, seq_len, device="cuda", holdout=0, gpu_fraction=0.25, verbose=True):
        self.seq_len = seq_len
        self.device = torch.device(device)
        store = EpisodeStore(root)
        files = store.files()
        if not files:
            raise FileNotFoundError(f"no episodes in {root}; run `pixi run collect-data` first")
        self.meta = store.read_meta() if (Path(root) / "meta.json").exists() else {}

        self.eval_episodes = []
        episodes = []
        for i, f in enumerate(files):
            d = np.load(f)
            ep = {k: d[k] for k in self.KEYS}
            if ep["obs"].shape[0] < seq_len + 1:
                continue
            (self.eval_episodes if i < holdout else episodes).append(ep)
        if not episodes:
            raise ValueError(f"every episode in {root} is shorter than seq_len={seq_len}")

        self.lengths = np.array([e["obs"].shape[0] for e in episodes])
        self.total_steps = int(self.lengths.sum())
        self.obs_shape = episodes[0]["obs"].shape[1:]
        self.action_dim = episodes[0]["action"].shape[1]
        self.n_episodes = len(episodes)

        flat = {k: np.concatenate([e[k] for e in episodes], 0) for k in self.KEYS}
        self.nbytes = int(sum(v.nbytes for v in flat.values()))

        # A window may not straddle an episode boundary, so valid starts are enumerated
        # once here rather than rejection-sampled every step.
        offsets = np.concatenate([[0], np.cumsum(self.lengths)])[:-1]
        starts = np.concatenate(
            [o + np.arange(L - seq_len + 1) for o, L in zip(offsets, self.lengths)]
        )

        self.storage = self._choose_storage(gpu_fraction)
        pin = self.storage == "pinned"
        target = self.device if self.storage == "gpu" else torch.device("cpu")

        def put(arr, dtype):
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)
            t = t.to(target)
            return t.pin_memory() if pin else t

        self.obs = put(flat["obs"], torch.uint8)
        self.action = put(flat["action"], torch.float32)
        self.reward = put(flat["reward"], torch.float32)
        self.cont = put(flat["cont"], torch.float32)
        self.is_first = put(flat["is_first"], torch.float32)
        self.starts = torch.from_numpy(starts).to(self.device).long()
        self.arange = torch.arange(seq_len, device=self.device)
        self.n_windows = int(self.starts.shape[0])

        # Kept as plain numpy for the eval code, which wants whole episodes, not windows.
        self.episodes = episodes

        if verbose:
            print(
                f"[data] {self.n_episodes} train / {len(self.eval_episodes)} holdout episodes, "
                f"{self.total_steps} transitions, {self.n_windows} valid windows, "
                f"{self.nbytes / 2**20:.0f} MiB held in {self.storage}"
            )

    def _choose_storage(self, gpu_fraction):
        if self.device.type != "cuda":
            return "cpu"
        free, _ = torch.cuda.mem_get_info(self.device.index or 0)
        return "gpu" if self.nbytes < gpu_fraction * free else "pinned"

    def sample(self, batch_size, generator=None):
        pick = torch.randint(
            0, self.n_windows, (batch_size,), device=self.device, generator=generator
        )
        # [B, T] absolute indices into the flat arrays. Every valid start is equally
        # likely, so every transition is equally likely - not every episode.
        idx = self.starts[pick][:, None] + self.arange[None, :]

        if self.storage == "gpu":
            gather = lambda t: t[idx]
        else:
            cpu_idx = idx.cpu()
            gather = lambda t: t[cpu_idx].to(self.device, non_blocking=True)

        return self._format(
            gather(self.obs),
            gather(self.action),
            gather(self.reward),
            gather(self.cont),
            gather(self.is_first),
        )

    def _format(self, obs, action, reward, cont, is_first):
        # NHWC uint8 -> NCHW float centred on zero; DreamerV3 predicts pixels in
        # [-0.5, 0.5] with a unit-variance Gaussian, so the target must match.
        obs = obs.permute(0, 1, 4, 2, 3).float().div_(255.0).sub_(0.5)
        return {
            "obs": obs,
            "action": action.float(),
            "reward": reward.float(),
            "cont": cont.float(),
            "is_first": is_first.float(),
        }

    def episode_slice(self, ep, start, length, eval_set=False):
        source = self.eval_episodes if eval_set else self.episodes
        e = source[ep]
        sl = slice(start, start + length)
        obs = torch.from_numpy(e["obs"][sl][None]).to(self.device)
        to = lambda k: torch.from_numpy(e[k][sl][None]).to(self.device)
        return self._format(obs, to("action"), to("reward"), to("cont"), to("is_first"))
