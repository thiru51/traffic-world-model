from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    cnn_depth: int = 24
    deter: int = 512
    stoch: int = 32
    classes: int = 32
    hidden: int = 384
    unimix: float = 0.01
    n_bins: int = 255
    kl_free: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    image_scale: float = 1.0
    reward_scale: float = 1.0
    cont_scale: float = 1.0


@dataclass
class ACConfig:
    hidden: int = 384
    n_bins: int = 255
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_coef: float = 3e-4
    slow_reg: float = 1.0
    slow_critic_every: int = 1
    slow_critic_tau: float = 0.02


@dataclass
class EnvConfig:
    resolution: int = 64
    frame_stack: int = 3
    map_size: int = 3
    traffic_density: float = 0.2
    horizon: int = 500
    start_seed: int = 0
    num_scenarios: int = 40


@dataclass
class DataConfig:
    dir: str = "data/metadrive64"
    episodes: int = 60
    max_steps_per_episode: int = 250
    action_noise: float = 0.25
    seed: int = 0


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 16
    seq_len: int = 50
    lr: float = 1e-4
    ac_lr: float = 3e-5
    eps: float = 1e-8
    grad_clip: float = 1000.0
    ac_grad_clip: float = 100.0
    weight_decay: float = 0.0
    amp: bool = True
    grad_checkpoint: bool = False
    train_actor_critic: bool = True
    log_every: int = 25
    ckpt_every: int = 500
    seed: int = 0
    out_dir: str = "checkpoints/run1"
    device: str = "cuda"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    ac: ACConfig = field(default_factory=ACConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge(dc, overrides):
    known = {f.name for f in fields(dc)}
    for k, v in overrides.items():
        if k not in known:
            raise KeyError(f"unknown config key '{k}' for {type(dc).__name__}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__"):
            _merge(cur, v)
        else:
            setattr(dc, k, type(cur)(v) if cur is not None else v)
    return dc


def load_config(path=None, overrides=None):
    cfg = Config()
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    path = Path(path)
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        _merge(cfg, raw)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def parse_cli_overrides(args):
    """Turn `--train.steps=500` style flags into a nested dict."""
    out = {}
    for a in args:
        if not a.startswith("--"):
            raise ValueError(f"expected --section.key=value, got {a!r}")
        key, _, val = a[2:].partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def config_to_dict(cfg):
    return asdict(cfg)
