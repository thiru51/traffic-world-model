import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from twm.utils.device import MODEL_SIZES, apply_model_size

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    size: str = "s"
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
    action_noise: float = 0.35
    seed: int = 0


@dataclass
class TrainConfig:
    steps: int = 3000
    # 0 means "pick from free VRAM at startup"; any positive value is taken literally.
    batch_size: int = 0
    seq_len: int = 50
    lr: float = 1e-4
    ac_lr: float = 3e-5
    eps: float = 1e-8
    grad_clip: float = 1000.0
    ac_grad_clip: float = 100.0
    weight_decay: float = 0.0
    amp: bool = True
    compile: bool = False
    tf32: bool = True
    cudnn_benchmark: bool = True
    grad_checkpoint: bool = False
    train_actor_critic: bool = True
    log_every: int = 25
    ckpt_every: int = 500
    seed: int = 0
    out_dir: str = "checkpoints/run1"
    device: str = "auto"


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


def common_parser(description):
    """Flags shared by every entry point.

    Dotted `--section.key=value` overrides still work alongside these and are applied
    last, so `--batch-size 32 --train.batch_size=8` ends up at 8. The named flags exist
    because they are the ones you actually reach for at the command line.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    p.add_argument("--device", default=None, help="cuda | cuda:1 | cpu | auto (default auto)")
    p.add_argument("--batch-size", type=int, default=None, help="0 = auto-scale from free VRAM")
    p.add_argument("--seq-len", type=int, default=None, help="timesteps per training sequence")
    p.add_argument("--model-size", choices=list(MODEL_SIZES), default=None)
    p.add_argument("--steps", type=int, default=None, help="gradient steps")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="torch.compile the world model (off by default: slow first step, few gains "
        "on runs this short)",
    )
    return p


def config_from_args(argv, description="", add_args=None):
    p = common_parser(description)
    if add_args is not None:
        add_args(p)
    args, rest = p.parse_known_args(list(argv))

    cfg = load_config(args.config)
    if args.model_size:
        apply_model_size(cfg, args.model_size)
        cfg.model.size = args.model_size
    for flag, section, key in [
        ("device", "train", "device"),
        ("batch_size", "train", "batch_size"),
        ("seq_len", "train", "seq_len"),
        ("steps", "train", "steps"),
        ("out_dir", "train", "out_dir"),
        ("seed", "train", "seed"),
        ("amp", "train", "amp"),
        ("compile", "train", "compile"),
    ]:
        val = getattr(args, flag)
        if val is not None:
            setattr(getattr(cfg, section), key, val)

    _merge(cfg, parse_cli_overrides(rest))
    return cfg, args
