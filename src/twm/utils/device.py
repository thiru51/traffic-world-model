"""Single place where device, precision and memory-format policy is decided.

Everything else in the repo imports from here rather than touching torch.backends or
torch.autocast directly, so there is exactly one answer to "what precision is this run
using" and nothing can silently disagree with the training loop.
"""

import contextlib
import os

import torch

# Scale presets. `classes` and `stoch` stay fixed across sizes on purpose: the categorical
# latent geometry (32 x 32) is the part of DreamerV3 that transfers, so what scales is the
# capacity around it, not the latent itself.
MODEL_SIZES = {
    "xs": {"cnn_depth": 16, "deter": 384, "stoch": 32, "classes": 32, "hidden": 256},
    "s": {"cnn_depth": 24, "deter": 512, "stoch": 32, "classes": 32, "hidden": 384},
    "m": {"cnn_depth": 32, "deter": 768, "stoch": 32, "classes": 32, "hidden": 512},
}

AC_HIDDEN = {"xs": 256, "s": 384, "m": 512}


def resolve_device(requested=None):
    """`requested` may be None, 'auto', 'cuda', 'cuda:1', 'cpu'. Never assumes CUDA."""
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"[device] '{requested}' asked for but no CUDA runtime is visible; using CPU")
        return torch.device("cpu")
    return dev


def setup_backends(device, tf32=True, benchmark=True):
    """TF32 + cudnn autotune. Both are throughput levers that cost nothing here.

    TF32 keeps fp32's exponent range and drops mantissa bits on the matmul units only;
    the loss terms we actually care about numerically are forced to real fp32 elsewhere,
    so this is free speed rather than a precision trade.
    """
    if device.type != "cuda":
        return {"tf32": False, "cudnn_benchmark": False}
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = benchmark
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return {
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": "high" if tf32 else "highest",
    }


def pick_amp_dtype(device):
    """bf16 where the hardware has it, fp16 otherwise, None on CPU.

    The distinction matters downstream: bf16 has fp32's exponent range so it never needs
    loss scaling, while fp16 does. Getting that backwards is the classic way to make an
    RSSM produce NaNs a few hundred steps in.
    """
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast(device, dtype, enabled=True):
    if not enabled or dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype, enabled=True)


@contextlib.contextmanager
def fp32():
    """Force real fp32 inside an autocast region.

    Used around every KL, log-softmax, entropy and straight-through step in the RSSM.
    Those are the terms where bf16's 8-bit mantissa quietly costs accuracy: the damage
    shows up as a KL that plateaus and a latent that stops carrying scene information,
    not as a crash, so it is worth being explicit rather than trusting autocast's
    op-level policy.
    """
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
        yield


def make_grad_scaler(device, dtype, enabled=True):
    """GradScaler is only ever constructed for fp16. bf16 must not use one."""
    need = enabled and device.type == "cuda" and dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=need)


def vram_bytes(device):
    """(free, total) in bytes. Reads the driver, so it accounts for other processes."""
    if device.type != "cuda":
        return (0, 0)
    return torch.cuda.mem_get_info(device.index or 0)


def auto_batch_size(device, seq_len=50, model_size="s", floor=4, ceiling=128):
    """Pick a sequence-batch size from whatever VRAM is actually free right now.

    Calibrated against a measured point rather than a formula: 'S' at batch 16 / seq 50
    peaks at ~2.9 GiB of torch allocation on this repo's 64x64x5 observations (see
    results/train_summary.json). Everything below scales linearly off that and keeps
    ~35% headroom for fragmentation and the imagination pass.
    """
    if device.type != "cuda":
        return floor
    free, _ = vram_bytes(device)
    free_gib = free / 2**30
    # GiB of peak torch allocation per (batch=1, seq=50) sequence, measured per size.
    per_seq = {"xs": 0.11, "s": 0.18, "m": 0.30}[model_size] * (seq_len / 50.0)
    usable = max(free_gib * 0.65 - 0.6, 0.3)  # 0.6 GiB for weights, optimiser state, ctx
    bs = int(usable / per_seq)
    bs = max(floor, min(ceiling, bs - bs % 4 if bs >= 8 else bs))
    return bs


def apply_model_size(cfg, size):
    for k, v in MODEL_SIZES[size].items():
        setattr(cfg.model, k, v)
    cfg.ac.hidden = AC_HIDDEN[size]
    return cfg


def maybe_compile(module, enabled, name=""):
    """torch.compile is opt-in. It helps steady-state throughput but the RSSM's Python
    loop over 50 timesteps makes the first compile expensive, and it hides shape bugs
    behind graph breaks, so short runs are usually faster without it."""
    if not enabled:
        return module
    print(f"[compile] compiling {name or type(module).__name__} (first step will be slow)")
    return torch.compile(module)


def runtime_report(device, amp_dtype, use_amp, backends):
    r = {
        "device": str(device),
        "device_type": device.type,
        "torch": torch.__version__,
        "amp": bool(use_amp),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if use_amp else "float32",
        "grad_scaler": bool(use_amp and amp_dtype == torch.float16),
    }
    r.update(backends)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = vram_bytes(device)
        r.update(
            {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_mib": round(total / 2**20),
                "free_vram_mib": round(free / 2**20),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "cuda": torch.version.cuda,
                "driver_cuda": os.environ.get("CUDA_VERSION", ""),
            }
        )
    return r
