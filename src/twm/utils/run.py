import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def nvidia_smi_used_mib():
    """Process-external VRAM reading, so the reported peak includes CUDA context and
    fragmentation that torch's own allocator counters do not see."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def cuda_mem_report():
    if not torch.cuda.is_available():
        return {}
    return {
        "torch_alloc_mib": torch.cuda.memory_allocated() / 2**20,
        "torch_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "nvidia_smi_used_mib": nvidia_smi_used_mib(),
    }


def device_info():
    if not torch.cuda.is_available():
        return {"device": "cpu", "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": props.name,
        "total_vram_mib": props.total_memory / 2**20,
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a")
        self.t0 = time.time()

    def log(self, step, metrics, echo=True):
        row = {"step": step, "wall_s": round(time.time() - self.t0, 2)}
        row.update({k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()})
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()
        if echo:
            short = " ".join(
                f"{k.split('/')[-1]}={v:.4g}" for k, v in row.items() if isinstance(v, float)
            )
            print(f"step {step:>6} | {short}", flush=True)
        return row

    def close(self):
        self.fh.close()


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
