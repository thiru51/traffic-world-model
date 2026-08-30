"""Run this first on any new machine.

Prints what torch actually sees - not what you hope it sees - and finishes with a small
matmul benchmark so you know the GPU is being used rather than merely detected. If this
script is unhappy, nothing else in the repo will work.

    python scripts/check_gpu.py
"""

import platform
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from twm.utils.device import (  # noqa: E402
    MODEL_SIZES,
    auto_batch_size,
    pick_amp_dtype,
    resolve_device,
    setup_backends,
    vram_bytes,
)


def line(k, v):
    print(f"{k:<26} {v}")


def bench_matmul(device, n=4096, iters=30, dtype=torch.float32):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(5):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # 2*n^3 flops per matmul.
    return 2 * n**3 * iters / dt / 1e12


def main():
    print("=" * 62)
    print("traffic-world-model :: GPU doctor")
    print("=" * 62)
    line("python", platform.python_version())
    line("platform", platform.platform())
    line("torch", torch.__version__)
    line("torch built for CUDA", torch.version.cuda or "cpu-only build")
    line("cuda available", torch.cuda.is_available())

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"], text=True, timeout=10).strip()
        line("nvidia-smi", smi)
    except Exception as e:
        line("nvidia-smi", f"not available ({type(e).__name__})")

    device = resolve_device(None)
    line("resolved device", device)

    if device.type != "cuda":
        print("\nNo CUDA device. Everything still runs on CPU, just slowly:")
        print("  python -m twm.smoke --device cpu --model-size xs --seq-len 16")
        print("\nIf you expected a GPU, check `nvidia-smi` first, then that torch was")
        print("installed with a CUDA build (torch.version.cuda must not be None).")
        return

    props = torch.cuda.get_device_properties(device)
    free, total = vram_bytes(device)
    line("device name", props.name)
    line("compute capability", f"{props.major}.{props.minor}")
    line("multiprocessors", props.multi_processor_count)
    line("total VRAM", f"{total / 2**30:.2f} GiB")
    line("free VRAM", f"{free / 2**30:.2f} GiB")
    line("bf16 supported", torch.cuda.is_bf16_supported())

    backends = setup_backends(device)
    line("TF32 matmul", torch.backends.cuda.matmul.allow_tf32)
    line("TF32 cudnn", torch.backends.cudnn.allow_tf32)
    line("cudnn.benchmark", torch.backends.cudnn.benchmark)
    line("float32 matmul precision", backends.get("matmul_precision"))

    amp = pick_amp_dtype(device)
    line("AMP dtype this repo picks", str(amp).replace("torch.", ""))
    line("needs GradScaler", amp == torch.float16)

    print("\nSuggested batch size (seq_len 50), from free VRAM right now:")
    for size in MODEL_SIZES:
        line(f"  --model-size {size}", auto_batch_size(device, 50, size))

    print("\nMatmul benchmark (4096^3, 30 iters):")
    torch.backends.cuda.matmul.allow_tf32 = False
    line("  fp32 (TF32 off)", f"{bench_matmul(device):.1f} TFLOP/s")
    torch.backends.cuda.matmul.allow_tf32 = True
    line("  fp32 (TF32 on)", f"{bench_matmul(device):.1f} TFLOP/s")
    if amp is not None:
        line(f"  {str(amp).replace('torch.', '')}", f"{bench_matmul(device, dtype=amp):.1f} TFLOP/s")

    line("peak alloc during bench", f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print("\nGPU looks usable. Next: python -m twm.smoke")


if __name__ == "__main__":
    main()
