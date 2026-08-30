"""Find where this GPU actually saturates, by measuring instead of guessing.

Runs a short training run at each batch size in turn, in a fresh subprocess so the CUDA
context and the caching allocator start clean every time, and records steps/sec and peak
VRAM. A size that runs out of memory is recorded as an OOM row rather than killing the
sweep, so the table always ends with the first size that does not fit.

    python scripts/sweep_batch.py --sizes 8,16,32,48,64 --steps 120
    python scripts/sweep_batch.py --seq-lens 25,50,100 --sizes 32
    python scripts/sweep_batch.py --model-sizes xs,s,m --sizes 32

A laptop GPU under a power cap runs its first minute of work at close to double the clock
it can sustain, so a sweep run back to back would make whichever size went first look
fastest. Two things guard against that: `--warmup-seconds` loads the card before the
first measured point, and the SM clock is sampled throughout every point and reported
alongside it, so an unfair row is visible rather than silent.

Writes results/batch_sweep.json and prints the markdown table that goes in the README.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _smi(fields):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return [x.strip() for x in out.strip().splitlines()[0].split(",")]
    except Exception:
        return None


class ClockSampler(threading.Thread):
    """Polls SM clock and temperature while a run is in flight."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period = period
        self.clocks = []
        self.temps = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = _smi("clocks.sm,temperature.gpu")
            if v:
                try:
                    self.clocks.append(int(v[0]))
                    self.temps.append(int(v[1]))
                except ValueError:
                    pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        return {
            "sm_clock_mhz_mean": round(statistics.mean(self.clocks)) if self.clocks else None,
            "sm_clock_mhz_min": min(self.clocks) if self.clocks else None,
            "gpu_temp_c_max": max(self.temps) if self.temps else None,
        }


def run_one(batch_size, seq_len, model_size, steps, extra=()):
    """One training run in its own process. Returns the run's train_summary.json, or an
    'oom' row if the allocator gave up."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "twm.train",
            "--batch-size",
            str(batch_size),
            "--seq-len",
            str(seq_len),
            "--model-size",
            model_size,
            "--steps",
            str(steps),
            "--out-dir",
            tmp,
            # The actor-critic is part of the real step cost, so it stays on; what is
            # switched off is checkpointing, which would otherwise write a 50 MB file
            # per sweep point for no reason.
            "--train.ckpt_every=1000000",
            *extra,
        ]
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        sampler = ClockSampler()
        sampler.start()
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        clocks = sampler.stop()

        out = proc.stdout + proc.stderr
        base = {"batch_size": batch_size, "seq_len": seq_len, "model_size": model_size}
        if "out of memory" in out.lower() or "OutOfMemoryError" in out:
            return {**base, "oom": True, **clocks}
        summary_path = Path(tmp) / "train_summary.json"
        if proc.returncode != 0 or not summary_path.exists():
            last = out.strip().splitlines()[-1] if out.strip() else "no output"
            return {**base, "error": last, **clocks}
        row = json.loads(summary_path.read_text())
        row.update(clocks)
        # The runtime line is the only place the pre-run free-VRAM reading surfaces.
        m = re.search(r"free_vram_mib=(\d+)", out)
        if m:
            row["free_vram_mib_at_start"] = int(m.group(1))
        return row


def fmt_table(rows):
    head = (
        "| batch | seq | model | transitions/step | steps/s | transitions/s "
        "| peak torch alloc (MiB) | peak nvidia-smi (MiB) | SM clock (MHz) |"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        stem = f"| {r['batch_size']} | {r['seq_len']} | {r['model_size']} | "
        if r.get("oom"):
            lines.append(stem + f"{r['batch_size'] * r['seq_len']} | OOM | OOM | OOM | OOM | - |")
            continue
        if r.get("error"):
            lines.append(stem + "- | error | - | - | - | - |")
            continue
        lines.append(
            stem
            + f"{r['transitions_per_step']} | {r['steps_per_sec']:.2f} | "
            f"{r['transitions_per_sec']:.0f} | {r['torch_peak_alloc_mib']:.0f} | "
            f"{r['peak_nvidia_smi_used_mib']} | {r.get('sm_clock_mhz_mean') or '-'} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="32", help="comma-separated batch sizes")
    p.add_argument("--seq-lens", default="50", help="comma-separated sequence lengths")
    p.add_argument("--model-sizes", default="s", help="comma-separated xs/s/m")
    p.add_argument("--steps", type=int, default=120, help="gradient steps per point")
    p.add_argument(
        "--warmup-seconds",
        type=float,
        default=90.0,
        help="load the GPU before the first measured point so every point is timed at "
        "the same sustained clock (0 to skip)",
    )
    p.add_argument("--out", default="results/batch_sweep.json")
    args, extra = p.parse_known_args()

    batches = [int(x) for x in args.sizes.split(",")]
    seqs = [int(x) for x in args.seq_lens.split(",")]
    models = args.model_sizes.split(",")

    if args.warmup_seconds > 0:
        print(f"warming the GPU for {args.warmup_seconds:.0f}s so clocks settle...", flush=True)
        _warmup(args.warmup_seconds)
        print(f"  SM clock after warmup: {_smi('clocks.sm,temperature.gpu')}", flush=True)

    rows = []
    for model_size in models:
        for seq_len in seqs:
            for bs in batches:
                print(f"\n>>> batch {bs}  seq {seq_len}  model {model_size}", flush=True)
                row = run_one(bs, seq_len, model_size, args.steps, extra)
                rows.append(row)
                if row.get("oom"):
                    print("    OOM", flush=True)
                elif row.get("error"):
                    print(f"    error: {row['error']}", flush=True)
                else:
                    print(
                        f"    {row['steps_per_sec']:.2f} steps/s  "
                        f"{row['transitions_per_sec']:.0f} transitions/s  "
                        f"peak {row['torch_peak_alloc_mib']:.0f} MiB torch / "
                        f"{row['peak_nvidia_smi_used_mib']} MiB nvidia-smi  "
                        f"@ {row.get('sm_clock_mhz_mean')} MHz",
                        flush=True,
                    )

    out = Path(REPO / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "steps_per_point": args.steps,
                "warmup_seconds": args.warmup_seconds,
                "gpu": _smi("name,memory.total"),
                "rows": rows,
            },
            indent=2,
        )
    )
    print("\n" + fmt_table(rows))
    print(f"\nwrote {out}")


def _warmup(seconds):
    import torch

    if not torch.cuda.is_available():
        return
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = (a @ a).div_(4096**0.5)
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
