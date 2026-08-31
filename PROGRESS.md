# Progress

Real state of the project. `[x]` means done and verified. `[ ]` means not done.

Last updated after the first training run (`logs/train_run1.log`, 2,725 steps, stopped
before the configured 3,000).

---

## Environment and tooling

- [x] pixi environment, locked (`pixi.toml` + `pixi.lock`, linux-64, python 3.11)
- [x] Plain venv + pip path (`requirements.txt`, pinned to what was actually installed)
- [x] Dockerfile, pinned to the same torch 2.6.0 / cu124 as the pixi environment
- [x] `.gitignore` excludes `data/`, `checkpoints/`, `.pixi/`, `*.pt`, `*.npz`
- [x] Headless rendering handled in all three places (pixi `[activation.env]`, Dockerfile
      `ENV`, and defensively in `traffic_env.py`)
- [x] `scripts/check_gpu.py` — runs, reports device / precision / suggested batch sizes,
      benchmarks matmul. Verified on the RTX 4080 Laptop.
- [ ] Tested on any GPU other than the RTX 4080 Laptop 12GB
- [ ] Tested on any OS other than Linux

## Model code

- [x] `RSSM` with categorical latents (32 categories x 32 classes), LayerNorm GRU cell,
      straight-through sampling, unimix, KL with free bits — `models/rssm.py`
- [x] CNN encoder and decoder for 64x64 observations — `models/nets.py`
- [x] Symlog two-hot discrete regression head over 255 bins — `models/nets.py`
- [x] World model: encoder + RSSM + image / reward / continue heads and the combined
      loss — `models/world_model.py`
- [x] Imagination actor-critic: actor, critic, slow target critic, percentile return
      normaliser, lambda-returns — `models/actor_critic.py`
- [x] Precision policy centralised in `utils/device.py`; every KL, log-softmax, entropy
      and straight-through step forced to fp32 inside bf16 autocast
- [x] Model-size presets `xs` / `s` / `m` with the latent geometry held fixed
- [x] Smoke test asserting every loss returns fp32 and finite — `smoke.py`, passes
- [ ] Any unit tests. There is no test suite; `smoke.py` is the only automated check.

## Data

- [x] MetaDrive top-down env wrapper — `envs/traffic_env.py`
- [x] Scripted `NoisyLaneFollower` collector with OU noise and per-episode lateral bias —
      `envs/scripted_policy.py`
- [x] Episode store (one `.npz` per episode) and on-device sequence sampler —
      `data/buffer.py`
- [x] Dataset collected: 200 episodes, 32,643 transitions, mean episode 163.2 steps,
      mean return 44.3, crash rate 4%, 165 s to collect, 24 MB on disk
- [x] Holdout split — the first 6 episode files are reserved and never trained on
- [ ] Dataset committed. It is gitignored; regenerate with `pixi run collect-data`.
- [ ] Anything better than one scripted policy as the data source

## Training

- [x] Training loop with bf16 autocast, gradient clipping, and correct GradScaler
      handling (constructed only on the fp16 path) — `train.py`
- [x] Throughput and VRAM instrumentation, logged to `metrics.jsonl` every 25 steps
- [x] Auto batch sizing from free VRAM (`--batch-size 0`, the default)
- [x] Replay kept on device when it fits, pinned host memory when it does not
- [x] **One training run completed.** 2,725 gradient steps of a configured 3,000,
      1,039 s wall clock, batch 32 x seq 50, model size `s`, 11.21M world model
      parameters. Log committed at `logs/train_run1.log`.
- [ ] **Converged.** No. Loss still trending down and noisy: the last three logged steps
      were 11.83, 8.89, 10.50.
- [ ] A run that reaches the configured step count. Run 1 stopped at 2,725, so
      `train_summary.json` was never written for it.
- [ ] More than one run
- [ ] Any hyperparameter tuning

## Evaluation

- [x] `eval_rollout.py` written: imagined vs persistence vs posterior over 15 steps, with
      occupancy IoU, centroid error and reward error
- [x] Confirmed to execute end to end against the run-1 checkpoint (2-window plumbing
      check only — not a measurement, and no numbers from it are reported anywhere)
- [ ] **Fidelity evaluation run properly.** No result exists. This is the single most
      important missing piece.
- [ ] Side-by-side frame strip produced from a real evaluation
- [ ] `prescreen.py` run end to end against a trained checkpoint
- [ ] Any comparison of pre-screening against the actor acting directly

## Compute scoping

- [x] Peak VRAM measured: 3,789 MiB peak torch allocation, 5,098 MiB peak reserved,
      about 6,300 MiB reported by `nvidia-smi`
- [x] Throughput measured: 3.62 steps/s in the last logged window, 2.62 steps/s averaged
      over the whole run, 5,795 transitions/s at the end
- [x] Parameter counts measured: 11.21M world model, 2.41M actor-critic
- [x] `scripts/sweep_batch.py` written, with GPU-clock sampling and OOM rows instead of
      crashes
- [ ] **Batch-size sweep run.** No sweep table exists.
- [ ] Sequence-length sweep
- [ ] Model-size (`xs` / `s` / `m`) comparison
- [ ] Any measurement on a GPU other than the 12 GB laptop card

## Documentation

- [x] `END_GOAL.md` — what the project is trying to be and explicitly is not
- [x] `README.md` — world models explained, RSSM architecture, measured numbers, install,
      every command, sweep guidance, troubleshooting, references
- [x] `PROGRESS.md` — this file
- [x] `HANDOFF.md` — what to run next, known issues, design decisions and why
- [x] Integration with `cooperative-negotiation-marl` documented as an interface only
- [ ] Loss-curve figure committed. `scripts/plot_curves.py` works, but `results/` is
      empty; nothing has been generated into it.
- [ ] Any results section with prediction-quality numbers, because there are none

---

## The honest one-line summary

The code is complete and runs. The science has not been done. One short, unconverged
training run exists; nothing has been evaluated; this is a feasibility study at the point
where feasibility of *training* has been shown and feasibility of *predicting* has not.
