# Handoff

For whoever picks this up next, including future me. Read `README.md` first for what the
project is. This file is about what state it is in, what to do next, what is broken or
suspicious, and why things were built the way they were.

---

## Hardware, stated once so it is not wrong anywhere

**All measurements in this repository were taken on an NVIDIA GeForce RTX 4080 Laptop GPU,
12 GB.** Reported by torch as 11,851 MiB total, compute capability 8.9, 58 SMs, driver
595.84, torch 2.6.0+cu124.

If you find a note anywhere — in an older doc, a commit message, a slide, a CV bullet —
saying **"RTX 5060 Ti 16GB"**, that is **wrong**. It was never the machine any of these
numbers came from. Correct it wherever you see it. `logs/train_run1.log` is the source of
truth and it records the 4080 Laptop on its `[runtime]` line.

---

## Where things stand

Code: complete and working. Science: not done.

- One training run exists: 2,725 gradient steps of a configured 3,000, stopped early.
  ~17 minutes. Log committed at `logs/train_run1.log`.
- It is **not converged**. The loss is still trending down and is noisy step to step.
- The imagination fidelity evaluation has **not been run**. There is no prediction-quality
  number anywhere in this project.
- `prescreen.py` (imagine-then-act) has **not been run** against the trained checkpoint.
- No batch-size sweep has been run.

Nothing in this repository supports any claim about how well the model predicts. It
supports claims about what the model costs to train, and that is all.

## What is implemented

Verified by reading the code and by running `check_gpu.py`, `smoke.py`, `--help` on every
entry point, an import of every module, and `plot_curves.py`.

| Piece | File | State |
|---|---|---|
| RSSM: categorical latents, LN-GRU, straight-through, unimix, KL with free bits | `src/twm/models/rssm.py` | Works |
| CNN encoder/decoder, symlog two-hot head, MLP factory | `src/twm/models/nets.py` | Works |
| World model: encoder + RSSM + image/reward/continue heads and loss | `src/twm/models/world_model.py` | Works |
| Actor-critic trained purely in imagination | `src/twm/models/actor_critic.py` | Works |
| Device/precision policy, auto batch sizing, size presets | `src/twm/utils/device.py` | Works |
| Seeding, param counts, VRAM reporting, JSONL logger | `src/twm/utils/run.py` | Works |
| MetaDrive top-down wrapper | `src/twm/envs/traffic_env.py` | Works |
| Scripted noisy lane follower | `src/twm/envs/scripted_policy.py` | Works |
| Episode store and on-device sequence sampler | `src/twm/data/buffer.py` | Works |
| Training loop with instrumentation | `src/twm/train.py` | Works. Run once, stopped early. |
| Data collection | `src/twm/collect.py` | Works. 200 episodes collected. |
| Open-loop fidelity evaluation | `src/twm/eval_rollout.py` | Executes. Never run as a measurement. |
| Imagine-then-act pre-screening | `src/twm/prescreen.py` | Imports and parses. Never run. |
| Shape/precision smoke test | `src/twm/smoke.py` | Passes in ~2 s |
| GPU doctor | `scripts/check_gpu.py` | Works |
| Batch sweep harness | `scripts/sweep_batch.py` | Never run |
| Loss-curve plot | `scripts/plot_curves.py` | Works (produced a figure from run 1's metrics) |

2,630 lines of Python across 23 files.

## What exists on disk but not in git

`.gitignore` excludes these on purpose. A fresh clone will not have them.

- `data/metadrive64/` — 200 episodes, 24 MB. Regenerate: `pixi run collect-data`.
- `checkpoints/run1/` — `latest.pt` (52 MB, **saved at step 2500**, not 2725, because
  `ckpt_every` is 500), `metrics.jsonl` (110 logged points), `config.json`.
- `results/` — exists but is **empty**. Nothing has been generated into it.

`checkpoints/run1/train_summary.json` does **not** exist. `train.py` writes it only after
the loop finishes, and run 1 was stopped at 2,725 of 3,000 steps. If you want that file,
finish a run.

---

## Exactly what to run next

In this order. Each step should be finished before the next is worth doing.

### 1. Confirm the machine is sane

```bash
pixi run check-gpu
pixi run smoke
```

Two seconds each. `smoke` must print `smoke test OK`. If the GPU doctor reports anything
other than `bfloat16` for `AMP dtype this repo picks` on a modern card, stop and read the
README's troubleshooting section before training.

### 2. Train properly — this is the real next step

Run 1 stopped at 2,725 steps and had not converged. Do a long run, into a new directory so
run 1 is preserved:

```bash
pixi run python -m twm.train --steps 30000 --out-dir checkpoints/run2
```

At the measured 2.62 steps/s whole-run average, 30,000 steps is roughly three hours on the
same laptop card. Watch two numbers in the log:

- `recon_mse` should keep falling. It was ~8e-4 at step 2,725.
- `kl_dyn` should settle somewhere and stay there. It sat around 2.4-2.9 through the
  second half of run 1, which looks stable, not runaway.

If the loss plateaus long before 30,000 steps, that is useful information and worth
recording rather than immediately tuning around.

### 3. Run the fidelity evaluation — the single most important missing number

```bash
pixi run python -m twm.eval_rollout --windows 256 \
  --checkpoint checkpoints/run2/latest.pt --out-dir checkpoints/run2
```

Read `imagined_mse_step15` against `persistence_mse_step15` in the printed summary. That
comparison is the whole project:

- imagined clearly **below** persistence at step 15 → the model predicts. Continue.
- imagined at or **above** persistence → the model has learned to copy the last frame, not
  to predict. Nothing further is worth doing until that is fixed. Look at more training,
  more data, or a longer sequence length before anything else.

Also read the gap between imagined and posterior. Posterior is the ceiling the
encoder/decoder can represent at all, so that gap is the part the dynamics model owns.

Then commit the result — the JSON and the frame strip — and put the numbers in the README.

### 4. Run the batch sweep and put the table in the README

```bash
pixi run python scripts/sweep_batch.py --sizes 8,16,32,48,64,96 --steps 120 --warmup-seconds 120
```

This is the compute-scoping contribution and it is currently missing. It writes
`results/batch_sweep.json` and prints a ready-to-paste markdown table. Watch the SM clock
column; if it falls on the larger rows, that is thermal throttling and the row is not a
fair comparison.

### 5. Only then, imagine-then-act

```bash
pixi run python -m twm.prescreen --episodes 10 --out-dir checkpoints/run2 \
  --checkpoint checkpoints/run2/latest.pt
```

Pre-screening candidate actions inside a model that cannot predict just adds latency. This
step is meaningless before step 3 comes back positive.

---

## Known issues and things to be suspicious of

### 1. The occupancy IoU and centroid metrics look broken, or at least badly calibrated

Measured directly from `data/metadrive64/`:

- channel 0 (road network): mean 24/255, **maximum 126**
- channels 1-4 (ego trail, three traffic-flow snapshots): mean under 0.4/255, fewer than
  0.2% of pixels above 128

`eval_rollout.py` thresholds at `thresh=0.0` on observations that have been centred to
`[-0.5, 0.5]`. A pixel value of 126 maps to -0.006, which is *just below* that threshold.
So the road channel contributes essentially nothing to occupancy IoU and centroid error —
those two metrics see only the sparse vehicle markings, and they are very sensitive to any
blur in the decoder output.

In the 2-window plumbing check against the run-1 checkpoint, imagined occupancy IoU came
back at roughly 0.007 against a persistence baseline of 0.25, while imagined MSE was
*better* than persistence. Those two statements disagree, and the threshold is the likely
reason: the undertrained decoder spreads about 0.55% of pixels above the threshold where
the real frames have about 0.01%.

Before trusting those two metrics, either pick a threshold per channel, or compute them on
the traffic channels only, or drop them and rely on MSE plus the frame strip. Do not
report them as they stand.

### 2. Nearly a third of the collected data is thrown away

`SequenceSampler` drops any episode shorter than `seq_len + 1`. At the default
`seq_len 50`, **63 of the 200 collected episodes** were dropped, leaving 132 training and 5
holdout episodes and 29,600 of 32,643 transitions. The scripted policy's episode lengths
are bimodal — most reach the 250-step cap, a tail terminate early on `out_of_road`.

Options: collect more episodes, train with a shorter `--seq-len`, or change the sampler to
pad short episodes. None is obviously right; the current behaviour is at least honest,
since it never samples a window that straddles an episode boundary.

### 3. `HOLDOUT_EPISODES = 6` gives 5 holdout episodes, not 6

Defined in both `train.py` and `eval_rollout.py`. It reserves the first 6 episode *files*,
but short files are dropped before the split is applied, so at `seq_len 50` one of the
first six vanished and the training log correctly reports "132 train / 5 holdout". Not a
bug — the train/eval split is still clean and there is no leakage, because both files use
the same constant and the same file ordering. Just do not be surprised by the number.

Related: `eval_rollout.py` builds its sampler with `seq_len = context + horizon = 20`, so a
different, larger set of episodes qualifies there (193 train / 6 holdout). The holdout
episodes are still drawn from the first 6 files, so they are still episodes the model never
trained on.

### 4. Reconstruction MSE flatters the model

The observations are mostly empty. A model that predicts "black" everywhere already scores
a low pixel MSE. `recon_mse ≈ 8e-4` should not be read as "the model reconstructs the scene
well". This is precisely why the persistence baseline and the occupancy metrics exist — and
see issue 1 for why the occupancy metrics currently cannot be trusted either.

### 5. `train_summary.json` only appears when a run finishes

It is written after the loop, not incrementally. Any interrupted run leaves you with
`metrics.jsonl` and no summary. `plot_curves.py` degrades gracefully when the summary is
missing (the figure's subtitle just loses some fields), so this is an annoyance rather than
a failure.

### 6. Throughput varies a lot on a laptop

Steps/sec across run 1 ranged from 0.78 (first step, warm-up included) to 3.75, with the
whole-run average at 2.62 and the final windows at ~3.6. This is power and thermal capping,
not a code problem. Any timing comparison you intend to quote must go through
`sweep_batch.py --warmup-seconds` and must be read alongside the SM clock column.

### 7. No test suite

`smoke.py` is the only automated check. It covers shapes, finiteness and the fp32 guards,
which is the highest-value thing to cover, but it is not tests. If you refactor the RSSM,
`smoke.py` passing is necessary and not sufficient.

### 8. `results/` is empty

`plot_curves.py` and `sweep_batch.py` both write there and neither has been run for keeps.
`.gitignore` is already set up to allow committed figures under `results/`.

---

## Design decisions, and why

### Categorical latents instead of Gaussian

32 categorical variables of 32 classes each, following DreamerV3 (Hafner et al., 2023) and
DreamerV2 (Hafner et al., 2021), rather than the diagonal Gaussian of Dreamer v1.

Three reasons. Traffic futures are genuinely multi-modal — the car ahead brakes or it does
not — and a Gaussian's single peak is forced to predict the average of two things, which is
a third thing that never happens. The KL between two categoricals over a fixed class count
is bounded by log(classes), whereas the KL between two Gaussians can grow without limit,
and that unboundedness is where Gaussian latent models tend to destabilise. And a one-hot
sample is sparse, which suits a scene that is mostly empty road.

The 32x32 geometry is held **fixed** across the `xs` / `s` / `m` presets. What scales is
the capacity around the latent, not the latent itself, because the latent geometry is the
part of DreamerV3 that is claimed to transfer.

### Precision policy: bf16 everywhere except where it visibly hurts

bf16 for the bulk of the compute, because it has fp32's exponent range and therefore never
needs loss scaling — a GradScaler is constructed only on the fp16 path, and getting that
backwards is the classic way to make an RSSM produce NaNs a few hundred steps in.

But four things are forced into real fp32 by the `fp32()` context manager in
`utils/device.py`:

- the **straight-through estimator** (`draw + probs - probs.detach()`): it subtracts two
  nearly identical tensors, so almost all significant bits cancel and only the residual
  carries gradient. bf16's 8-bit mantissa turns that residual into rounding noise.
- the **categorical KL**: a sum of 32 terms of `p*log(p/q)`. In bf16 the free-bits clamp
  starts firing on rounding noise instead of on real information.
- the **255-way two-hot log-softmax**: the logsumexp over 255 bins rounds away exactly the
  differences the reward head is trying to learn.
- the **image reconstruction sum**: ~20,000 elements per frame; bf16 runs out of mantissa
  long before the sum finishes.

The failure mode from getting this wrong is not a crash. It is a KL that plateaus and a
latent that quietly stops carrying scene information. That is why it is explicit rather
than left to autocast's op-level policy, and why `smoke.py` asserts every loss comes back
fp32.

### Everything about device and precision lives in one file

`utils/device.py` is the only place that touches `torch.backends` or `torch.autocast`.
Nothing else in the repo makes its own decision about precision, so there is exactly one
answer to "what is this run using" and nothing can silently disagree with the training
loop.

### Batch size is measured, never hardcoded

`auto_batch_size` reads free VRAM through the driver at startup (so it accounts for other
processes) and scales off a measured allocation-per-sequence figure, keeping ~35% headroom
for fragmentation, the CUDA context and the imagination pass. `--batch-size 0` is the
default and means "work it out". No VRAM number is hardcoded anywhere, which is what makes
the repo portable to a different card.

### Replay lives on the GPU when it fits

The dataset is a few hundred MB by design. `SequenceSampler` concatenates everything into
flat uint8 tensors, parks them in VRAM if they take less than 25% of what is free, and
gathers each batch on-device — no per-item numpy slicing, no host-to-device copy, no
synchronisation point inside the step loop. On a laptop the CPU side of batch assembly is
the first thing that starves the GPU. When the dataset is too large the tensors fall back
to pinned host memory with `non_blocking=True` copies, which at least overlaps the transfer
with the previous step's compute.

Observations stay uint8 all the way to the GPU and are only converted to float there. That
is what keeps a 32x50 batch inside a few hundred MB.

### The scripted collector is a PD lane keeper with noise, not MetaDrive's IDM expert

A world model trained on a single deterministic policy never sees the action channel vary,
so the transition model learns to ignore actions entirely — which would make the entire
imagine-then-act idea impossible. The OU (correlated) noise and the per-episode lateral
bias exist purely to make the dataset action-conditioned.

The gains and the noise split were tuned against the thing that actually limits this
dataset, which is `out_of_road` terminations rather than crashes. Steering noise drives the
car off the road; throttle noise does not. So the two channels get different noise scales:
pushing noise into the throttle channel while a stiffer controller fights the remaining
steering noise produced a median episode of 250 steps instead of 35, with *more* action
variance rather than less.

### The decoder is training signal only

Everything downstream — imagination, pre-screening, the actor-critic — reads only
`feat = [deter, flatten(stoch)]`. The pixel decoder exists to give the latent something to
be about during training, and to let a human check predictions by eye. It is never called
at act time. That is what keeps pre-screening 24 candidates over 15 steps cheap.

### `torch.compile` is off by default

The RSSM's Python loop over 50 timesteps makes the first compile expensive, and compilation
hides shape bugs behind graph breaks. On runs of a few thousand steps the compile cost is
not repaid. `--compile` turns it on if you are doing a long run and want to measure whether
it helps.

### The actor-critic backward writes gradients into the world model, on purpose

Imagination runs through the RSSM, so backpropagating the actor and critic losses also
deposits gradients on world-model parameters. That is harmless here: only `ac_params` are
stepped, and the world model's gradients are zeroed at the top of the next iteration before
they are ever used. Worth knowing if you refactor the loop, because reordering those two
blocks would silently change training.

### Integration with `cooperative-negotiation-marl` is an interface, not a link

There is a sibling repo about multi-agent negotiation between vehicles. **Nothing is
imported across repositories and this repo has no dependency on that one.**

The seam is the candidate set in `prescreen.py`. `manoeuvre_primitives()` returns 8
hand-written constant manoeuvres and `Prescreener.candidates()` adds 16 sampled actor
rollouts. Everything downstream needs only a `[horizon, n_candidates, action_dim]` tensor
of actions in `[-1, 1]` plus a list of names, and returns a `[n_candidates]` tensor of
imagined values. Any upstream proposer emitting that shape can replace
`manoeuvre_primitives` without touching the world model.

Described that way in the README, and that is the extent of it. Do not claim the two
projects are integrated.

---

## Things not to claim

Worth being blunt about, because these are the easy mistakes to make when writing this up
or talking about it.

- Do **not** compare any number here to DreamerV3's Atari, DMC or Minecraft tables.
  Different domain, roughly a tenth of the parameters, a tiny fraction of the environment
  steps. The comparison would be meaningless.
- Do **not** describe the model as working, accurate or converged. One 2,725-step run
  exists, it was stopped early, and nothing has been evaluated.
- Do **not** quote `recon_mse` as evidence of prediction quality. See issue 4.
- Do **not** call this a safety result. Pre-screening inside an imperfect model moves the
  failure mode; it does not remove it.
- Do **not** say the hardware was an RTX 5060 Ti. It was an RTX 4080 Laptop 12GB.

What *can* be said, and is supported by `logs/train_run1.log`: an 11.21M-parameter
DreamerV3-style RSSM trains stably in bf16 at about 3.6 steps/s and roughly 6.3 GB of VRAM
on a 12 GB consumer laptop GPU, on 64x64x5 traffic observations, at 1,600 transitions per
gradient step. That is a compute-scoping result. It is a small one, and it is real.
