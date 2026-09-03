# traffic-world-model

A compact latent world model of a traffic scene, written from scratch, scoped to fit on
one consumer GPU. The model learns to predict what a top-down traffic view will look like
a second or two ahead. Candidate manoeuvres are then rolled forward 15 steps *inside the
model* and scored, so the car can throw away the bad ones before it commits to a real
action.

The architecture follows DreamerV3 (Hafner et al., 2023): a Recurrent State-Space Model
with categorical latents, a GRU deterministic state, and a CNN encoder/decoder over 64x64
observations from MetaDrive. The whole world model is 11.21M parameters.

See `END_GOAL.md` for the longer statement of what this is trying to be.

---

## Status

Read this before reading anything else.

**This is a small-scale feasibility study, not a result.** It is a test of whether this
class of model can be scoped down to one consumer GPU at all. It is not a reproduction of
DreamerV3, and it is not a driving policy anyone should take seriously as driving.

| Piece | State |
|---|---|
| Data collection | Done. 400 scripted episodes, 66,144 transitions, on disk. |
| Model code | Done. Imports clean, smoke test passes. |
| Training | Done. 15,000 gradient steps on 393 training episodes, 3,672 s. |
| Convergence | Partially. Loss fell 366.9 -> 7.6 and reconstruction MSE 0.035 -> 0.0005, still trending down at the end. |
| Fidelity evaluation | **Done, and positive.** Imagined 15-step rollouts are 4.0x more accurate than persistence on held-out episodes. See [RESULTS.md](RESULTS.md). |
| Imagine-then-act (`prescreen.py`) | **Not yet run** end to end against the trained checkpoint. |
| Batch-size sweep | **Not yet run.** No sweep table exists. |

**The headline.** On 6 held-out episodes and 200 windows, imagined rollouts reach MSE
0.000863 against persistence at 0.003417 -- 4.0x better on average, 3.9x at the 15-step
horizon, +5.98 dB PSNR. More importantly the advantage *widens* with horizon: over fifteen
steps the imagined error grows 2.5x while persistence grows 6.4x. Full table, the two
metric bugs that had to be fixed first, and the honest limits are in
[RESULTS.md](RESULTS.md).

**What it still does not show.** Low pixel error is not the same as better decisions. The
`prescreen.py` imagine-then-act loop is where that would be demonstrated and it has not
been run.

The numbers in the [Compute scoping](#compute-scoping-what-it-actually-costs) section
below are throughput and memory measurements from that one run. They say what this model
*costs to train*. They say nothing about how well it predicts, because that has not been
measured yet.

Nothing here is compared to DreamerV3's published Atari, DMC or Minecraft tables. That
comparison would be meaningless: different domain, roughly a tenth of the parameters, and
a tiny fraction of the environment steps.

---

## Contents

- [What a world model is, and why bother](#what-a-world-model-is-and-why-bother)
- [The architecture](#the-architecture)
- [Compute scoping: what it actually costs](#compute-scoping-what-it-actually-costs)
- [File-by-file layout](#file-by-file-layout)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Running it, in order](#running-it-in-order)
- [Finding the saturation point on a bigger GPU](#finding-the-saturation-point-on-a-bigger-gpu)
- [The integration seam](#the-integration-seam)
- [Troubleshooting](#troubleshooting)
- [Honest limitations](#honest-limitations)
- [References](#references)

---

## What a world model is, and why bother

Most reinforcement learning agents are **model-free**: they look at the current frame,
pick an action, and find out what happens by doing it. The only way such an agent learns
"do not swerve into that lane" is to swerve into that lane, many times, and collect the
penalty each time. In a simulator with unlimited budget that is fine. Anywhere else it is
useless.

A **world model** is a second network that learns to answer a different question: *given
the scene as it is now, and given an action I am considering, what does the scene look
like next?* Once you have that, the arithmetic changes. The agent can try ten candidate
manoeuvres in its head, watch each one play out for fifteen imagined steps, discard the
ones that end in a predicted crash, and only then move. Each of those ten trials costs one
forward pass through a small recurrent network. None of them costs a real environment
step, and none of them costs a real crash.

That is the **sample efficiency** argument — sample efficiency meaning how much useful
learning you extract per real interaction with the environment. It is the argument behind
the Dreamer line of work (Hafner et al., 2020, 2021, 2023), behind Ha and Schmidhuber's
original world-model paper (2018), and behind the driving-specific models MILE (Hu et al.,
2022) and GAIA-1 (Wayve, 2023).

Two details make this practical rather than merely appealing.

**The model predicts in a latent space, not in pixels.** A "latent" is a small compressed
vector that stands in for the full image — here 1,536 numbers instead of 64x64x5 = 20,480
pixel values. Imagining forward means stepping that small vector, which is cheap. The
pixel decoder exists to give the model a training signal and to let a human check the
predictions by eye. It is never needed when the agent is actually acting.

**Imagination is open-loop.** Feed the model a few real frames so its recurrent state is
warm, then cut off the observations and let it run on its own predictions. That is the
honest test, and it is what `src/twm/eval_rollout.py` measures: 5 context frames, then 15
steps with no further observations, against the actions that were really taken.

That evaluation is compared against two other curves so the number means something:

- **persistence** — just repeat the last observed frame for all 15 steps. This is the bar.
  A model that cannot beat "assume nothing moves" has learned to copy, not to predict.
- **posterior** — reconstruct with every real frame available. This is the ceiling: it is
  what the encoder/decoder pair can represent at all. The gap between imagined and
  posterior is the part the dynamics model is responsible for.

---

## The architecture

### The RSSM

RSSM stands for **Recurrent State-Space Model**. It is the core of every Dreamer variant.
The idea is that the model's belief about the world is split into two halves that do
different jobs.

```
                obs[t] --> CNN encoder --> embed[t]
                                             |
   stoch[t-1], action[t-1] --> img_in --> LN-GRU --> deter[t]
                                             |          |
                        prior logits <-------+          |
                    (predict without seeing obs[t])     |
                                                        v
                        posterior logits <-- [deter[t], embed[t]]
                                (correct using obs[t])
                                                        |
                                  sample --> stoch[t] --+
                                                        |
                              feat[t] = [deter[t], flatten(stoch[t])]
                                        |        |         |
                                  CNN decoder  reward    continue
```

The two halves:

- **`deter`** — the *deterministic* state, 512 numbers, carried by a GRU (Gated Recurrent
  Unit, a recurrent cell that decides at each step how much of its memory to keep versus
  overwrite). This is the part that remembers. Because it is fully determined by the
  history, gradients flow back through it cleanly over long unrolls.
- **`stoch`** — the *stochastic* state, sampled fresh at each step. This is the part that
  represents genuine uncertainty about what happens next: a car that might or might not
  pull out.

Together they form `feat = [deter, flatten(stoch)]`, a 1,536-dimensional vector
(512 + 32x32). Everything downstream — the decoder, the reward head, the continue head,
the actor, the critic, the pre-screener — reads only `feat`.

Two paths produce `stoch` at each step, and the difference between them is the whole
training signal:

- the **prior** (`img_step`) predicts the next latent from `deter` alone, without looking
  at the new observation. This is what imagination uses.
- the **posterior** (`obs_step`) corrects that prediction using the encoded observation.
  This is the "what actually happened" version.

Training pushes the prior towards the posterior with a KL divergence term — KL divergence
being a measure of how far one probability distribution is from another. Make the prior
good at matching the posterior, and you have a model that can predict without looking.

`src/twm/models/rssm.py` holds all of this.

### Why categorical latents rather than Gaussian

This is the single most important design choice inherited from DreamerV3, and it is worth
understanding properly.

The obvious way to build a stochastic latent is a **Gaussian**: the network outputs a mean
and a standard deviation, and you sample from a bell curve. Dreamer v1 (Hafner et al.,
2020) did exactly that. DreamerV2 (Hafner et al., 2021) switched to categorical latents
and DreamerV3 kept them.

Here the latent is **32 independent categorical variables, each with 32 classes** — think
of it as 32 dials, each of which snaps to one of 32 settings, rather than 32 continuous
knobs. This repo uses that same 32x32 geometry at every model size.

Three reasons it works better here:

1. **Multi-modality.** A Gaussian has one peak. If the car ahead might brake *or* might
   accelerate, a Gaussian's best answer is the average of the two — a prediction of
   something that will never happen. A categorical distribution can put probability mass
   on both outcomes and on neither of the points in between. Traffic is full of these
   either/or futures, so this matters more here than in a smooth control task.

2. **Better-behaved gradients.** The KL divergence between two Gaussians can grow without
   limit as their variances diverge, and in practice this is where Gaussian latent models
   destabilise. The KL between two categoricals over a fixed number of classes is bounded
   by log(number of classes). It cannot explode. Training gets a lot less fragile as a
   direct result.

3. **Sparsity fits the data.** A categorical sample is a one-hot vector — mostly zeros.
   That is a reasonable match for a scene that is mostly empty road with a few discrete
   objects in it.

The cost is that sampling a discrete value is not differentiable, so gradients cannot flow
through it normally. The standard fix is the **straight-through estimator**: sample the
hard one-hot value on the forward pass, but pretend on the backward pass that you passed
the soft probabilities through. In code that is the line

```python
return draw + probs - probs.detach()
```

in `RSSM._sample`. Numerically the forward value is exactly `draw` (the two `probs` terms
cancel); the gradient comes from the `probs` term that was not detached.

That line is also the most precision-sensitive operation in the whole model. It subtracts
two nearly identical tensors, so most of the significant bits cancel and only the small
residual carries information. In bfloat16, which has only 8 bits of mantissa, that
residual is mostly rounding noise. Every such operation in this repo is therefore forced
into real fp32 by the `fp32()` context manager in `src/twm/utils/device.py`, even while
the bulk of the compute runs in bf16.

One more DreamerV3 detail worth naming: **unimix**. Each categorical's probabilities are
mixed with a 1% uniform floor before use. Without it a class can collapse to exactly zero
probability, the KL to the prior blows up, and training falls over. It is one line and it
is cheap insurance.

### The rest of the world model

- **CNN encoder** (`nets.py`): four stride-2 convolutions, 64x64 down to 4x4, channel
  widths 24 / 48 / 96 / 192 at model size `s`. LayerNorm over the channel axis and SiLU
  activations throughout, matching DreamerV3.
- **CNN decoder**: the mirror image, four transposed convolutions back up to 64x64.
- **Reward head and critic**: these do not regress a scalar directly. They classify over
  255 fixed bins spaced in **symlog** space (symlog being a signed logarithm, so it
  compresses large magnitudes without breaking near zero), and the prediction is the
  expected value under that distribution. DreamerV3 introduced this **two-hot** encoding
  precisely because it removes the scale sensitivity of a plain MSE regression, which is
  why the same hyperparameters transfer across domains with wildly different reward
  magnitudes.
- **Continue head**: a single sigmoid predicting "is the episode still alive". This is what
  lets a predicted crash suppress everything a candidate manoeuvre would have earned after
  it.

### The imagination actor-critic

`src/twm/models/actor_critic.py` trains a policy entirely inside the model. Starting from
every posterior state in the training batch, it rolls the prior forward 15 steps under the
current actor, scores those imagined trajectories with the reward head and the critic, and
updates the actor from the result. No environment steps are involved.

DreamerV3's percentile return normalisation is included (`ReturnNormalizer`): advantages
are divided by the 5th-to-95th percentile spread of imagined returns, never scaled up.
That is what lets one entropy coefficient work regardless of reward magnitude.

### Imagine-then-act

`src/twm/prescreen.py` is the payoff. At every real step it:

1. proposes a candidate set — 8 hand-written manoeuvre primitives (`hold`, `brake`,
   `nudge_left`, `turn_right`, ...) plus 16 sampled actor rollouts,
2. rolls all 24 candidates 15 steps forward through the RSSM prior, in one batch, with no
   simulator calls and no pixels decoded,
3. scores each by discounted imagined reward, multiplied at every step by the model's own
   predicted survival probability, plus the critic's value at the end,
4. executes the first action of the winner.

The observation stays 5 channels of 64x64 from MetaDrive's top-down multi-channel view:
channel 0 is the road network, channel 1 is the ego vehicle's recent trail, channels 2-4
are three stacked snapshots of traffic flow.

---

## Compute scoping: what it actually costs

This is the part of the project that is a real, if small, contribution: measured numbers
for what this class of model costs on hardware people own.

> **All figures below: measured on an RTX 4080 Laptop GPU, 12GB, early run, not
> converged.** They come from `logs/train_run1.log`, which is committed. Every one can be
> checked against that file.

### The run

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9 |
| VRAM visible to torch | 11,851 MiB total, 10,738 MiB free at start |
| torch / CUDA | 2.6.0+cu124 |
| Precision | bfloat16 autocast, no GradScaler; KL, straight-through and two-hot log-softmax forced to fp32 |
| Model size | `s` |
| World model parameters | 11.21M (all trainable) |
| Actor-critic parameters | 2.41M (1.58M trainable) |
| Feature dimension | 1,536 (512 deterministic + 32x32 categorical) |
| Batch | 32 sequences x 50 timesteps = 1,600 transitions per gradient step, auto-selected from free VRAM |
| Training data | 132 train / 5 holdout episodes, 29,600 transitions, 23,132 valid 50-step windows, 579 MiB held in VRAM |

### Throughput

| | |
|---|---|
| Gradient steps completed | 2,725 (of 3,000 configured — the run was stopped early) |
| Wall clock | 1,039 s, about 17 minutes |
| Steps/sec, last logged window | 3.62 |
| Steps/sec, whole-run average | 2.62 (2,725 steps / 1,039 s) |
| Transitions/sec, last logged window | 5,795 |
| Steps/sec range across the run | 0.78 (first step, includes warm-up) to 3.75 |

The spread is real and worth knowing about: this is a laptop GPU under a power and thermal
cap, and its sustained clock is well below its boost clock. Any timing comparison between
two configurations on hardware like this has to warm the card first, which is exactly what
`scripts/sweep_batch.py --warmup-seconds` does.

### Memory

| | |
|---|---|
| Peak torch allocated | 3,789 MiB — flat across the entire run |
| Peak torch reserved | 5,098 MiB |
| `nvidia-smi` used, last logged step | 6,296 MiB |
| `nvidia-smi` used, range across the run | 5,205 to 7,791 MiB |

The gap between 3,789 MiB allocated and roughly 6,300 MiB reported by `nvidia-smi` is the
CUDA context, the caching allocator's reserved-but-unused blocks, and cuDNN workspaces.
That gap is why the auto batch sizer keeps about 35% headroom rather than filling free
VRAM. Peak allocation being flat from step 1 onwards is the expected shape: the recurrent
unroll allocates its full activation graph on the first step and reuses it thereafter.

Roughly: **about 6.3 GB of a 12 GB card**, leaving room for a desktop session alongside.

### Losses — what an unconverged run looks like

| Step | Total loss | Image term | Reconstruction MSE | Dynamics KL (nats) | Reward NLL |
|---|---|---|---|---|---|
| 1 | 3,681.8 | 3,673.6 | 3.59e-01 | 3.393 | 5.541 |
| 100 | 108.5 | 106.0 | 1.04e-02 | 1.000 | 1.834 |
| 500 | 24.9 | 22.0 | 2.15e-03 | 2.857 | 1.179 |
| 1,000 | 15.2 | 12.7 | 1.24e-03 | 2.739 | 0.811 |
| 1,500 | 11.4 | 8.9 | 8.74e-04 | 2.842 | 0.710 |
| 2,000 | 11.5 | 9.2 | 9.01e-04 | 2.772 | 0.649 |
| 2,500 | 9.3 | 7.1 | 6.91e-04 | 2.614 | 0.615 |
| 2,725 | 10.5 | 8.3 | 8.09e-04 | 2.689 | 0.609 |

The last three logged steps went 11.83, then 8.89, then 10.50. The trend is downward and
the step-to-step noise is larger than the trend over any 100-step window. **This is an
early run that was stopped, not a converged model.** Nothing about model quality should be
read into it.

One honest caveat about the reconstruction MSE. The observations are extremely dark: the
road channel averages 24 out of 255, and the four other channels are sparse markings
covering well under 1% of pixels. A pixel-space MSE on data like that is dominated by
correctly predicting empty space, so `recon_mse ≈ 8e-4` looks better than it is. This is
exactly why `eval_rollout.py` also reports occupancy IoU and centroid error, and why the
persistence baseline is reported next to the model. Until that evaluation is run properly,
no claim about prediction quality is supported.

### What was not measured

No batch-size sweep, no sequence-length sweep, no model-size comparison, no fidelity
number, no imagine-then-act number. Those runs have not happened.

---

## File-by-file layout

```
src/twm/
  config.py               Dataclass config, YAML loading, and the shared CLI flags
                          (--device, --batch-size, --model-size, ...) that every entry
                          point accepts, plus dotted --section.key=value overrides.

  models/
    nets.py               Building blocks: CNN encoder and decoder, the MLP factory,
                          symlog/symexp, and the TwoHotSymlog discrete-regression head.
    rssm.py               The RSSM itself: LayerNorm GRU cell, categorical prior and
                          posterior, straight-through sampling, KL with free bits,
                          observe() for filtering and imagine() for rollouts.
    world_model.py        Encoder + RSSM + image/reward/continue heads and the combined
                          training loss.
    actor_critic.py       Actor, critic, slow-updating target critic, percentile return
                          normaliser, lambda-returns, and the imagination training loop.

  envs/
    traffic_env.py        Thin wrapper over MetaDrive's TopDownMetaDrive. Emits uint8
                          observations and the (obs, action, reward, cont, is_first)
                          tuple layout. Sets SDL to a dummy driver so it works headless.
    scripted_policy.py    NoisyLaneFollower: a PD lane keeper with correlated (OU) noise
                          and a per-episode lateral bias, used to collect the dataset.

  data/
    buffer.py             EpisodeStore (one .npz per episode on disk) and SequenceSampler
                          (concatenates everything into flat uint8 tensors, parks them in
                          VRAM when they fit, and gathers each batch on-device).

  utils/
    device.py             The single place device, precision and memory policy is decided:
                          resolve_device, TF32/cudnn setup, bf16-vs-fp16 choice, the
                          fp32() guard, VRAM-based auto batch sizing, model-size presets.
    run.py                Seeding, parameter counting, VRAM reporting via both torch and
                          nvidia-smi, and the JSONL metrics logger.

  smoke.py                Shape and precision check on random tensors. No dataset needed,
                          runs in seconds, asserts every loss comes back fp32 and finite.
  collect.py              Drives the scripted policy through MetaDrive and writes episodes.
  train.py                The training loop: world model step, then imagination
                          actor-critic step, with throughput and VRAM instrumentation.
  eval_rollout.py         Open-loop fidelity: imagined vs persistence vs posterior over a
                          15-step horizon, on held-out episodes.
  prescreen.py            Imagine-then-act. Scores candidate manoeuvres in the model and
                          executes the winner.

scripts/
  check_gpu.py            Run first on any new machine. Prints what torch actually sees
                          and benchmarks a matmul so you know the GPU is being used.
  sweep_batch.py          Batch/sequence/model-size sweep in fresh subprocesses, with GPU
                          clock sampling and OOM rows instead of crashes.
  plot_curves.py          Turns a run's metrics.jsonl into a six-panel loss figure.

configs/default.yaml      Every tunable, with comments. batch_size 0 means auto-scale.
pixi.toml, pixi.lock      The reproducible environment.
requirements.txt          The plain venv + pip path.
Dockerfile                Container build, pinned to the same torch/CUDA as pixi.toml.
END_GOAL.md               What this project is trying to be, and explicitly is not.
logs/train_run1.log       The committed log of the one training run that has happened.
```

Not in git, because `.gitignore` excludes them: `data/` (the collected episodes, 24 MB),
`checkpoints/` (the trained weights, 52 MB), `.pixi/`. You regenerate those by running the
commands below.

---

## Prerequisites

- **Linux.** The pixi environment is built for `linux-64`. Everything else is portable in
  principle but has not been tested elsewhere.
- **Python 3.11** if you use the pip path. The pixi path brings its own.
- **An NVIDIA GPU with a driver new enough for CUDA 12.x.** Check with `nvidia-smi`. The
  code runs on CPU too, just far too slowly to train anything real.
- **About 3 GB of disk** for the environment, plus ~200 MB for MetaDrive's asset pack,
  which it downloads on first use.

Nothing anywhere hardcodes a VRAM number. Batch size is chosen from what is actually free
at startup, so a smaller or larger card just gets a different batch size.

---

## Install

### Path 1: pixi (recommended — this is the locked, reproducible one)

pixi is a package manager that reads `pixi.toml` and `pixi.lock` and builds the exact
environment that was tested, into a `.pixi/` folder inside the project. It does not touch
your system Python.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL          # reload your shell so `pixi` is on PATH
```

```bash
git clone <this-repo> traffic-world-model
cd traffic-world-model
pixi install
```

Then prefix commands with `pixi run`, or open a shell inside the environment once:

```bash
pixi shell
```

pixi also sets three environment variables for you, which matter:

- `PYTHONPATH=src` so `python -m twm.train` finds the package
- `SDL_VIDEODRIVER=dummy` so MetaDrive's pygame renderer does not try to open a window
- `MPLBACKEND=Agg` so matplotlib writes files instead of looking for a display

### Path 2: venv + pip

Use this if you would rather not install another tool. You have to set those three
environment variables yourself.

```bash
git clone <this-repo> traffic-world-model
cd traffic-world-model

python3.11 -m venv .venv
source .venv/bin/activate

# The --extra-index-url is not optional. Without it pip installs the CPU-only torch
# wheel and nothing will use your GPU.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

export PYTHONPATH=src
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
```

To make those permanent for the venv, append them to `.venv/bin/activate`.

Verify torch found CUDA before going further:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: `2.6.0+cu124 12.4 True`. If the middle value is `None`, you got the CPU wheel —
reinstall torch with the extra index URL above.

### Path 3: Docker

```bash
docker build -t traffic-world-model .
docker run --gpus all -it --rm -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" traffic-world-model
```

---

## Running it, in order

Commands are shown for pixi. On the venv path, drop the `pixi run` prefix and use the
plain `python ...` command instead — every `pixi run <task>` is just an alias, listed at
the bottom of `pixi.toml`.

### 1. Check the GPU

```bash
pixi run check-gpu
# venv equivalent: python scripts/check_gpu.py
```

This prints what torch actually sees rather than what you hope it sees, and finishes with
a matmul benchmark so you know the GPU is being used and not merely detected. Sample
output, measured on the RTX 4080 Laptop 12GB used for this project:

```
==============================================================
traffic-world-model :: GPU doctor
==============================================================
python                     3.11.16
platform                   Linux-7.0.0-30-generic-x86_64-with-glibc2.39
torch                      2.6.0+cu124
torch built for CUDA       12.4
cuda available             True
nvidia-smi                 NVIDIA GeForce RTX 4080 Laptop GPU, 595.84, 12282 MiB, 1373 MiB
resolved device            cuda
device name                NVIDIA GeForce RTX 4080 Laptop GPU
compute capability         8.9
multiprocessors            58
total VRAM                 11.57 GiB
free VRAM                  10.05 GiB
bf16 supported             True
TF32 matmul                True
TF32 cudnn                 True
cudnn.benchmark            True
float32 matmul precision   high
AMP dtype this repo picks  bfloat16
needs GradScaler           False

Suggested batch size (seq_len 50), from free VRAM right now:
  --model-size xs          52
  --model-size s           32
  --model-size m           16

Matmul benchmark (4096^3, 30 iters):
  fp32 (TF32 off)          13.7 TFLOP/s
  fp32 (TF32 on)           21.6 TFLOP/s
  bfloat16                 49.6 TFLOP/s
peak alloc during bench    200 MiB

GPU looks usable. Next: python -m twm.smoke
```

Two things to read here. `AMP dtype this repo picks` should say `bfloat16` on any card
from compute capability 8.0 upwards; bf16 has fp32's exponent range, so it never needs
loss scaling. On older cards it will say `float16` and `needs GradScaler` flips to True —
the code handles that automatically. And if bf16 is not roughly 2x the TF32 number, your
GPU is not actually being used for the fast path.

### 2. Smoke test

```bash
pixi run smoke
# venv equivalent: python -m twm.smoke
```

Builds the full model on random tensors, runs one forward and backward pass through the
world model and the actor-critic, and asserts that every loss comes back as fp32 and
finite. It needs no dataset and finishes in a couple of seconds. This is the first thing
to run after touching model code or moving to a new machine.

You should see `world model params : 11.21M`, `actor-critic params: 2.41M`,
`feature dim : 1536`, and `smoke test OK` at the end.

To check the CPU path works, or if you have no GPU:

```bash
pixi run python -m twm.smoke --device cpu --model-size xs --seq-len 16
```

### 3. Collect data

```bash
pixi run collect-data
# venv equivalent: python -m twm.collect
```

Drives the scripted `NoisyLaneFollower` through MetaDrive and writes one compressed `.npz`
per episode into `data/metadrive64/`. Defaults come from `configs/default.yaml`: 200
episodes, up to 250 steps each.

MetaDrive downloads its asset pack (~200 MB) the first time this runs. That is a one-off.

Measured for the committed dataset, from `data/metadrive64/meta.json`: 200 episodes,
32,643 transitions, mean episode length 163.2 steps, mean return 44.3, crash rate 4%,
collected in 165 seconds, 24 MB on disk.

To collect a smaller set for a quick test:

```bash
pixi run python -m twm.collect --episodes 20 --data-dir data/small
```

One thing worth knowing before you train: the sampler drops any episode shorter than
`seq_len + 1`. At the default `seq_len 50` that discarded 63 of the 200 episodes, leaving
132 training and 5 holdout episodes and 29,600 of the 32,643 transitions. If you want more
of your data used, either collect longer episodes or train with a shorter `--seq-len`.

### 4. Train — **this is the next thing to run**

```bash
pixi run train
# venv equivalent: python -m twm.train
```

This is where the project currently stands. One 2,725-step run exists and is not
converged. The obvious next move is a longer run.

```bash
# A longer run, everything else at defaults.
pixi run python -m twm.train --steps 20000

# Pin the batch size instead of auto-scaling, and write somewhere new so run1 is kept.
pixi run python -m twm.train --steps 20000 --batch-size 32 --out-dir checkpoints/run2

# Any config field can be overridden with a dotted flag.
pixi run python -m twm.train --steps 20000 --train.log_every=100 --train.ckpt_every=2000
```

Useful flags:

| Flag | What it does |
|---|---|
| `--steps N` | Gradient steps. Default 3,000. |
| `--batch-size N` | Sequences per step. `0` (the default) means pick from free VRAM. |
| `--seq-len N` | Timesteps per training sequence. Default 50. |
| `--model-size xs\|s\|m` | Capacity preset. The 32x32 latent geometry stays fixed; what scales is everything around it. |
| `--out-dir DIR` | Where checkpoints, `metrics.jsonl` and `train_summary.json` go. |
| `--device cpu` | Force CPU. Correct, but far too slow to train anything real. |
| `--no-amp` | Turn off mixed precision. Slower and uses more VRAM; useful when isolating a numerical problem. |
| `--compile` | `torch.compile` the world model. Off by default: the RSSM's Python loop over 50 timesteps makes the first compile expensive, so short runs are usually faster without it. |

What gets written to `--out-dir`:

- `latest.pt` — weights, config, obs shape and action dim, saved every `ckpt_every` steps
- `metrics.jsonl` — one JSON object per logged step
- `config.json` — the fully resolved config for this run
- `train_summary.json` — throughput and memory summary, **written only when the run
  finishes**. Run 1 was stopped early, so this file does not exist for it.

Expect roughly 17 minutes per 2,725 steps at these settings on a 12 GB laptop card. Watch
`wm/recon_mse` (should fall steadily) and `wm/kl_dyn` (should settle rather than run away).

To plot the curves afterwards:

```bash
pixi run plot
# venv equivalent: python scripts/plot_curves.py checkpoints/run1 results/loss_curves.png
```

### 5. Evaluate imagination fidelity — **NOT YET RUN**

```bash
pixi run eval-rollout
# venv equivalent: python -m twm.eval_rollout
```

**No fidelity result exists for this project yet.** The script has been confirmed to
execute end to end against the run-1 checkpoint, but only as a plumbing check on two
windows, which is not a measurement. Run it properly after a longer training run.

What it does: takes held-out episodes the model never trained on, feeds 5 real frames
through the posterior to warm the recurrent state, then rolls the prior forward 15 steps
using the actions that were really taken but with no further observations, and compares
against the real frames.

It reports three curves per step — **imagined**, **persistence** (repeat the last observed
frame) and **posterior** (reconstruct with every real frame available) — plus occupancy
IoU, centroid error in pixels, and reward prediction error. Results go to
`<out-dir>/rollout_fidelity.json`, with a side-by-side frame strip in
`rollout_comparison.png`.

```bash
# More windows for a tighter average, and a specific checkpoint.
pixi run python -m twm.eval_rollout --windows 256 --checkpoint checkpoints/run2/latest.pt --out-dir checkpoints/run2

# Push the horizon out to see where the model falls apart.
pixi run python -m twm.eval_rollout --windows 128 --context 5 --horizon 30
```

The number that matters is imagined MSE versus persistence MSE at step 15. If imagined is
not clearly below persistence, the model has learned to copy the last frame rather than to
predict, and nothing further in the project is worth doing.

### 6. Imagine-then-act — **NOT YET RUN**

```bash
pixi run prescreen
# venv equivalent: python -m twm.prescreen
```

Runs episodes twice: once with pre-screening on (24 candidate manoeuvres scored in
imagination, winner executed) and once with the actor acting directly, then prints both.
Writes `<out-dir>/prescreen.json`.

This has not been run against the trained checkpoint. It is only meaningful once the
fidelity check above says the model can actually predict — pre-screening inside a model
that cannot predict just adds latency.

```bash
pixi run python -m twm.prescreen --episodes 10 --actor-samples 32
```

---

## Finding the saturation point on a bigger GPU

`scripts/sweep_batch.py` answers "how large can I go on this card, and where do the extra
transitions per second stop arriving". It runs a short training run at each setting **in a
fresh subprocess**, so the CUDA context and the caching allocator start clean every time —
without that, an earlier point's fragmentation makes a later point look like it OOMs when
it would have fitted.

```bash
# The usual sweep: batch size, at the default sequence length and model size.
pixi run python scripts/sweep_batch.py --sizes 8,16,32,48,64,96 --steps 120

# Sequence length instead, at a fixed batch size.
pixi run python scripts/sweep_batch.py --seq-lens 25,50,100 --sizes 32

# Model capacity.
pixi run python scripts/sweep_batch.py --model-sizes xs,s,m --sizes 32
```

It writes `results/batch_sweep.json` and prints a markdown table with transitions/step,
steps/sec, transitions/sec, peak torch allocation, peak `nvidia-smi` usage and mean SM
clock, ready to paste into a README.

**No sweep has been run for this project.** The table does not exist yet.

Two things make the results trustworthy, and both matter on a laptop:

- `--warmup-seconds` (default 90) loads the GPU before the first measured point. A laptop
  GPU under a power cap runs its first minute at close to double the clock it can sustain,
  so without this whichever configuration went first would look fastest.
- The SM clock is sampled throughout every point and reported in its own column. An unfair
  row is then visible rather than silent.

**How to read the table.** Going up in batch size should raise transitions/sec while the
GPU still has idle capacity, then flatten. That flat point is saturation: past it you are
spending VRAM for nothing. Take the largest size that is still on the rising part of the
curve, leave one step of headroom, and use that. Watch the SM clock column while you do —
if it drops on the larger sizes, you are looking at thermal throttling rather than a real
saturation point.

### The OOM symptom, and the fix

**OOM** means out of memory: the GPU ran out of VRAM. You will see:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.
GPU 0 has a total capacity of 11.57 GiB of which 220.50 MiB is free.
```

The sweep script catches this and records the row as `OOM` rather than dying, so the table
always ends with the first size that does not fit — which is exactly the information you
wanted. In a normal training run it kills the process.

Fixes, cheapest first:

```bash
# 1. Let the auto-sizer choose (it keeps ~35% headroom for fragmentation).
pixi run python -m twm.train --batch-size 0

# 2. Pin something smaller.
pixi run python -m twm.train --batch-size 16

# 3. Shorten the sequence. The recurrent unroll holds activations for every timestep,
#    so halving seq_len saves roughly as much as halving the batch.
pixi run python -m twm.train --batch-size 32 --seq-len 25

# 4. Drop model capacity.
pixi run python -m twm.train --model-size xs

# 5. Last resort: turn off mixed precision to rule out a numerical problem.
#    This makes memory worse, not better. Diagnostic only.
pixi run python -m twm.train --no-amp
```

If it OOMs at batch 4, something else is holding VRAM. Check with `nvidia-smi` — a
browser, a previous run that did not exit, or a Jupyter kernel will each hold a GB or more.

---

## The integration seam

There is a sibling project, `cooperative-negotiation-marl`, about multi-agent negotiation
between vehicles. The two could meet, and `src/twm/prescreen.py` is where.

**They are not integrated. Nothing is imported across repositories, and this repo has no
dependency on that one.** What follows is a documented interface, not a working link.

The seam is the candidate set. `manoeuvre_primitives()` currently returns 8 hand-written
constant manoeuvres, and `Prescreener.candidates()` bolts 16 sampled actor rollouts onto
them. Everything downstream — `Prescreener.score()` and `Prescreener.choose()` — only
requires:

- **input:** a tensor of shape `[horizon, n_candidates, action_dim]`, actions in `[-1, 1]`,
  plus a list of `n_candidates` names for logging
- **output:** a `[n_candidates]` tensor of imagined discounted values, and the first action
  of the winner

Any upstream proposer that can emit action sequences in that shape can replace
`manoeuvre_primitives` without touching the world model at all. A negotiation policy would
plug in there: it proposes the manoeuvres the negotiation is about, and the world model
scores them for feasibility before the vehicle commits.

---

## Troubleshooting

### `torch.cuda.is_available()` is False

Run `pixi run check-gpu` first — it tells you which of the three usual causes it is.

- `torch built for CUDA: cpu-only build` — you installed the CPU wheel. On the pip path,
  reinstall with `--extra-index-url https://download.pytorch.org/whl/cu124`. On the pixi
  path, `pixi install` again; `pixi.toml` already points torch at that index.
- `nvidia-smi: not available` — the driver is missing or you are in a container started
  without `--gpus all`. Fix the driver first; torch cannot help.
- `nvidia-smi` works but torch still says False — the driver is older than the CUDA version
  torch was built for. Either update the driver or install a torch built for an older CUDA.

Everything still runs on CPU, just far too slowly to train:

```bash
pixi run python -m twm.smoke --device cpu --model-size xs --seq-len 16
```

### Out of memory

See [the OOM section above](#the-oom-symptom-and-the-fix).

One extra case: OOM *partway through* a run that started fine usually means fragmentation
rather than a genuinely too-large batch. Restarting with a slightly smaller batch is the
practical fix. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` sometimes helps and costs
nothing to try.

### MetaDrive fails to import, or crashes on first use

- `ModuleNotFoundError: No module named 'metadrive'` — you are outside the environment. Use
  `pixi run ...`, or `source .venv/bin/activate`.
- A numpy error mentioning `np.float` or a dtype that no longer exists — MetaDrive 0.4.x is
  not numpy-2 clean. Both `pixi.toml` and `requirements.txt` pin numpy below 2. If you
  upgraded numpy by hand, put it back: `pip install "numpy<2"`.
- An asset error on first run — MetaDrive downloads its asset pack (~200 MB) on first use
  and needs network access. Force it once with
  `pixi run python -m metadrive.pull_asset`.

### Rendering / headless display errors

MetaDrive's top-down observation renders through pygame, which by default wants a real
window and dies over SSH or in a container.

```
pygame.error: No available video device
```

The fix is already applied in three places — `pixi.toml`'s `[activation.env]`, the
Dockerfile's `ENV`, and defensively at the top of `src/twm/envs/traffic_env.py` — but if
you are running Python directly without any of them:

```bash
export SDL_VIDEODRIVER=dummy
```

A matplotlib equivalent shows up when saving figures without a display:

```bash
export MPLBACKEND=Agg
```

In Docker, missing shared libraries give
`ImportError: libGL.so.1: cannot open shared object file`. The Dockerfile installs `libgl1`,
`libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1` and `libx11-6` for exactly this. On a
bare host, install the same packages.

### `ModuleNotFoundError: No module named 'twm'`

`PYTHONPATH` is not set. pixi sets it; a plain venv does not.

```bash
export PYTHONPATH=src
```

### `FileNotFoundError: no episodes in data/metadrive64`

Collect data first: `pixi run collect-data`. `data/` is gitignored, so a fresh clone has
none.

### `every episode in data/... is shorter than seq_len=50`

Your episodes are too short for the training window. Either collect longer ones
(`--data.max_steps_per_episode=250`) or train with `--seq-len 25`.

### Throughput varies a lot between runs

Expected on a laptop. See the [Compute scoping](#compute-scoping-what-it-actually-costs)
section — steps/sec across run 1 ranged from 0.78 to 3.75. Use
`scripts/sweep_batch.py --warmup-seconds 120` for any timing comparison you intend to
quote, and read the SM clock column.

---

## Honest limitations

- **One short training run, not converged.** 2,725 steps. The loss is still falling. No
  claim about model quality is supported by anything in this repository yet.
- **Prediction quality is unmeasured.** The fidelity evaluation has not been run. Until
  imagined MSE is shown to beat the persistence baseline, "the model can imagine traffic"
  is a hypothesis, not a finding.
- **Reconstruction MSE flatters the model.** The observations are mostly empty space, so a
  low pixel MSE is easy. The occupancy and centroid metrics in `eval_rollout.py` exist to
  catch this, and they have not been read yet.
- **The data comes from one scripted policy.** A PD lane keeper with noise. It is not
  expert driving, the crash rate is 4%, and the model has never seen a competent driver.
- **Almost a third of the collected data is unused** at the default sequence length — 63 of
  200 episodes are shorter than 51 steps and get dropped.
- **No comparison to published results.** DreamerV3's benchmark tables are a different
  domain at a different scale. Any comparison would be meaningless and none is made.
- **Not a safety claim.** Pre-screening actions inside a model that is itself imperfect
  moves the failure mode. It does not remove it.
- **Not integrated with `cooperative-negotiation-marl`.** The seam is documented above and
  that is all it is.

---

## References

The design follows these papers. All numbers reported in those papers are theirs, measured
on their hardware and their domains, and none of them are compared to anything here.

- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World Models*
  (DreamerV3), 2023. arXiv:2301.04104. The direct template: categorical latents, symlog
  two-hot heads, free bits, unimix, percentile return normalisation.
- Hafner, Lillicrap, Norouzi, Ba. *Mastering Atari with Discrete World Models*
  (DreamerV2), ICLR 2021. arXiv:2010.02193. Where the switch from Gaussian to categorical
  latents was made and justified.
- Hafner, Lillicrap, Ba, Norouzi. *Dream to Control: Learning Behaviors by Latent
  Imagination* (Dreamer), ICLR 2020. arXiv:1912.01603. Learning a policy purely inside the
  model.
- Ha, Schmidhuber. *Recurrent World Models Facilitate Policy Evolution*, NeurIPS 2018.
  arXiv:1803.10122. The original "train in the model's dream" result.
- Hu, Corrado, Griffiths, Murez, Gurau, Yeo, Kendall, Cipolla, Shotton. *Model-Based
  Imitation Learning for Urban Driving* (MILE), NeurIPS 2022. A world model for driving,
  from Wayve.
- Wayve. *GAIA-1: A Generative World Model for Autonomous Driving*, 2023.
  arXiv:2309.17080. The large-scale end of this idea, for contrast with how small this is.
- Li, Peng, Feng, Zhang, Xue, Zhou. *MetaDrive: Composing Diverse Driving Scenarios for
  Generalizable Reinforcement Learning*, IEEE TPAMI 2022. arXiv:2109.12674. The simulator
  used here.
