# Results

Run 3 September 2026. Every number here comes from `checkpoints/run2/rollout_fidelity.json`
and `results/*.log`, produced by the scripts in this repo. Nothing is estimated.

## What was run

| | |
|---|---|
| Data | 400 MetaDrive episodes, 66,144 transitions |
| Split | 393 training / 6 held out, 58,677 valid windows |
| Training | 15,000 gradient steps, batch 32 x seq-len 50 |
| Model | RSSM, categorical latents, 11.21M + 2.41M parameters |
| Hardware | RTX 4080 Laptop, 12 GB, bf16 autocast |
| Throughput | ~3.8 optimiser steps/s, 3,672 s wall clock |
| Peak VRAM | 4.4 GB allocated, 5.7 GB reserved |

The earlier run (`run1`, 2,725 steps on 200 episodes) is superseded. The training set was
doubled because the `seq_len 50` filter was dropping 63 of the original 200 episodes as too
short.

## The fidelity result

This is the measurement the project exists for: can the model imagine 15 steps ahead
accurately enough to be worth planning against? It is scored on **held-out episodes the
model never trained on**, 200 windows, 5 context frames, 15-step imagination horizon.

The baseline is **persistence** -- freeze the last observed frame and predict no change.
That is the bar any dynamics model has to clear to have earned its existence.

| metric | imagined | persistence | posterior |
|---|---|---|---|
| MSE (mean over horizon) | **0.000863** | 0.003417 | 0.000484 |
| MSE at step 15 | **0.001224** | 0.004786 | -- |
| PSNR | **30.64 dB** | 24.66 dB | 33.15 dB |
| traffic-channel IoU | **0.381** | 0.323 | -- |
| centroid error | **1.22 px** | 1.57 px | -- |
| reward absolute error | 0.0438 | -- | -- |

The imagined rollout is **4.0x more accurate than persistence** on average and **3.9x at
the 15-step horizon**, a 5.98 dB PSNR gain.

`posterior` is the cheating upper bound: it keeps seeing real observations instead of
imagining. Imagination lands 1.8x worse than that, which is the honest cost of closing
the loop and running blind.

### The part that matters most: the gap widens with horizon

| step | 1 | 5 | 10 | 15 | growth |
|---|---|---|---|---|---|
| imagined MSE | 0.00049 | 0.00071 | 0.00098 | 0.00122 | **2.5x** |
| persistence MSE | 0.00075 | 0.00287 | 0.00419 | 0.00479 | **6.4x** |

Persistence degrades 6.4x over fifteen steps. The world model degrades 2.5x. Error still
compounds -- it always does in a latent rollout -- but it compounds far more slowly than
doing nothing, and the advantage grows rather than decays with horizon. That is the
property that makes a 15-step imagined rollout usable for pre-screening manoeuvres.

## Two fixes this measurement required

**The occupancy metric was broken.** `occupancy_iou` and `centroid_error_px` thresholded at
`0.0` on observations centred to `[-0.5, 0.5]`. Channel 0 is the static road network and
its maximum raw value is 126/255, which centres to -0.006 -- just below the threshold. So
the road never registered, while an undertrained decoder did spray stray pixels above it,
inflating the union and collapsing IoU. Earlier plumbing checks reported IoU near 0.007
while MSE said the model beat persistence; those two statements disagreed, which is what
exposed the bug.

Both metrics now score `TRAFFIC_CHANNELS = slice(1, None)`. That is also the more
principled choice: the road is identical in every frame of an episode, so including it
rewards a model for copying static background rather than for predicting where the traffic
goes.

**Results were written to the wrong directory.** `eval_rollout.py` wrote
`rollout_fidelity.json` to `cfg.train.out_dir` regardless of which checkpoint `--checkpoint`
pointed at, so scoring run 2 silently overwrote run 1's results. It now writes beside the
checkpoint it actually scored.

## Verifying the results

**The 53 MB weights are not committed** -- they would bloat every clone of this repo. What
is committed is everything else needed to check the work: the training config, the
per-step `metrics.jsonl`, `rollout_fidelity.json` with the full result, and
`rollout_comparison.png` showing imagined against real frames side by side (628 KB in
total, under `checkpoints/run2/`).

This is a weaker guarantee than the sibling `cooperative-negotiation-marl` repo, where the
networks are 170 KB and the weights ship with the code so any number can be re-scored
directly. Here you have the full evidence trail but must retrain to reproduce from scratch.
Training is deterministic and takes about an hour on a 12 GB GPU:

```bash
python -m twm.collect --episodes 400
python -m twm.train --steps 15000 --out-dir checkpoints/run2
```

With a checkpoint in hand, re-derive the table above:

```bash
python -m twm.eval_rollout \
  --checkpoint checkpoints/run2/latest.pt \
  --windows 200 --horizon 15
```

This writes `checkpoints/run2/rollout_fidelity.json` and prints the same summary. The run
is deterministic -- executing it twice gives byte-identical output, including
`reward_abs_err_mean = 0.043772317469120026`. There is no seed flag to get wrong.

`rollout_fidelity.json` records the checkpoint path, the training step it came from
(15,000), the window count, the context and horizon lengths, and confirms
`evaluated_on_held_out_episodes: true`.

## Honest limits

**Six held-out episodes is a small test set.** 200 windows are drawn from them, so the
windows are not independent samples. The 4x margin over persistence is large enough that
this is unlikely to flip, but the precise figures should not be quoted to three digits as
though they came from a large benchmark.

**Persistence is a weak baseline.** It is the right floor -- a dynamics model that cannot
beat "assume nothing changes" is worthless -- but clearing it is not evidence of being good
in absolute terms. No comparison against a published world model has been run.

**Pixel metrics are not planning metrics.** Low MSE on 64x64 frames does not by itself
establish that imagined rollouts support better decisions. The `prescreen.py` imagine-then-act
loop, which is where that would be demonstrated, has still not been run as a measurement.

**MetaDrive at 64x64, scripted-policy data.** No real driving data, no learned exploration
policy, and the observations are deliberately low resolution to fit a consumer GPU.

## Summary

The world model imagines 15 steps ahead about four times more accurately than assuming
nothing moves, on episodes it has never seen, and its advantage widens with the horizon
rather than decaying. That is the result this repository was built to test, and it is
positive. What it does not yet show is that this accuracy converts into better decisions;
that requires the pre-screening experiment, which remains unrun.
