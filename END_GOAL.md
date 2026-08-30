# What this project is trying to be

## The one-sentence version

Train a compact latent world model of a traffic scene on a single consumer GPU, and use
15-step imagined rollouts inside that model to pre-screen candidate manoeuvres before the
car commits to a real action.

## Why bother

A driving policy that only ever reacts to the current frame has to learn "do not do that"
by actually doing it. That is fine in a simulator with unlimited budget and useless
anywhere else. A world model changes the arithmetic: once the model can predict what the
scene looks like a second or two ahead given an action, the agent can try ten candidate
manoeuvres in its own head, throw away the ones that end badly, and only then move. Every
one of those ten trials costs a forward pass through a small recurrent net instead of an
environment step, and none of them cost a crash.

This is the sample-efficiency argument behind the Dreamer line of work (Hafner et al.,
2020/2021/2023) and behind the driving-specific models MILE (Hu et al., 2022) and GAIA-1
(Wayve, 2023). None of that is being reproduced here at scale. What is being tested is
narrower and more practical.

## The actual question

**Can a DreamerV3-style RSSM be scoped down far enough to train on one consumer GPU and
still imagine a traffic scene 15 steps ahead usefully better than a trivial baseline?**

"Usefully better" needs a number, so the fidelity check compares three curves over the
same 15-step window:

- **imagined** - the model rolls its own prior forward from 5 context frames, no further
  observations, using the actions that were really taken
- **persistence** - just repeat the last observed frame for all 15 steps
- **posterior** - reconstruct with every real frame available

Persistence is the bar. A model that cannot beat "assume nothing moves" has learned to
copy, not to predict. Posterior is the ceiling: it is what the encoder/decoder pair can
represent at all, so the gap between imagined and posterior is the part the dynamics
model is responsible for.

## What "done" looks like

1. `git clone`, one install command, and every script in the README runs on a machine
   with any CUDA GPU - no VRAM number hardcoded anywhere, batch size scaled from what is
   actually free.
2. A world model in the 8-18M parameter range that trains without diverging, with the
   precision-sensitive parts (categorical KL, straight-through estimator, two-hot
   log-softmax) provably in fp32 while the bulk of the compute runs in bf16.
3. A real training run whose throughput and peak VRAM are measured, logged, and committed
   - not estimated.
4. A real open-loop fidelity number on held-out episodes, with the persistence baseline
   next to it, plus side-by-side frames.
5. `imagine-then-act` running end to end: candidate manoeuvres scored inside the model,
   winner executed in the real env.

## What this project is explicitly not

- Not a reproduction of DreamerV3. Different domain, ~1/10th the parameters, ~1/1000th
  the environment steps, and no attempt at the Atari/DMC/Minecraft benchmark suite. Any
  comparison to those published tables would be meaningless.
- Not a driving policy anyone should take seriously as driving. The scripted data
  collector is a PD lane keeper with noise; the learned actor is trained purely in
  imagination on a tiny offline dataset.
- Not a claim about safety. Pre-screening actions in a model that is itself imperfect
  moves the failure mode, it does not remove it.
- Not integrated with the sibling `cooperative-negotiation-marl` repo. The candidate-set
  interface in `src/twm/prescreen.py` is documented as a seam so the two *could* meet,
  and nothing is imported across repos.

## The compute-scoping angle

The part of this that is actually a contribution, small as it is, is the measurement: what
does it cost to run this class of model on hardware people own? Parameter count, peak
VRAM, steps per second, and how those move with batch size, sequence length and model
scale. Those numbers are in the README and every one of them comes from a run that was
executed, with the log committed alongside.
