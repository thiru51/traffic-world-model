import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from twm.models.nets import TwoHotSymlog, make_bins, mlp


class Actor(nn.Module):
    """Diagonal Gaussian over continuous actions, squashed into the env's [-1, 1] box."""

    def __init__(self, feat_dim, action_dim, hidden, min_std=0.1, max_std=1.0):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        self.net = mlp(feat_dim, hidden, 2 * action_dim, layers=2)

    def dist(self, feat):
        out = self.net(feat)
        mean, std = out.chunk(2, -1)
        mean = torch.tanh(mean)
        # Bounded std: an unbounded softplus lets the policy widen without limit early on
        # and the resulting actions saturate the steering channel.
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
        return Independent(Normal(mean, std), 1)

    def act(self, feat, sample=True):
        d = self.dist(feat)
        a = d.rsample() if sample else d.mean
        return a.clamp(-1.0, 1.0)


class Critic(nn.Module):
    def __init__(self, feat_dim, hidden, n_bins):
        super().__init__()
        self.net = mlp(feat_dim, hidden, n_bins, layers=2, out_zero_init=True)
        self.register_buffer("bins", make_bins(n_bins))

    def dist(self, feat):
        return TwoHotSymlog(self.net(feat), self.bins)

    def value(self, feat):
        return self.dist(feat).mean()


class ReturnNormalizer(nn.Module):
    """DreamerV3's percentile return scaling.

    Dividing advantages by the 5th-95th percentile spread (never scaling *up*, hence the
    max with 1) is what lets one entropy coefficient work across environments whose
    return magnitudes differ by orders of magnitude.
    """

    def __init__(self, decay=0.99, low=5.0, high=95.0):
        super().__init__()
        self.decay = decay
        self.low = low
        self.high = high
        self.register_buffer("scale", torch.ones(()))

    @torch.no_grad()
    def update(self, returns):
        flat = returns.detach().flatten().float()
        lo = torch.quantile(flat, self.low / 100.0)
        hi = torch.quantile(flat, self.high / 100.0)
        self.scale.mul_(self.decay).add_((hi - lo) * (1 - self.decay))

    def __call__(self, x):
        return x / self.scale.clamp(min=1.0)


def lambda_return(reward, value, cont, gamma, lam):
    """Bootstrapped lambda-returns over an imagined trajectory. Tensors are [H, B]."""
    horizon = reward.shape[0]
    out = [None] * horizon
    nxt = value[-1]
    for t in reversed(range(horizon)):
        disc = gamma * cont[t]
        boot = value[t] if t == horizon - 1 else (1 - lam) * value[t] + lam * nxt
        nxt = reward[t] + disc * boot
        out[t] = nxt
    return torch.stack(out, 0)


class ImaginationActorCritic(nn.Module):
    def __init__(self, feat_dim, action_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.actor = Actor(feat_dim, action_dim, cfg.hidden)
        self.critic = Critic(feat_dim, cfg.hidden, cfg.n_bins)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.norm = ReturnNormalizer()
        self._updates = 0

    def update_slow(self):
        self._updates += 1
        if self._updates % self.cfg.slow_critic_every:
            return
        with torch.no_grad():
            for a, b in zip(self.slow_critic.parameters(), self.critic.parameters()):
                a.data.lerp_(b.data, self.cfg.slow_critic_tau)

    def losses(self, wm, start_state):
        cfg = self.cfg
        states, actions = wm.rssm.imagine(start_state, self.actor.act, cfg.horizon)
        feat = wm.rssm.to_feat(states)

        with torch.no_grad():
            reward = wm.reward_mean(feat)
            cont = wm.cont_prob(feat)
        value = self.critic.value(feat)
        ret = lambda_return(reward, value.detach(), cont, cfg.gamma, cfg.lam)

        # Discount by the model's own survival probability so imagined steps past a
        # predicted crash barely count.
        weight = torch.cumprod(
            torch.cat([torch.ones_like(cont[:1]), cfg.gamma * cont[:-1]], 0), 0
        ).detach()

        self.norm.update(ret)
        adv = self.norm(ret - value.detach())
        dist = self.actor.dist(feat)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        actor_loss = -(weight * (logp * adv.detach() + cfg.entropy_coef * entropy)).mean()

        critic_dist = self.critic.dist(feat)
        critic_loss = -critic_dist.log_prob(ret.detach())
        # Anchor to a slowly-updated copy of itself; without it the critic chases its own
        # bootstrap target and drifts.
        with torch.no_grad():
            slow = self.slow_critic.value(feat)
        critic_loss = critic_loss - critic_dist.log_prob(slow) * cfg.slow_reg
        critic_loss = (weight * critic_loss).mean()

        metrics = {
            "ac/actor_loss": actor_loss.detach(),
            "ac/critic_loss": critic_loss.detach(),
            "ac/imag_return": ret.mean().detach(),
            "ac/imag_reward": reward.mean().detach(),
            "ac/imag_value": value.mean().detach(),
            "ac/entropy": entropy.mean().detach(),
            "ac/ret_scale": self.norm.scale.detach(),
        }
        return actor_loss, critic_loss, metrics
