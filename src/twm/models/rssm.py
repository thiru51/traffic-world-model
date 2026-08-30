import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, OneHotCategorical

from twm.models.nets import mlp
from twm.utils.device import fp32


class LayerNormGRUCell(nn.Module):
    """The GRU variant used in DreamerV3.

    Two things differ from nn.GRUCell and both matter: a single LayerNorm is applied
    to the concatenated gate pre-activations (keeps the recurrence well conditioned
    over 50-step unrolls), and the update gate is shifted by -1 so the cell defaults
    to carrying its state forward rather than overwriting it.
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.layer = nn.Linear(in_dim + hidden, 3 * hidden, bias=False)
        self.norm = nn.LayerNorm(3 * hidden, eps=1e-3)

    def forward(self, x, h):
        parts = self.norm(self.layer(torch.cat([x, h], -1)))
        reset, cand, update = parts.chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1 - update) * h


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim,
        action_dim,
        deter=512,
        stoch=32,
        classes=32,
        hidden=384,
        unimix=0.01,
    ):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        self.feat_dim = deter + stoch * classes

        self.img_in = nn.Sequential(
            nn.Linear(stoch * classes + action_dim, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3),
            nn.SiLU(),
        )
        self.cell = LayerNormGRUCell(hidden, deter)
        self.img_out = mlp(deter, hidden, stoch * classes, layers=1)
        self.obs_out = mlp(deter + embed_dim, hidden, stoch * classes, layers=1)

    def initial(self, batch, device):
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "logit": torch.zeros(batch, self.stoch, self.classes, device=device),
            "stoch": torch.zeros(batch, self.stoch, self.classes, device=device),
        }

    def _sample(self, logits, sample=True):
        # fp32 throughout: the straight-through estimator subtracts two nearly identical
        # tensors (draw + probs - probs.detach()), and in bf16 the residual gradient path
        # loses most of its significant bits. This is the single most precision-sensitive
        # line in the model.
        with fp32():
            logits = logits.float()
            if not sample:
                probs = torch.softmax(logits, -1)
                return F.one_hot(probs.argmax(-1), self.classes).float()
            dist = self.get_dist(logits)
            draw = dist.sample()
            probs = dist.base_dist.probs
            return draw + probs - probs.detach()

    def get_dist(self, logits):
        with fp32():
            logits = logits.float()
            probs = torch.softmax(logits, -1)
            if self.unimix > 0:
                # Mixing in a uniform floor keeps every class at nonzero probability.
                # Without it a class can collapse to exactly 0, the KL to the prior blows
                # up, and training destabilises - DreamerV3's fix, and it is cheap.
                probs = (1 - self.unimix) * probs + self.unimix / self.classes
                logits = torch.log(probs)
            return Independent(OneHotCategorical(logits=logits), 1)

    def img_step(self, prev_state, prev_action, sample=True):
        x = torch.cat([prev_state["stoch"].flatten(-2), prev_action], -1)
        deter = self.cell(self.img_in(x), prev_state["deter"])
        logit = self.img_out(deter).reshape(*deter.shape[:-1], self.stoch, self.classes)
        return {"deter": deter, "logit": logit, "stoch": self._sample(logit, sample)}

    def obs_step(self, prev_state, prev_action, embed, is_first=None, sample=True):
        if is_first is not None:
            # initial() is all zeros, so zeroing the carried state is exactly a reset.
            keep = 1.0 - is_first.float()
            prev_action = prev_action * keep[:, None]
            prev_state = {
                k: v * keep.reshape(-1, *([1] * (v.dim() - 1))) for k, v in prev_state.items()
            }
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([prior["deter"], embed], -1)
        logit = self.obs_out(x).reshape(*embed.shape[:-1], self.stoch, self.classes)
        post = {"deter": prior["deter"], "logit": logit, "stoch": self._sample(logit, sample)}
        return post, prior

    def observe(self, embeds, actions, is_first):
        """Filter a batch of sequences. Tensors are [B, T, ...]; returns stacked states."""
        b, t = embeds.shape[:2]
        state = self.initial(b, embeds.device)
        posts, priors = [], []
        for i in range(t):
            state, prior = self.obs_step(state, actions[:, i], embeds[:, i], is_first[:, i])
            posts.append(state)
            priors.append(prior)
        stack = lambda seq: {k: torch.stack([s[k] for s in seq], 1) for k in seq[0]}
        return stack(posts), stack(priors)

    def imagine(self, state, policy, horizon):
        """Roll the prior forward `horizon` steps under `policy(feat) -> action`."""
        states, actions = [], []
        for _ in range(horizon):
            action = policy(self.to_feat(state))
            state = self.img_step(state, action)
            states.append(state)
            actions.append(action)
        stack = lambda seq: {k: torch.stack([s[k] for s in seq], 0) for k in seq[0]}
        return stack(states), torch.stack(actions, 0)

    def to_feat(self, state):
        return torch.cat([state["deter"], state["stoch"].flatten(-2)], -1)

    def kl_loss(self, post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1):
        # KL between two 32x32 categoricals is a sum of 32 terms of p*log(p/q); under
        # bf16 the log ratio is where the free-bits clamp starts firing on rounding noise
        # instead of on real information, so the whole term is computed in fp32.
        with fp32():
            sg = lambda d: {k: v.detach().float() for k, v in d.items()}
            dyn = self._kl(sg(post)["logit"], prior["logit"].float())
            rep = self._kl(post["logit"].float(), sg(prior)["logit"])
            # Free bits are applied to the summed KL, so the model is not penalised for
            # the first nat of information it puts in the latent - only for exceeding it.
            dyn = dyn.clamp(min=free)
            rep = rep.clamp(min=free)
            return dyn_scale * dyn.mean() + rep_scale * rep.mean(), dyn.mean(), rep.mean()

    def _kl(self, logits_q, logits_p):
        with fp32():
            q = self.get_dist(logits_q)
            p = self.get_dist(logits_p)
            return torch.distributions.kl_divergence(q, p)
