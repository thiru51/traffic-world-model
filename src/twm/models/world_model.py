import torch
import torch.nn as nn
import torch.nn.functional as F

from twm.models.nets import CNNDecoder, CNNEncoder, TwoHotSymlog, make_bins, mlp
from twm.models.rssm import RSSM
from twm.utils.device import fp32


class WorldModel(nn.Module):
    """Encoder + RSSM + (image, reward, continue) heads.

    Everything downstream - imagination, action pre-screening, the actor-critic - reads
    only `feat = [deter, flatten(stoch)]`, so the pixel decoder exists purely as a
    training signal and for the fidelity check. It is never needed at act time.
    """

    def __init__(self, obs_shape, action_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = CNNEncoder(obs_shape[0], cfg.cnn_depth, obs_shape[-1])
        self.rssm = RSSM(
            embed_dim=self.encoder.out_dim,
            action_dim=action_dim,
            deter=cfg.deter,
            stoch=cfg.stoch,
            classes=cfg.classes,
            hidden=cfg.hidden,
            unimix=cfg.unimix,
        )
        feat = self.rssm.feat_dim
        self.decoder = CNNDecoder(feat, obs_shape[0], cfg.cnn_depth)
        self.reward_head = mlp(feat, cfg.hidden, cfg.n_bins, layers=2, out_zero_init=True)
        self.cont_head = mlp(feat, cfg.hidden, 1, layers=2)
        self.register_buffer("bins", make_bins(cfg.n_bins))

    def to_channels_last(self):
        """Call after .to(device). NHWC-strided conv weights let cuDNN pick its tensor-core
        kernels instead of transposing every activation on the way in."""
        self.encoder.to(memory_format=torch.channels_last)
        self.decoder.to(memory_format=torch.channels_last)
        return self

    def loss(self, batch):
        obs, action, reward, cont, is_first = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["cont"],
            batch["is_first"],
        )
        embed = self.encoder(obs)
        post, prior = self.rssm.observe(embed, action, is_first)
        feat = self.rssm.to_feat(post)

        recon = self.decoder(feat)
        # Unit-variance Gaussian log-likelihood up to a constant, summed over pixels -
        # the image term has to outweigh the scalar heads or the latent stops encoding
        # the scene at all. Accumulated in fp32: this is a sum over ~20k elements per
        # frame and bf16 runs out of mantissa long before the sum finishes.
        with fp32():
            err = recon.float() - obs.float()
            image_loss = 0.5 * (err**2).sum(dim=(-3, -2, -1)).mean()
            recon_mse = (err**2).mean().detach()

        reward_dist = TwoHotSymlog(self.reward_head(feat), self.bins)
        reward_loss = -reward_dist.log_prob(reward).mean()

        cont_logit = self.cont_head(feat).squeeze(-1)
        with fp32():
            cont_loss = F.binary_cross_entropy_with_logits(cont_logit.float(), cont.float()).mean()

        kl, dyn, rep = self.rssm.kl_loss(
            post, prior, self.cfg.kl_free, self.cfg.dyn_scale, self.cfg.rep_scale
        )
        total = (
            self.cfg.image_scale * image_loss
            + self.cfg.reward_scale * reward_loss
            + self.cfg.cont_scale * cont_loss
            + kl
        )
        metrics = {
            "wm/loss": total.detach(),
            "wm/image": image_loss.detach(),
            "wm/reward": reward_loss.detach(),
            "wm/cont": cont_loss.detach(),
            "wm/kl_dyn": dyn.detach(),
            "wm/kl_rep": rep.detach(),
            "wm/recon_mse": recon_mse,
        }
        return total, post, metrics

    @torch.no_grad()
    def decode(self, feat):
        return self.decoder(feat)

    @torch.no_grad()
    def predict_reward(self, feat):
        return TwoHotSymlog(self.reward_head(feat), self.bins).mean()

    def reward_mean(self, feat):
        return TwoHotSymlog(self.reward_head(feat), self.bins).mean()

    def cont_prob(self, feat):
        return torch.sigmoid(self.cont_head(feat).squeeze(-1))
