import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    # Clamped because expm1 overflows fp32 around |x| = 88; nothing we predict is
    # anywhere near that, so hitting the clamp means something upstream diverged.
    return torch.sign(x) * torch.expm1(x.abs().clamp(max=20.0))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor.

    DreamerV3 keeps activations in NHWC and normalises the last (feature) axis; the
    equivalent under PyTorch's NCHW convolutions is a per-pixel normalisation over C.
    """

    def __init__(self, channels, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def mlp(in_dim, hidden, out_dim, layers=2, act=nn.SiLU, out_zero_init=False):
    dims = [in_dim] + [hidden] * layers
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b, bias=False), nn.LayerNorm(b, eps=1e-3), act()]
    head = nn.Linear(dims[-1], out_dim)
    if out_zero_init:
        # Value / reward heads start at "predict the middle bin" so early imagination
        # is not dominated by whatever noise the last layer happened to initialise to.
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    mods.append(head)
    return nn.Sequential(*mods)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=24, resolution=64):
        super().__init__()
        assert resolution == 64, "stride-2 stack below is sized for 64x64"
        chans = [in_channels, depth, depth * 2, depth * 4, depth * 8]
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv2d(a, b, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(b),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*blocks)
        self.out_dim = depth * 8 * 4 * 4

    def forward(self, x):
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = self.net(x)
        return x.reshape(*lead, -1)


class CNNDecoder(nn.Module):
    def __init__(self, feat_dim, out_channels=3, depth=24):
        super().__init__()
        self.depth = depth
        self.linear = nn.Linear(feat_dim, depth * 8 * 4 * 4)
        chans = [depth * 8, depth * 4, depth * 2, depth]
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.ConvTranspose2d(a, b, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(b),
                nn.SiLU(),
            ]
        blocks.append(nn.ConvTranspose2d(depth, out_channels, 4, stride=2, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, feat):
        lead = feat.shape[:-1]
        x = self.linear(feat.reshape(-1, feat.shape[-1]))
        x = x.reshape(-1, self.depth * 8, 4, 4)
        x = self.net(x)
        return x.reshape(*lead, *x.shape[-3:])


class TwoHotSymlog:
    """Discrete regression head over symlog-spaced bins (DreamerV3 sec. 3).

    Regressing a scalar with cross-entropy over a fixed set of bins removes the scale
    sensitivity of MSE, which is the reason the same hyperparameters transfer across
    domains with wildly different reward magnitudes.
    """

    def __init__(self, logits, bins):
        self.logits = logits
        self.bins = bins

    def mean(self):
        probs = torch.softmax(self.logits, -1)
        return symexp((probs * self.bins).sum(-1))

    def log_prob(self, x):
        x = symlog(x)
        below = (self.bins <= x[..., None]).sum(-1) - 1
        above = len(self.bins) - (self.bins > x[..., None]).sum(-1)
        below = below.clamp(0, len(self.bins) - 1)
        above = above.clamp(0, len(self.bins) - 1)
        equal = below == above
        d_below = torch.where(equal, torch.ones_like(x), (self.bins[below] - x).abs())
        d_above = torch.where(equal, torch.ones_like(x), (self.bins[above] - x).abs())
        total = d_below + d_above
        w_below = (d_above / total)[..., None]
        w_above = (d_below / total)[..., None]
        target = (
            F.one_hot(below, len(self.bins)) * w_below
            + F.one_hot(above, len(self.bins)) * w_above
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


def make_bins(n_bins, lo=-20.0, hi=20.0, device=None):
    return torch.linspace(lo, hi, n_bins, device=device)
