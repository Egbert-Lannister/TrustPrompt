"""Module A of the mainline (docs/plan_omnidc.md sec. 14): a gate that judges each reading before OMNI-DC sees it.

The gate is T inside the network. It has T's inputs and architecture (tp/trust/model.py: 9 scale-free channels,
the small U-Net without encoder features, 3 outputs clean / measured too far / measured too near), with OMNI-DC's
own relative depth (Depth Anything V2, already computed by OMNI-DC v1.1) in place of T's UniDepth prior. The relative
map is turned into a metric prior D_g by a robust affine fit to the readings in disparity (iteratively reweighted
least squares with Tukey weights; no gradient), as T's aligned prior was. Output: the three-class logits; the gate
g = softmax[clean] is what tp.omnidc.gates.soft_gated applies to OMNI-DC's three entry points.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tp.trust.model import IN_CH, TrustNetV0

CLASSES = ("clean", "far", "near")


@torch.no_grad()
def dav2_relative(net, rgb_normalised):
    """OMNI-DC v1.1's relative depth input, computed exactly as OGNIDC.forward does at test time (src/model/
    ognidc.py: resize to 518, Depth Anything V2, ReLU, resize back, 2-98 % quantile normalisation). In training
    OGNIDC reads this map from sample['mono_dep'], which the released data loaders do not provide."""
    from align_utils import resize_image          # OMNI-DC src/model; in sys.modules once OGNIDC has been imported
    B, _, H, W = rgb_normalised.shape
    raw = F.relu(net.depth_module.forward(resize_image(rgb_normalised, size=net.depth_module_input_size)).unsqueeze(1))
    raw = F.interpolate(raw, (H, W), mode="bilinear", align_corners=True)
    lo = torch.quantile(raw.reshape(B, -1), 0.02, dim=1).reshape(B, 1, 1, 1)
    hi = torch.quantile(raw.reshape(B, -1), 0.98, dim=1).reshape(B, 1, 1, 1)
    return (raw - lo) / (hi - lo)


@torch.no_grad()
def metric_prior(rel, prompt, iters=5, c=4.685):
    """Fit 1/P ~ a * rel + b on valid readings (Tukey IRLS, scale from the median absolute residual); returns D_g
    [B,1,H,W] in metres (NaN where the fitted disparity is not positive)."""
    B = rel.shape[0]
    out = torch.full_like(rel, float("nan"))
    for i in range(B):
        m = prompt[i] > 0
        x, y = rel[i][m].double(), 1.0 / prompt[i][m].double()
        if x.numel() < 10:
            continue
        w = torch.ones_like(x)
        for _ in range(iters):
            A = torch.stack([x * w.sqrt(), w.sqrt()], 1)
            sol = torch.linalg.lstsq(A, (y * w.sqrt()).unsqueeze(1)).solution.squeeze(1)
            r = y - (sol[0] * x + sol[1])
            s = 1.4826 * r.abs().median().clamp(min=1e-12)
            u = r / (c * s)
            w = ((1 - u ** 2).clamp(min=0)) ** 2
        disp = (sol[0] * rel[i] + sol[1]).to(rel.dtype)
        out[i] = torch.where(disp > 0, 1.0 / disp.clamp(min=1e-6), torch.full_like(disp, float("nan")))
    return out


def gate_inputs(prompt, prior, rgb01):
    """T's 9 input channels (tp.trust.model.make_inputs) in torch: valid prompt, valid prior, log P - m, log D_g - m,
    clip(log P - log D_g, +-1), its absolute value, RGB - 0.5; m = median log D_g per image."""
    pv = prompt > 0
    dv = torch.isfinite(prior) & (prior > 0)
    lp = torch.where(pv, torch.log(prompt.clamp(min=1e-6)), torch.zeros_like(prompt))
    ldg = torch.where(dv, torch.log(torch.nan_to_num(prior, nan=1.0).clamp(min=1e-6)), torch.zeros_like(prompt))
    m = torch.stack([ldg[i][dv[i]].median() if dv[i].any() else ldg.new_zeros(()) for i in range(len(ldg))]).view(-1, 1, 1, 1)
    r = torch.where(pv & dv, (lp - ldg).clamp(-1, 1), torch.zeros_like(lp))
    x = torch.cat([pv.float(), dv.float(), torch.where(pv, lp - m, 0 * lp), torch.where(dv, ldg - m, 0 * ldg), r, r.abs()], 1)
    return torch.cat([x, rgb01 - 0.5], 1)


class Gate(nn.Module):
    def __init__(self, width=32, open_logit=4.0):
        """open_logit: initial bias of the clean class, so the untrained gate passes readings (g ~ 0.96) and OMNI-DC
        starts from its pretrained behaviour instead of a uniformly third-weighted prompt."""
        super().__init__()
        self.net = TrustNetV0(in_ch=IN_CH, use_features=False, width=width, out_ch=len(CLASSES))
        with torch.no_grad():
            self.net.head.bias.zero_(); self.net.head.bias[0] = open_logit

    def forward(self, prompt, prior, rgb01):
        """Logits [B,3,H,W] on the input grid (H, W multiples of 8)."""
        return self.net(gate_inputs(prompt, prior, rgb01))


def gate_targets(labels, valid_prompt):
    """T labels (tp.sim.corrupt.type_labels: 0 clean, 1 far, 2 near, 3 hole, 255 no GT) -> class index, -100 where
    the gate is not supervised (no reading or no GT)."""
    t = labels.long().clone()
    t[(t > 2) | ~valid_prompt] = -100
    return t
