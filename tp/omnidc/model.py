"""OMNI-DC v1.1 with the gate (module A, docs/plan_omnidc.md sec. 14) in front of its three entry points.

    model = GatedOMNIDC(omnidc_baseline.net, Gate())       # OGNIDC from baselines.completion.OmniDC
    out = model(sample)            # OGNIDC's output dict + gate_logits [B,3,H,W], gate [B,1,H,W], prior [B,1,H,W]
    model = GatedOMNIDC(net, None) # no gate (configurations F0 / F1): OMNI-DC as released, plus the relative depth

`sample` is OGNIDC's: rgb ImageNet-normalised and dep (metres, 0 = missing), both padded to a multiple of
4 * 2^(R - 1), K, pattern. The relative depth is computed once here and handed to OGNIDC as sample['mono_dep'] (what
it reads in training). The backbone's first depth convolution gets one extra, zero-initialised channel for g.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gate import dav2_relative, metric_prior
from .gates import extend_depth_input, soft_gated

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GatedOMNIDC(nn.Module):
    def __init__(self, net, gate, paths=("input", "ddi", "spn")):
        super().__init__()
        self.net, self.gate, self.paths = net, gate, tuple(paths) if gate is not None else ()
        if "input" in self.paths:
            extend_depth_input(net.backbone)
        self.rate = net.args.backbone_output_downsample_rate

    def forward(self, sample):
        rgb = sample["rgb"]
        mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        rel = dav2_relative(self.net, rgb)
        if self.gate is None:
            return self.net(dict(sample, mono_dep=rel))
        prior = metric_prior(rel, sample["dep"])
        logits = self.gate(sample["dep"], prior, (rgb * std + mean).clamp(0, 1))
        g = torch.softmax(logits, 1)[:, :1]
        with soft_gated(self.net, g, self.rate, self.paths):
            out = self.net(dict(sample, mono_dep=rel))
        out.update(gate_logits=logits, gate=g, prior=prior)
        return out
