"""T v0: a small U-Net on the 192x256 prompt grid with frozen ViT-S features injected at 1/4 and 1/8.

Inputs (9 channels, per image, scale-free): prompt valid, prior valid, log P - m, log D_g - m,
clip(log P - log D_g, +-1), its absolute value, RGB - 0.5 (m = median log D_g). Output: one
logit per pixel for "prompt pixel is contaminated" (|P - GT| / GT > tau); trust = 1 - sigmoid.
T v2 (out_ch=3): logits for clean / measured too far / measured too near; trust = softmax[clean].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH = 9


def make_inputs(prompt, valid, dg, rgb):
    prompt = np.asarray(prompt, np.float32); dg = np.asarray(dg, np.float32)
    pv = np.asarray(valid, bool) & np.isfinite(prompt) & (prompt > 0)
    dv = np.isfinite(dg) & (dg > 0)
    ldg = np.where(dv, np.log(np.where(dv, dg, 1)), 0).astype(np.float32)
    lp = np.where(pv, np.log(np.where(pv, prompt, 1)), 0).astype(np.float32)
    m = float(np.median(ldg[dv])) if dv.any() else 0.0
    both = pv & dv
    r = np.where(both, np.clip(lp - ldg, -1, 1), 0).astype(np.float32)
    rgbn = np.asarray(rgb, np.float32).transpose(2, 0, 1) / 255.0 - 0.5
    x = np.stack([pv, dv, np.where(pv, lp - m, 0), np.where(dv, ldg - m, 0), r, np.abs(r)]).astype(np.float32)
    return np.concatenate([x, rgbn.astype(np.float32)], 0)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class TrustNetV0(nn.Module):
    def __init__(self, in_ch=IN_CH, feat_ch=384, width=32, use_features=True, feat_proj=64, feat_dropout=0.0, out_ch=1):
        super().__init__()
        w = width; self.use_features = use_features; fp = feat_proj if use_features else 0
        # T v1: optional Dropout2d on the projected encoder features (no parameters; v0 checkpoints still load)
        self.fproj = nn.Sequential(nn.Conv2d(feat_ch, feat_proj, 1), nn.GELU(), nn.Dropout2d(feat_dropout)) if use_features else None
        self.e1 = Block(in_ch, w); self.e2 = Block(w, 2 * w)
        self.e3 = Block(2 * w + fp, 4 * w); self.e4 = Block(4 * w + fp, 4 * w)
        self.d3 = Block(8 * w, 2 * w); self.d2 = Block(4 * w, w); self.d1 = Block(2 * w, w)
        self.head = nn.Conv2d(w, out_ch, 1)   # 1: contamination logit (v0/v1); 3: clean / far / near logits (v2)

    def forward(self, x, feat=None):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        x3 = F.max_pool2d(e2, 2); x4_size = (x3.shape[-2] // 2, x3.shape[-1] // 2)
        if self.use_features:
            f = self.fproj(feat)
            x3 = torch.cat([x3, F.interpolate(f, size=x3.shape[-2:], mode="bilinear", align_corners=False)], 1)
        e3 = self.e3(x3)
        x4 = F.max_pool2d(e3, 2)
        if self.use_features:
            x4 = torch.cat([x4, F.interpolate(f, size=x4.shape[-2:], mode="bilinear", align_corners=False)], 1)
        e4 = self.e4(x4)
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.head(d1)


def auroc(scores, positive):
    """Rank AUROC with half credit for ties; positives = clean (trusted) pixels. None if one class is empty."""
    scores = np.asarray(scores, np.float64); positive = np.asarray(positive, bool)
    npos = int(positive.sum()); nneg = positive.size - npos
    if npos == 0 or nneg == 0:
        return None
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv.ravel()]
    return float((ranks[positive].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
