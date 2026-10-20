"""Where wrong readings enter OMNI-DC, and how to gate them there.

A prompt reaches the OMNI-DC prediction through three paths (src/model/ognidc.py @ a5ecd35):
  input  the (log, median-whitened) prompt is a network input channel (backbone features, GRU);
  ddi    the depth integration fits the prompt with per-cell weight confidence_input x weights_input (1/4 res);
  spn    the final DySPN refinement re-injects the full-resolution prompt with its own confidence each iteration.

`patched(net, backbone=..., spn=...)` replaces the forward of `net.backbone` / `net.prop_layer` of one network
instance for the duration of a `with` block (each wrapper receives the original forward as its first argument);
nothing outside the block or on other instances changes. Two uses:
  gated       hard, for the D1b diagnostic (scripts/omnidc_gates.py): DDI weight x keep per cell, prompt zeroed for
              DySPN where flagged; the network input is left alone.
  soft_gated  module A of the mainline (docs/plan_omnidc.md sec. 14): one per-pixel gate g in [0, 1] on all three
              paths - network input channel 0 (log prompt) x g with g appended as an extra channel, DDI weight x g
              (cell minimum), DySPN confidence sigmoid(c) -> g sigmoid(c).
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def patched(net, backbone=None, spn=None):
    replaced = []
    if backbone is not None:
        orig_backbone = net.backbone.forward
        net.backbone.forward = lambda *a, **k: backbone(orig_backbone, *a, **k)
        replaced.append(net.backbone)
    if spn is not None:
        if not hasattr(net, "prop_layer"):
            raise RuntimeError("this OMNI-DC has no SPN refinement")
        orig_spn = net.prop_layer.forward
        net.prop_layer.forward = lambda *a: spn(orig_spn, *a)
        replaced.append(net.prop_layer)
    try:
        yield
    finally:
        for m in replaced:
            del m.forward                                  # back to the class method


def _with_confidence(outputs, fn):
    out = list(outputs)
    if out[4] is None:
        raise RuntimeError("this OMNI-DC has no input-confidence head (pred_confidence_input = 0)")
    out[4] = fn(out[4])
    return tuple(out)


def cell_keep(flag, rate):
    """Full-res bool flag map [B,1,H,W] (padded like the network input) -> keep weight per backbone cell
    (0 where any pixel of the cell is flagged; OMNI-DC's own ConfInput labels max-pool the same way)."""
    import torch.nn.functional as F
    return 1.0 - F.max_pool2d(flag.float(), rate)


def gated(net, ddi_keep=None, spn_keep=None):
    """ddi_keep: [B,1,H/r,W/r] multiplier of the DDI input confidence; spn_keep: [B,1,H,W] 0/1 mask of the prompt
    pixels DySPN may re-inject. None leaves that path as it is."""
    backbone = None if ddi_keep is None else (lambda f, *a, **k: _with_confidence(f(*a, **k), lambda c: c * ddi_keep))
    spn = None if spn_keep is None else (lambda f, feat, guide, sp_dep, conf: f(feat, guide, sp_dep * spn_keep, conf))
    return patched(net, backbone=backbone, spn=spn)


def soft_gated(net, g, rate, paths=("input", "ddi", "spn")):
    """g: [B,1,H,W] gate in [0, 1] on the padded network grid (1 = trust the reading). The backbone must accept one
    extra depth channel when "input" is gated (extend_depth_input)."""
    import torch
    import torch.nn.functional as F
    g_cell = -F.max_pool2d(-g, rate)                       # a cell is as trusted as its least trusted pixel

    def backbone(f, rgb, depth, pattern, *rest):
        if "input" in paths:
            depth = torch.cat([depth[:, :1] * g, depth[:, 1:], g], 1)
        out = f(rgb, depth, pattern, *rest)
        return _with_confidence(out, lambda c: c * g_cell) if "ddi" in paths else out

    def spn(f, feat, guide, sp_dep, conf):
        s = (g * torch.sigmoid(conf)).clamp(1e-6, 1 - 1e-6)
        return f(feat, guide, sp_dep, torch.log(s) - torch.log1p(-s))     # sigmoid of the new logit = g sigmoid(c)
    return patched(net, backbone=backbone, spn=spn if "spn" in paths else None)


def extend_depth_input(backbone):
    """Give the backbone's first depth convolution one more input channel (the gate), with zero weights, so the
    network computes exactly what it computed before until training moves those weights."""
    import torch
    conv = backbone.conv1_dep[0]
    new = torch.nn.Conv2d(conv.in_channels + 1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                          bias=conv.bias is not None).to(conv.weight.device, conv.weight.dtype)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    backbone.conv1_dep[0] = new
    return backbone
