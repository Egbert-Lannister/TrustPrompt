"""Where do wrong readings enter OMNI-DC? Switch each entry point off for flagged pixels, without retraining.

A prompt reaches the OMNI-DC prediction through three paths (src/model/ognidc.py @ a5ecd35):
  input  the (log, median-whitened) prompt is a network input channel (backbone features, GRU);
  ddi    the depth integration fits the prompt with per-cell weight confidence_input x weights_input (1/4 res);
  spn    the final DySPN refinement re-injects the full-resolution prompt with its own confidence each iteration.
`gated(net, ddi_keep=..., spn_keep=...)` patches the two explicit paths of one network instance for the
duration of a `with` block: the DDI input confidence is multiplied by `ddi_keep` (per backbone cell) and the
prompt handed to DySPN is zeroed where `spn_keep` is 0 (DySPN ignores zero pixels: confidence x sign(sp_dep)).
The input path is unchanged; deleting the pixels from the prompt (the `tmask` / `oracle_mask` variants) closes
all three. Nothing is patched outside the block.
"""
from __future__ import annotations

import contextlib


def cell_keep(flag, rate):
    """Full-res bool flag map [B,1,H,W] (padded like the network input) -> keep weight per backbone cell
    (0 where any pixel of the cell is flagged; OMNI-DC's own ConfInput labels max-pool the same way)."""
    import torch.nn.functional as F
    return 1.0 - F.max_pool2d(flag.float(), rate)


@contextlib.contextmanager
def gated(net, ddi_keep=None, spn_keep=None):
    """ddi_keep: [B,1,H/r,W/r] multiplier of the DDI input confidence; spn_keep: [B,1,H,W] 0/1 mask of the prompt
    pixels DySPN may re-inject. None leaves that path as it is."""
    patched = []
    if ddi_keep is not None:
        backbone_forward = net.backbone.forward

        def backbone(*args, **kwargs):
            out = list(backbone_forward(*args, **kwargs))
            if out[4] is None:
                raise RuntimeError("this OMNI-DC has no input-confidence head (pred_confidence_input = 0)")
            out[4] = out[4] * ddi_keep
            return tuple(out)
        net.backbone.forward = backbone
        patched.append(net.backbone)
    if spn_keep is not None:
        if not hasattr(net, "prop_layer"):
            raise RuntimeError("this OMNI-DC has no SPN refinement")
        spn_forward = net.prop_layer.forward

        def spn(feat, guide, sp_dep, confidence):
            return spn_forward(feat, guide, sp_dep * spn_keep, confidence)
        net.prop_layer.forward = spn
        patched.append(net.prop_layer)
    try:
        yield
    finally:
        for m in patched:
            del m.forward                                  # back to the class method
