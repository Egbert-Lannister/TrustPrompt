"""PromptDA baseline wrapper (plan section 6; revision v2 R6 = C1 first).

Follows the official inference recipe (DepthAnything/PromptDA @ 9161547, read
2026-10-09): RGB in [0,1]; if the longer side exceeds 1008 it is resized with
cv2.INTER_AREA to multiples of 14; prompt depth in metres at 192x256. Our
deviations, all recorded per prediction:
  * images whose sides are not multiples of 14 (NYU 456x608) are resized to
    the nearest multiple of 14 (462x602) because the official loader only
    enforces multiples when downscaling;
  * the prediction is resized back to the RGB grid with bilinear interpolation;
  * holes are filled by nearest neighbour on the native prompt grid before the
    192x256 nearest resize (plan section 6), since min=0 would break the
    model's min/max prompt normalisation.

Prompt variants (only `raw` is a deployable method; the others read GT-derived
masks and are labelled oracle analyses):
  raw          sensor prompt as is
  focus_prior  focus pixels (NYU mirror / HAMMER objects) removed, filled with
               the global robust-aligned prior D_g ("GT material mask" baseline)
  oracle_prior every contaminated prompt pixel C (|P-GT|/GT > tau) removed and
               filled with D_g: upper bound of a perfect trust map with k0 fill
  detector_prior pixels flagged by the DEPTHOR++-style detector (tp/models/depthor_det.py,
               no GT) removed and filled with D_g: deployable baseline (revision R14)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tp.geom.camera import nearest_fill, resize_nearest

PROMPT_HW = (192, 256)
MAX_SIZE = 1008
VARIANTS = ("raw", "focus_prior", "oracle_prior", "detector_prior")
DEPLOYABLE = ("raw", "detector_prior")


def model_input_size(h, w, max_size=MAX_SIZE, multiple=14):
    max_size = max_size // multiple * multiple
    if max(h, w) > max_size:                       # official branch (floor to multiples)
        s = max_size / max(h, w)
        return int(h * s // multiple * multiple), int(w * s // multiple * multiple), "official_area_downscale"
    rh, rw = max(multiple, int(round(h / multiple)) * multiple), max(multiple, int(round(w / multiple)) * multiple)
    return rh, rw, ("none" if (rh, rw) == (h, w) else "nearest_multiple_of_14_resize")


def prepare_prompt(prompt, prompt_valid, *, variant="raw", remove=None, fill_depth=None, hw=PROMPT_HW):
    """Return (prompt_hw float32 metres, info). `remove` marks pixels to drop
    (variant != raw); they and sensor holes are filled from `fill_depth` where it
    is finite and positive, otherwise by nearest neighbour from kept pixels."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    p = np.asarray(prompt, np.float32)
    keep = np.asarray(prompt_valid, bool).copy()
    info = dict(variant=variant, native_valid=int(keep.sum()))
    if variant != "raw":
        if remove is None or fill_depth is None:
            raise ValueError("non-raw variants need `remove` and `fill_depth`")
        keep &= ~remove
        p = p.copy()
        fd = np.asarray(fill_depth, np.float32)
        usable = ~keep & np.isfinite(fd) & (fd > 0)
        p[usable] = fd[usable]
        keep = keep | usable
        info.update(removed=int((np.asarray(prompt_valid, bool) & remove).sum()), filled_from_prior=int(usable.sum()))
    if not keep.any():
        return np.full(hw, np.nan, np.float32), dict(info, status="empty_prompt")
    filled = nearest_fill(p, keep)
    return resize_nearest(filled, hw).astype(np.float32), dict(info, status="ok")


class PromptDAPredictor:
    """Loads the official model from a local source checkout and checkpoint (no download)."""

    def __init__(self, source_dir, ckpt_path, encoder, device="cuda"):
        import torch
        self.torch = torch
        source_dir = Path(source_dir).resolve()
        sys.dont_write_bytecode = True
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from promptda.promptda import PromptDA
        self.model = PromptDA(encoder=encoder, ckpt_path=str(source_dir / "__no_ckpt__"))  # skip its loader
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.load_mode = "weights_only"
        except Exception:
            # Lightning checkpoints may carry non-tensor metadata; the file hash is recorded by the caller.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.load_mode = "full_pickle_official_checkpoint"
        state = {k[9:]: v for k, v in ckpt["state_dict"].items()}        # same prefix strip as upstream
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval()
        self.device = device

    def predict(self, rgb, prompt_hw):
        import cv2
        torch = self.torch
        h, w = rgb.shape[:2]
        th, tw, mode = model_input_size(h, w)
        image = rgb.astype(np.float32) / 255.0
        if (th, tw) != (h, w):
            interp = cv2.INTER_AREA if mode == "official_area_downscale" else cv2.INTER_LINEAR
            image = cv2.resize(image, (tw, th), interpolation=interp)
        x = torch.from_numpy(image).permute(2, 0, 1)[None].to(self.device)
        pr = torch.from_numpy(np.ascontiguousarray(prompt_hw))[None, None].to(self.device)
        with torch.no_grad():
            out = self.model.predict(x, pr)[0, 0].float().cpu().numpy()
        if out.shape != (th, tw):
            raise ValueError(f"unexpected PromptDA output {out.shape} for input {(th, tw)}")
        depth = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) if (th, tw) != (h, w) else out
        return depth.astype(np.float32), dict(model_input_hw=[th, tw], resize_mode=mode,
                                              output_resize="bilinear" if (th, tw) != (h, w) else "none",
                                              prompt_hw=list(prompt_hw.shape), cv2=cv2.__version__)
