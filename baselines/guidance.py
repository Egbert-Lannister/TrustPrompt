"""Marigold-DC with a per-pixel guidance objective: the prompt-error condition inside the test-time optimisation.

Upstream: prs-eth/Marigold-DC @ c6d756b, marigold_dc.py, MarigoldDepthCompletionPipeline.__call__ (Apache-2.0).
At every DDIM step the Tweedie preview of the depth latent is decoded to metric depth D and fitted to the prompt P
on all valid prompt pixels with  mean|D - P| + mean (D - P)^2 ; the latent gradient is rescaled to the norm of the
predicted noise, Adam moves the latent (lr 0.05) and the affine scale/shift (lr 0.005), then the DDIM step runs.

`complete` repeats that loop step for step (torch is imported inside, so `Guide` stays numpy-only). A `Guide` changes exactly two things:
  objective  L = sum_i [ w_i rho(e_i) + h_i rho(max(e_i, 0)) ] / n,   rho(e) = |e| + e^2,  e = D - P,
             n = number of valid prompt pixels. w = equality weight; h = one-sided weight: D may lie in front of
             the reading but not behind it (a mirror or glass reading is the depth of what is reflected in or seen
             through the surface, which is farther than the surface).
  anchor     the pixels whose min and max set the affine parametrisation (upstream: all valid pixels), so a far
             reflected reading does not stretch the initial depth range.
`Guide.equality()` is upstream's objective and anchor; `complete` then evaluates upstream's loss expression
itself, so it reproduces the upstream pipeline (tests/test_guidance.py with a tiny random pipeline,
scripts/check_marigold_guidance.py with the real weights).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

UPSTREAM = "prs-eth/Marigold-DC@c6d756b marigold_dc.py MarigoldDepthCompletionPipeline.__call__"


@dataclass(frozen=True)
class Guide:
    """Per-pixel guidance on the RGB grid. None = upstream (w = 1 and h = 0 on valid pixels, anchor = valid)."""
    weight: np.ndarray | None = None      # float32 >= 0, equality weight w
    hinge: np.ndarray | None = None       # float32 >= 0, one-sided weight h
    anchor: np.ndarray | None = None      # bool, pixels that set the affine range

    @classmethod
    def equality(cls):
        return cls()

    @property
    def is_equality(self):
        return self.weight is None and self.hinge is None and self.anchor is None

    def check(self, valid):
        for name in ("weight", "hinge", "anchor"):
            a = getattr(self, name)
            if a is None:
                continue
            if a.shape != valid.shape:
                raise ValueError(f"guide {name} {a.shape} != prompt {valid.shape}")
            if name == "anchor":
                if a.dtype != np.bool_ or np.any(a & ~valid):
                    raise ValueError("guide anchor must be a boolean subset of the valid prompt pixels")
            elif a.dtype != np.float32 or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"guide {name} must be finite float32 >= 0")
        if self.anchor is not None and not self.anchor.any():
            raise ValueError("empty guide anchor")

    def sha256(self):
        h = hashlib.sha256()
        for name in ("weight", "hinge", "anchor"):
            a = getattr(self, name)
            h.update(name.encode() + (b"none" if a is None else np.ascontiguousarray(a).tobytes()))
        return h.hexdigest()


def guidance_loss(estimate, target, valid, weight=None, hinge=None):
    """estimate, target: [1,1,H,W] metric depth; valid: bool [1,1,H,W]; weight, hinge: [1,1,H,W] or None."""
    import torch
    d, p = estimate[valid], target[valid]
    if weight is None and hinge is None:            # upstream loss_l1l2, same expression
        return torch.nn.functional.l1_loss(d, p) + torch.nn.functional.mse_loss(d, p)
    e = d - p
    w = torch.ones_like(e) if weight is None else weight[valid]
    loss = (w * (e.abs() + e ** 2)).sum()
    if hinge is not None:
        ep = e.clamp(min=0)
        loss = loss + (hinge[valid] * (ep + ep ** 2)).sum()
    return loss / e.numel()


def complete(pipe, image, sparse_depth, guide=None, num_inference_steps=50, ensemble_size=1,
             processing_resolution=768, seed=2024):
    """Upstream MarigoldDepthCompletionPipeline.__call__ with the objective and anchor taken from `guide`.
    image: PIL RGB; sparse_depth: float32 HxW, 0 = missing. Returns float32 HxW metric depth."""
    import torch
    guide = guide or Guide.equality()
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
        raise ValueError("Invalid num_inference_steps")
    if type(sparse_depth) is not np.ndarray or sparse_depth.ndim != 2:
        raise ValueError("Sparse depth should be a 2D numpy ndarray with zeros at missing positions")
    if ensemble_size < 1:
        raise ValueError("Ensemble size must be at least 1")
    guide.check(sparse_depth > 0)

    with torch.no_grad():
        if pipe.empty_text_embedding is None:
            text_inputs = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
            pipe.empty_text_embedding = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    image, padding, original_resolution = pipe.image_processor.preprocess(
        image, processing_resolution=processing_resolution, device=device, dtype=pipe.dtype)
    if sparse_depth.shape != original_resolution:
        raise ValueError(f"Sparse depth dimensions ({sparse_depth.shape}) must match that of the image ({image.shape[-2:]})")
    with torch.no_grad():
        image_latent, pred_latent = pipe.prepare_latents(image, None, generator, ensemble_size, 1)
    del image

    as_map = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None].to(device)
    sparse_depth = as_map(sparse_depth).float()
    sparse_mask = sparse_depth > 0
    weight = None if guide.weight is None else as_map(guide.weight).float()
    hinge = None if guide.hinge is None else as_map(guide.hinge).float()
    anchor = sparse_mask if guide.anchor is None else as_map(guide.anchor)

    def latent_to_metric(latent):
        prediction = pipe.decode_prediction(latent)
        prediction = (scale ** 2) * sparse_range * prediction + (shift ** 2) * sparse_lower
        prediction = pipe.image_processor.unpad_image(prediction, padding)
        return pipe.image_processor.resize_antialias(prediction, original_resolution, "bilinear", is_aa=False)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ensemble_predictions = []
    for ensemble_idx in pipe.progress_bar(range(ensemble_size), desc="Processing ensemble members...", leave=False):
        current_image_latent = image_latent[ensemble_idx:ensemble_idx + 1]
        current_pred_latent = pred_latent[ensemble_idx:ensemble_idx + 1]
        scale = torch.nn.Parameter(torch.ones(1, device=device))
        shift = torch.nn.Parameter(torch.ones(1, device=device))
        current_pred_latent = torch.nn.Parameter(current_pred_latent)
        sparse_range = (sparse_depth[anchor].max() - sparse_depth[anchor].min()).item()
        sparse_lower = (sparse_depth[anchor].min()).item()
        optimizer = torch.optim.Adam([{"params": [scale, shift], "lr": 0.005},
                                      {"params": [current_pred_latent], "lr": 0.05}])
        for t in pipe.progress_bar(pipe.scheduler.timesteps, desc=f"Marigold-DC steps ({device})...", leave=False):
            optimizer.zero_grad()
            batch_latent = torch.cat([current_image_latent, current_pred_latent], dim=1)
            noise = pipe.unet(batch_latent, t, encoder_hidden_states=pipe.empty_text_embedding, return_dict=False)[0]
            with torch.no_grad():
                alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
                pred_epsilon = (alpha_prod_t ** 0.5) * noise + ((1 - alpha_prod_t) ** 0.5) * current_pred_latent
            step_output = pipe.scheduler.step(noise, t, current_pred_latent, generator=generator)
            current_metric_estimate = latent_to_metric(step_output.pred_original_sample)
            loss = guidance_loss(current_metric_estimate, sparse_depth, sparse_mask, weight, hinge)
            loss.backward()
            with torch.no_grad():
                scaling_factor = torch.linalg.norm(pred_epsilon).item() / max(torch.linalg.norm(current_pred_latent.grad).item(), 1e-8)
                current_pred_latent.grad *= scaling_factor
            optimizer.step()
            with torch.no_grad():
                current_pred_latent.data = pipe.scheduler.step(noise, t, current_pred_latent, generator=generator).prev_sample
            del current_metric_estimate, step_output, pred_epsilon, noise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with torch.no_grad():
            ensemble_predictions.append(latent_to_metric(current_pred_latent.detach()))
    del image_latent
    if ensemble_size > 1:
        prediction = torch.cat(ensemble_predictions, dim=0).median(dim=0, keepdim=True).values
    else:
        prediction = ensemble_predictions[0]
    prediction = pipe.image_processor.pt_to_numpy(prediction)
    pipe.maybe_free_model_hooks()
    return prediction.squeeze()
