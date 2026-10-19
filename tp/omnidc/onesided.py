"""Module B of the mainline (docs/plan_omnidc.md sec. 3): one-sided depth integration.

OMNI-DC's depth integration (DDI, `DepthGradOptimLayer`) solves, in log depth x,
    min_x  sum_e c_e (A x - g)_e^2  +  alpha * sum_i w_i (x_i - p_i)^2          (p = log prompt, valid pixels)
with predicted gradients g, gradient confidences c and per-pixel input weights w (OMNI-DC: confidence_input x
GRU weights_input). Module B adds a one-sided term for readings that are likely too far:
    + alpha * lam * sum_i f_i * max(0, x_i - p_i)^2                               (f = far weight, e.g. p_far)
i.e. the estimate may lie in front of the reading but is pulled back when it lies behind it.

The objective is convex and piecewise quadratic. With the set S = {i : x_i > p_i} held fixed the hinge is the
quadratic (x_i - p_i)^2 on S, so each round is one ordinary DDI solve with input weights w + lam * f * 1_S
(same target p). `integrate` alternates "solve" and "update S" (a primal active-set / semismooth Newton scheme);
when S no longer changes, the solution is the exact minimiser. Gradients flow through every weight of the last
solve via OMNI-DC's own backward; S itself is piecewise constant and treated as a constant.
"""
from __future__ import annotations


def input_weights(w_equal, w_far, active, lam):
    """DDI input-term weights [B, R, H, W]: channel 0 (the only one OMNI-DC uses) gets the active hinge added."""
    import torch
    return torch.cat([w_equal[:, :1] + lam * w_far * active, w_equal[:, 1:]], dim=1)


def integrate(solve, log_prompt, valid, w_equal, w_far, lam=1.0, max_rounds=3):
    """solve(weights, x_init) -> x [B,1,H,W] is one DDI solve (weights as `input_weights`; x_init may be None).
    log_prompt, valid, w_far: [B,1,H,W]; w_equal: [B,R,H,W]. Returns (x, active set of the last solve, rounds).
    Round 0 is the plain OMNI-DC solve; at most `max_rounds` hinge rounds follow, stopping when S is stable."""
    import torch
    active = torch.zeros_like(log_prompt)
    x = solve(input_weights(w_equal, w_far, active, lam), None)
    for rounds in range(1, max_rounds + 1):
        new = ((valid > 0) & (x.detach() > log_prompt)).to(log_prompt.dtype)
        if torch.equal(new, active):
            return x, active, rounds - 1
        active = new
        x = solve(input_weights(w_equal, w_far, active, lam), x.detach())
    return x, active, max_rounds
