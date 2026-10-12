"""Module C: low-degree-of-freedom region fill (plan section 5.3), rule v1.

For every connected low-trust region Omega (dilated by `grow` px) we fit
candidates on the trusted ring R = dilate(Omega, ring) minus Omega, where ring
pixels must hold a valid, trusted prompt. All fits are in inverse depth:

  k0  D_a                       base aligned prior, 0 parameters
  k1  1/Z = s*m                 s = median of (1/P)/m on R          (1 parameter)
  k2  1/Z = s*m + t             IRLS-Huber on R                     (2 parameters)
  k3  1/Z = a*u + b*v + c       IRLS-Huber plane on R               (3 parameters)

m is the prior's inverse depth (relative disparity). A plane in camera space
has inverse depth affine in pixel coordinates, so k3 is exact for planar
surfaces such as mirrors and glass panes.

Rule v1 (no learning): BIC = n log(RSS/n) + k log n on the ring, RSS of the
relative inverse-depth residual. k3 is admissible only if k2's prediction
inside Omega is itself near-planar (relative plane-fit residual < 2%), so a
curved object is not flattened. Fewer than `min_ring` ring points -> k0.

Selection rules added in revision R14 (all use the same candidates and the same
k3 interior-planarity admissibility check; only the scoring differs):
  aic             n log(RSS/n) + 2k on the ring
  gric            Torr-style robust criterion: sum min(e^2/sigma^2, 2) + k log(3n),
                  sigma = 1.4826 MAD of the most general admissible fit; the
                  lambda1*d*n term is identical for all candidates (all are surfaces)
  holdout_sector  ring split into `sectors` angular sectors about the region
                  centroid; fit on the others, score = mean over folds of the
                  median relative inverse-depth error on the held-out sector
  holdout_radial  fit on the outer half of the ring, score on the inner half
                  (Chebyshev distance to Omega <= ring // 2): radial extrapolation
`diag_gt` (optional) only adds per-candidate GT errors to the records for
analysis of selection regret; no rule except oracle_best reads ground truth.

Choices are returned per region for analysis. Candidates that produce
nonpositive inverse depth inside Omega are inadmissible (never clamped).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

CANDIDATES = ("k0", "k1", "k2", "k3")
SCORED_RULES = ("v1", "aic", "gric", "holdout_sector", "holdout_radial")
N_PARAMS = {"k0": 0, "k1": 1, "k2": 2, "k3": 3}


def _irls(design, y, iters=20, delta=0.05):
    """Huber IRLS on relative residual (design @ beta - y) / y; returns beta or None."""
    w = 1.0 / (y * y)
    beta = None
    for _ in range(iters):
        sw = np.sqrt(w)
        try:
            new, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new)):
            return None
        r = np.abs((design @ new - y) / y)
        hub = np.ones_like(r); big = r > delta; hub[big] = delta / r[big]
        w = hub / (y * y)
        if beta is not None and np.allclose(new, beta, rtol=1e-8, atol=1e-12):
            beta = new; break
        beta = new
    return beta


def _plane_residual(u, v, inv):
    a = np.stack([u, v, np.ones_like(u)], 1)
    beta = _irls(a, inv)
    if beta is None:
        return np.inf
    return float(np.median(np.abs((a @ beta - inv) / inv)))


def _fit_predict(kind, y, m, u, v, m_eval, u_eval, v_eval):
    """Fit candidate `kind` on (y, m, u, v) and return inverse-depth predictions at the eval points."""
    if kind == "k1":
        ok = np.isfinite(y / m) & (m > 0)
        if not ok.any():
            return None
        s1 = float(np.median((y / m)[ok]))
        return s1 * m_eval if np.isfinite(s1) and s1 > 0 else None
    if kind == "k2":
        b = _irls(np.stack([m, np.ones_like(m)], 1), y)
        return None if b is None else b[0] * m_eval + b[1]
    if kind == "k3":
        if y.size < 3:
            return None
        b = _irls(np.stack([u, v, np.ones_like(u)], 1), y)
        return None if b is None else b[0] * u_eval + b[1] * v_eval + b[2]
    raise ValueError(kind)


def _heldout_error(kind, train, val, k0_val):
    """Median |pred - y| / y on the validation points (inverse depth)."""
    y, m, u, v = train
    yv, mv, uv, vv = val
    pred = k0_val if kind == "k0" else _fit_predict(kind, y, m, u, v, mv, uv, vv)
    if pred is None or not np.all(np.isfinite(pred)):
        return np.inf
    return float(np.median(np.abs(pred - yv) / yv))


def fill_regions(base_depth, prior_inv, prompt, trusted, low_trust, *, grow=3, ring=10, min_ring=50,
                 min_area=20, planar_tol=0.02, rule="v1", gt=None, max_ring_points=4096, seed=0,
                 sectors=8, min_fold_points=10, diag_gt=None):
    """Return (filled_depth, region_records).

    base_depth: D_a (float32, NaN where invalid). prior_inv: m. prompt/trusted: sensor
    prompt and trust mask (trusted pixels keep the prompt). low_trust: pixels to refill.
    rule: one of SCORED_RULES, a fixed candidate 'k0'..'k3', or 'oracle_best' (needs gt; picks the
    candidate with the lowest AbsRel inside the region -- an upper bound, not a method).
    Pixels outside every region keep the prompt where trusted, otherwise base_depth.
    """
    if rule not in SCORED_RULES + CANDIDATES + ("oracle_best",):
        raise ValueError(f"unknown rule {rule}")
    rng = np.random.default_rng(seed)
    base = np.asarray(base_depth, np.float64)
    out = np.where(trusted, prompt, base).astype(np.float64)
    h, w = base.shape
    vv, uu = np.mgrid[0:h, 0:w]
    square = lambda r: np.ones((2 * r + 1, 2 * r + 1), bool)
    labels, n = ndimage.label(low_trust, structure=np.ones((3, 3), bool))
    records = []
    objects = ndimage.find_objects(labels)
    inv_prompt = np.where(trusted & (prompt > 0), 1.0 / np.where(prompt > 0, prompt, 1.0), np.nan)
    for lab, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        pad = grow + ring + 1
        y0, y1 = max(0, sl[0].start - pad), min(h, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(w, sl[1].stop + pad)
        comp = labels[y0:y1, x0:x1] == lab
        area = int(comp.sum())
        omega = ndimage.binary_dilation(comp, square(grow)) if grow else comp
        outer = ndimage.binary_dilation(omega, square(ring))
        ringm = outer & ~omega & trusted[y0:y1, x0:x1] & np.isfinite(inv_prompt[y0:y1, x0:x1]) \
            & np.isfinite(prior_inv[y0:y1, x0:x1])
        rec = dict(label=lab, area=area, ring_points=int(ringm.sum()), choice="k0", reason=None, candidates={})
        region = (slice(y0, y1), slice(x0, x1))
        target = omega
        cand = {"k0": base[region]}
        if area < min_area:
            rec["reason"] = "below_min_area"
        elif ringm.sum() < min_ring:
            rec["reason"] = "insufficient_ring"
        else:
            idx = np.flatnonzero(ringm.ravel())
            if idx.size > max_ring_points:
                idx = rng.choice(idx, max_ring_points, replace=False)
            y = inv_prompt[region].ravel()[idx]; m = prior_inv[region].ravel()[idx]
            u = uu[region].ravel()[idx].astype(float); v = vv[region].ravel()[idx].astype(float)
            mm = prior_inv[region]
            fits = {}
            ratio = y / m
            s1 = float(np.median(ratio[np.isfinite(ratio) & (m > 0)])) if np.any(m > 0) else np.nan
            if np.isfinite(s1) and s1 > 0:
                fits["k1"] = (lambda s=s1: s * mm, np.array([s1]) * m)
            b2 = _irls(np.stack([m, np.ones_like(m)], 1), y)
            if b2 is not None:
                fits["k2"] = (lambda b=b2: b[0] * mm + b[1], b2[0] * m + b2[1])
            b3 = _irls(np.stack([u, v, np.ones_like(u)], 1), y)
            if b3 is not None:
                fits["k3"] = (lambda b=b3: b[0] * uu[region] + b[1] * vv[region] + b[2], b3[0] * u + b3[1] * v + b3[2])
            n_r = idx.size
            bic = {}
            k0_ring = 1.0 / base[region].ravel()[idx]
            rss0 = np.nansum(((k0_ring - y) / y) ** 2)
            bic["k0"] = n_r * np.log(max(rss0, 1e-30) / n_r)
            for k, (field, ring_pred) in fits.items():
                inv_field = field()
                if not np.all(np.isfinite(inv_field[target])) or np.any(inv_field[target] <= 0):
                    rec["candidates"][k] = "inadmissible_nonpositive"; continue
                cand[k] = 1.0 / np.where(inv_field > 0, inv_field, np.nan)
                rss = np.sum(((ring_pred - y) / y) ** 2)
                bic[k] = n_r * np.log(max(rss, 1e-30) / n_r) + N_PARAMS[k] * np.log(n_r)
            if "k3" in bic and "k2" in cand:
                inside = target
                k2_inv = 1.0 / cand["k2"][inside]
                pr = _plane_residual(uu[region][inside].astype(float), vv[region][inside].astype(float), k2_inv) \
                    if inside.sum() >= 3 else np.inf
                rec["k2_interior_plane_residual"] = pr
                if not pr < planar_tol:
                    bic.pop("k3"); rec["candidates"]["k3"] = "rejected_interior_not_planar"
            rec["bic"] = {k: float(b) for k, b in bic.items()}
            if rule in SCORED_RULES and rule != "v1":
                admissible = list(bic)            # same admissibility as v1 (incl. k3 planarity check)
                score = {}
                if rule in ("aic", "gric"):
                    preds = {"k0": k0_ring, **{k: fits[k][1] for k in admissible if k != "k0"}}
                    res = {k: (preds[k] - y) / y for k in admissible}
                    if rule == "aic":
                        for k in admissible:
                            rss = np.nansum(res[k] ** 2)
                            score[k] = n_r * np.log(max(rss, 1e-30) / n_r) + 2 * N_PARAMS[k]
                    else:
                        general = max(admissible, key=lambda k: N_PARAMS[k])
                        sigma = max(1.4826 * float(np.nanmedian(np.abs(res[general]))), 1e-6)
                        rec["gric_sigma"] = sigma
                        for k in admissible:
                            score[k] = float(np.nansum(np.minimum(res[k] ** 2 / sigma ** 2, 2.0))) \
                                + N_PARAMS[k] * np.log(3 * n_r)
                else:
                    pts = (y, m, u, v)
                    if rule == "holdout_sector":
                        cy, cx = ndimage.center_of_mass(omega)
                        ang = np.arctan2(v - (y0 + cy), u - (x0 + cx))
                        fold = np.floor((ang + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
                        folds = [fold == f for f in range(sectors)]
                    else:
                        dist = ndimage.distance_transform_cdt(~omega, metric="chessboard")
                        dist_pts = dist.ravel()[idx]
                        folds = [dist_pts <= ring // 2]
                    errs = {k: [] for k in admissible}
                    used = 0
                    for vmask in folds:
                        tmask = ~vmask
                        if vmask.sum() < min_fold_points or tmask.sum() < min_fold_points:
                            continue
                        used += 1
                        tr = tuple(a[tmask] for a in pts); va = tuple(a[vmask] for a in pts)
                        for k in admissible:
                            errs[k].append(_heldout_error(k, tr, va, k0_ring[vmask]))
                    rec["holdout_folds_used"] = used
                    if used:
                        score = {k: float(np.mean(e)) for k, e in errs.items()}
                    else:
                        score = dict(bic); rec["holdout_fallback"] = "no_usable_fold_bic"
                rec["selection_scores"] = {k: float(x) for k, x in score.items()}
                finite = {k: x for k, x in score.items() if np.isfinite(x)}
                rec["choice"] = min(finite, key=finite.get) if finite else "k0"
                rec["reason"] = f"min_{rule}"
            if rule == "v1":
                rec["choice"] = min(bic, key=bic.get); rec["reason"] = "min_bic"
            elif rule in SCORED_RULES:
                pass
            elif rule in CANDIDATES:
                rec["choice"] = rule if rule in cand else "k0"
                rec["reason"] = "forced" if rule in cand else f"forced_{rule}_unavailable"
            elif rule == "oracle_best":
                if gt is None:
                    raise ValueError("oracle_best needs gt")
                g = gt[region]; ok = target & (g > 0)
                errs = {k: float(np.nanmean(np.abs(c[ok] - g[ok]) / g[ok])) for k, c in cand.items() if ok.any()}
                rec["choice"] = min(errs, key=errs.get) if errs else "k0"; rec["reason"] = "oracle_best"; rec["oracle_errors"] = errs
            else:
                raise ValueError(f"unknown rule {rule}")
        if diag_gt is not None and "oracle_errors" not in rec:
            g = diag_gt[region]; ok = target & (g > 0) & np.isfinite(g)
            if ok.any():
                rec["oracle_errors"] = {k: float(np.nanmean(np.abs(c[ok] - g[ok]) / g[ok])) for k, c in cand.items()}
        chosen = cand[rec["choice"]]
        sub = out[region]
        sub[target] = chosen[target]
        records.append(rec)
    return out.astype(np.float32), records
