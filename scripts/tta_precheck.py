"""CAPA pre-check (docs/plan_tta.md sec. 2): does test-time adaptation inherit wrong readings, and does the
adaptation's loss trajectory tell wrong from correct readings better than the frozen prior does? GPU; development
data only, test never read.

  python -m scripts.tta_precheck                          # NYU 95 + HAMMER 30 images (pre-registered population)
  python -m scripts.tta_precheck --limit 2                # engineering check: 2 images per dataset
  python -m scripts.tta_precheck --resume results/tta_precheck_<time> [--max-minutes 120]
  python -m scripts.tta_precheck --resolution-level 0     # pre-registered fallback if UniDepthV2-L runs out of memory

First run fetches the pinned CAPA source into external/baselines/capa and the UniDepthV2-L weights into
checkpoints/baselines/lpiccinelli__unidepth-v2-vitl14 (public, ungated, CC BY-NC 4.0; --hf-endpoint for a mirror).
Needs `peft` in the environment (pip install peft); colorlog is optional.

PRE-REGISTERED RULES (fixed before any run; docs/plan_tta.md):
  (a) premise: IR at step 100 minus IR at step 0 (main-scope C, pixel-pooled per dataset, paired scene bootstrap,
      2000 draws, seed 20261030) has CI low > 0 and IR(100) >= 0.25 on at least one dataset.
  (b) dynamics: AUROC(s_10) - AUROC(s_0) >= 0.03 with scene-bootstrap CI low > 0 on both datasets, where
      s_k = |D_k - P| / P on main-scope readings with valid GT; positives = wrong readings (C), negatives = correct.
  go = (a) and (b). Everything else in the report is descriptive.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json

TAU, BAND, SEED, DRAWS = 0.05, 5, 20261030, 2000
STEPS = (0, 5, 10, 20, 50, 100)
SCORES = tuple(str(k) for k in STEPS) + ("mean", "ratio10_0")      # AUROC columns (the last two descriptive)
H_LO, H_HI, H_BINS = -5.0, 1.0, 1200          # log10 residual histogram for AUROC
DATASETS = {"nyu": dict(processed="data/processed/mirror3d_dev_v1", per_group=None),
            "hammer": dict(processed="data/processed/hammer_dev_v1", per_group=1)}
CAPA = dict(key="capa", label="CAPA (UniDepthV2-L + LoRA)", repo="https://github.com/nv-dvl/capa",
            commit="080739287e18509c3b046ac7e8b3e0593c2d4e7d", config="config/unidepth_lora.yaml",
            hf_repo="lpiccinelli/unidepth-v2-vitl14", hf_revision="main",   # what CAPA loads; resolved to a commit and locked

            files=("config.json", "model.safetensors"), code_license="CC BY-NC 4.0 (CAPA), UniDepth CC BY-NC 4.0",
            weights_license="CC BY-NC 4.0 (UniDepth v2)")
FAITHFUL_MAX = 0.01
# Deviation from config/unidepth_lora.yaml (15000): CAPA's exact robust-L1 scale-shift alignment builds n x n tensors
# (about 50 n^2 bytes, ~11 GB at 15000, measured 0.8 GB at 4000), which cannot fit an 8 GB GPU next to UniDepthV2-L.
ALIGN_POINTS = 4000


# ---------------------------------------------------------------- statistics (pure numpy, unit-tested)

def hist(scores):
    """Counts of log10(score) in H_BINS bins over [H_LO, H_HI]; values outside are clipped to the ends."""
    s = np.log10(np.clip(np.asarray(scores, np.float64), 10 ** H_LO, 10 ** H_HI))
    return np.histogram(s, bins=H_BINS, range=(H_LO, H_HI))[0].astype(np.int64)


def auroc_hist(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie within a bin) from binned counts; None if a class is empty."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def scene_bootstrap(scenes, stat, rng, draws=DRAWS):
    """scenes: list of per-scene payloads; stat(list) -> float or None. Returns point, 2.5 %, 97.5 %."""
    point = stat(scenes)
    if point is None:
        return None, None, None
    vals = []
    n = len(scenes)
    for _ in range(draws):
        v = stat([scenes[i] for i in rng.integers(0, n, n)])
        if v is not None:
            vals.append(v)
    return float(point), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def image_stats(P, pv, G, gv, scope, preds, tau=TAU, band=BAND):
    """Per-image counts for step 0 / final metrics and residual histograms for every recorded step."""
    R = regions(P, pv, G, gv, tau=tau, band=band)
    C = R["C"] & scope
    ok = R["clean_all"] & scope
    clean = R["clean"] & scope
    out = dict(C=int(C.sum()), correct=int(ok.sum()), clean=int(clean.sum()), steps={})
    for k, D in preds.items():
        valid = np.isfinite(D) & (D > 0)
        Ps = np.where(P > 0, P, 1.0)
        Gs = np.where(G > 0, G, 1.0)
        rel_p, rel_g = np.abs(D - P) / Ps, np.abs(D - G) / Gs
        inh = C & valid & (rel_p < 0.05) & (rel_g > 0.05)
        st = dict(inherited=int(inh.sum()), invalid=int((gv & ~valid).sum()),
                  absrel_C_sum=float(rel_g[C & valid].sum()), absrel_C_n=int((C & valid).sum()),
                  absrel_clean_sum=float(rel_g[clean & valid].sum()), absrel_clean_n=int((clean & valid).sum()),
                  pos_main=hist(rel_p[C & valid]).tolist(), neg_main=hist(rel_p[ok & valid]).tolist(),
                  pos_full=hist(rel_p[R["C"] & valid]).tolist(), neg_full=hist(rel_p[R["clean_all"] & valid]).tolist())
        out["steps"][str(k)] = st
    # descriptive trajectory scores: mean residual over the recorded steps, and s_10 / s_0
    valid = np.all([np.isfinite(D) & (D > 0) for D in preds.values()], 0)
    Ps = np.where(P > 0, P, 1.0)
    res = {k: np.abs(D - P) / Ps for k, D in preds.items()}
    derived = {"mean": np.mean(list(res.values()), 0)}
    if 10 in res and 0 in res:
        derived["ratio10_0"] = res[10] / np.maximum(res[0], 1e-6)
    for k, sc in derived.items():
        out["steps"][k] = dict(pos_main=hist(sc[C & valid]).tolist(), neg_main=hist(sc[ok & valid]).tolist(),
                               pos_full=hist(sc[R["C"] & valid]).tolist(), neg_full=hist(sc[R["clean_all"] & valid]).tolist())
    return out


def _by_scene(rows):
    d = {}
    for r in rows:
        d.setdefault(r["scene"], []).append(r)
    return [d[k] for k in sorted(d)]


def _ir(scenes, k):
    c = sum(r["stats"]["C"] for s in scenes for r in s)
    return None if c == 0 else sum(r["stats"]["steps"][k]["inherited"] for s in scenes for r in s) / c


def _auc(scenes, k, pop="main"):
    pos = np.sum([r["stats"]["steps"][k][f"pos_{pop}"] for s in scenes for r in s], 0)
    neg = np.sum([r["stats"]["steps"][k][f"neg_{pop}"] for s in scenes for r in s], 0)
    return auroc_hist(pos, neg)


def summarise(rows, rng):
    res = {}
    for ds in DATASETS:
        xs = [r for r in rows if r["dataset"] == ds]
        if not xs:
            continue
        sc = _by_scene(xs)
        last = str(STEPS[-1])
        ir0, ir1 = _ir(sc, "0"), _ir(sc, last)
        d_ir = scene_bootstrap(sc, lambda s: None if _ir(s, last) is None else _ir(s, last) - _ir(s, "0"), rng)
        d_auc = scene_bootstrap(sc, lambda s: None if _auc(s, "10") is None or _auc(s, "0") is None
                                else _auc(s, "10") - _auc(s, "0"), rng)
        absrel = lambda k, reg: (lambda n: None if n == 0 else sum(r["stats"]["steps"][k][f"absrel_{reg}_sum"] for r in xs) / n)(
            sum(r["stats"]["steps"][k][f"absrel_{reg}_n"] for r in xs))
        res[ds] = dict(images=len(xs), scenes=len(sc), C_pixels=sum(r["stats"]["C"] for r in xs),
                       correct_pixels=sum(r["stats"]["correct"] for r in xs),
                       IR={str(k): _ir(sc, str(k)) for k in STEPS}, delta_IR=d_ir,
                       AUROC_main={k: _auc(sc, k) for k in SCORES},
                       AUROC_full={k: _auc(sc, k, "full") for k in SCORES}, delta_AUROC_10_0=d_auc,
                       absrel_C={k: absrel(k, "C") for k in ("0", last)},
                       absrel_clean={k: absrel(k, "clean") for k in ("0", last)},
                       invalid_pixels={k: sum(r["stats"]["steps"][k]["invalid"] for r in xs) for k in ("0", last)},
                       seconds_median=float(np.median([r["seconds"] for r in xs])),
                       gpu_peak_gib_max=max(r["gpu_peak_gib"] or 0 for r in xs))
    return res


def verdict(res):
    a = {ds: x["delta_IR"][1] is not None and x["delta_IR"][1] > 0 and x["IR"]["100"] is not None and x["IR"]["100"] >= 0.25
         for ds, x in res.items()}
    b = {ds: x["delta_AUROC_10_0"][0] is not None and x["delta_AUROC_10_0"][0] >= 0.03 and x["delta_AUROC_10_0"][1] > 0
         for ds, x in res.items()}
    pa = any(a.values())
    pb = len(b) == len(DATASETS) and all(b.values())
    return dict(premise_a=pa, premise_a_by_dataset=a, dynamics_b=pb, dynamics_b_by_dataset=b,
                outcome="go" if pa and pb else ("a_without_b" if pa else "premise_fails"))


# ---------------------------------------------------------------- CAPA (GPU)

@contextmanager
def _sys_path(p):
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        if str(p) in sys.path:
            sys.path.remove(str(p))


def _wandb_stub():
    """UniDepth's utils import wandb for logging only; a stub keeps it out of the environment (as baselines/unidepth.py)."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        stub = ModuleType("wandb")
        stub.Image = lambda value: value
        stub.log = lambda *a, **k: None
        sys.modules["wandb"] = stub


def fetch_capa(hf_endpoint=None):
    """Pinned CAPA source (scripts.setup_baselines.ensure_source). Weights: CAPA loads the hub's `main` of
    lpiccinelli/unidepth-v2-vitl14; on first use `main` is resolved to its commit, the files are downloaded at that
    commit into checkpoints/baselines/<repo>/<commit[:12]>/ and their size + SHA256 are written to a receipt; every
    later run uses that commit and checks the files (trust on first use, disclosed in docs/plan_tta.md)."""
    from scripts.setup_baselines import ensure_source
    from baselines.completion import source_dir, weight_dir
    commit = ensure_source(CAPA["key"], dict(repo=CAPA["repo"], commit=CAPA["commit"]))
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    receipt_path = ROOT / "checkpoints" / "baselines" / "capa_weights_receipt_v2.json"
    old = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    if old is None:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(CAPA["hf_repo"], revision=CAPA["hf_revision"]).sha
        print(f"{CAPA['hf_repo']}: '{CAPA['hf_revision']}' resolved to {revision}", flush=True)
    else:
        revision = old["revision"]
    d = weight_dir(CAPA["hf_repo"]) / revision[:12]
    files = {}
    for name in CAPA["files"]:
        p = d / name
        if not p.is_file():
            from huggingface_hub import hf_hub_download
            print(f"download {CAPA['hf_repo']}@{revision[:8]} {name}", flush=True)
            hf_hub_download(repo_id=CAPA["hf_repo"], filename=name, revision=revision, local_dir=str(d))
        files[name] = [p.stat().st_size, sha256(p)]
    model_cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))["model"]
    if "layer_scale" not in model_cfg:
        raise RuntimeError(f"{d / 'config.json'} is the old UniDepthV2 format (no model.layer_scale); "
                           "CAPA's vendored UniDepth needs the current checkpoint")
    if old is not None:
        if old["files"] != files:
            raise RuntimeError(f"UniDepthV2-L weights differ from the first-use receipt {receipt_path}")
    else:
        write_json(receipt_path, dict(repo=CAPA["hf_repo"], requested=CAPA["hf_revision"], revision=revision, files=files,
                                      note="trust on first use: hub 'main' resolved and locked at first download",
                                      recorded_at=datetime.now().isoformat()))
    return dict(source=str(source_dir(CAPA["key"])), commit=commit, weights_dir=str(d), weights_revision=revision,
                weights=files)


def load_protocol(device, weights_dir, resolution_level=None):
    import yaml
    from baselines.completion import source_dir
    src = source_dir(CAPA["key"])
    _wandb_stub()
    sys.path.insert(0, str(src))           # `import capa`; CAPA itself adds third_party/UniDepth_VPT
    from capa.protocol import CAPAProtocol
    cfg = yaml.safe_load((src / CAPA["config"]).read_text(encoding="utf-8"))
    cfg["ckpt_path"] = str(weights_dir)
    cfg["alignment"]["n_points"] = ALIGN_POINTS
    proto = CAPAProtocol(cfg, device)
    if resolution_level is not None:
        reset = proto.model.reset

        def reset_with_level():
            reset()
            proto.model.model.resolution_level = int(resolution_level)
        proto.model.reset = reset_with_level
    return proto, cfg


def run_recorded(proto, rgb_n3hw, cond_nhw, mask_nhw, record=STEPS):
    """CAPAProtocol._run_optimization (CAPA commit 0807392) copied line by line, plus a record of the aligned
    prediction at the steps in `record` (the step-k forward is the one CAPA computes anyway; nothing extra runs).
    Returns (prediction, {step: aligned prediction}, losses) or None on NaN, like the original."""
    import torch
    from capa.utils.alignment import align_depth_to_condition
    from capa.protocol import _CAPA_ROOT
    self = proto
    device = self.device
    N, _, H, W = rgb_n3hw.shape
    self.model.reset()
    if self.tuning_mode == "lora":
        self.model.inject_lora(rank=self.config["lora_rank"], alpha=self.config["lora_alpha"],
                               target_modules=self.config["lora_target_modules"])
    elif self.tuning_mode == "vpt":
        init_method = self.config.get("vpt_init_method", "xavier")
        if init_method.startswith("load_"):
            init_method = f"load_{_CAPA_ROOT / init_method[len('load_'):]}"
        self.model.inject_vpt(token_len=self.config["vpt_token_len"], init_method=init_method)
    else:
        raise ValueError(f"Unknown tuning mode: {self.tuning_mode}")
    optimizer = torch.optim.AdamW(self.model.get_trainable_params(lr=self.lr))
    optimizer.zero_grad()
    lr_scheduler = None
    if self.lr_scheduler_cfg is not None:
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_cfg["type"])(optimizer, **self.lr_scheduler_cfg["kwargs"])
    preprocess_inputs = getattr(self.model, "preprocess_inputs", None)
    predict_depth_from_processed = getattr(self.model, "predict_depth_from_processed", None)
    rgb_processed_n3hw = None
    if callable(preprocess_inputs) and callable(predict_depth_from_processed):
        rgb_processed_n3hw = preprocess_inputs(rgb_n3hw.contiguous().to(device))
    frame_sampling_rng = torch.Generator(device="cpu").manual_seed(self.align_seed)
    recorded, losses = {}, []
    for step in range(self.n_steps + 1):
        is_last_step = step == self.n_steps
        if is_last_step or self.pct_random_key_frame <= 0:
            frame_idxs = list(range(N))
        else:
            n_sample = max(1, math.ceil(N * 0.01 * self.pct_random_key_frame))
            frame_idxs = torch.randperm(N, generator=frame_sampling_rng)[:n_sample].tolist()
        if is_last_step:
            sub_batches = [frame_idxs] if self.max_bs_infer is None else self._make_sub_batches(frame_idxs, self.max_bs_infer)
            depth_pred_nhw = torch.full((N, H, W), torch.nan, device=device)
        else:
            sub_batches = self._make_sub_batches(frame_idxs, self.max_bs_optimize)
        loss_accum = 0.0
        for sub_idxs in sub_batches:
            rgb_sub = (rgb_n3hw[sub_idxs].contiguous().to(device) if rgb_processed_n3hw is None
                       else rgb_processed_n3hw[sub_idxs].clone().contiguous())
            depth_cond_sub = cond_nhw[sub_idxs].contiguous().to(device)
            mask_cond_sub = mask_nhw[sub_idxs].contiguous().to(device)
            with (torch.no_grad() if is_last_step else torch.enable_grad()):
                depth_pred_sub = (self.model.predict_depth(rgb_sub) if rgb_processed_n3hw is None
                                  else predict_depth_from_processed(rgb_sub, output_size=(H, W)))
            if torch.any(torch.isnan(depth_pred_sub)):
                return None
            with torch.no_grad():
                scale, shift = align_depth_to_condition(depth_pred_sub.detach(), depth_cond_sub.detach(), mask_cond_sub,
                                                        n_points=self.align_n_points, seed=self.align_seed)
            scale, shift = scale.detach(), shift.detach()
            depth_pred_sub = depth_pred_sub * scale.view(-1, 1, 1) + shift.view(-1, 1, 1)
            if step in record and N == 1:
                recorded[step] = depth_pred_sub.detach()[0].float().cpu().numpy()
            if is_last_step:
                depth_pred_nhw[sub_idxs] = depth_pred_sub.detach()
                continue
            loss = torch.abs(depth_pred_sub - depth_cond_sub)[mask_cond_sub].mean()
            (loss / len(sub_batches)).backward()
            loss_accum += loss.item()
        if is_last_step:
            break
        losses.append(loss_accum / len(sub_batches))
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=self.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return depth_pred_nhw, recorded, losses


def run_with_retries(proto, *args):
    for _ in range(proto.max_nan_retries + 1):
        out = run_recorded(proto, *args)
        if out is not None:
            return out
    raise RuntimeError("CAPA prediction still contains NaN after the retries")


# ---------------------------------------------------------------- driver

def select(ds, per_group, limit):
    d = DATASETS[ds]
    rows = json.loads((ROOT / d["processed"] / "manifest.json").read_text(encoding="utf-8"))["records"]
    rows = per_group_subsample([r for r in rows if r["split"] in {"train", "val"}], per_group)   # test locked
    return rows[:limit] if limit else rows


def tensors(s):
    import torch
    rgb = torch.from_numpy(s.rgb.copy()).permute(2, 0, 1)[None].float() / 255.0
    mask = s.prompt_valid & (s.prompt > 0)
    cond = torch.from_numpy(np.where(mask, s.prompt, 0).astype(np.float32))[None]
    return rgb, cond, torch.from_numpy(mask)[None]


def report(out, res, ver, args, info):
    f = lambda x, d=3: "NA" if x is None else f"{x:.{d}f}"
    ci = lambda t: f"{f(t[0])} ({f(t[1])} to {f(t[2])})"
    lines = ["# CAPA pre-check: inheritance and loss trajectory (pre-registered, development data)", "",
             f"Generated by `python -m scripts.tta_precheck` ({out.name}); rules fixed in docs/plan_tta.md sec. 2 and the "
             f"script docstring before the run. CAPA {info['capa']['commit'][:7]}, UniDepthV2-L + LoRA, "
             f"config/unidepth_lora.yaml except {ALIGN_POINTS} alignment points (memory); resolution level "
             f"{'default' if args.resolution_level is None else args.resolution_level}; test never read.", "",
             f"**Outcome: {ver['outcome']}** (a: {ver['premise_a']}, b: {ver['dynamics_b']}).", "",
             "## (a) Inheritance: IR on main-scope wrong readings", "",
             "| dataset | images / scenes | IR step 0 (prior, aligned) | IR step 100 (CAPA) | difference (95 % CI) |",
             "|---|---:|---:|---:|---:|"]
    for ds, x in res.items():
        lines.append(f"| {ds} | {x['images']} / {x['scenes']} | {f(x['IR']['0'])} | {f(x['IR']['100'])} | {ci(x['delta_IR'])} |")
    lines += ["", "## (b) Wrong vs correct readings by residual at step k (AUROC, main scope)", "",
              "| dataset | " + " | ".join(f"s_{k}" for k in SCORES) + " | s_10 - s_0 (95 % CI) |",
              "|---|" + "---:|" * (len(SCORES) + 1)]
    for ds, x in res.items():
        lines.append(f"| {ds} | " + " | ".join(f(x["AUROC_main"][k]) for k in SCORES) + f" | {ci(x['delta_AUROC_10_0'])} |")
    lines += ["", "s_mean = mean residual over the recorded steps, s_ratio10_0 = s_10 / s_0 (both descriptive).", "",
              "Full image (descriptive):", "", "| dataset | " + " | ".join(f"s_{k}" for k in SCORES) + " |",
              "|---|" + "---:|" * len(SCORES)]
    for ds, x in res.items():
        lines.append(f"| {ds} | " + " | ".join(f(x["AUROC_full"][k]) for k in SCORES) + " |")
    lines += ["", "## Descriptive", "",
              "| dataset | AbsRel C step 0 | AbsRel C step 100 | AbsRel clean step 0 | AbsRel clean step 100 | invalid px (0 / 100) | s / image | GPU peak GiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, x in res.items():
        lines.append(f"| {ds} | {f(x['absrel_C']['0'], 4)} | {f(x['absrel_C']['100'], 4)} | {f(x['absrel_clean']['0'], 4)} | "
                     f"{f(x['absrel_clean']['100'], 4)} | {x['invalid_pixels']['0']} / {x['invalid_pixels']['100']} | "
                     f"{x['seconds_median']:.1f} | {x['gpu_peak_gib_max']:.2f} |")
    fa = info.get("faithfulness")
    lines += ["", f"Faithfulness (first image, CAPA's own `run` vs the recording copy): median relative difference "
              f"{'NA' if fa is None else format(fa['median_rel_diff'], '.2e')} (limit {FAITHFUL_MAX}).",
              "NYU main-scope correct readings are rare (Mirror3D mirrors hold ~3 % correct readings), so NYU AUROC rests "
              "on few negatives; counts are in tta_precheck.json.", ""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="engineering check: first N images per dataset")
    ap.add_argument("--resolution-level", type=int, default=None, choices=range(10))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--hf-endpoint", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import importlib.util
    missing = [m for m in ("torch", "peft", "yaml", "huggingface_hub") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"missing packages {missing}; install with: python -m pip install {' '.join('pyyaml' if m == 'yaml' else m.replace('_', '-') for m in missing)}")
    import torch
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment (use the GPU venv, e.g. .venvs/baselines)")
    config = dict(tau=TAU, band=BAND, seed=SEED, bootstrap_draws=DRAWS, steps=list(STEPS), datasets=a.datasets,
                  per_group={k: DATASETS[k]["per_group"] for k in a.datasets}, limit=a.limit,
                  resolution_level=a.resolution_level, align_points=ALIGN_POINTS, capa=CAPA, hist=[H_LO, H_HI, H_BINS], splits=["train", "val"])
    command = [sys.executable, "-m", "scripts.tta_precheck", *sys.argv[1:]]
    if a.resume is None:
        out = ROOT / "results" / f"tta_precheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, config, command, SEED)
    else:
        out = a.resume if a.resume.is_absolute() else ROOT / a.resume
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        if receipt["config"] != json.loads(json.dumps(config)):
            raise ValueError("resume requires the identical configuration (same flags)")
        if sha256(ROOT / "scripts" / "tta_precheck.py") != receipt["source_sha256"]["scripts/tta_precheck.py"]:
            raise ValueError("scripts/tta_precheck.py changed since the run started; start a new run")
        receipt.setdefault("resumed", []).append(command)
    t0 = time.time()
    try:
        info = receipt.get("capa_setup") or {}
        info["capa"] = fetch_capa(a.hf_endpoint)
        receipt["capa_setup"] = info
        receipt["model_version"] = f"CAPA {CAPA['commit']} + {CAPA['hf_repo']}@{info['capa']['weights_revision']}"
        device = torch.device(a.device)
        proto, cfg = load_protocol(device, info["capa"]["weights_dir"], a.resolution_level)
        receipt["capa_config"] = cfg
        per = out / "per_image"
        per.mkdir(exist_ok=True)
        preds_dir = out / "predictions"
        preds_dir.mkdir(exist_ok=True)
        todo = [(ds, r) for ds in a.datasets for r in select(ds, DATASETS[ds]["per_group"], a.limit)]
        for i, (ds, r) in enumerate(todo):
            name = f"{ds}__{Path(r['path']).stem}"
            if (per / f"{name}.json").is_file():
                continue
            if a.max_minutes is not None and time.time() - t0 > 60 * a.max_minutes:
                receipt.update(stage="partial_time_budget")
                print(f"time budget reached ({sum(1 for _ in per.glob('*.json'))} / {len(todo)} images). Resume with: "
                      f"python -m scripts.tta_precheck --resume {out.relative_to(ROOT).as_posix()} " + " ".join(
                          x for x in sys.argv[1:] if x not in {"--resume", str(a.resume)}), flush=True)
                return
            path = ROOT / DATASETS[ds]["processed"] / r["path"]
            if sha256(path) != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            s = load_sample(path)
            if s.split == "test":
                raise ValueError("test sample reached the pre-check")
            rgb, cond, mask = tensors(s)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(SEED)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(SEED)
            t1 = time.time()
            try:
                pred, rec, losses = run_with_retries(proto, rgb, cond, mask)
            except torch.cuda.OutOfMemoryError:
                receipt.update(stage="failed_out_of_memory", image=name)
                print("CUDA out of memory. Pre-registered fallback: start a NEW run with --resolution-level 0 "
                      "(the whole run then uses it).", flush=True)
                raise
            seconds = time.time() - t1
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
            if "faithfulness" not in info:
                torch.manual_seed(SEED)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(SEED)
                ref = proto.run(rgb, cond, mask)[0].float().cpu().numpy()
                mine = pred[0].float().cpu().numpy()
                rel = np.abs(ref - mine) / np.maximum(np.abs(ref), 1e-6)
                info["faithfulness"] = dict(image=name, median_rel_diff=float(np.median(rel)), max_rel_diff=float(rel.max()))
                write_json(out / "receipt.json", receipt)
                if info["faithfulness"]["median_rel_diff"] > FAITHFUL_MAX:
                    raise RuntimeError(f"recording copy differs from CAPA's run: {info['faithfulness']}")
            final = pred[0].float().cpu().numpy()
            if not np.allclose(rec[STEPS[-1]], final, equal_nan=True):
                raise AssertionError("step-100 record differs from the returned prediction")
            _, scope = focus_scope(s)
            stats = image_stats(s.prompt, s.prompt_valid, s.gt, s.gt_valid, scope, rec)
            np.savez_compressed(preds_dir / f"{name}.npz", **{f"step_{k}": v.astype(np.float16) for k, v in rec.items()})
            write_json(per / f"{name}.json", dict(dataset=ds, scene=s.scene, sensor=s.sensor, frame=s.frame, seconds=seconds,
                                                 gpu_peak_gib=peak, losses=losses, stats=stats))
            print(f"[{i + 1}/{len(todo)}] {name}: {seconds:.1f} s, peak {peak if peak is None else round(peak, 2)} GiB, "
                  f"loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
        rows = [json.loads((per / f"{ds}__{Path(r['path']).stem}.json").read_text(encoding="utf-8")) for ds, r in todo]
        res = summarise(rows, np.random.default_rng(SEED))
        ver = verdict(res) if a.limit is None and set(a.datasets) == set(DATASETS) else dict(
            outcome="engineering check (no verdict)", premise_a=None, dynamics_b=None)
        write_json(out / "tta_precheck.json", dict(results=res, verdict=ver))
        lines = report(out, res, ver, a, info)
        doc = ROOT / "docs" / ("tta_precheck.md" if a.limit is None else "tta_precheck_engineering.md")
        doc.write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="complete" if a.limit is None else "engineering_check_complete", verdict=ver,
                       result_sha256=sha256(out / "tta_precheck.json"), doc=doc.relative_to(ROOT).as_posix())
        print("\n".join(lines))
        print("TTA PRECHECK DONE" if a.limit is None else "TTA ENGINEERING CHECK DONE", flush=True)
    except BaseException as e:
        if receipt.get("stage") in {"started", None}:
            receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
