"""Depth-completion baselines behind one interface (plan revision v2 sec. 3.7.5; docs/baselines.md).

Every wrapper loads the official code from a pinned checkout under external/baselines/<key> and pinned weights
under checkpoints/baselines/<hf repo>/ (configs/baselines/registry.json; scripts/windows/setup_baselines.ps1
downloads and verifies them). Nothing is downloaded here.

    model = build("priorda", device="cuda")
    depth, report = model.predict(rgb_uint8_hw3, depth_in_m_hw, K_rgb)      # depth_in: 0 = missing

Inputs follow each project's own inference script; every place where we differ is written into `deviations`
and copied into the cache receipt:
  priorda     PriorDepthAnything(version 1.1, vitb/vitb).infer_one_sample(image, prior, pattern=None).
              torch_cluster.knn is replaced by an exact KD-tree k-NN with the CUDA kernel's tie rule (lower index
              first), see TorchClusterShim; torch_cluster has no wheels for our torch.
  omnidc      OGNIDC v1.1 with the demo.sh flags; RGB ImageNet-normalised, sparse depth in metres, padded to a
              multiple of 16 like demo.py. mmcv/mmseg are imported by pvt.py only to load an ImageNet init that the
              v1.1 weights overwrite; they are stubbed. DAv2-L is read from our pinned copy instead of a relative path.
  cdm         RGBDDepth(vitl) per camera; inputs as cdm/infer.py (depth > 25 m dropped, inverse depth in,
              1/prediction out, RGB array passed to infer_image exactly as infer.py does).
  lingbot     MDMModel v0.5 .infer(image/255, depth, normalised K, apply_mask=False) -> dense depth; mask share kept.
  any2full    Any2Full(vitl) forward({'rgb','dep'}) as run_any2full.py (init_scailing, max_depth 1000, no denoise).
  marigold_dc MarigoldDepthCompletionPipeline (bf16, DDIM trailing, 50 steps, ensemble 1, 768 px, seed 2024).
torch >= 2.6 flipped torch.load's default to weights_only=True; projects written before that call torch.load
without the argument. For those we set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (only files verified by SHA256 are read).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "baselines" / "registry.json"


def registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def source_dir(key):
    return ROOT / "external" / "baselines" / key


def weight_dir(hf_repo):
    return ROOT / "checkpoints" / "baselines" / hf_repo.replace("/", "__")


def weight_file(ref):
    """'owner/repo/path/in/repo' -> local path."""
    owner, repo, rest = ref.split("/", 2)
    return weight_dir(f"{owner}/{repo}") / rest


def weight_files(key):
    out = {}
    for h in registry()["models"][key]["hf"]:
        for name in h["files"]:
            out[f"{h['repo']}/{name}"] = weight_dir(h["repo"]) / name
    return out


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def verify_weights(key, files=None):
    """Size + SHA256 (when the registry has one) of every pinned file this model reads; returns the record."""
    spec = registry()["models"][key]
    out = {}
    for h in spec["hf"]:
        for name, (size, digest) in h["files"].items():
            ref = f"{h['repo']}/{name}"
            if files is not None and ref not in files:
                continue
            p = weight_dir(h["repo"]) / name
            if not p.is_file():
                raise FileNotFoundError(f"{p} missing; run scripts/windows/setup_baselines.ps1")
            if p.stat().st_size != size:
                raise ValueError(f"{p}: size {p.stat().st_size} != pinned {size}")
            got = _sha256(p)
            if digest is not None and got != digest:
                raise ValueError(f"{p}: sha256 {got} != pinned {digest}")
            out[ref] = dict(path=str(p), size=size, sha256=got, revision=h["revision"])
    return out


@contextlib.contextmanager
def _sys_path(*paths):
    added = [str(p) for p in paths if str(p) not in sys.path]
    for p in reversed(added):
        sys.path.insert(0, p)
    try:
        yield
    finally:
        pass        # imported modules keep working from their own __file__; leaving the path is harmless per process


@contextlib.contextmanager
def _pushd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------------------------------------------------------------------------------------- exact k-NN
def exact_knn(points, queries, k):
    """Indices (len(queries), k) of the k nearest `points` for each query, ordered by (distance, index):
    the order torch_cluster's CUDA brute-force kernel produces (strict '<' insertion keeps the lower index
    on ties). Exact for any input; ties at the k-th distance are resolved by re-querying wider."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, np.float64); queries = np.asarray(queries, np.float64)
    n = len(points)
    if not 0 < k <= n:
        raise ValueError(f"k={k} with {n} points")
    tree = cKDTree(points)
    out = np.empty((len(queries), k), np.int64)
    todo = np.arange(len(queries)); m = min(n, k + 16)
    while len(todo):
        d, i = tree.query(queries[todo], k=m)
        d = d.reshape(len(todo), m); i = i.reshape(len(todo), m)
        # A row is final when every point at the k-th distance was retrieved: the last retrieved distance is
        # strictly larger, or all points were retrieved.
        ok = (d[:, -1] > d[:, k - 1]) | (m == n)
        dd, ii = d[ok], i[ok]
        order = np.lexsort((ii, dd), axis=-1)[:, :k]
        out[todo[ok]] = np.take_along_axis(ii, order, axis=1)
        todo = todo[~ok]; m = min(n, 2 * m)
    return out


def torch_cluster_shim():
    """Module object exposing torch_cluster.knn(x, y, k, batch_x, batch_y) with the CUDA kernel's output
    layout: [2, M*k] long tensor, row 0 = query (y) index, row 1 = index into x, neighbours by (distance, index)."""
    import torch
    mod = types.ModuleType("torch_cluster")
    mod.__trustprompt_shim__ = "exact KD-tree k-NN, CUDA tie rule (baselines/completion.py)"

    def knn(x, y, k, batch_x=None, batch_y=None, cosine=False, num_workers=1, batch_size=None):
        if cosine:
            raise NotImplementedError("cosine k-NN is not used by PriorDA")
        xs = x.detach().double().cpu().numpy(); ys = y.detach().double().cpu().numpy()
        bx = np.zeros(len(xs), np.int64) if batch_x is None else batch_x.detach().cpu().numpy().astype(np.int64)
        by = np.zeros(len(ys), np.int64) if batch_y is None else batch_y.detach().cpu().numpy().astype(np.int64)
        rows, cols = [], []
        for b in np.unique(by):
            xi = np.flatnonzero(bx == b); yi = np.flatnonzero(by == b)
            if len(xi) == 0:
                continue
            kk = min(k, len(xi))
            idx = exact_knn(xs[xi], ys[yi], kk)
            rows.append(np.repeat(yi, kk)); cols.append(xi[idx].ravel())
        r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        return torch.from_numpy(np.stack([r, c])).to(x.device)

    mod.knn = knn
    return mod


# ----------------------------------------------------------------------------------------------- wrappers
class Baseline:
    key = None
    min_points = 1          # fewer valid prompt pixels than this: the image is recorded as an invalid prediction

    def __init__(self, device="cuda"):
        self.device = device
        self.spec = registry()["models"][self.key]
        self.deviations = []
        self.load_info = {}

    def predict(self, rgb, depth, K):
        raise NotImplementedError

    def _check_inputs(self, rgb, depth):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be uint8 HxWx3")
        if depth.shape != rgb.shape[:2] or depth.dtype != np.float32:
            raise ValueError("depth must be float32 on the RGB grid")
        if np.any(~np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError("depth must be finite and >= 0 (0 = missing)")


def _force_legacy_torch_load(info):
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    info["torch_load"] = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (pre-2.6 default for calls without weights_only)"


class PriorDA(Baseline):
    key = "priorda"

    def __init__(self, device="cuda", load_weights=True):
        super().__init__(device)
        import torch
        s = self.spec["settings"]
        _force_legacy_torch_load(self.load_info)
        sys.modules["torch_cluster"] = torch_cluster_shim()
        self.deviations.append("torch_cluster.knn replaced by an exact KD-tree k-NN with the CUDA tie rule")
        with _sys_path(source_dir(self.key)):
            from prior_depth_anything import PriorDepthAnything
        wd = weight_dir("Rain729/Prior-Depth-Anything")
        dev = "cuda:0" if device == "cuda" else device
        if not load_weights:
            raise NotImplementedError("PriorDA builds its frozen DAv2 from the checkpoint; the k-NN shim is tested separately")
        self.model = PriorDepthAnything(device=dev, version=s["version"], mde_dir=str(wd), ckpt_dir=str(wd),
                                        frozen_model_size=s["frozen_model_size"],
                                        conditioned_model_size=s["conditioned_model_size"])
        if self.model.args.K != s["K"]:
            raise ValueError(f"PriorDA K={self.model.args.K}, registry says {s['K']}")
        self.torch = torch

    def predict(self, rgb, depth, K):
        self._check_inputs(rgb, depth)
        out = self.model.infer_one_sample(image=np.ascontiguousarray(rgb), prior=np.ascontiguousarray(depth),
                                          geometric=None, pattern=None, double_global=False, prior_cover=False,
                                          visualize=False)
        d = out.float().cpu().numpy()
        if d.shape != depth.shape:
            raise ValueError(f"PriorDA output {d.shape} != input {depth.shape}")
        return d.astype(np.float32), dict(knn="exact_kdtree_cuda_tie_rule")

    # PriorDA raises when fewer than K prompt points remain; the cache records that image as an invalid prediction.
    min_points = 5


class OmniDC(Baseline):
    key = "omnidc"

    def __init__(self, device="cuda", load_weights=True):
        super().__init__(device)
        import torch
        _force_legacy_torch_load(self.load_info)
        src = source_dir(self.key) / "src"
        for name, attr in (("mmcv", None), ("mmcv.runner", "load_checkpoint"), ("mmseg", None), ("mmseg.utils", "get_root_logger")):
            if name not in sys.modules:
                m = types.ModuleType(name)
                if attr:
                    def _unused(*a, _n=f"{name}.{attr}", **k):
                        raise RuntimeError(f"{_n} is stubbed (OMNI-DC v1.1 inference never calls it)")
                    setattr(m, attr, _unused)
                sys.modules[name] = m
        self.deviations.append("mmcv/mmseg stubbed (pvt.py imports them only for an ImageNet init overwritten by the v1.1 weights)")
        spec = importlib.util.spec_from_file_location("omnidc_config", src / "config.py")
        cfg = importlib.util.module_from_spec(spec)
        argv = sys.argv
        sys.argv = ["omnidc", *self.spec["settings"]["args"]]
        try:
            spec.loader.exec_module(cfg)
        finally:
            sys.argv = argv
        self.args = cfg.args
        self.args.num_gpus = 1
        dav2 = weight_dir("Rain729/Prior-Depth-Anything") / "depth_anything_v2_vitl.pth"
        orig_load = torch.load

        import torchvision

        def redirect(f, *a, **k):
            name = str(f).replace("\\", "/") if isinstance(f, (str, os.PathLike)) else ""
            if name.endswith("depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth"):     # cwd-relative in OGNIDC
                return orig_load(str(dav2), *a, **k)
            if name.endswith(("pretrained/resnet34.pth", "pretrained/resnet18.pth")) and not Path(name).exists():
                # ImageNet init of PVT's embed layers; every one of these tensors is overwritten by the v1.1
                # weights (checked below: no key of the network may be missing from model.safetensors).
                arch = torchvision.models.resnet34 if name.endswith("resnet34.pth") else torchvision.models.resnet18
                return arch(weights=None).state_dict()
            return orig_load(f, *a, **k)
        self.deviations.append("DAv2-L read from the pinned checkpoints/baselines copy instead of src/depth_models/.../checkpoints")
        self.deviations.append("ResNet ImageNet init files (src/pretrained, Google Drive) not downloaded: random init, "
                               "fully overwritten by the v1.1 weights (no missing keys allowed)")
        with _sys_path(src, src / "model"), _pushd(src):
            from model.ognidc import OGNIDC
            torch.load = redirect
            try:
                if load_weights:
                    wd = weight_dir("zuoym15/OMNI-DC")
                    net = OGNIDC.from_pretrained(str(wd), args=self.args)
                    from safetensors import safe_open
                    with safe_open(str(wd / "model.safetensors"), "pt") as f:
                        stored = set(f.keys())
                    have = set(net.state_dict().keys())
                    self.load_info.update(missing_keys=sorted(have - stored)[:50], n_missing=len(have - stored),
                                          n_unexpected=len(stored - have))
                    if have - stored:
                        raise RuntimeError(f"OMNI-DC weights miss {len(have - stored)} keys, e.g. {sorted(have - stored)[:5]}")
                else:
                    self.args.load_dav2 = 0
                    net = OGNIDC(self.args)
            finally:
                torch.load = orig_load
        self.net = net.to(device).eval()
        self.torch = torch

    def predict(self, rgb, depth, K):
        self._check_inputs(rgb, depth)
        torch = self.torch
        import torchvision.transforms as T
        t_rgb = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        x = t_rgb(rgb)[None].to(self.device)
        d = torch.from_numpy(depth)[None, None].to(self.device)
        H, W = depth.shape
        div = int(4 * 2 ** (self.args.num_resolution - 1))
        ph, pw = (-H) % div, (-W) % div
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph)); d = torch.nn.functional.pad(d, (0, pw, 0, ph))
        sample = dict(rgb=x, dep=d, K=torch.eye(3, device=self.device).reshape(1, 3, 3), pattern=0)
        with torch.no_grad():
            out = self.net(sample)["pred"][..., :H, :W]
        return out[0, 0].float().cpu().numpy().astype(np.float32), dict(pad=[int(ph), int(pw)], divisor=div)


class CDM(Baseline):
    key = "cdm"

    def __init__(self, device="cuda", load_weights=True):
        super().__init__(device)
        import torch
        self.torch = torch
        _force_legacy_torch_load(self.load_info)
        with _sys_path(source_dir(self.key) / "cdm"):
            from rgbddepth.dpt import RGBDDepth
        self.RGBDDepth = RGBDDepth
        self.load_weights = load_weights
        self.current = None; self.model = None
        self.load_info["per_checkpoint"] = {}
        self.deviations.append("Azure Kinect CDM used for HAMMER Helios and NYU Kinect v1 (no model for these cameras)")

    def _model_for(self, sensor):
        ref = self.spec["settings"]["sensor_weights"].get(sensor)
        if ref is None:
            raise KeyError(f"no CDM weights mapped for sensor {sensor}")
        if ref == self.current:
            return self.model, ref
        torch = self.torch
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = self.RGBDDepth(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        if self.load_weights:
            ck = torch.load(str(weight_file(ref)), map_location="cpu")
            if "model" in ck:
                states = {k[7:]: v for k, v in ck["model"].items()}
            elif "state_dict" in ck:
                states = {k[9:]: v for k, v in ck["state_dict"].items()}
            else:
                states = ck
            res = m.load_state_dict(states, strict=False)          # official: strict=False
            self.load_info["per_checkpoint"][ref] = dict(n_missing=len(res.missing_keys), n_unexpected=len(res.unexpected_keys),
                                                         missing_keys=list(res.missing_keys)[:50])
            if res.missing_keys:
                raise RuntimeError(f"CDM {ref} misses {len(res.missing_keys)} keys, e.g. {res.missing_keys[:5]}")
            del ck, states
        self.model = m.to(self.device).eval(); self.current = ref
        return self.model, ref

    def predict(self, rgb, depth, K, sensor=None):
        self._check_inputs(rgb, depth)
        model, ref = self._model_for(sensor)
        s = self.spec["settings"]
        d = depth.copy(); d[d > s["max_depth_in"]] = 0.0
        simi = np.zeros_like(d); simi[d > 0] = 1.0 / d[d > 0]
        with self.torch.no_grad():
            inv = model.infer_image(np.ascontiguousarray(rgb), simi, input_size=s["input_size"])
        with np.errstate(divide="ignore"):
            out = np.where(inv > 0, 1.0 / inv, np.inf).astype(np.float32)
        return out, dict(weights=ref, nonpositive_inverse=int((inv <= 0).sum()))


class LingBot(Baseline):
    key = "lingbot"

    def __init__(self, device="cuda", load_weights=True):
        super().__init__(device)
        import torch
        self.torch = torch
        with _sys_path(source_dir(self.key)):
            from mdm.model.v2 import MDMModel
        self.deviations.append("apply_mask=False: dense depth kept (official default masks low-confidence pixels to inf)")
        if load_weights:
            ck = torch.load(str(weight_dir("robbyant/lingbot-depth-pretrain-vitl-14-v0.5") / "model.pt"),
                            map_location="cpu", weights_only=True)               # as MDMModel.from_pretrained
            model = MDMModel(**ck["model_config"])
            res = model.load_state_dict(ck["model"], strict=False)
            self.load_info.update(n_missing=len(res.missing_keys), n_unexpected=len(res.unexpected_keys),
                                  missing_keys=list(res.missing_keys)[:50])
            if res.missing_keys:
                raise RuntimeError(f"LingBot weights miss {len(res.missing_keys)} keys, e.g. {res.missing_keys[:5]}")
        else:
            raise NotImplementedError("LingBot needs its model_config from the checkpoint")
        self.model = model.to(device).eval()

    def predict(self, rgb, depth, K):
        self._check_inputs(rgb, depth)
        torch = self.torch
        s = self.spec["settings"]
        H, W = depth.shape
        img = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
        d = torch.tensor(depth, dtype=torch.float32, device=self.device)[None]
        Kn = np.asarray(K, np.float64).copy(); Kn[0] /= W; Kn[1] /= H
        Kt = torch.tensor(Kn, dtype=torch.float32, device=self.device)[None]
        out = self.model.infer(img, depth_in=d, intrinsics=Kt, resolution_level=s["resolution_level"],
                               apply_mask=s["apply_mask"], use_fp16=s["use_fp16"])
        pred = out["depth"].squeeze().float().cpu().numpy().astype(np.float32)
        mask = out.get("mask")
        share = None if mask is None else float(mask.squeeze().float().mean().cpu())
        return pred, dict(model_mask_share=share)


class Any2Full(Baseline):
    key = "any2full"

    def __init__(self, device="cuda", load_weights=True):
        super().__init__(device)
        import torch
        self.torch = torch
        s = self.spec["settings"]
        with _sys_path(source_dir(self.key)):
            from model.ours.any2full import Any2Full as A2F
        self.args = types.SimpleNamespace(init_scailing=s["init_scailing"], stage=s["stage"],
                                          max_depth=s["max_depth"], min_depth=s["min_depth"])
        model = A2F(encoder=s["encoder"], da_ckpt_path=None, args=self.args)
        if load_weights:
            ck = torch.load(str(weight_dir("zhiyuandaily/Any2Full") / "checkpoints" / "Any2Full_vitl.pth.tar"),
                            map_location="cpu", weights_only=False)                # as run_any2full.py
            state = ck.get("state_dict", ck)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
        self.model = model.to(device).eval()

    def predict(self, rgb, depth, K):
        self._check_inputs(rgb, depth)
        torch = self.torch
        import torchvision.transforms as T
        x = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(rgb)[None].to(self.device)
        d = torch.from_numpy(depth)[None, None].to(self.device)
        with torch.no_grad():
            pred = self.model({"rgb": x, "dep": d})["pred"].squeeze(0).squeeze(0).float().cpu().numpy()
        if pred.shape != depth.shape:
            raise ValueError(f"Any2Full output {pred.shape} != input {depth.shape}")
        return pred.astype(np.float32), {}


class MarigoldDC(Baseline):
    key = "marigold_dc"

    def __init__(self, device="cuda", load_weights=True):
        super().__init__(device)
        import torch
        self.torch = torch
        if not load_weights:
            raise NotImplementedError("Marigold-DC smoke tests need the real pipeline")
        with _sys_path(source_dir(self.key)):
            from marigold_dc import MarigoldDepthCompletionPipeline
        from diffusers import DDIMScheduler
        s = self.spec["settings"]
        dtype = torch.bfloat16 if s["dtype"] == "bfloat16" else torch.float32
        pipe = MarigoldDepthCompletionPipeline.from_pretrained(str(weight_dir("prs-eth/marigold-depth-v1-0")),
                                                               prediction_type="depth").to(device, dtype=dtype)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe = pipe

    def predict(self, rgb, depth, K):
        self._check_inputs(rgb, depth)
        from PIL import Image
        s = self.spec["settings"]
        pred = self.pipe(image=Image.fromarray(rgb), sparse_depth=np.ascontiguousarray(depth),
                         num_inference_steps=s["num_inference_steps"], ensemble_size=s["ensemble_size"],
                         processing_resolution=s["processing_resolution"], seed=s["seed"])
        pred = np.asarray(pred, np.float32)
        if pred.shape != depth.shape:
            raise ValueError(f"Marigold-DC output {pred.shape} != input {depth.shape}")
        return pred, {}


MODELS = {c.key: c for c in (PriorDA, OmniDC, CDM, LingBot, Any2Full, MarigoldDC)}


def build(key, device="cuda", load_weights=True):
    return MODELS[key](device=device, load_weights=load_weights)
