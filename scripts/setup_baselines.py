"""Fetch and verify the pinned baseline sources and weights (run by scripts/windows/setup_baselines.ps1).

  python -m scripts.setup_baselines [--models priorda omnidc ...] [--with-optional] [--hf-endpoint URL]

Sources: git clone into external/baselines/<key>, checkout the registry commit, refuse a dirty or different tree.
Weights: huggingface_hub.hf_hub_download(repo, file, revision=<pinned sha>, local_dir=checkpoints/baselines/<repo>);
a file already present with the pinned size is not downloaded again; every file is then checked for size and
SHA256 (baselines.completion.verify_weights). Writes checkpoints/baselines/setup_receipt.json.
Licences are recorded, not accepted on the user's behalf: all repos used here are public and ungated.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import subprocess
import sys

from baselines.completion import registry, source_dir, verify_weights, weight_dir
from tp.provenance import ROOT, write_json


def git(src, *args, check=True):
    safe = src.resolve().as_posix()
    p = subprocess.run(["git", "-c", f"safe.directory={safe}", "-C", str(src), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {src}: {p.stderr.strip()}")
    return p.stdout.strip()


def ensure_source(key, spec):
    src = source_dir(key)
    if src.exists() and not (src / ".git").exists():
        raise RuntimeError(f"{src} exists but is not a git checkout; rename it and re-run")
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"clone {spec['repo']} -> {src}", flush=True)
        p = subprocess.run(["git", "clone", "--quiet", spec["repo"], str(src)])
        if p.returncode:
            raise RuntimeError(f"git clone {spec['repo']} failed")
    head = git(src, "rev-parse", "HEAD")
    if head != spec["commit"]:
        git(src, "fetch", "--quiet", "origin", spec["commit"], check=False)
        git(src, "checkout", "--quiet", spec["commit"])
        head = git(src, "rev-parse", "HEAD")
    if head != spec["commit"]:
        raise RuntimeError(f"{src} at {head}, pinned {spec['commit']}")
    status = git(src, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise RuntimeError(f"{src} has local modifications:\n{status}")
    return head


def ensure_weights(key, spec):
    from huggingface_hub import hf_hub_download
    for h in spec["hf"]:
        d = weight_dir(h["repo"])
        for name, (size, _) in h["files"].items():
            p = d / name
            if p.is_file() and p.stat().st_size == size:
                continue
            print(f"download {h['repo']}@{h['revision'][:8]} {name} ({size / 1e9:.2f} GB)", flush=True)
            hf_hub_download(repo_id=h["repo"], filename=name, revision=h["revision"], local_dir=str(d))
    print(f"verify {key} weights (size + SHA256)", flush=True)
    return verify_weights(key)


def main():
    reg = registry()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--with-optional", action="store_true")
    ap.add_argument("--hf-endpoint", default="")
    a = ap.parse_args()
    if a.hf_endpoint:
        os.environ["HF_ENDPOINT"] = a.hf_endpoint
    keys = a.models or [k for k, m in reg["models"].items() if a.with_optional or not m.get("optional")]
    out = ROOT / "checkpoints" / "baselines" / "setup_receipt.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    rec = dict(old, schema="trustprompt.baselines.setup.v1", updated_at_utc=datetime.now(timezone.utc).isoformat(),
               python=sys.version, packages={}, models=old.get("models", {}))
    for pkg in ("torch", "torchvision", "numpy", "scipy", "huggingface_hub", "safetensors", "einops", "timm",
                "omegaconf", "trimesh", "diffusers", "transformers", "accelerate", "opencv-python"):
        try:
            rec["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            rec["packages"][pkg] = None
    for k in keys:
        spec = reg["models"][k]
        print(f"=== {k}: {spec['label']}", flush=True)
        commit = ensure_source(k, spec)
        weights = ensure_weights(k, spec)
        rec["models"][k] = dict(label=spec["label"], commit=commit, weights=weights, code_license=spec["code_license"],
                                weights_license=spec["weights_license"], verified_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out, rec)
    print("BASELINE SOURCES AND WEIGHTS VERIFIED: " + ", ".join(keys), flush=True)


if __name__ == "__main__":
    main()
