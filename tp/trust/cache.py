"""Read a verified T trust cache (scripts/infer_trust.py) for one sample."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tp.provenance import ROOT, sha256
from tp.trust.data import upsample_prob


class TrustCache:
    def __init__(self, root):
        root = Path(root); self.root = root if root.is_absolute() else ROOT / root
        self.receipt = json.loads((self.root / "receipt.json").read_text(encoding="utf-8"))
        if self.receipt.get("stage") != "trust_cache_verified":
            raise ValueError(f"trust cache stage {self.receipt.get('stage')}")
        if sha256(self.root / "index.json") != self.receipt["index_sha256"]:
            raise ValueError("trust cache index hash mismatch")
        self.index = {e["key"]: e for e in json.loads((self.root / "index.json").read_text(encoding="utf-8"))}
        self.threshold = float(self.receipt["threshold"])

    def trust(self, sample, sample_sha256, shape=None):
        """Full-resolution trust map (probability the prompt pixel is clean)."""
        key = f"{sample.sensor}__{sample.scene}__{sample.frame}"
        e = self.index.get(key)
        if e is None:
            raise KeyError(f"{key} not in trust cache")
        if e["sample_sha256"] != sample_sha256:
            raise ValueError("trust cache was built from a different sample file")
        lr = np.load(self.root / e["path"])["trust"].astype(np.float32)
        return upsample_prob(lr, shape or sample.gt.shape)

    def direction_far(self, sample, sample_sha256, shape=None):
        """T v2: full-resolution boolean map 'predicted measured-too-far' (p_far >= p_near)."""
        key = f"{sample.sensor}__{sample.scene}__{sample.frame}"
        e = self.index.get(key)
        if e is None or e["sample_sha256"] != sample_sha256:
            raise ValueError(f"{key}: not in trust cache or different sample")
        z = np.load(self.root / e["path"])
        if "p_far" not in z.files:
            raise ValueError("trust cache has no direction (needs a 3-class T v2 model)")
        far = upsample_prob(z["p_far"].astype(np.float32), shape or sample.gt.shape)
        near = upsample_prob(z["p_near"].astype(np.float32), shape or sample.gt.shape)
        return far >= near
