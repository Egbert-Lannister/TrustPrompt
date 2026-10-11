"""Read verified RGB-only prior predictions, never use a stale frame cache."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import numpy as np
from tp.provenance import sha256


class CachedPrior:
    def __init__(self, root, expected_model="UniDepthV2-ViT-S14"):
        self.root=Path(root)
        self.receipt=json.loads((self.root/"receipt.json").read_text(encoding="utf-8"))
        if self.receipt["stage"] not in {"real_train_smoke_verified","real_dev_verified"}:
            raise ValueError("prior cache not verified/complete")
        if self.receipt.get("model_version",{}).get("name")!=expected_model:
            raise ValueError(f"prediction cache model mismatch: expected {expected_model}")
        if self.receipt.get("images_sha256") and sha256(self.root/"images.json")!=self.receipt["images_sha256"]:
            raise ValueError("prior index hash mismatch")
        rows=json.loads((self.root/"images.json").read_text(encoding="utf-8"))
        self.rows={tuple(r[k] for k in ("dataset","sensor","scene","frame")):r for r in rows}
        if len(self.rows)!=len(rows):raise ValueError("duplicate prior identity")
        inputs=self.receipt.get("inputs",[])
        if len(rows)!=self.receipt.get("frame_count") or len(inputs)!=len(rows):
            raise ValueError("prior index/receipt frame count mismatch")
        # Sensor is part of the identity when present (HAMMER: 3 sensor samples share one RGB).
        ident=lambda r:(r["dataset"],r.get("sensor"),r["scene"],r["frame"])
        bound={ident(r):r for r in inputs}
        if len(bound)!=len(inputs):raise ValueError("duplicate receipt input")
        for row in rows:
            r=bound.get(ident(row)) or (bound.get((row["dataset"],None,row["scene"],row["frame"])) if "sensor" not in inputs[0] else None)
            if r is None or any(row[k]!=r[k] for k in ("split","rgb_sha256","sample_npz_sha256")):
                raise ValueError("prior index identity/hash differs from receipt input")
            if "sensor" in r and row["sensor"]!=r["sensor"]:
                raise ValueError("prior index sensor differs from receipt input")

    def record(self,sample):
        return self.rows[tuple(getattr(sample,k) for k in ("dataset","sensor","scene","frame"))]

    def predict(self,sample,sample_sha256):
        key=tuple(getattr(sample,k) for k in ("dataset","sensor","scene","frame"))
        row=self.rows[key]
        if row["split"]!=sample.split or row["sample_npz_sha256"]!=sample_sha256:
            raise ValueError("prior cache split/sample hash mismatch")
        if row["rgb_sha256"]!=hashlib.sha256(sample.rgb.tobytes()).hexdigest():
            raise ValueError("prior cache RGB preprocessing mismatch")
        # Cache folder can move; only use the prediction basename within it.
        # PureWindowsPath splits on both separators, so receipts written on
        # Windows resolve on POSIX hosts too.
        path=self.root/"predictions"/PureWindowsPath(row["prediction_path"]).name
        if sha256(path)!=row["prediction_sha256"]:raise ValueError("corrupted prior prediction")
        with np.load(path,allow_pickle=False) as z:depth=z["depth"].copy()
        if depth.shape!=sample.rgb.shape[:2]:raise ValueError("prior grid mismatch")
        return depth
