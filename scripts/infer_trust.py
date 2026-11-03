"""Run a trained T v0 on the real prompts of the train/val splits and cache trust maps.

  python -m scripts.infer_trust --model results/trust_t_v0_real_sim --data cache/trust_data_v0_hammer \
      --features cache/encoder_vits_hammer_dev_v1 --output cache/trust_t_v0_real_sim_hammer_dev_v1

Output: predictions/{sensor}__{scene}__{frame}.npz with `trust` (192x256 float16, probability the
prompt pixel is clean), index.json (sample hashes) and a receipt carrying the model receipt hash and
the threshold chosen on the train-split inner holdout. Consumers upsample with
tp.trust.data.upsample_prob and flag valid prompt pixels with trust < threshold.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from scripts.train_trust import load_features, load_items, predict_trust, selection_metrics
from tp.provenance import ROOT, sha256, start_run, write_json


def run(a):
    import torch
    from tp.trust.model import TrustNetV0
    model_dir = Path(a.model); model_dir = model_dir if model_dir.is_absolute() else ROOT / model_dir
    mrec = json.loads((model_dir / "receipt.json").read_text(encoding="utf-8"))
    if mrec.get("stage") != "trust_model_trained":
        raise ValueError(f"model stage {mrec.get('stage')}")
    out = Path(a.output); out = out if out.is_absolute() else ROOT / out
    formal = None
    if a.formal_test:                   # the one formal test evaluation (docs/plan_test.md)
        from tp.formal import guard
        formal = guard()
    splits = ("test",) if formal else ("train", "val")
    receipt = start_run(out, dict(vars(a), purpose="T v0 inference cache", **(dict(formal_test=formal) if formal else {})),
                        [sys.executable, "-m", "scripts.infer_trust", *sys.argv[1:]], 0)
    try:
        if sha256(model_dir / "best.pt") != mrec["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        use_features = not mrec["config"]["no_features"]
        items, data_receipts = load_items(a.data, allow_test=bool(formal))
        feats, feat_receipts = load_features(a.features) if use_features else ({}, {})
        items = [e for e in items if e["variant"] == "real" and e["split"] in splits]
        classes = int(mrec["config"].get("classes") or 2)
        model = TrustNetV0(use_features=use_features, feat_dropout=float(mrec["config"].get("feat_dropout") or 0.0),
                           out_ch=1 if classes == 2 else 3)
        model.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True))
        model = model.to(a.device)
        (out / "predictions").mkdir()
        full = predict_trust(model, items, feats, use_features, a.device, return_probs=True)
        preds = [(t, y) for t, y, _ in full]
        probs = [pr for _, _, pr in full]
        index = []
        post = bool(mrec["best"].get("post", False)); thr = float(mrec["best"]["threshold"])
        if post:   # v1: region-consistent post-processing chosen on the train holdout, baked into the cached map
            from tp.trust.post import postprocess_trust
            preds = [(postprocess_trust(prob, thr, np.load(e["file"])["valid"]), lab) for (prob, lab), e in zip(preds, items)]
        for (prob, _), pr, e in zip(preds, probs, items):
            key = f"{e['sensor']}__{e['scene']}__{e['frame']}"
            f = out / "predictions" / f"{key}.npz"
            extra = {} if pr is None else dict(p_far=pr[1].astype(np.float16), p_near=pr[2].astype(np.float16))
            with f.open("xb") as fh:
                np.savez_compressed(fh, trust=prob.astype(np.float16), **extra)
            index.append(dict(key=key, dataset=e["dataset"], split=e["split"], scene=e["scene"], frame=e["frame"],
                              sensor=e["sensor"], sample_sha256=e["sample_sha256"], path=f"predictions/{f.name}", sha256=sha256(f)))
        write_json(out / "index.json", index)
        lowres = {}
        for split in splits:
            sel = [(p, e) for p, e in zip(preds, items) if e["split"] == split]
            if sel:
                m = selection_metrics([p for p, _ in sel], [e for _, e in sel])
                lowres[split] = dict(per_dataset_auroc={k: v["auroc"] for k, v in m["per_dataset"].items()},
                                     f1_at_model_threshold={k: v["f1_by_threshold"].get(f"{mrec['best']['threshold']:.2f}")
                                                            for k, v in m["per_dataset"].items()})
        receipt.update(stage="trust_cache_verified", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       model_receipt_sha256=sha256(model_dir / "receipt.json"), model_dir=str(model_dir),
                       checkpoint_sha256=mrec["checkpoint_sha256"], threshold=mrec["best"]["threshold"], post=post, classes=classes,
                       regime=mrec["config"]["regime"], data_receipts_sha256=data_receipts, feature_receipts_sha256=feat_receipts,
                       items=len(index), index_sha256=sha256(out / "index.json"),
                       lowres_diagnostics=lowres, note="lowres diagnostics on the 192x256 grid; train rows include fitted scenes")
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); traceback.print_exc(); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--features", nargs="+", default=[]); ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--formal-test", action="store_true", help="test split only (needs the method v1 freeze)")
    run(ap.parse_args())
