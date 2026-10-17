"""Assemble the cross-fitted HAMMER trust cache (configs/trust/hammer_crossfit_v1.yaml).

  python -m scripts.crossfit_trust [--config configs/trust/hammer_crossfit_v1.yaml]

For each fold: check that its model was trained with exactly that fold (fold 0 = T v1, same fit/selection sets),
calibrate the conformal threshold (alpha, danger P > (1 + tau_d) GT) on the fold's hold scene, and report the
conformal loss on the fold's own test scenes. Then copy, for every HAMMER dev image (scenes 2-11, real prompts),
the trust map from the fold whose test set contains its scene, with that fold's threshold, into one cache with
per-item thresholds (tp.trust.cache.TrustCache.per_item). Every image must be covered exactly once.
Writes the cache, results/trust_t_v1cf_crc.json and docs/crossfit_trust.md.
"""
import argparse
from datetime import datetime, timezone
import shutil
import sys

import numpy as np
import yaml

from scripts.conformal_threshold import collect
from scripts.train_trust import load_items
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.trust.cache import TrustCache
from tp.trust.conformal import GRID, crc_threshold


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/trust/hammer_crossfit_v1.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text(encoding="utf-8"))
    items = [e for e in load_items(cfg["data"][1:])[0] if e["variant"] == "real" and e["dataset"] == "hammer"]
    out = ROOT / cfg["output"]
    receipt = start_run(out, dict(cfg, config_sha256=sha256(ROOT / a.config)),
                        [sys.executable, "-m", "scripts.crossfit_trust", *sys.argv[1:]], 0)
    try:
        import json
        al, dt = float(cfg["crc"]["alpha"]), float(cfg["crc"]["danger_tau"])
        tested = [s for f in cfg["folds"] for s in f["test"]]
        if len(tested) != len(set(tested)) or set(tested) != {e["scene"] for e in items}:
            raise ValueError(f"folds must test every HAMMER dev scene exactly once: {sorted(tested)}")
        (out / "predictions").mkdir()
        index, folds = [], []
        for f in cfg["folds"]:
            mdir = ROOT / f["model"]
            mrec = json.loads((mdir / "receipt.json").read_text(encoding="utf-8"))
            if mrec.get("stage") != "trust_model_trained":
                raise ValueError(f"{mdir}: stage {mrec.get('stage')}")
            hf = mrec.get("hammer_fold")
            if hf is not None:
                if hf["test_scenes"] != sorted(f["test"]) or hf["selection_scenes"] != sorted(f["hold"]):
                    raise ValueError(f"{mdir} was trained for fold {hf}, config says {f}")
                excluded = set(hf["fit_scenes"]) & (set(f["test"]) | set(f["hold"]))
            else:            # fold 0 = T v1: its own receipt names the selection scenes; fit = train minus those
                sel = sorted(s for d, s in mrec["counts"]["selection_scenes"] if d == "hammer")
                if sel != sorted(f["hold"]) or any(e["split"] != "val" for e in items if e["scene"] in f["test"]):
                    raise ValueError(f"fold {f['id']} is not T v1's split")
                excluded = set()
            if excluded:
                raise ValueError(f"fold {f['id']} fitted on {excluded}")
            cache = TrustCache(f.get("cache") or cfg["fold_cache"].format(id=f["id"]))
            if cache.receipt.get("checkpoint_sha256") != mrec["checkpoint_sha256"]:
                raise ValueError(f"fold {f['id']}: trust cache was made by another checkpoint")
            cal = [e for e in items if e["scene"] in f["hold"]]
            tst = [e for e in items if e["scene"] in f["test"]]
            Lc, _, _, _ = collect(cal, cache, dt); Lt, Ft, Nt, _ = collect(tst, cache, dt)
            lam, bound = crc_threshold(Lc, al)
            if lam is None:
                raise ValueError(f"fold {f['id']}: no conformal threshold at alpha {al} (best bound {bound:.3f})")
            k = int(np.flatnonzero(GRID == lam)[0])
            fr = dict(id=f["id"], test=f["test"], hold=f["hold"], model=str(mdir), checkpoint_sha256=mrec["checkpoint_sha256"],
                      cache=str(cache.root), cache_receipt_sha256=sha256(cache.root / "receipt.json"),
                      model_threshold=cache.threshold, lambda_hat=lam, calibration_bound=bound, calibration_images=len(cal),
                      test_images=len(tst), test_risk=float(Lt[:, k].mean()), test_false_flags=float(Ft[:, k].mean()),
                      test_images_with_far=int((Nt > 0).sum()))
            folds.append(fr)
            for e in tst:
                key = f"{e['sensor']}__{e['scene']}__{e['frame']}"
                src = cache.index[key]
                if src["sample_sha256"] != e["sample_sha256"]:
                    raise ValueError(f"{key}: sample hash differs between trust data and fold cache")
                dst = out / "predictions" / f"{key}.npz"
                shutil.copyfile(cache.root / src["path"], dst)
                if sha256(dst) != src["sha256"]:
                    raise ValueError(f"{key}: copy hash mismatch")
                index.append(dict(src, path=f"predictions/{dst.name}", fold=f["id"], threshold=lam,
                                  model_threshold=cache.threshold))
            print(f"fold {f['id']}: test {f['test']} hold {f['hold']} lambda {lam} test risk {fr['test_risk']:.3f}", flush=True)
        if len(index) != len(items) or len({e["key"] for e in index}) != len(items):
            raise ValueError("cross-fitted cache does not cover every HAMMER dev image exactly once")
        write_json(out / "index.json", index)
        write_json(ROOT / "results" / "trust_t_v1cf_crc.json", dict(alpha=al, danger_tau=dt, folds=folds))
        receipt.update(stage="trust_cache_verified", per_item_threshold=True, threshold=None, folds=folds,
                       items=len(index), index_sha256=sha256(out / "index.json"),
                       completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       note="per-item threshold = conformal lambda_hat of the item's fold (alpha, danger tau_d)")
        lines = ["# HAMMER cross-fitted T v1 (leave-scene-out, 5 folds)", "",
                 f"Generated by `python -m scripts.crossfit_trust` from `{a.config}`. Conformal alpha = {al}, danger = "
                 f"P > {1 + dt:.2f}·GT, calibrated on each fold's hold scene; test risk = mean conformal loss on the fold's "
                 "own test scenes (never seen by that fold's T).", "",
                 "| fold | test scenes | hold scene | model thr | λ̂ | test images | test risk | test false flags |",
                 "|---:|---|---|---:|---:|---:|---:|---:|"]
        for fr in folds:
            lines.append(f"| {fr['id']} | {', '.join(fr['test'])} | {', '.join(fr['hold'])} | {fr['model_threshold']} | "
                         f"{fr['lambda_hat']:.3f} | {fr['test_images']} | {fr['test_risk']:.3f} | {fr['test_false_flags']:.3f} |")
        (ROOT / "docs" / "crossfit_trust.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
