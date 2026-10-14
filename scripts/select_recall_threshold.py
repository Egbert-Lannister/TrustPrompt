"""High-recall threshold for a cached T (R15, item 2): on train-holdout real items only, per dataset,
the largest-trust-preserving threshold (= smallest t) whose mean main-scope recall of contaminated
prompt pixels is >= target; ties broken by fewest false flags. Grid 0.01..0.99. Val is not read.

  python -m scripts.select_recall_threshold --data cache/trust_data_v1_nyu cache/trust_data_v1_hammer \
      --cache cache/trust_t_v1_real_sim_nofeat_{tag}_dev_v1 --target 0.95 --output results/trust_t_v1_highrecall_threshold.json
"""
import argparse

import numpy as np

from scripts.train_trust import load_items
from scripts.trust_mask_diagnostics import TAGS, masks_for
from tp.provenance import ROOT, sha256, write_json
from tp.trust.cache import TrustCache
from tp.trust.data import scene_inner_holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--target", type=float, default=0.95); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    items = [e for e in load_items(a.data)[0] if e["variant"] == "real" and e["split"] == "train"
             and scene_inner_holdout(e["dataset"], e["scene"])]
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    out = dict(target_recall=a.target, population="train holdout scenes, real prompts", grid=[0.01, 0.99, 0.01], datasets={})
    for ds, tag in TAGS.items():
        c = TrustCache(a.cache.replace("{tag}", tag))
        rows = []
        for e in [x for x in items if x["dataset"] == ds]:
            arr = np.load(e["file"]); arr = {k: arr[k] for k in arr.files}
            t = np.load(c.root / c.index[f"{e['sensor']}__{e['scene']}__{e['frame']}"]["path"])["trust"].astype(np.float32)
            rows.append((arr, t))
        stats = []
        for th in grid:
            ms = [masks_for(arr, arr["valid"] & (t < th)) for arr, t in rows]
            rec = [m["recall_scope"] for m in ms if m["recall_scope"] is not None]
            ff = [m["false_flag"] for m in ms if m["false_flag"] is not None]
            stats.append(dict(threshold=float(th), recall_scope=float(np.mean(rec)), false_flag=float(np.mean(ff))))
        ok = [s for s in stats if s["recall_scope"] >= a.target]
        pick = min(ok, key=lambda s: (s["false_flag"], s["threshold"])) if ok else max(stats, key=lambda s: s["recall_scope"])
        out["datasets"][ds] = dict(cache=str(c.root), cache_receipt_sha256=sha256(c.root / "receipt.json"),
                                   model_threshold=c.threshold, chosen=pick, reached_target=bool(ok), images=len(rows),
                                   curve=stats[::5])
        print(ds, "model thr", c.threshold, "->", pick, "reached" if ok else "NOT reached")
    write_json(ROOT / a.output if not a.output.startswith("/") else a.output, out)


if __name__ == "__main__":
    main()
