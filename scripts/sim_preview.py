"""Visual check of simulator v0 on real training frames (train split only).

  python -m scripts.sim_preview --manifest data/processed/mirror3d_dev_v1/manifest.json --n 4 --output results/sim_preview_v1
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from tp.data.schema import load_sample
from tp.provenance import start_run, write_json
from tp.sim.generate import simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    receipt = start_run(a.output, dict(manifests=[str(m) for m in a.manifest], n=a.n, seed=a.seed),
                        [sys.executable, "-m", "scripts.sim_preview", *sys.argv[1:]], a.seed)
    rng = np.random.default_rng(a.seed); records = []
    for man in a.manifest:
        recs = [r for r in json.loads(man.read_text())["records"] if r["split"] == "train"]
        pick = rng.choice(len(recs), min(a.n, len(recs)), replace=False)
        for i in pick:
            s = load_sample(man.parent / recs[i]["path"])
            out = simulate(s, rng, p_corrupt=1.0)
            vmax = float(np.nanpercentile(np.where(s.gt_valid, s.gt, np.nan), 99) * 1.6)
            fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True); ax = ax.ravel()
            cm = plt.get_cmap("turbo").copy(); cm.set_bad("black")
            ax[0].imshow(s.rgb); ax[0].set_title(f"{s.dataset}/{s.sensor}/{s.scene}/{s.frame}")
            ax[1].imshow(np.where(s.gt_valid, s.gt, np.nan), cmap=cm, vmin=0, vmax=vmax); ax[1].set_title("GT [m]")
            ax[2].imshow(np.where(out["V_clean"], out["P_clean"], np.nan), cmap=cm, vmin=0, vmax=vmax); ax[2].set_title("P_clean (sensor model)")
            ax[3].imshow(np.where(out["V_corrupt"], out["P_corrupt"], np.nan), cmap=cm, vmin=0, vmax=vmax); ax[3].set_title("P_corrupt")
            ax[4].imshow(out["applied"], cmap=ListedColormap(["black", "#29b6f6", "#ab47bc", "#ffa726"]), vmin=0, vmax=3)
            ax[4].set_title("applied: blue penetration / purple reflection / orange metal")
            ax[5].imshow(np.where(out["labels"] == 255, 4, out["labels"]),
                         cmap=ListedColormap(["#43a047", "#e53935", "#f9a825", "#1976d2", "#000000"]), vmin=0, vmax=4)
            ax[5].set_title("T labels: green clean / red far / gold near / blue hole")
            for x in ax: x.set_axis_off()
            name = f"{s.dataset}__{s.sensor}__{s.scene}__{s.frame}.png"
            fig.savefig(a.output / name, dpi=70); plt.close(fig)
            lab = out["labels"]; valid = lab != 255
            records.append(dict(sample=name, info=out["info"],
                                label_frac={k: float((lab[valid] == v).mean()) for k, v in (("clean", 0), ("far", 1), ("near", 2), ("hole", 3))}))
            print(name, [ (r.get("type"), r.get("status", ""), round(r.get("area_frac", 0), 3)) for r in out["info"]["regions"]], flush=True)
    write_json(a.output / "preview.json", records)
    receipt.update(stage="complete", samples=len(records)); write_json(a.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
