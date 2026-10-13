"""Train T v0 (GPU, Windows 4070; CPU works for smoke tests).

  python -m scripts.train_trust --data cache/trust_data_v0_nyu cache/trust_data_v0_hammer \
      --features cache/encoder_vits_nyu_dev_v1 cache/encoder_vits_hammer_dev_v1 \
      --regime real_sim --output results/trust_t_v0_real_sim [--steps 4000]

Training uses the train split only. Model selection (checkpoint and the binary threshold) uses
train-split scenes held out by tp.trust.data.scene_inner_holdout (real prompts only); the val
split is never read here, so val stays a clean development report.
  --selection f1     (v0) checkpoint by mean per-image AUROC, threshold by pooled pixel F1
  --selection error  (v1) checkpoint, threshold and post-processing jointly by the replacement
                     error (tp.trust.data.replacement_error; needs v1 data with gt/focus):
                     mean over datasets of the per-image mean of (scope error + full error) / 2 Regimes: real (real sensor
prompts), sim (simulated variants only), real_sim (both). Datasets are sampled with equal weight.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from tp.provenance import ROOT, sha256, start_run, write_json
from tp.trust.data import scene_inner_holdout

GOOD_DATA = "trust_data_verified"; GOOD_FEAT = "features_verified"


def load_items(data_dirs):
    items, receipts = [], {}
    for d in data_dirs:
        d = Path(d); d = d if d.is_absolute() else ROOT / d
        shards = sorted(p for p in d.glob("shard_*") if p.is_dir() and "_failed" not in p.name)
        if not shards:
            raise ValueError(f"no shards in {d}")
        for s in shards:
            rec = json.loads((s / "receipt.json").read_text(encoding="utf-8"))
            if rec.get("stage") != GOOD_DATA:
                raise ValueError(f"{s} has stage {rec.get('stage')}")
            receipts[str(s)] = sha256(s / "receipt.json")
            for e in json.loads((s / "index.json").read_text(encoding="utf-8")):
                if e["split"] not in ("train", "val"):
                    raise ValueError("unexpected split in trust data")
                items.append(dict(e, file=str(s / e["path"])))
    return items, receipts


def load_features(feat_dirs):
    feats, receipts = {}, {}
    for d in feat_dirs:
        d = Path(d); d = d if d.is_absolute() else ROOT / d
        rec = json.loads((d / "receipt.json").read_text(encoding="utf-8"))
        if rec.get("stage") != GOOD_FEAT:
            raise ValueError(f"{d} has stage {rec.get('stage')}")
        receipts[str(d)] = sha256(d / "receipt.json")
        for h, e in json.loads((d / "index.json").read_text(encoding="utf-8")).items():
            feats[h] = str(d / e["path"])
    return feats, receipts


def batch_arrays(entries, feats, use_features):
    from tp.trust.model import make_inputs
    xs, ys, fs = [], [], []
    for e in entries:
        a = np.load(e["file"])
        xs.append(make_inputs(a["prompt"], a["valid"], a["dg"], a["rgb"])); ys.append(a["label"])
        if use_features:
            fs.append(np.load(feats[e["rgb_sha256"]]).astype(np.float32))
    return np.stack(xs), np.stack(ys), (np.stack(fs) if use_features else None)


def predict_trust(model, entries, feats, use_features, device, bs=8):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(entries), bs):
            x, y, f = batch_arrays(entries[i:i + bs], feats, use_features)
            logit = model(torch.from_numpy(x).to(device), torch.from_numpy(f).to(device) if f is not None else None)
            out += [(1 - torch.sigmoid(l[0]).cpu().numpy(), yy) for l, yy in zip(logit, y)]
    model.train()
    return out


def selection_metrics(preds, entries, thresholds=np.round(np.arange(0.05, 0.96, 0.05), 2)):
    """Per dataset: mean per-image AUROC (clean = positive) and pooled F1 of 'contaminated' for each threshold."""
    from tp.trust.model import auroc
    by = {}
    for (prob, lab), e in zip(preds, entries):
        ok = lab != 255
        d = by.setdefault(e["dataset"], dict(aucs=[], tp=np.zeros(len(thresholds)), fp=np.zeros(len(thresholds)),
                                             fn=np.zeros(len(thresholds))))
        a = auroc(prob[ok], lab[ok] == 0)
        if a is not None:
            d["aucs"].append(a)
        bad = lab[ok] == 1; p = prob[ok]
        for k, t in enumerate(thresholds):
            flag = p < t
            d["tp"][k] += (flag & bad).sum(); d["fp"][k] += (flag & ~bad).sum(); d["fn"][k] += (~flag & bad).sum()
    res = {}
    for ds, d in by.items():
        f1 = 2 * d["tp"] / np.maximum(2 * d["tp"] + d["fp"] + d["fn"], 1)
        res[ds] = dict(auroc=float(np.mean(d["aucs"])) if d["aucs"] else None, images=len(d["aucs"]),
                       f1_by_threshold={f"{t:.2f}": float(v) for t, v in zip(thresholds, f1)})
    aucs = [v["auroc"] for v in res.values() if v["auroc"] is not None]
    mean_f1 = np.mean([[v["f1_by_threshold"][f"{t:.2f}"] for t in thresholds] for v in res.values()], 0)
    best = int(np.argmax(mean_f1))
    return dict(per_dataset=res, score=float(np.mean(aucs)) if aucs else None,
                threshold=float(thresholds[best]), mean_f1_at_threshold=float(mean_f1[best]))


def error_selection(preds, entries, thresholds=np.round(np.arange(0.05, 0.96, 0.05), 2), posts=(False, True)):
    from tp.trust.data import replacement_error
    arrays = [np.load(e["file"]) for e in entries]
    if "gt" not in arrays[0]:
        raise ValueError("--selection error needs v1 trust data (gt, gt_valid, focus)")
    arrays = [{k: a[k] for k in ("prompt", "dg", "gt", "valid", "gt_valid", "focus")} for a in arrays]
    best = None; table = {}
    for post in posts:
        for t in thresholds:
            per = {}
            for (prob, _), e, a in zip(preds, entries, arrays):
                sc, full = replacement_error(prob, t, a, post=post)
                if full is None:
                    continue
                per.setdefault(e["dataset"], []).append(full if sc is None else 0.5 * (sc + full))
            score = float(np.mean([np.mean(v) for v in per.values()]))
            table[f"{'post' if post else 'raw'}@{t:.2f}"] = score
            if best is None or score < best["objective"]:
                best = dict(objective=score, threshold=float(t), post=bool(post),
                            per_dataset={k: float(np.mean(v)) for k, v in per.items()})
    return best, table


def run(a):
    import torch
    from tp.trust.model import TrustNetV0
    out = Path(a.output); out = out if out.is_absolute() else ROOT / out
    config = dict(vars(a), purpose="T v0 training", selection="train-split inner holdout scenes, real prompts only")
    receipt = start_run(out, config, [sys.executable, "-m", "scripts.train_trust", *sys.argv[1:]], a.seed)
    log = (out / "train.log").open("w", encoding="utf-8")
    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    try:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        items, data_receipts = load_items(a.data)
        feats, feat_receipts = load_features(a.features) if not a.no_features else ({}, {})
        train_items = [e for e in items if e["split"] == "train"]
        hold = [e for e in train_items if scene_inner_holdout(e["dataset"], e["scene"])]
        fit = [e for e in train_items if not scene_inner_holdout(e["dataset"], e["scene"])]
        sel = [e for e in hold if e["variant"] == "real"]
        keep = {"real": lambda e: e["variant"] == "real", "sim": lambda e: e["variant"].startswith("sim"),
                "real_sim": lambda e: True}[a.regime]
        fit = [e for e in fit if keep(e)]
        if not fit or not sel:
            raise ValueError("empty fit or selection set")
        by_ds = {}
        for e in fit:
            by_ds.setdefault(e["dataset"], []).append(e)
        pos = sum(e["contaminated"] for e in fit); neg = sum(e["clean"] for e in fit)
        pos_weight = float(np.clip(neg / max(pos, 1), 1.0, 20.0))
        receipt.update(data_receipts_sha256=data_receipts, feature_receipts_sha256=feat_receipts,
                       counts=dict(fit={k: len(v) for k, v in by_ds.items()}, selection=len(sel),
                                   selection_scenes=sorted({(e["dataset"], e["scene"]) for e in sel}),
                                   fit_pixels=dict(clean=neg, contaminated=pos)), pos_weight=pos_weight,
                       device=a.device, torch=torch.__version__,
                       gpu=torch.cuda.get_device_name() if a.device.startswith("cuda") else None)
        write_json(out / "receipt.json", receipt)
        model = TrustNetV0(use_features=not a.no_features, feat_dropout=a.feat_dropout).to(a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=a.device), reduction="none")
        names = sorted(by_ds)
        history, best = [], None
        t0 = time.time()
        for step in range(1, a.steps + 1):
            batch = [by_ds[names[i]][rng.integers(len(by_ds[names[i]]))] for i in rng.integers(len(names), size=a.batch)]
            x, y, f = batch_arrays(batch, feats, not a.no_features)
            if rng.random() < 0.5:                       # horizontal flip of inputs, labels and the feature grid
                x, y = x[..., ::-1].copy(), y[..., ::-1].copy()
                f = f[..., ::-1].copy() if f is not None else None
            x = torch.from_numpy(x).to(a.device); yt = torch.from_numpy(y.astype(np.int64)).to(a.device)
            if f is not None and a.feat_drop_sample > 0:      # v1: drop whole feature maps for some samples
                f = f * (rng.random(len(f)) >= a.feat_drop_sample)[:, None, None, None].astype(np.float32)
            ft = torch.from_numpy(f).to(a.device) if f is not None else None
            logit = model(x, ft)[:, 0]
            mask = (yt != 255).float()
            loss = (lossf(logit, (yt == 1).float()) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 50 == 0:
                say(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
            if step % a.eval_every == 0 or step == a.steps:
                preds = predict_trust(model, sel, feats, not a.no_features, a.device)
                m = selection_metrics(preds, sel)
                h = dict(step=step, loss=float(loss.item()), **{k: m[k] for k in ("score", "threshold", "mean_f1_at_threshold")},
                         per_dataset={k: v["auroc"] for k, v in m["per_dataset"].items()})
                if a.selection == "error":
                    eb, etable = error_selection(preds, sel)
                    h.update(error=eb, error_table=etable)
                    better = best is None or eb["objective"] < best["error"]["objective"]
                    say(f"select step {step}: AUROC {m['score']:.4f} | replacement error {eb['objective']:.4f} "
                        f"thr {eb['threshold']} post {eb['post']} {eb['per_dataset']}")
                else:
                    better = best is None or m["score"] > best["score"]
                    say(f"select step {step}: AUROC {m['score']:.4f} {h['per_dataset']} thr {m['threshold']} F1 {m['mean_f1_at_threshold']:.3f}")
                history.append(h)
                if better:
                    best = dict(step=step, **m, error=h.get("error"))
                    torch.save(model.state_dict(), out / "best.pt")
        write_json(out / "history.json", history)
        receipt.update(stage="trust_model_trained", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       best=dict(step=best["step"], selection_auroc=best["score"], mean_f1=best["mean_f1_at_threshold"],
                                 per_dataset=best["per_dataset"], selection=a.selection,
                                 threshold=best["error"]["threshold"] if a.selection == "error" else best["threshold"],
                                 post=bool(best["error"]["post"]) if a.selection == "error" else False,
                                 objective=best["error"]["objective"] if a.selection == "error" else None,
                                 objective_per_dataset=best["error"]["per_dataset"] if a.selection == "error" else None),
                       checkpoint_sha256=sha256(out / "best.pt"), train_seconds=time.time() - t0,
                       parameters=int(sum(p.numel() for p in model.parameters())))
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); traceback.print_exc(file=log); raise
    finally:
        log.close(); write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--features", nargs="+", default=[])
    ap.add_argument("--regime", choices=["real", "sim", "real_sim"], required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--selection", choices=["f1", "error"], default="f1")
    ap.add_argument("--feat-dropout", type=float, default=0.0); ap.add_argument("--feat-drop-sample", type=float, default=0.0)
    run(ap.parse_args())
