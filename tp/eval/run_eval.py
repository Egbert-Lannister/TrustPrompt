"""Reproducible NPZ-only inference/evaluation, main results never aligned."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import yaml
from tp.data.schema import load_sample,validate_scene_splits
from tp.geom import resize_nearest
from tp.provenance import ROOT,sha256,write_json,start_run
from .regions import regions
from .metrics import evaluate_depth
from .aggregate import aggregate
from .visualize import alignment_panel


def evaluate_manifest(config, output, command):
    seed=int(config.get("seed",0));np.random.seed(seed)
    receipt=start_run(output,config,command,seed)
    started=time.perf_counter()
    logs=[]
    def log(message):
        logs.append(message);print(message,flush=True)
        (output/"run.log").write_text("\n".join(logs)+"\n",encoding="utf-8")
    try:
        if config.get("alignment","none")!="none":
            raise ValueError("main evaluation requires no alignment")
        method=config["method"]
        if method=="raw_prompt":
            from baselines.raw_prompt import predict
            receipt["model_version"]="raw_prompt.native_grid_edt_then_center_nearest.v1; no checkpoint"
        elif method in {"unidepth_v2_s","robust_global","robust_local"}:
            from baselines.cached_prior import CachedPrior
            from tp.align.robust import align_depth
            cache_root=Path(config["prior_cache"])
            if not cache_root.is_absolute():cache_root=ROOT/cache_root
            prior=CachedPrior(cache_root)
            receipt["model_version"]=prior.receipt["model_version"]
            receipt["prior_cache_receipt_sha256"]=sha256(cache_root/"receipt.json")
            (output/"alignment").mkdir()
        else:
            raise ValueError(f"unsupported method {method}")
        manifest=Path(config["manifest"])
        if not manifest.is_absolute():manifest=ROOT/manifest
        data=json.loads(manifest.read_text(encoding="utf-8"))
        records=data["records"];validate_scene_splits(records)
        selected=[r for r in records if r["split"]==config["split"]]
        if not selected:raise ValueError("empty requested split")
        receipt.update(manifest_path=str(manifest),manifest_sha256=sha256(manifest),synthetic=data["synthetic"])
        write_json(output/"split_manifest.json",data)
        output.joinpath("predictions").mkdir()
        rows=[];mirror_rows=[]
        for i,r in enumerate(selected):
            sample_path=(manifest.parent/r["path"]).resolve()
            if sha256(sample_path)!=r["sha256"]:raise ValueError("sample checksum mismatch")
            s=load_sample(sample_path)
            if any(getattr(s,k)!=r[k] for k in ("dataset","scene","frame","split")):
                raise ValueError("sample identity differs from manifest")
            if "sensor" in r and s.sensor!=r["sensor"]:
                raise ValueError("sample sensor differs from manifest")
            if s.metadata["synthetic"]!=data["synthetic"]:raise ValueError("real/synthetic labeling mismatch")
            qualified=s.metadata.get("strict_rgb_z_qualified",s.metadata["synthetic"])
            if config.get("experiment_role")=="formal_main_result" and not qualified:
                raise ValueError("sample is not qualified for strict RGB-axis Z main results")
            p=resize_nearest(s.prompt,s.gt.shape)
            pv=resize_nearest(s.prompt_valid,s.gt.shape)
            masks=regions(p,pv,s.gt,s.gt_valid,tau=config["tau"],band=config["band"])
            t=time.perf_counter()
            if method=="raw_prompt":
                pred=predict(s)
            else:
                prior_depth=prior.predict(s,r["sha256"])
                if method=="unidepth_v2_s":pred=prior_depth
                else:
                    aligned=align_depth(prior_depth,p,pv,seed=seed,**config.get("robust",{}))
                    pred=aligned["global_depth" if method=="robust_global" else "local_depth"]
                    write_json(output/"alignment"/f"{s.sensor}__{s.scene}__{s.frame}.json",aligned["diagnostics"])
                    np.savez_compressed(output/"alignment"/f"{s.sensor}__{s.scene}__{s.frame}.npz",
                                        **{k:v for k,v in aligned.items() if isinstance(v,np.ndarray)})
            elapsed=time.perf_counter()-t
            if pred.shape!=s.gt.shape:raise ValueError("prediction shape mismatch")
            key=f"{s.dataset}__{s.sensor}__{s.scene}__{s.frame}"
            np.save(output/"predictions"/f"{key}.npy",pred,allow_pickle=False)
            row={k:getattr(s,k) for k in ("dataset","scene","frame","split","sensor")}
            row.update(method=method,sample_sha256=r["sha256"],prediction_sha256=sha256(output/"predictions"/f"{key}.npy"),
                       evaluation_prediction_seconds=elapsed,
                       timing_component="raw_prompt_cpu" if method=="raw_prompt" else "cache_load" if method=="unidepth_v2_s" else "cache_load_and_cpu_alignment",
                       original_prior_inference_seconds=None if method=="raw_prompt" else prior.record(s)["seconds"],
                       end_to_end_inference_timing_measured=method=="raw_prompt",
                       synthetic=s.metadata["synthetic"],gt_scope=s.metadata["gt_scope"],
                       camera_reference=s.metadata.get("camera_reference","synthetic exact camera Z"),
                       strict_rgb_z_qualified=qualified,
                       metrics=evaluate_depth(pred,s.gt,p,masks),
                       trust_quality=dict(status="not_provided",reason="no calibrated prompt-trust output from this baseline"),
                       population={name:int(mask.sum()) for name,mask in masks.items()})
            rows.append(row)
            mirror_masks={name:mask&(s.material==2) for name,mask in masks.items()}
            mirror_rows.append({**row,"gt_scope":"precise_mirror_only","metrics":evaluate_depth(pred,s.gt,p,mirror_masks),
                                "population":{name:int(mask.sum()) for name,mask in mirror_masks.items()}})
            if i<int(config.get("visualizations",5)):
                alignment_panel(s,p,pred,masks,output/"visualizations"/f"{key}.png")
            log(f"{i+1}/{len(selected)} {s.key} C={row['population']['C']} holes={row['population']['hole']} invalid={row['metrics']['regions']['all']['invalid_count']}")
        write_json(output/"per_image.json",rows)
        write_json(output/"per_image_mirror_only.json",mirror_rows)
        summary=aggregate(rows);mirror_summary=aggregate(mirror_rows)
        qualification=dict(strict_rgb_z_qualified=all(r["strict_rgb_z_qualified"] for r in rows),
                           camera_references=sorted({r["camera_reference"] for r in rows}),
                           experiment_role=config.get("experiment_role"),synthetic=data["synthetic"],
                           main_metric_alignment="none",scientific_claim="none; engineering smoke")
        summary["qualification"]=qualification;mirror_summary["qualification"]=qualification
        receipt["qualification"]=qualification
        write_json(output/"summary.json",summary)
        write_json(output/"summary_mirror_only.json",mirror_summary)
        write_json(output/"statistics.json",dict(paired_scene_bootstrap="NA: only one method; do not fabricate paired CI",
                    affine_aligned_results="not requested; main metric is unaligned",confidence="NA: no confidence output"))
        invalid_count=sum(r["metrics"]["regions"]["all"]["invalid_count"] for r in rows)
        receipt.update(stage="completed_with_invalid_predictions" if invalid_count else "small_scale_validation_passed",
                       invalid_prediction_count=invalid_count,finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter()-started,frames=len(rows),formal_experiment_complete=False)
        log("COMPLETE: real/synthetic identity recorded, no scale alignment, fixed denominators.")
        return summary
    except Exception as e:
        receipt.update(stage="failed",error=repr(e),elapsed_seconds=time.perf_counter()-started)
        log(traceback.format_exc());raise
    finally:
        write_json(output/"receipt.json",receipt)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--method",choices=["raw_prompt","unidepth_v2_s","robust_global","robust_local"],default=None)
    args=p.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.method:config["method"]=args.method
    evaluate_manifest(config,args.output,[sys.executable,"-m","tp.eval.run_eval",*sys.argv[1:]])


if __name__=="__main__":main()
