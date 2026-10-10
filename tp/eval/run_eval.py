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
from .metrics import evaluate_depth,trust_quality
from .aggregate import aggregate
from .visualize import alignment_panel


# Plan revision R4/R5: v1 local kept; v2 local and gate variants added.
ALIGNED_SOURCE={"robust_global":"global_depth","robust_local":"local_depth","robust_local_v2":"local_depth_v2",
                "gate_global":"global_depth","gate_local_v2":"local_depth_v2"}
PRIOR_METHODS={"unidepth_v2_s",*ALIGNED_SOURCE}
# Plan revision R2: NYU background reference is inpainted from the same raw prompt.
NYU_SEMANTICS=dict(clean="deviation_from_inpainted_sensor_reference_not_gt_accuracy",
                   hole="not_reported_for_nyu_inpainted_reference",
                   main_regions="precise_mirror_only C/Cfar/band (summary_mirror_only.json)")


def _resume_receipt(output,config,command):
    """Continue an interrupted run only under identical config and source code."""
    receipt=json.loads((output/"receipt.json").read_text(encoding="utf-8"))
    if receipt.get("stage") not in {"partial_time_budget","started"}:
        raise ValueError(f"cannot resume a run in stage {receipt.get('stage')!r}")
    if json.loads((output/"config.json").read_text(encoding="utf-8"))!=json.loads(json.dumps(config)):
        raise ValueError("resume requires the identical configuration")
    current={k:sha256(ROOT/k) for k in receipt["source_sha256"] if (ROOT/k).is_file()}
    if current!=receipt["source_sha256"]:
        raise ValueError("source files changed since the run started; start a new run instead")
    receipt.setdefault("resumed_sessions",[]).append(dict(command=command,at_utc=datetime.now(timezone.utc).isoformat()))
    return receipt


def evaluate_manifest(config, output, command, resume=False, time_budget_s=None):
    if config.get("split")=="test" and config.get("experiment_role")!="formal_main_result":
        raise ValueError("test split is locked until method freeze (formal_main_result only)")
    seed=int(config.get("seed",0));np.random.seed(seed)
    resuming=resume and (output/"receipt.json").is_file()
    receipt=_resume_receipt(output,config,command) if resuming else start_run(output,config,command,seed)
    started=time.perf_counter()
    logs=(output/"run.log").read_text(encoding="utf-8").splitlines() if resuming and (output/"run.log").is_file() else []
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
        elif method in PRIOR_METHODS:
            from baselines.cached_prior import CachedPrior
            from tp.align.robust import align_depth,gate_prompt
            cache_root=Path(config["prior_cache"])
            if not cache_root.is_absolute():cache_root=ROOT/cache_root
            prior=CachedPrior(cache_root)
            receipt["model_version"]=prior.receipt["model_version"]
            receipt["prior_cache_receipt_sha256"]=sha256(cache_root/"receipt.json")
            (output/"alignment").mkdir(exist_ok=resuming)
        else:
            raise ValueError(f"unsupported method {method}")
        manifest=Path(config["manifest"])
        if not manifest.is_absolute():manifest=ROOT/manifest
        data=json.loads(manifest.read_text(encoding="utf-8"))
        records=data["records"];validate_scene_splits(records)
        selected=[r for r in records if r["split"]==config["split"]]
        if not selected:raise ValueError("empty requested split")
        receipt.update(manifest_path=str(manifest),manifest_sha256=sha256(manifest),synthetic=data["synthetic"])
        if resuming:
            if json.loads((output/"split_manifest.json").read_text(encoding="utf-8"))!=data:
                raise ValueError("manifest changed since the run started")
        else:
            write_json(output/"split_manifest.json",data)
        output.joinpath("predictions").mkdir(exist_ok=resuming)
        partial_path=output/"partial_rows.jsonl"
        rows=[];mirror_rows=[]
        if resuming and partial_path.is_file():
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                pair=json.loads(line);rows.append(pair["row"]);mirror_rows.append(pair["mirror_row"])
        done={(x["scene"],x["frame"]) for x in rows}
        for i,r in enumerate(selected):
            if (r["scene"],r["frame"]) in done:continue
            if time_budget_s is not None and time.perf_counter()-started>time_budget_s:
                receipt.update(stage="partial_time_budget",frames_done=len(rows),frames_total=len(selected))
                log(f"PARTIAL: {len(rows)}/{len(selected)} frames; rerun with --resume")
                return None
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
            trust=dict(status="not_provided",reason="no prompt-trust output from this baseline")
            if method=="raw_prompt":
                pred=predict(s)
            else:
                prior_depth=prior.predict(s,r["sha256"])
                if method=="unidepth_v2_s":pred=prior_depth
                else:
                    aligned=align_depth(prior_depth,p,pv,seed=seed,compute_local=ALIGNED_SOURCE[method]!="global_depth",
                                        **config.get("robust",{}))
                    pred=aligned[ALIGNED_SOURCE[method]]
                    if method.startswith("gate_"):
                        pred,w=gate_prompt(p,pv,pred,tau_gate=float(config.get("tau_gate",0.05)))
                        aligned["gate_w"]=w
                        trust=trust_quality(w,p,s.gt,pv,s.gt_valid,tau=config["tau"])
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
                       execution_platform=sys.platform,
                       original_prior_inference_seconds=None if method=="raw_prompt" else prior.record(s)["seconds"],
                       end_to_end_inference_timing_measured=method=="raw_prompt",
                       synthetic=s.metadata["synthetic"],gt_scope=s.metadata["gt_scope"],
                       camera_reference=s.metadata.get("camera_reference","synthetic exact camera Z"),
                       strict_rgb_z_qualified=qualified,
                       metrics=evaluate_depth(pred,s.gt,p,masks),
                       trust_quality=trust,
                       population={name:int(mask.sum()) for name,mask in masks.items()})
            rows.append(row)
            mirror_masks={name:mask&(s.material==2) for name,mask in masks.items()}
            mirror_rows.append({**row,"gt_scope":"precise_mirror_only","metrics":evaluate_depth(pred,s.gt,p,mirror_masks),
                                "population":{name:int(mask.sum()) for name,mask in mirror_masks.items()}})
            with partial_path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(dict(row=rows[-1],mirror_row=mirror_rows[-1]),allow_nan=False)+"\n")
            if i<int(config.get("visualizations",5)):
                alignment_panel(s,p,pred,masks,output/"visualizations"/f"{key}.png")
            log(f"{i+1}/{len(selected)} {s.key} C={row['population']['C']} holes={row['population']['hole']} invalid={row['metrics']['regions']['all']['invalid_count']}")
        write_json(output/"per_image.json",rows)
        write_json(output/"per_image_mirror_only.json",mirror_rows)
        summary=aggregate(rows);mirror_summary=aggregate(mirror_rows)
        qualification=dict(strict_rgb_z_qualified=all(r["strict_rgb_z_qualified"] for r in rows),
                           camera_references=sorted({r["camera_reference"] for r in rows}),
                           experiment_role=config.get("experiment_role"),synthetic=data["synthetic"],
                           main_metric_alignment="none",scientific_claim="none; engineering smoke",
                           metric_semantics=NYU_SEMANTICS if all(r["dataset"]=="mirror3d_nyu" for r in rows) else None)
        summary["qualification"]=qualification;mirror_summary["qualification"]=qualification
        receipt["qualification"]=qualification
        write_json(output/"summary.json",summary)
        write_json(output/"summary_mirror_only.json",mirror_summary)
        write_json(output/"statistics.json",dict(paired_scene_bootstrap="NA: only one method; do not fabricate paired CI",
                    affine_aligned_results="not requested; main metric is unaligned",confidence="binary gate agreement w (per_image trust_quality)" if method.startswith("gate_") else "NA: no confidence output"))
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
    p.add_argument("--method",choices=["raw_prompt",*sorted(PRIOR_METHODS)],default=None)
    p.add_argument("--split",choices=["train","val","test"],default=None)
    p.add_argument("--resume",action="store_true",help="continue an interrupted/time-budgeted run in --output")
    p.add_argument("--time-budget-s",type=float,default=None,help="stop cleanly after this many seconds (resumable)")
    args=p.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.method:config["method"]=args.method
    if args.split:config["split"]=args.split
    evaluate_manifest(config,args.output,[sys.executable,"-m","tp.eval.run_eval",*sys.argv[1:]],
                      resume=args.resume,time_budget_s=args.time_budget_s)


if __name__=="__main__":main()
