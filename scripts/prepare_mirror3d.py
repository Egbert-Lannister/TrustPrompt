"""Prepare five real training frames and a fresh numbering regression."""
import argparse
import json
from pathlib import Path
import sys
import yaml
from tp.data.mirror3d import Mirror3DNYU
from tp.data.schema import save_sample
from tp.provenance import ROOT,write_json,sha256,start_run


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--paths",type=Path,default=ROOT/"configs/paths.example.yaml")
    p.add_argument("--split-file",type=Path,default=ROOT/"configs/splits/mirror3d_nyu_v1.json")
    p.add_argument("--output",type=Path,default=ROOT/"data/processed/mirror3d_smoke_v1")
    p.add_argument("--ids",nargs="+",default=["45","184","245","440","654"])
    p.add_argument("--dev",action="store_true",help="all train+val ids present in the source cache; test refused")
    p.add_argument("--formal-test",action="store_true",help="the one formal test read: all test ids of --source (needs the method v1 freeze)")
    p.add_argument("--source",type=Path,default=None,help="source h5/MAT instead of the paths file entry")
    args=p.parse_args()
    formal=None
    if args.formal_test:
        if args.dev:raise ValueError("--formal-test and --dev are exclusive")
        from tp.formal import guard
        formal=guard()
    paths=yaml.safe_load(args.paths.read_text(encoding="utf-8"))["mirror3d"]
    source=Path(args.source or paths["mat"] or paths["subset_h5"])
    if not source.is_absolute():source=ROOT/source
    config=dict(paths=paths,ids=args.ids,split_file=str(args.split_file),split_sha256=sha256(args.split_file),
                source_sha256=sha256(source),annotation_zip_sha256=sha256(paths["annotation_zip"]),
                metadata_csv_sha256=sha256(paths["metadata_csv"]),seed=20261009,formal_test=formal)
    receipt=start_run(args.output,config,[sys.executable,"-m","scripts.prepare_mirror3d",*sys.argv[1:]],20261009)
    try:
        loader=Mirror3DNYU(source,paths["annotation_zip"],paths["metadata_csv"],args.split_file,allow_test=bool(formal))
        allowed={"test"} if formal else ({"train","val"} if args.dev else {"train"})
        has_previous=None
        if args.dev or formal:
            import h5py
            with h5py.File(source,"r") as h:
                args.ids=sorted((k for k in h["frames"]),key=int)
                has_previous={k:"depths_previous" in h["frames"][k] for k in args.ids}
            config["ids"]=args.ids
        rows=[];audits=[]
        for sid in args.ids:
            if loader.mapping[str(sid)]["split"] not in allowed:raise ValueError("test/unapproved split refused")
            regression=True if has_previous is None else has_previous[str(sid)]
            sample,audit=loader.load(sid,regression=regression)
            path=args.output/f"{sid}.npz"
            save_sample(sample,path)
            rows.append(dict(path=path.name,sha256=sha256(path),dataset=sample.dataset,scene=sample.scene,
                             frame=sample.frame,split=sample.split,sensor=sample.sensor,source_index=int(sid)))
            if audit is not None:audits.append(dict(frame=str(sid),scene=sample.scene,split=sample.split,**audit))
            print(f"Prepared REAL {sample.split} {sample.key}: prompt_valid={sample.prompt_valid.mean():.4f}",flush=True)
        totals={key:dict(count=sum(r[key]["count"] for r in audits),within_1mm=sum(r[key]["within_1mm"] for r in audits)) for key in ("N","N_minus_1")}
        for t in totals.values():t["within_1mm_fraction"]=t["within_1mm"]/t["count"]
        write_json(args.output/"index_regression.json",dict(rows=audits,totals=totals,tolerance_mm=1.0,
                   frames_without_regression=[str(s) for s in args.ids if has_previous and not has_previous[str(s)]]))
        if totals["N"]["within_1mm_fraction"]<=.99:raise ValueError("numbering regression failed >99% within 1mm")
        write_json(args.output/"manifest.json",dict(schema="trustprompt.manifest.v1",synthetic=False,records=rows,
                    split_source=config["split_sha256"],seed=20261009,gt_scope="mirror_refined_background_inpainted",
                    strict_rgb_z_qualified=False,source_config=config))
        receipt.update(stage="complete_real_data_preparation",frames=len(rows))
    except Exception as e:
        receipt.update(stage="failed",error=repr(e));raise
    finally:
        write_json(args.output/"receipt.json",receipt)


if __name__=="__main__":main()
