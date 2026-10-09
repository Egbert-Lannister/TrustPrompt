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
    args=p.parse_args()
    paths=yaml.safe_load(args.paths.read_text(encoding="utf-8"))["mirror3d"]
    source=Path(paths["mat"] or paths["subset_h5"])
    if not source.is_absolute():source=ROOT/source
    config=dict(paths=paths,ids=args.ids,split_file=str(args.split_file),split_sha256=sha256(args.split_file),
                source_sha256=sha256(source),annotation_zip_sha256=sha256(paths["annotation_zip"]),
                metadata_csv_sha256=sha256(paths["metadata_csv"]),seed=20261009)
    receipt=start_run(args.output,config,[sys.executable,"-m","scripts.prepare_mirror3d",*sys.argv[1:]],20261009)
    try:
        loader=Mirror3DNYU(source,paths["annotation_zip"],paths["metadata_csv"],args.split_file)
        rows=[];audits=[]
        for sid in args.ids:
            if loader.mapping[str(sid)]["split"]!="train":raise ValueError("this smoke command is training-only")
            sample,audit=loader.load(sid,regression=True)
            path=args.output/f"{sid}.npz"
            save_sample(sample,path)
            rows.append(dict(path=path.name,sha256=sha256(path),dataset=sample.dataset,scene=sample.scene,
                             frame=sample.frame,split=sample.split,sensor=sample.sensor,source_index=int(sid)))
            audits.append(dict(frame=str(sid),scene=sample.scene,**audit))
            print(f"Prepared REAL train {sample.key}: prompt_valid={sample.prompt_valid.mean():.4f}",flush=True)
        totals={key:dict(count=sum(r[key]["count"] for r in audits),within_1mm=sum(r[key]["within_1mm"] for r in audits)) for key in ("N","N_minus_1")}
        for t in totals.values():t["within_1mm_fraction"]=t["within_1mm"]/t["count"]
        write_json(args.output/"index_regression.json",dict(rows=audits,totals=totals,tolerance_mm=1.0))
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
