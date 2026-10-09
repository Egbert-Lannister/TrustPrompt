"""Generate reviewable CSV directly from per-image JSON, no manual numbers."""
import argparse
import csv
import json
from pathlib import Path
from .aggregate import aggregate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-image",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    summary=aggregate(json.loads(args.per_image.read_text(encoding="utf-8")))
    columns=["dataset","scope","scene","region","count","invalid_count","status","absrel","rmse","log_rmse","delta_1_05","delta_1_10","delta_1_25"]
    with args.output.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for ds,data in summary["datasets"].items():
            entries=[("dataset","ALL",data["metrics"])]+[("scene",s,r["metrics"]) for s,r in data["scenes"].items()]
            for scope,scene,metrics in entries:
                for region,m in metrics["regions"].items():
                    row=dict(dataset=ds,scope=scope,scene=scene,region=region)
                    row.update({k:("NA" if m[k] is None else m[k]) for k in columns[4:]})
                    w.writerow(row)


if __name__=="__main__":main()
