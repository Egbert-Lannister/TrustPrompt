"""Compare fixed-cohort runs with paired scene bootstrap, never omit failures."""
import argparse
import json
from pathlib import Path
import sys
from tp.provenance import start_run,write_json,sha256
from .aggregate import aggregate
from .bootstrap import paired_scene_bootstrap


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidates",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261009)
    args=p.parse_args()
    config=dict(reference=str(args.reference),candidates=[str(p) for p in args.candidates],
                seed=args.seed,n_resamples=10000,scope="full_and_precise_mirror",stage="train_smoke_descriptive_only")
    receipt=start_run(args.output,config,[sys.executable,"-m","tp.eval.compare",*sys.argv[1:]],args.seed)
    results={}
    try:
        for scope,name in (("full","per_image.json"),("mirror_only","per_image_mirror_only.json")):
            reference=json.loads((args.reference/name).read_text())
            key=lambda r:(r["dataset"],r["sensor"],r["scene"],r["frame"],r["split"])
            ref_by_key={key(r):r for r in reference}
            ref=aggregate(reference)
            scope_results={}
            for path in args.candidates:
                candidate=json.loads((path/name).read_text())
                if set(map(key,candidate))!=set(ref_by_key):raise ValueError("comparison cohort mismatch")
                for r in candidate:
                    rr=ref_by_key[key(r)]
                    if r["sample_sha256"]!=rr["sample_sha256"] or r["population"]!=rr["population"]:
                        raise ValueError("comparison source/region denominator mismatch")
                cand=aggregate(candidate)
                paired={}
                for dataset,rd in ref["datasets"].items():
                    cd=cand["datasets"][dataset]
                    for region in ("C","Cfar","Cnear","clean","hole","band","all"):
                        vals=lambda d:{s:r["metrics"]["regions"][region]["absrel"] for s,r in d["scenes"].items()}
                        rv,cv=vals(rd),vals(cd)
                        k=f"{dataset}.{region}.absrel"
                        if any(x is None for x in [*rv.values(),*cv.values()]):
                            paired[k]=dict(status="NA_empty_or_invalid_scene",reference=rv,candidate=cv)
                        else:
                            paired[k]=paired_scene_bootstrap(rv,cv,seed=args.seed)
                            paired[k]["interpretation"]="descriptive training smoke; not a confirmatory test"
                    for metric in ("IR","DOR"):
                        vals=lambda d:{s:r["metrics"][metric]["value"] for s,r in d["scenes"].items()}
                        rv,cv=vals(rd),vals(cd)
                        paired[f"{dataset}.{metric}"]=(dict(status="NA_empty_or_invalid_scene",reference=rv,candidate=cv)
                            if any(x is None for x in [*rv.values(),*cv.values()]) else paired_scene_bootstrap(rv,cv,seed=args.seed))
                scope_results[path.name]=dict(candidate_per_image_sha256=sha256(path/name),paired=paired,
                    qualification="NYU registered-Z exploratory; inpainted background; five train scenes")
            results[scope]=scope_results
        write_json(args.output/"paired_bootstrap.json",results)
        receipt.update(stage="descriptive_comparison_complete",formal_experiment_complete=False)
        (args.output/"run.log").write_text("Fixed cohort/sample hashes and method-independent populations checked.\n10,000 paired scene resamples; empty/invalid scenes explicitly NA.\n",encoding="utf-8")
    except Exception as e:
        receipt.update(stage="failed",error=repr(e));raise
    finally:
        write_json(args.output/"receipt.json",receipt)


if __name__=="__main__":main()
