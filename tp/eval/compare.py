"""Compare fixed-cohort runs with paired scene bootstrap, never omit failures."""
import argparse
import json
from pathlib import Path
import sys
from tp.provenance import start_run,write_json,sha256
from .aggregate import aggregate
from .bootstrap import paired_scene_bootstrap


def paired_or_na(ref_entries, cand_entries, field, seed):
    """Scenes whose region is empty (method-independent population, so empty for
    both) are excluded and listed, exactly as the aggregation excludes them. Any
    scene with invalid predictions or a missing value makes the comparison NA:
    failures are never dropped."""
    empty=sorted(s for s in ref_entries if ref_entries[s].get("status")=="empty")
    if any(cand_entries[s].get("status")!="empty" for s in empty):
        raise ValueError("empty-region mismatch between methods (denominators must be method-independent)")
    keep=[s for s in ref_entries if s not in empty]
    rv={s:ref_entries[s].get(field) for s in keep};cv={s:cand_entries[s].get(field) for s in keep}
    if not keep or any(v is None for v in [*rv.values(),*cv.values()]):
        return dict(status="NA_invalid_or_missing_scene" if keep else "NA_all_scenes_empty",
                    excluded_empty_scenes=empty,reference=rv,candidate=cv)
    result=paired_scene_bootstrap(rv,cv,seed=seed)
    result.update(excluded_empty_scenes=empty,interpretation="descriptive development comparison; not confirmatory")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidates",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261009)
    args=p.parse_args()
    config=dict(reference=str(args.reference),candidates=[str(p) for p in args.candidates],
                seed=args.seed,n_resamples=10000,scope="full_and_precise_mirror",stage="development_descriptive_only")
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
                        get=lambda d:{sc:r["metrics"]["regions"][region] for sc,r in d["scenes"].items()}
                        paired[f"{dataset}.{region}.absrel"]=paired_or_na(get(rd),get(cd),"absrel",args.seed)
                    for metric in ("IR","DOR"):
                        get=lambda d:{sc:r["metrics"][metric] for sc,r in d["scenes"].items()}
                        paired[f"{dataset}.{metric}"]=paired_or_na(get(rd),get(cd),"value",args.seed)
                scope_results[path.name]=dict(candidate_per_image_sha256=sha256(path/name),paired=paired,
                    qualification="NYU registered-Z (gap <= 0.473% on dev); inpainted background; development split only")
            results[scope]=scope_results
        write_json(args.output/"paired_bootstrap.json",results)
        receipt.update(stage="descriptive_comparison_complete",formal_experiment_complete=False)
        (args.output/"run.log").write_text("Fixed cohort/sample hashes and method-independent populations checked.\n10,000 paired scene resamples; empty scenes excluded and listed; invalid scenes force NA.\n",encoding="utf-8")
    except Exception as e:
        receipt.update(stage="failed",error=repr(e));raise
    finally:
        write_json(args.output/"receipt.json",receipt)


if __name__=="__main__":main()
