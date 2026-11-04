"""The one formal test evaluation of method v1, end to end (docs/plan_test.md). Run on the Windows machine:

  .\\.venv\\Scripts\\python.exe -m scripts.run_test_eval --dry-run        # list the steps and what is already done
  .\\.venv\\Scripts\\python.exe -m scripts.run_test_eval                  # run everything that is not done yet
  .\\.venv\\Scripts\\python.exe -m scripts.run_test_eval --with-capa      # also the secondary CAPA comparison (GPU, ~6 h)

Re-running continues: a step whose output already has its success stage is skipped; a step whose output exists with
another stage is moved aside to <name>_failed_<time> and redone. Every step that reads a test sample first checks the
method v1 freeze (tp.formal.guard: configs/freeze_method_v1.json). Order:
  1  NYU test frames: byte ranges of the official NYU Depth v2 MAT -> data/source_cache/nyu_test_v1.h5
  2a Mirror3D NYU annotations nyu.zip / nyu.csv (official release, SHA256 must equal the dev copy) -> data/source_cache/mirror3d
  2  NYU test samples -> data/processed/mirror3d_test_v1
  3  HAMMER scenes 12-14, every 5th frame, from the official archive (MIT) -> data/source_cache/hammer_test_v1
  4  HAMMER test samples -> data/processed/hammer_test_v1
  5  UniDepthV2-S priors (GPU)          6  T arrays (CPU)          7  T v1 trust maps (GPU)
  8  PromptDA-L raw / + T v1 (crc10) / GT mask (GPU)
  9  PriorDA, OMNI-DC, CDM, LingBot-Depth, Any2Full: raw / + T v1 / GT mask (GPU, .venvs/baselines)
  10 evaluation of every cache (CPU, parallel)    11 Holm family (configs/baselines/holm_family_test_v1.yaml)
  12 tables (docs/test_results.md)               13 conformal guarantee on test (docs/conformal_test.md)
  14 optional: CAPA with and without T v1 (scripts.tta_test)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from tp.provenance import ROOT

WIN = sys.platform.startswith("win")
PY = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
BPY = ROOT / ".venvs" / "baselines" / ("Scripts/python.exe" if WIN else "bin/python")
DATASETS = {
    "nyu": dict(processed="data/processed/mirror3d_test_v1", eval="configs/mirror3d_test_eval.yaml", prefix="mirror3d_test",
                prior="cache/unidepth_nyu_test_v1", trust_data="cache/trust_data_v1_nyu_test",
                trust="cache/trust_t_v1_real_sim_nofeat_nyu_test_v1", lam="0.72", pda="nyu"),
    "hammer": dict(processed="data/processed/hammer_test_v1", eval="configs/hammer_test_eval.yaml", prefix="hammer_test",
                   prior="cache/unidepth_hammer_test_v1", trust_data="cache/trust_data_v1_hammer_test",
                   trust="cache/trust_t_v1_real_sim_nofeat_hammer_test_v1", lam="0.46", pda="hammer"),
}
BASELINES = {"priorda": "PriorDA-B v1.1", "omnidc": "OMNI-DC-v1.1", "cdm": "CDM", "lingbot": "LingBot-Depth v0.5",
             "any2full": "Any2Full-L"}
VARIANTS = ("raw", "tmask", "oracle_mask")
PDA = (("raw", "nyu_vitl_raw", "promptda_vitl_raw", []),
       ("trust", "nyu_vitl_trust_prior", "promptda_vitl_trust_v1_crc10_prior", ["trust"]),
       ("oracle", "nyu_vitl_oracle_prior", "promptda_vitl_oracle_prior", []))
EVAL_OK = ("small_scale_validation_passed", "completed_with_invalid_predictions")


def stage(path):
    p = ROOT / path / "receipt.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("stage")
    except json.JSONDecodeError:
        return "unreadable"


def registry_label(key):
    return json.loads((ROOT / "configs/baselines/registry.json").read_text(encoding="utf-8"))["models"][key]["label"]


def steps(with_capa):
    """(name, interpreter, args, output, success stages); output None = always run (idempotent)."""
    S = []
    nyu_json = ROOT / "data/source_cache/nyu_test_v1.json"
    S.append(("1 NYU test frames", PY, ["-m", "scripts.fetch_nyu_subset", "--formal-test", "--output",
                                        "data/source_cache/nyu_test_v1.h5", "--max-download-mib", "1500"], "nyu_fetch", None))
    S.append(("2a Mirror3D NYU annotations (hash = dev copy)", PY, ["-m", "scripts.fetch_mirror3d_meta"], "mirror3d_meta", None))
    S.append(("2 NYU test samples", PY, ["-m", "scripts.prepare_mirror3d", "--formal-test", "--paths", "configs/paths.dev.yaml",
                                         "--source", "data/source_cache/nyu_test_v1.h5", "--output", DATASETS["nyu"]["processed"]],
              DATASETS["nyu"]["processed"], ("complete_real_data_preparation",)))
    S.append(("3 HAMMER test files", PY, ["-m", "scripts.hammer_remote", "pfetch", "--index", "data/source_cache/hammer_index_v1.json",
                                          "--select", "configs/hammer_test_v1.yaml", "--dest", "data/source_cache/hammer_test_v1",
                                          "--workers", "6", "--max-download-gib", "2"], "hammer_fetch", None))
    S.append(("4 HAMMER test samples", PY, ["-m", "scripts.prepare_hammer", "--formal-test", "--root", "data/source_cache/hammer_test_v1",
                                            "--output", DATASETS["hammer"]["processed"], "--stride-test", "5"],
              DATASETS["hammer"]["processed"], ("complete_real_data_preparation",)))
    for ds, d in DATASETS.items():
        S.append((f"5 UniDepth prior {ds}", PY, ["-m", "scripts.cache_unidepth", "--config", f"configs/unidepth_{ds}_test.yaml"],
                  d["prior"], ("real_dev_verified",)))
    for ds, d in DATASETS.items():
        S.append((f"6 T arrays {ds}", PY, ["-m", "scripts.build_trust_data", "--eval-config", d["eval"], "--output", d["trust_data"],
                                           "--version", "v1", "--formal-test"], d["trust_data"] + "/shard_0of1", ("trust_data_verified",)))
    for ds, d in DATASETS.items():
        S.append((f"7 T v1 trust maps {ds}", PY, ["-m", "scripts.infer_trust", "--model", "results/trust_t_v1_real_sim_nofeat",
                                                  "--data", d["trust_data"], "--output", d["trust"], "--formal-test"],
                  d["trust"], ("trust_cache_verified",)))
    for ds, d in DATASETS.items():
        for tag, cfg, run, extra in PDA:
            sets = ["--set", f"processed={d['processed']}", "--set", f"prior_cache={d['prior']}", "--set", "splits=[test]"]
            if extra:
                sets += ["--set", f"trust_cache={d['trust']}", "--set", f"trust_threshold={d['lam']}"]
            out = f"cache/{run}_{ds}_test_v1"
            S.append((f"8 PromptDA-L {tag} {ds}", PY, ["-m", "scripts.cache_promptda", "--config",
                                                      f"configs/promptda/{cfg.replace('nyu', d['pda'])}.yaml", "--output", out,
                                                      "--formal-test", *sets], out, ("real_dev_verified",)))
    for mk in BASELINES:
        for ds, d in DATASETS.items():
            for v in VARIANTS:
                out = f"cache/bl_{mk}_{v}_{ds}_test_v1"
                extra = ["--trust-cache", d["trust"]] if v == "tmask" else []
                S.append((f"9 {mk} {v} {ds}", BPY, ["-m", "scripts.cache_completion", "--model", mk, "--dataset", ds, "--variant", v,
                                                   "--formal-test", "--processed", d["processed"], "--output", out, *extra],
                          out, ("real_dev_verified",)))
    return S


def eval_jobs():
    J = []
    for ds, d in DATASETS.items():
        runs = [(f"cache/{run}_{ds}_test_v1", run, "PromptDA-L") for _, _, run, _ in PDA]
        runs += [(f"cache/bl_{mk}_{v}_{ds}_test_v1", f"bl_{mk}_{v}", registry_label(mk)) for mk in BASELINES for v in VARIANTS]
        for cache, label, model in runs:
            out = f"results/{d['prefix']}_{label}_test_v1"
            J.append((out, ["-m", "tp.eval.run_eval", "--config", d["eval"], "--method", "cached_prediction", "--split", "test",
                            "--output", out, "--set", f"prediction_cache={cache}", "--set", f"expected_model={model}",
                            "--set", f"method_label={label}"]))
    return J


def move_aside(path):
    p = ROOT / path
    dst = p.with_name(f"{p.name}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.rename(dst)
    print(f"moved {path} aside -> {dst.name}", flush=True)


def hammer_fetch_done():
    """Step 3 is done when the fetch receipt's last run had no error and every selected member is on disk at its size."""
    import yaml
    from scripts.hammer_remote import select_members
    dest = ROOT / "data/source_cache/hammer_test_v1"
    rp = dest / "hammer_fetch_receipt.json"
    if not rp.is_file():
        return False
    r = json.loads(rp.read_text(encoding="utf-8"))
    if not r.get("runs") or r["runs"][-1].get("error") is not None:
        return False
    index = json.loads((ROOT / "data/source_cache/hammer_index_v1.json").read_text(encoding="utf-8"))
    rule = yaml.safe_load((ROOT / "configs/hammer_test_v1.yaml").read_text(encoding="utf-8"))
    for m in select_members(index["members"], rule):
        e, f = r["members"].get(m["name"]), dest / m["name"]
        if e is None or not f.is_file() or f.stat().st_size != e["size"]:
            return False
    return True


def is_done(out, ok):
    if out == "nyu_fetch":
        j = ROOT / "data/source_cache/nyu_test_v1.json"
        return j.is_file() and json.loads(j.read_text(encoding="utf-8")).get("status") == "COMPLETE_REAL_DATA_SUBSET"
    if out == "mirror3d_meta":
        from scripts.fetch_mirror3d_meta import ok as m3d_ok
        return m3d_ok()
    if out == "hammer_fetch":
        return hammer_fetch_done()
    return stage(out) in ok


def run_step(name, interp, args, out, ok, dry):
    if is_done(out, ok):
        print(f"[done] {name}", flush=True)
        return
    if dry:
        print(f"[todo] {name}: {Path(interp).name} {' '.join(args)}", flush=True)
        return
    if out == "nyu_fetch":
        j = ROOT / "data/source_cache/nyu_test_v1.json"
        if j.is_file() and json.loads(j.read_text(encoding="utf-8")).get("status") == "INCOMPLETE":
            args = [*args, "--resume"]
    elif out not in ("hammer_fetch", "mirror3d_meta") and (ROOT / out).exists():
        move_aside(out)
    print(f"=== {name}  ({datetime.now():%H:%M:%S})", flush=True)
    t = time.time()
    rc = subprocess.run([str(interp), *args], cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"FAILED: {name} (exit {rc}). Fix the cause and re-run the same command; finished steps are kept.")
    if not is_done(out, ok):
        raise SystemExit(f"FAILED: {name} finished but its output does not show the success stage {ok}.")
    print(f"    {name}: {time.time() - t:.0f} s", flush=True)


def run_parallel(jobs, n, dry):
    todo = [(o, a) for o, a in jobs if stage(o) not in EVAL_OK]
    print(f"=== 10 evaluation: {len(todo)} of {len(jobs)} to do, {n} at a time", flush=True)
    if dry:
        for o, _ in todo:
            print(f"[todo] eval {o}", flush=True)
        return
    for o, _ in todo:
        if (ROOT / o).exists():
            move_aside(o)
    (ROOT / "results/_logs").mkdir(parents=True, exist_ok=True)
    running = []
    for o, a in todo:
        while len([p for p in running if p[1].poll() is None]) >= n:
            time.sleep(3)
        log = ROOT / "results/_logs" / (o.replace("/", "_") + ".log")
        fh = log.open("w", encoding="utf-8")
        running.append((o, subprocess.Popen([str(PY), *a], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT), fh))
    for o, p, fh in running:
        p.wait(); fh.close()
    bad = [o for o, _ in jobs if stage(o) not in EVAL_OK]
    if bad:
        raise SystemExit("FAILED evaluations (logs in results/_logs): " + ", ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-capa", action="store_true")
    ap.add_argument("--parallel", type=int, default=4, help="evaluations at a time (CPU)")
    ap.add_argument("--min-free-gib", type=float, default=30)
    a = ap.parse_args()
    for interp in (PY, BPY):
        if not interp.is_file():
            raise SystemExit(f"interpreter missing: {interp}")
    import shutil
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    print(f"free disk space at {ROOT}: {free:.0f} GiB (the test caches need about 30 GiB)", flush=True)
    if not a.dry_run:
        if free < a.min_free_gib:
            raise SystemExit(f"only {free:.0f} GiB free; free space or pass --min-free-gib")
        from tp.formal import guard
        print("method v1 freeze verified:", guard(), flush=True)
    started = time.time()
    for s in steps(a.with_capa):
        run_step(*s, a.dry_run)
    run_parallel(eval_jobs(), a.parallel, a.dry_run)
    tail = [("11 Holm family", PY, ["-m", "scripts.holm_family", "--spec", "configs/baselines/holm_family_test_v1.yaml"]),
            ("12 tables", PY, ["-m", "scripts.summarize_test"]),
            ("13 conformal guarantee", PY, ["-m", "scripts.conformal_test_check"])]
    if a.with_capa:
        tail.append(("14 CAPA with and without T v1", BPY, ["-m", "scripts.tta_test"] +
                     (["--resume", "latest"] if sorted((ROOT / "results").glob("tta_test_*")) else [])))
    for name, interp, args in tail:
        if a.dry_run:
            print(f"[todo] {name}: {Path(interp).name} {' '.join(args)}", flush=True)
            continue
        print(f"=== {name}  ({datetime.now():%H:%M:%S})", flush=True)
        if subprocess.run([str(interp), *args], cwd=ROOT).returncode != 0:
            raise SystemExit(f"FAILED: {name}. Re-run the same command; finished steps are kept.")
    if not a.dry_run:
        print(f"TEST EVALUATION DONE ({(time.time() - started) / 3600:.1f} h). Results: docs/holm_test_v1.md, docs/test_results.md, "
              "docs/conformal_test.md" + (", docs/tta_test.md" if a.with_capa else ""), flush=True)


if __name__ == "__main__":
    main()
