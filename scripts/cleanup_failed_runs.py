"""Delete failed / superseded run folders listed below (2026-10-30 review). Dry run unless --yes.

  python -m scripts.cleanup_failed_runs                 # show what would be deleted (tier 1)
  python -m scripts.cleanup_failed_runs --tier 1 2 --yes
  python -m scripts.cleanup_failed_runs --tier 3 --yes  # only the .pt checkpoints of the P1 v2 pre-checks

Tier 1: failed, interrupted or superseded runs whose successful re-run exists; no doc or config cites them.
Tier 2: failed runs that docs/progress.md mentions as records; a note is appended to progress.md when deleted.
Tier 3: model checkpoints (*.pt) of the concluded P1 v2 pre-checks; receipts, configs and metrics stay.
Only the folders named here are touched; every deletion is appended to docs/progress.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

from tp.provenance import ROOT

TIER1 = """
_test_temp
hammer_dev_promptda_vitl_detector_nogate_prior_val_v1_failed_disk_full
hammer_dev_promptda_vitl_detector_prior_val_v1_failed_disk_full
hammer_dev_trustfill_depthor_holdout_sector_train_v1_failed_disk_full
hammer_dev_trustfill_depthor_holdout_sector_val_v1_failed_disk_full
hammer_dev_trustfill_depthor_k0_train_v1_failed_disk_full
hammer_dev_trustfill_depthor_k0_val_v1_failed_disk_full
hammer_dev_trustfill_depthor_nogate_k0_train_v1_failed_disk_full
hammer_dev_trustfill_depthor_nogate_k0_val_v1_failed_disk_full
hammer_dev_trustfill_depthor_nogate_v1_train_v1_failed_disk_full
hammer_dev_trustfill_depthor_nogate_v1_val_v1_failed_disk_full
hammer_dev_trustfill_depthor_v1_train_v1_failed_disk_full
hammer_dev_trustfill_depthor_v1_val_v1_failed_disk_full
hammer_dev_trustfill_oracle_c_gric_val_v1_failed_disk_full
hammer_dev_trustfill_oracle_c_holdout_radial_val_v1_failed_disk_full
hammer_dev_trustfill_oracle_c_holdout_sector_val_v1_failed_disk_full
mirror3d_dev_paired_val_v1_superseded_empty_scene_rule
mirror3d_dev_promptda_vitl_detector_nogate_prior_train_v1_failed_disk_full
mirror3d_dev_promptda_vitl_detector_prior_val_v1_failed_disk_full
mirror3d_dev_robust_global_val_v1_abandoned_code_change
mirror3d_dev_trustfill_depthor_holdout_sector_train_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_holdout_sector_val_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_k0_train_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_k0_val_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_nogate_k0_train_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_nogate_k0_val_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_nogate_v1_train_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_nogate_v1_val_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_v1_train_v1_failed_disk_full
mirror3d_dev_trustfill_depthor_v1_val_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_aic_train_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_aic_val_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_gric_train_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_gric_val_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_holdout_radial_train_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_holdout_radial_val_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_holdout_sector_train_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_holdout_sector_val_v1_failed_disk_full
mirror3d_dev_trustfill_oracle_c_k0_train_v1_abandoned_moved_to_windows
mirror3d_gate_global_smoke_v1
mirror3d_gate_local_v2_smoke_v1
mirror3d_robust_local_v2_smoke_v1
onesided_validity_20261018_134044_interrupted_old_script
onesided_validity_20261018_134407_interrupted_old_script
p1_coupling_20261029_043518
p1_trust_floor_20261029_055723
p1_trust_floor_20261029_060118
layer2_check_20261030_035421
layer2_check_20261030_035510
""".split()
TIER2 = """
hammer_dev_gate_local_v2_val_v1_failed_disk_full
_peek_20261017
nyu_rgbz_bound_dev_v1_killed_background
nyu_rgbz_bound_smoke_v1
""".split()
TIER3_PT = """
p1v2pre_f1_20261025_085441
p1v2pre_f1s_20261025_101533
p1v2long_f1_20261025_222326
p1v2long_f1s_20261027_050647
""".split()


def size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else (p.stat().st_size if p.exists() else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()
    res = ROOT / "results"
    targets = []
    for t, names in ((1, TIER1), (2, TIER2)):
        if t in a.tier:
            targets += [(t, res / n) for n in names if (res / n).is_dir()]
    if 3 in a.tier:
        for n in TIER3_PT:
            targets += [(3, f) for f in sorted((res / n).rglob("*.pt"))]
    total = 0
    for t, p in targets:
        s = size(p)
        total += s
        print(f"tier {t}  {s / 2 ** 20:9.1f} MB  {p.relative_to(ROOT).as_posix()}")
    print(f"{len(targets)} items, {total / 2 ** 30:.2f} GB" + ("" if a.yes else "  (dry run; add --yes to delete)"))
    if not a.yes or not targets:
        return
    done = []
    for t, p in targets:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        done.append(f"tier {t}: {p.relative_to(ROOT).as_posix()}")
    note = (f"\n\n## {datetime.now():%Y-%m-%d} 清理失败或被替代的运行（用户执行 scripts.cleanup_failed_runs）\n\n"
            f"- 删除 {len(done)} 项，共 {total / 2 ** 30:.2f} GB（tier {sorted(a.tier)}）。tier 1 都已有成功的重跑结果，"
            "且没有被文档引用；tier 2 原先作为失败记录在本文件中提到，现已删除，相关描述保留；tier 3 只删 P1 v2 预检的 .pt 权重，"
            "receipt 与指标保留（receipt 中仍有权重的 SHA256）。\n" + "".join(f"  - {x}\n" for x in done))
    prog = ROOT / "docs" / "progress.md"
    prog.write_text(prog.read_text(encoding="utf-8").rstrip("\n") + note, encoding="utf-8")
    print(f"deleted {len(done)} items; recorded in docs/progress.md")


if __name__ == "__main__":
    main()
