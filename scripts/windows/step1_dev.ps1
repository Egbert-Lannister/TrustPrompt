# Plan revision v2, step 1 — the parts that need the Windows machine (NYU network, E: drive, CUDA).
# Run from D:\Python_Script\TrustPrompt in PowerShell. Every step refuses to overwrite existing outputs.
# Test-split depth is never downloaded or decoded; see docs/plan_revision_v2.md §2.1.
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = ".\.venv\Scripts\python.exe"

# 1a. Official NYU toolbox (31 KB). Hashes of the files used are checked against docs/intrinsics_audit.md.
New-Item -ItemType Directory -Force data\source_cache\nyu_toolbox | Out-Null
if (-not (Test-Path data\source_cache\nyu_toolbox\toolbox_nyu_depth_v2.zip)) {
  Invoke-WebRequest https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip -OutFile data\source_cache\nyu_toolbox\toolbox_nyu_depth_v2.zip
}

# 1c-1. Train+val slices (95 frames) via HTTP Range from the official MAT; capped at 900 MiB.
& $py -m scripts.fetch_nyu_subset --dev --output data/source_cache/nyu_dev_v1.h5 --max-download-mib 900

# 1c-2. Convert to Sample NPZ + numbering regression on frames whose N-1 is not in a test scene.
& $py -m scripts.prepare_mirror3d --dev --paths configs/paths.dev.yaml --output data/processed/mirror3d_dev_v1

# 1c-3. UniDepthV2-S RGB-only cache: 1-frame preflight, then all 95 frames.
& $py -m scripts.cache_unidepth --config configs/unidepth_dev.yaml --limit 1 --output cache/unidepth_dev_preflight_v1
& $py -m scripts.cache_unidepth --config configs/unidepth_dev.yaml

Write-Host "Done. CPU evaluation (run_eval for train/val) can now be run in the Linux workspace or here."
