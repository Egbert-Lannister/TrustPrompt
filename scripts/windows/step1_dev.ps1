# Plan revision v2, step 1 - the parts that need the Windows machine (NYU network, E: drive, CUDA).
# Run from D:\Python_Script\TrustPrompt in PowerShell. Steps refuse to overwrite finished outputs.
# Test-split depth is never downloaded or decoded; see docs/plan_revision_v2.md section 2.1.
# Each step stops the script on failure (native commands do not honour $ErrorActionPreference).
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = ".\.venv\Scripts\python.exe"
function Invoke-Step([string]$Name, [scriptblock]$Body) {
  Write-Host "=== $Name"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Name (exit $LASTEXITCODE). Fix and re-run this script; finished steps are skipped."; exit $LASTEXITCODE }
}

# 1a. Official NYU toolbox (31 KB). Hashes of the files used are checked against docs/intrinsics_audit.md.
New-Item -ItemType Directory -Force data\source_cache\nyu_toolbox | Out-Null
if (-not (Test-Path data\source_cache\nyu_toolbox\toolbox_nyu_depth_v2.zip)) {
  Invoke-WebRequest https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip -OutFile data\source_cache\nyu_toolbox\toolbox_nyu_depth_v2.zip
}

# 1c-1. Train+val slices (95 frames) via HTTP Range from the official MAT.
# An interrupted run leaves an INCOMPLETE cache; --resume keeps hash-verified frames and fetches the rest.
$cacheDone = $false
if (Test-Path data\source_cache\nyu_dev_v1.json) {
  $cacheDone = (Get-Content data\source_cache\nyu_dev_v1.json -Raw | ConvertFrom-Json).status -eq "COMPLETE_REAL_DATA_SUBSET"
}
if (-not $cacheDone) {
  $resume = @(); if (Test-Path data\source_cache\nyu_dev_v1.h5) { $resume = @("--resume") }
  Invoke-Step "fetch NYU train+val" { & $py -m scripts.fetch_nyu_subset --dev @resume --output data/source_cache/nyu_dev_v1.h5 --max-download-mib 1200 --retries 8 }
}

# 1c-2. Convert to Sample NPZ + numbering regression on frames whose N-1 is not in a test scene.
if (-not (Test-Path data\processed\mirror3d_dev_v1\manifest.json)) {
  if (Test-Path data\processed\mirror3d_dev_v1) { Write-Host "data\processed\mirror3d_dev_v1 exists without manifest (failed earlier run). Rename it, e.g. to mirror3d_dev_v1_failed, then re-run."; exit 1 }
  Invoke-Step "prepare Mirror3D dev" { & $py -m scripts.prepare_mirror3d --dev --paths configs/paths.dev.yaml --output data/processed/mirror3d_dev_v1 }
}

# 1c-3. UniDepthV2-S RGB-only cache: 1-frame preflight, then all 95 frames.
function Test-Verified([string]$Dir) {
  if (-not (Test-Path "$Dir\receipt.json")) { return $false }
  $stage = (Get-Content "$Dir\receipt.json" -Raw | ConvertFrom-Json).stage
  if ($stage -eq "real_dev_verified") { return $true }
  Write-Host "$Dir exists with stage '$stage'. Rename it (e.g. append _failed), then re-run."; exit 1
}
if (-not (Test-Verified "cache\unidepth_dev_preflight_v1")) {
  Invoke-Step "UniDepth preflight" { & $py -m scripts.cache_unidepth --config configs/unidepth_dev.yaml --limit 1 --output cache/unidepth_dev_preflight_v1 }
}
if (-not (Test-Verified "cache\unidepth_dev_v1")) {
  Invoke-Step "UniDepth dev cache" { & $py -m scripts.cache_unidepth --config configs/unidepth_dev.yaml }
}
Write-Host "ALL STEPS SUCCEEDED. Tell Claude; CPU evaluation runs next."
