# HAMMER development run: materialize subset -> UniDepth cache -> 6 baselines x val/train -> tables.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\hammer_dev.ps1
# Needs ~4 GB free on D: (about 1,100 samples x 3 MB). Test scenes 12-14 are never read.
# Each step stops on failure; finished steps are skipped on re-run.
param([string]$Root = "E:\TrustPrompt_data\hammer_subset_v1")
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = ".\.venv\Scripts\python.exe"
function Invoke-Step([string]$Name, [scriptblock]$Body) {
  Write-Host "=== $Name  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Name (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
function Test-Stage([string]$Receipt, [string[]]$Good) {
  if (-not (Test-Path $Receipt)) { return $false }
  $stage = (Get-Content $Receipt -Raw | ConvertFrom-Json).stage
  if ($Good -contains $stage) { return $true }
  Write-Host "$Receipt has stage '$stage'. Rename that folder (append _failed) and re-run."; exit 1
}
$freeGB = [math]::Floor((Get-PSDrive D).Free / 1GB)
if (-not (Test-Path data\processed\hammer_dev_v1\manifest.json) -and $freeGB -lt 6) { Write-Host "D: has $freeGB GB free; need ~6 GB."; exit 1 }

if (-not (Test-Stage data\processed\hammer_dev_v1\receipt.json @("complete_real_data_preparation"))) {
  Invoke-Step "prepare HAMMER dev subset" { & $py -m scripts.prepare_hammer --root $Root --output data/processed/hammer_dev_v1 --stride-train 20 --stride-val 5 }
}
if (-not (Test-Stage cache\unidepth_hammer_dev_preflight_v1\receipt.json @("real_dev_verified"))) {
  Invoke-Step "UniDepth preflight" { & $py -m scripts.cache_unidepth --config configs/unidepth_hammer_dev.yaml --limit 3 --output cache/unidepth_hammer_dev_preflight_v1 }
}
if (-not (Test-Stage cache\unidepth_hammer_dev_v1\receipt.json @("real_dev_verified"))) {
  Invoke-Step "UniDepth HAMMER dev cache" { & $py -m scripts.cache_unidepth --config configs/unidepth_hammer_dev.yaml }
}
# Local (8x8-tile) alignment costs ~10-20 s per 832x1088 sample on CPU; it runs on val only.
$methods = @{ "val" = @("raw_prompt", "unidepth_v2_s", "robust_global", "gate_global", "robust_local_v2", "gate_local_v2");
              "train" = @("raw_prompt", "unidepth_v2_s", "robust_global", "gate_global") }
foreach ($s in @("val", "train")) {
  foreach ($m in $methods[$s]) {
    $out = "results/hammer_dev_${m}_${s}_v1"
    if (Test-Stage "$out\receipt.json" @("small_scale_validation_passed", "completed_with_invalid_predictions")) { continue }
    $resume = @(); if (Test-Path "$out\receipt.json") { $resume = @("--resume") }
    Invoke-Step "eval $m $s" { & $py -m tp.eval.run_eval --config configs/hammer_dev_eval.yaml --method $m --split $s --output $out @resume }
  }
}
if (-not (Test-Path results\hammer_dev_paired_val_v1\receipt.json)) {
  Invoke-Step "paired val comparison" { & $py -m tp.eval.compare --reference results/hammer_dev_robust_global_val_v1 --candidates results/hammer_dev_gate_global_val_v1 results/hammer_dev_robust_local_v2_val_v1 results/hammer_dev_unidepth_v2_s_val_v1 --output results/hammer_dev_paired_val_v1 }
}
Invoke-Step "tables" { & $py -m scripts.summarize_dev --dataset hammer }
Write-Host "ALL HAMMER DEV STEPS SUCCEEDED ($(Get-Date -Format HH:mm:ss)). Tell Claude."
