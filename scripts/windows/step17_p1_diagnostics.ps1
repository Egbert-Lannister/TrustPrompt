# Mainline P1 diagnostics after G3 failed (docs/plan_omnidc.md sec. 20). Descriptive; development data only, test never read.
#   CPU  scripts.p1_diagnostics: A readings inside mirrors (real vs synthetic), B where F1's clean error lies,
#        D share of confidence cells labelled noisy (F0 vs F1 label rule)            -> docs\omnidc_p1_diagnostics.md
#   GPU  scripts.omnidc_confidence with the F0 and F1 weights (AUROC of 1 - confidence, mean confidence on correct
#        readings; same populations as D1)                       -> docs\omnidc_confidence_f0.md, docs\omnidc_confidence_f1.md
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step17_p1_diagnostics.ps1 [-SkipGpu]
param([switch]$SkipGpu)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$startTime = Get-Date
Invoke-Step "P1 diagnostics A, B, D (CPU)" { & $py -m scripts.p1_diagnostics }
if (-not $SkipGpu) {
  $Job = Get-Content results\autodl_full_20261024\mirror\job\operator.json -Raw | ConvertFrom-Json
  foreach ($v in @("f0", "f1")) {
    $run = $Job.run_dirs.$v
    $sha = $Job.checkpoints.$v.final.sha256
    if (-not $sha) { Write-Host "No final.pt SHA-256 for $v in the job record."; exit 1 }
    Invoke-Step "input confidence of $v (GPU)" { & $bpy -m scripts.omnidc_confidence --run-dir $run --variant $v --checkpoint-sha256 $sha }
  }
}
Write-Host ("P1 DIAGNOSTICS DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1_diagnostics.md, docs\omnidc_confidence_f0.md, docs\omnidc_confidence_f1.md. Tell Claude." -f ((Get-Date) - $startTime))
